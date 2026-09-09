from __future__ import annotations

import gc
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Thread
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from .config import Decoding
from .formats import GENERIC, CallFormat, detect_format, starts_inside_thinking
from .grammar import call_processor

logger = logging.getLogger(__name__)

# Enough tokens to hold the longest stop marker several times over.
STOP_WINDOW_TOKENS = 24


@dataclass(frozen=True, slots=True)
class Generation:
    text: str
    prompt_tokens: int
    generated_tokens: int
    seconds: float


def control_tokens(tokenizer) -> tuple[str, ...]:
    """Every angle-bracket control token this tokenizer knows.

    `all_special_tokens` is not enough: K2-Horizon ends a turn with
    <|ifm|im_end|>, which lives in the added vocabulary instead. Restricting to
    bracketed tokens keeps a large added vocabulary from turning decoding into
    thousands of string replacements.
    """
    tokens = set(tokenizer.all_special_tokens) | set(tokenizer.get_added_vocab())
    return tuple(token for token in tokens if token.startswith("<") and token.endswith(">"))


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class StopOnStrings(StoppingCriteria):
    """Stop generation once a stop string appears in the newly generated text.

    Only the last few tokens are decoded. Decoding everything generated so far
    on every token is quadratic in the answer length, which is invisible on a
    thirty-token reply and real on the code agent's four hundred. A stop string
    is short, so it is always complete inside that window by the time it
    matters. Never looks before the prompt, so a marker already in the
    conversation does not end generation immediately. Assumes batch 1.
    """

    def __init__(self, tokenizer, stop_strings: list[str], prompt_len: int) -> None:
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.prompt_len = prompt_len
        self.window = max(STOP_WINDOW_TOKENS, *(len(stop) for stop in stop_strings))

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
        start = max(self.prompt_len, input_ids.shape[-1] - self.window)
        generated = self.tokenizer.decode(input_ids[0, start:], skip_special_tokens=False)
        return any(stop in generated for stop in self.stop_strings)


def build_prompt(
    tokenizer,
    messages: list[dict[str, str]],
    tools: list[dict] | None,
    prefill: str | None,
) -> str:
    """Render a conversation with the model's own chat template.

    Every engine templates locally rather than letting a server do it: that is
    what keeps the prefill trick, the tool schemas and the stop markers
    identical whichever engine generates the tokens.
    """
    if prefill:
        messages = [*messages, {"role": "assistant", "content": prefill}]
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=not prefill,
        continue_final_message=bool(prefill),
        tokenize=False,
        # Hybrid reasoning models read this; templates without it ignore it.
        enable_thinking=False,
    )


def opens_thinking(tokenizer) -> bool:
    """Whether this model starts every turn inside a reasoning block."""
    probe = build_prompt(tokenizer, [{"role": "user", "content": "hi"}], None, None)
    return starts_inside_thinking(probe)


def strip_control(text: str, control: tuple[str, ...], protected: str) -> str:
    """Drop control tokens the caller does not need to see.

    Tool-call markers and thinking tags are protected, because the parser and
    the answer cleanup both still need them.
    """
    for token in control:
        if token not in protected:
            text = text.replace(token, "")
    return text


@dataclass(slots=True)
class LocalModel:
    model_id: str
    device: str
    tokenizer: Any
    model: Any
    call_format: CallFormat
    control: tuple[str, ...]
    thinks_first: bool

    @classmethod
    def load(
        cls, model_id: str, device: str | None = None, trust_remote_code: bool = False
    ) -> LocalModel:
        """Load a model onto the best available device.

        `trust_remote_code` runs Python published in the model repository. It is
        off unless the model's entry in configs/models.yaml asks for it, and the
        repository is worth reading before you turn it on.
        """
        device = device or pick_device()
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        logger.info("loading model=%s device=%s dtype=%s", model_id, device, dtype)
        if trust_remote_code:
            logger.warning("running custom code from the %s repository", model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, trust_remote_code=trust_remote_code
        ).to(device)
        model.eval()
        call_format = detect_format(tokenizer.get_chat_template() or "")
        logger.info("loaded model=%s tool-call format=%s", model_id, call_format.name)
        if call_format is GENERIC:
            logger.warning(
                "%s has no tool-call markers in its chat template: tool-using agents will be "
                "unreliable with it, and a call cannot be forced. Try /agent chat or /agent code",
                model_id,
            )
        return cls(
            model_id=model_id,
            device=device,
            tokenizer=tokenizer,
            model=model,
            call_format=call_format,
            control=control_tokens(tokenizer),
            thinks_first=opens_thinking(tokenizer),
        )

    def _generate_streaming(self, arguments: dict, on_token: Callable[[str], None]):
        """Run generation on a worker thread, forwarding text as it arrives.

        The streamer only yields decoded text, so the token ids still come from
        the finished call; the thread is joined before they are read.
        """
        # Special tokens stay in the stream so the caller can see a tool-call
        # marker coming; every other control token is dropped here, since the
        # usual post-decode cleanup never runs on streamed text.
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=False)
        noise = tuple(token for token in self.control if token not in self.call_format.protected)
        produced: list = []

        def run() -> None:
            with torch.inference_mode():
                produced.append(self.model.generate(**arguments, streamer=streamer))

        worker = Thread(target=run, daemon=True)
        worker.start()
        for chunk in streamer:
            for token in noise:
                chunk = chunk.replace(token, "")
            if chunk:
                on_token(chunk)
        worker.join()
        return produced[0]

    def complete_batch(self, prompts: list[str], decoding: Decoding | None = None) -> list[str]:
        """Answer many independent prompts in one pass.

        The agent loop wants one turn at a time; a benchmark wants a thousand,
        and running them one at a time leaves the GPU idle between short
        prompts. Padding is on the left because a decoder-only model continues
        from the last position: right-padded rows would be asked to continue
        from padding.

        A model with no chat template gets the bare prompt, which is the only
        thing a base model can be given.
        """
        decoding = decoding or Decoding()
        if decoding.seed is not None:
            torch.manual_seed(decoding.seed)

        template = self.tokenizer.get_chat_template()
        rendered = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            if template
            else prompt
            for prompt in prompts
        ]

        side = self.tokenizer.padding_side
        pad_token = self.tokenizer.pad_token
        self.tokenizer.padding_side = "left"
        if pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            inputs = self.tokenizer(
                rendered, return_tensors="pt", padding=True, add_special_tokens=not template
            ).to(self.device)
        finally:
            self.tokenizer.padding_side = side

        prompt_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                **decoding.as_generate_kwargs(),
            )

        answers = []
        for row in output[:, prompt_len:]:
            text = self.tokenizer.decode(row, skip_special_tokens=False)
            for token in self.control:
                text = text.replace(token, "")
            answers.append(text.strip())
        return answers

    def unload(self) -> None:
        """Drop the weights so a second model can be loaded in the same session."""
        del self.model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        decoding: Decoding | None = None,
        stop: list[str] | None = None,
        prefill: str | None = None,
        on_token: Callable[[str], None] | None = None,
        constrain: bool = False,
    ) -> Generation:
        """Generate one assistant turn.

        `prefill` opens the assistant turn with fixed text the model must
        continue, which is how a small model is made to commit to a tool call
        instead of answering from memory. `on_token` receives text as it is
        produced; generation then runs on a worker thread so this one can drain
        the stream. `constrain` restricts the output to a well-formed call, and
        is only meaningful on a turn that is already committed to making one.
        """
        decoding = decoding or Decoding()
        if decoding.seed is not None:
            torch.manual_seed(decoding.seed)
        if prefill:
            messages = [*messages, {"role": "assistant", "content": prefill}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=not prefill,
            continue_final_message=bool(prefill),
            return_tensors="pt",
            return_dict=True,
            # Hybrid reasoning models read this; templates without it ignore it.
            enable_thinking=False,
        ).to(self.device)
        prompt_len = inputs["input_ids"].shape[-1]

        criteria = None
        if stop:
            criteria = StoppingCriteriaList([StopOnStrings(self.tokenizer, stop, prompt_len)])

        arguments = {
            **inputs,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "stopping_criteria": criteria,
            **decoding.as_generate_kwargs(),
        }
        if constrain:
            processor = call_processor(self.model, self.tokenizer, self.call_format, tools)
            if processor is not None:
                arguments["logits_processor"] = [processor]

        started = time.monotonic()
        if on_token is None:
            with torch.inference_mode():
                output = self.model.generate(**arguments)
        else:
            output = self._generate_streaming(arguments, on_token)
        seconds = time.monotonic() - started

        # Tool-call markers are special tokens, so decoding has to keep them and
        # drop every other special token by hand.
        text = self.tokenizer.decode(output[0, prompt_len:], skip_special_tokens=False)
        # Thinking tags survive here so the answer can have whole blocks removed
        # rather than being left with the reasoning and no tags around it.
        protected = self.call_format.protected
        for token in self.control:
            if token not in protected:
                text = text.replace(token, "")
        return Generation(
            text=((prefill or "") + text).strip(),
            prompt_tokens=prompt_len,
            generated_tokens=int(output.shape[-1] - prompt_len),
            seconds=seconds,
        )
