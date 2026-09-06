from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from .formats import CallFormat, detect_format

logger = logging.getLogger(__name__)


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

    Decodes only the tokens past the prompt, so a stop string already present in
    the conversation does not end generation on the first step. Assumes batch 1.
    """

    def __init__(self, tokenizer, stop_strings: list[str], prompt_len: int) -> None:
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
        generated = self.tokenizer.decode(
            input_ids[0, self.prompt_len :], skip_special_tokens=False
        )
        return any(stop in generated for stop in self.stop_strings)


@dataclass(slots=True)
class LocalModel:
    model_id: str
    device: str
    tokenizer: Any
    model: Any
    call_format: CallFormat
    control: tuple[str, ...]

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
        return cls(
            model_id=model_id,
            device=device,
            tokenizer=tokenizer,
            model=model,
            call_format=call_format,
            control=control_tokens(tokenizer),
        )

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
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        stop: list[str] | None = None,
        prefill: str | None = None,
    ) -> Generation:
        """Generate one assistant turn.

        `prefill` opens the assistant turn with fixed text the model must
        continue, which is how a small model is made to commit to a tool call
        instead of answering from memory.
        """
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

        sampling = {"do_sample": False}
        if temperature > 0:
            sampling = {"do_sample": True, "temperature": temperature, "top_p": 0.9}

        criteria = None
        if stop:
            criteria = StoppingCriteriaList([StopOnStrings(self.tokenizer, stop, prompt_len)])

        started = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                stopping_criteria=criteria,
                **sampling,
            )
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
