from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import requests
from transformers import AutoTokenizer

from .backend import Generation, build_prompt, control_tokens, opens_thinking, strip_control
from .config import Decoding
from .formats import CallFormat, detect_format

logger = logging.getLogger(__name__)

TRANSFORMERS = "transformers"
AUTO = "auto"
ENGINES = (AUTO, TRANSFORMERS, "vllm", "sglang", "tokenspeed", "ollama")

# Fastest first, by the measured throughput in the README. `auto` walks this
# list and takes the first engine that is actually serving the model, which is
# almost always none of them, so the local engine is the usual answer.
PREFERENCE = ("ollama", "vllm", "sglang", "tokenspeed", TRANSFORMERS)
PROBE_TIMEOUT = 2

DEFAULT_URLS = {
    "vllm": "http://127.0.0.1:8000",
    "sglang": "http://127.0.0.1:30000",
    # TokenSpeed also defaults to 8000, so `auto` cannot tell the two apart by
    # probing. They speak the same API and take the same code path, so the only
    # cost is the label; run both and set ROOT_TOKENSPEED_URL to separate them.
    "tokenspeed": "http://127.0.0.1:8000",
    "ollama": "http://127.0.0.1:11434",
}


def default_url(engine: str) -> str:
    """Where an engine is served, overridable per engine from the environment.

    ROOT_OLLAMA_URL and friends are how you point at a server on another port
    or another host without passing a flag through every command.
    """
    return os.environ.get(f"ROOT_{engine.upper()}_URL", DEFAULT_URLS[engine])


REQUEST_TIMEOUT = 600


class EngineUnavailable(RuntimeError):
    """The engine's server is not reachable, or does not have the model."""


@dataclass(slots=True)
class ServedModel:
    """A model generating on another process, templated here.

    vLLM and SGLang speak the OpenAI completions API; Ollama has its own. Both
    are given an already-rendered prompt rather than a message list, so the tool
    schemas, the forced-call prefill and the stop markers behave exactly as they
    do on the local engine.
    """

    model_id: str
    served_as: str
    engine: str
    base_url: str
    tokenizer: Any
    call_format: CallFormat
    control: tuple[str, ...]
    thinks_first: bool

    @classmethod
    def load(
        cls,
        model_id: str,
        engine: str,
        base_url: str | None = None,
        served_as: str | None = None,
    ) -> ServedModel:
        url = (base_url or default_url(engine)).rstrip("/")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        served = served_as or _served_name(model_id, engine, url)
        logger.info("using %s at %s for %s", engine, url, served)
        return cls(
            model_id=model_id,
            served_as=served,
            engine=engine,
            base_url=url,
            tokenizer=tokenizer,
            call_format=detect_format(tokenizer.get_chat_template() or ""),
            control=control_tokens(tokenizer),
            thinks_first=opens_thinking(tokenizer),
        )

    @property
    def device(self) -> str:
        return f"{self.engine} @ {self.base_url}"

    def unload(self) -> None:
        """Nothing to free: the weights live in the server's process."""

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
        """Generate one assistant turn on the server.

        `constrain` is accepted and ignored: the grammar is built against a
        local tokenizer and vocabulary, so it only applies to the in-process
        engine. A served call is parsed and repaired as it always was.
        """
        decoding = decoding or Decoding()
        prompt = build_prompt(self.tokenizer, messages, tools, prefill)
        prompt_tokens = len(self.tokenizer(prompt)["input_ids"])

        started = time.monotonic()
        pieces: list[str] = []
        for chunk in self._stream(prompt, decoding, stop):
            cleaned = strip_control(chunk, self.control, self.call_format.protected)
            pieces.append(cleaned)
            if on_token and cleaned:
                on_token(cleaned)
        seconds = time.monotonic() - started

        text = "".join(pieces)
        return Generation(
            text=((prefill or "") + text).strip(),
            prompt_tokens=prompt_tokens,
            generated_tokens=len(self.tokenizer(text)["input_ids"]) if text else 0,
            seconds=seconds,
        )

    def _stream(self, prompt: str, decoding: Decoding, stop: list[str] | None) -> Iterator[str]:
        if self.engine == "ollama":
            yield from self._stream_ollama(prompt, decoding, stop)
        else:
            yield from self._stream_openai(prompt, decoding, stop)

    def _stream_openai(
        self, prompt: str, decoding: Decoding, stop: list[str] | None
    ) -> Iterator[str]:
        payload: dict[str, Any] = {
            "model": self.served_as,
            "prompt": prompt,
            "max_tokens": decoding.max_new_tokens,
            "temperature": decoding.temperature,
            "stream": True,
        }
        if decoding.samples:
            payload["top_p"] = decoding.top_p
            if decoding.top_k:
                payload["top_k"] = decoding.top_k
            if decoding.min_p:
                payload["min_p"] = decoding.min_p
        if decoding.repetition_penalty != 1.0:
            payload["repetition_penalty"] = decoding.repetition_penalty
        if stop:
            payload["stop"] = stop
        if decoding.seed is not None:
            payload["seed"] = decoding.seed

        response = self._post("/v1/completions", payload)
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            body = line.removeprefix("data: ")
            if body == "[DONE]":
                break
            choices = json.loads(body).get("choices") or [{}]
            yield choices[0].get("text", "")

    def _stream_ollama(
        self, prompt: str, decoding: Decoding, stop: list[str] | None
    ) -> Iterator[str]:
        # `raw` skips Ollama's own template, so the prompt built here is the one
        # the model sees.
        options: dict[str, Any] = {
            "num_predict": decoding.max_new_tokens,
            "temperature": decoding.temperature,
        }
        if decoding.samples:
            options["top_p"] = decoding.top_p
            if decoding.top_k:
                options["top_k"] = decoding.top_k
            if decoding.min_p:
                options["min_p"] = decoding.min_p
        if decoding.repetition_penalty != 1.0:
            options["repeat_penalty"] = decoding.repetition_penalty
        if decoding.seed is not None:
            options["seed"] = decoding.seed
        if stop:
            options["stop"] = stop

        payload = {"model": self.served_as, "prompt": prompt, "raw": True, "options": options}
        response = self._post("/api/generate", payload)
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            body = json.loads(line)
            if body.get("error"):
                raise EngineUnavailable(f"ollama: {body['error']}")
            yield body.get("response", "")
            if body.get("done"):
                break

    def _post(self, path: str, payload: dict) -> requests.Response:
        try:
            response = requests.post(
                f"{self.base_url}{path}", json=payload, stream=True, timeout=REQUEST_TIMEOUT
            )
        except requests.RequestException as exc:
            raise EngineUnavailable(f"{self.engine} at {self.base_url} is not reachable") from exc
        if response.status_code >= 400:
            raise EngineUnavailable(
                f"{self.engine} returned {response.status_code}: {response.text[:200]}"
            )
        return response


def _flatten(name: str) -> str:
    """Compare model names without punctuation getting in the way.

    Both sides have to be flattened the same way. Stripping the dot from one
    side only made `qwen2.5-coder` fail to match `Qwen2.5-Coder-0.5B-Instruct`
    while it was sitting installed.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


def ollama_tag(model_id: str) -> str:
    """Guess the Ollama library tag for a Hugging Face id.

    Ollama names a model `family:size`, so `Qwen/Qwen2.5-Coder-0.5B-Instruct`
    becomes `qwen2.5-coder:0.5b`. A guess, but a cheap one to try before giving
    up, and the pull says exactly what it tried when it is wrong.
    """
    name = model_id.split("/")[-1].lower()
    size = re.search(r"(\d+(?:\.\d+)?[bm])\b", name)
    if not size:
        return name
    family = name[: size.start()].rstrip("-_.")
    for suffix in ("-instruct", "-chat", "-it"):
        family = family.removesuffix(suffix)
    return f"{family}:{size.group(1)}"


def pull_ollama(url: str, reference: str, report: Callable[[str], None] = print) -> bool:
    """Ask Ollama to fetch a model, reporting progress as it goes."""
    report(f"pulling {reference} into ollama")
    try:
        response = requests.post(
            f"{url}/api/pull", json={"model": reference}, stream=True, timeout=REQUEST_TIMEOUT
        )
    except requests.RequestException:
        return False

    last = ""
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        body = json.loads(line)
        if body.get("error"):
            report(f"  {body['error']}")
            return False
        status = body.get("status", "")
        if status != last:
            report(f"  {status}")
            last = status
        if status == "success":
            return True
    return False


def _served_name(model_id: str, engine: str, url: str, allow_pull: bool = True) -> str:
    """What the server calls this model, fetching it if Ollama does not have it.

    vLLM, SGLang and TokenSpeed serve under the Hugging Face id. Ollama uses
    its own short names, so an installed tag is matched first, then the library
    tag is guessed and pulled, then the Hugging Face repository is tried
    directly, which works when it holds GGUF files.
    """
    if engine != "ollama":
        return model_id

    def installed() -> list[str]:
        try:
            found = requests.get(f"{url}/api/tags", timeout=10).json().get("models", [])
        except requests.RequestException as exc:
            raise EngineUnavailable(f"ollama at {url} is not reachable") from exc
        return [entry["name"] for entry in found]

    wanted = _flatten(model_id.split("/")[-1])
    names = installed()
    for name in names:
        if _flatten(name.split(":")[0]) in wanted:
            return name

    tried = []
    if allow_pull:
        for reference in (ollama_tag(model_id), f"hf.co/{model_id}"):
            tried.append(reference)
            if pull_ollama(url, reference):
                for name in installed():
                    if _flatten(name.split(":")[0]) in wanted or name == reference:
                        return name
                return reference

    attempted = f" Tried pulling {tried}." if tried else ""
    raise EngineUnavailable(
        f"ollama has no model matching {model_id}.{attempted} Installed: {names or 'none'}"
    )


def reachable(engine: str, model_id: str, base_url: str | None = None) -> bool:
    """Whether this engine is up and already serving the model.

    Probing has to be quick: it runs before every session that did not name an
    engine, and the common answer is no.
    """
    url = (base_url or default_url(engine)).rstrip("/")
    try:
        if engine == "ollama":
            # Never pull while probing: `auto` asks every engine whether it is
            # already serving the model, and a question should not download
            # two gigabytes or print a failed pull on every startup.
            _served_name(model_id, engine, url, allow_pull=False)
            return True
        response = requests.get(f"{url}/v1/models", timeout=PROBE_TIMEOUT)
        served = {entry["id"] for entry in response.json().get("data", [])}
        return model_id in served
    except (requests.RequestException, EngineUnavailable, KeyError, ValueError):
        return False


def choose_engine(model_id: str, base_url: str | None = None) -> str:
    """The fastest engine that can serve this model right now."""
    for engine in PREFERENCE:
        if engine == TRANSFORMERS:
            return TRANSFORMERS
        if reachable(engine, model_id, base_url):
            logger.info("auto-selected the %s engine", engine)
            return engine
    return TRANSFORMERS


def load_engine(model_id: str, engine: str, device: str | None = None, **kwargs) -> Any:
    """Build whichever engine will do the generating."""
    if engine == AUTO:
        engine = choose_engine(model_id, kwargs.get("base_url"))
    if engine == TRANSFORMERS:
        kwargs.pop("base_url", None)
        from .backend import LocalModel

        return LocalModel.load(model_id, device=device, **kwargs)
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}; try {list(ENGINES)}")
    kwargs.pop("trust_remote_code", None)
    return ServedModel.load(model_id, engine=engine, **kwargs)
