from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

DEFAULT_MODEL = "LiquidAI/LFM2.5-350M"


@dataclass(frozen=True, slots=True)
class AgentSpec:
    name: str
    system_prompt: str
    tools: tuple[str, ...] = ()
    max_steps: int = 4
    max_new_tokens: int = 256
    temperature: float = 0.3
    force_first_call: bool = False

    def __post_init__(self) -> None:
        if not self.system_prompt.strip():
            raise ValueError(f"agent {self.name}: system_prompt is empty")
        if self.max_steps < 1:
            raise ValueError(f"agent {self.name}: max_steps must be >= 1")
        if self.max_new_tokens < 1:
            raise ValueError(f"agent {self.name}: max_new_tokens must be >= 1")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"agent {self.name}: temperature must be in [0, 2]")
        if self.force_first_call and not self.tools:
            raise ValueError(f"agent {self.name}: force_first_call needs at least one tool")


def load_agents(path: Path) -> dict[str, AgentSpec]:
    """Read configs/agents.yaml into validated specs.

    `defaults` applies to every agent; per-agent keys override it. Unknown keys
    are an error rather than a silently ignored typo.
    """
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = document.get("defaults") or {}
    raw_agents = document.get("agents") or {}
    if not raw_agents:
        raise ValueError(f"no agents defined in {path}")

    allowed = {f.name for f in fields(AgentSpec)} - {"name"}
    specs: dict[str, AgentSpec] = {}
    for name, overrides in raw_agents.items():
        merged = {**defaults, **(overrides or {})}
        unknown = set(merged) - allowed
        if unknown:
            raise ValueError(f"agent {name}: unknown keys {sorted(unknown)}")
        merged["tools"] = tuple(merged.get("tools") or ())
        specs[name] = AgentSpec(name=name, **merged)
    return specs
