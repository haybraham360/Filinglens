"""Load versioned prompts from ``llm/prompts/<name>_v<N>.md``."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib import resources


@dataclass(frozen=True)
class Prompt:
    """A prompt's text and its version id (logged with every call)."""

    name: str
    version: str
    text: str

    @property
    def id(self) -> str:
        """``answer_v1``-style identifier."""
        return f"{self.name}_{self.version}"


@cache
def load_prompt(name: str, version: str = "v1") -> Prompt:
    """Read ``prompts/{name}_{version}.md`` bundled with the package."""
    path = resources.files("filinglens.llm").joinpath("prompts", f"{name}_{version}.md")
    return Prompt(name=name, version=version, text=path.read_text().strip())
