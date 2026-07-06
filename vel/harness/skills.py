from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from vel.tools import ToolSpec

from .config import HarnessBudgetConfig, SkillRef


@dataclass
class Skill:
    name: str
    instructions: str
    tools: List[ToolSpec] = field(default_factory=list)
    model: Optional[Dict[str, Any]] = None
    budget: Optional[HarnessBudgetConfig] = None
    description: str = ""


class SkillRegistry:
    def __init__(self):
        self._skills: Dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        if name not in self._skills:
            raise KeyError(f"Skill not found: {name}")
        return self._skills[name]

    def has(self, name: str) -> bool:
        return name in self._skills

    def list(self) -> List[str]:
        return sorted(self._skills.keys())


default_registry = SkillRegistry()


def resolve_skills(refs: List[SkillRef], registry: SkillRegistry = default_registry) -> List[Skill]:
    resolved: List[Skill] = []
    for ref in refs:
        resolved.append(ref.skill if ref.skill is not None else registry.get(ref.name))
    return resolved


__all__ = [
    'Skill',
    'SkillRegistry',
    'default_registry',
    'resolve_skills',
]
