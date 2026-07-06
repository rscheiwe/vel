"""Load a Vel :class:`~vel.agent.Agent` from an on-disk agent directory.

Filesystem-first authoring (inspired by Vercel eve): an agent is a directory of
conventional files, so a project is inspectable, diffable, and easy to operate.

Layout::

    my_agent/
      agent.toml         # [model], [agent], [harness] (+ nested [harness.*])
      instructions.md    # system prompt (optional)
      tools/*.py         # one tool per file; exports `tool` (ToolSpec or callable)
      skills/*.py        # one skill per file; exports `skill` (a Skill)

``load_agent("my_agent")`` compiles these into an ``Agent`` plus a
``HarnessConfig``. Everything maps onto the existing programmatic API — this is
pure sugar over ``Agent(...)`` and adds no new runtime behavior.
"""
from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .agent import Agent
from .tools.registry import ToolSpec

__all__ = ['load_agent']


def load_agent(path: Union[str, Path]) -> Agent:
    """Build an :class:`Agent` from an agent directory.

    Args:
        path: Path to a directory containing at least ``agent.toml``.

    Returns:
        A fully constructed :class:`Agent` (with a ``HarnessConfig`` when the
        directory declares ``[harness]`` or ships skills).

    Raises:
        FileNotFoundError: If ``path`` is not a directory or ``agent.toml`` is
            missing.
        ValueError: If required config (``[model]``) is absent or a tool/skill
            module does not export the expected symbol.
    """
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"agent directory not found: {root}")

    config_path = root / 'agent.toml'
    if not config_path.is_file():
        raise FileNotFoundError(f"missing agent.toml in {root}")
    with config_path.open('rb') as fh:
        config: Dict[str, Any] = tomllib.load(fh)

    model = config.get('model')
    if not isinstance(model, dict) or not model.get('model'):
        raise ValueError(
            f"{config_path}: [model] must define at least 'model' "
            "(and usually 'provider')"
        )

    agent_cfg: Dict[str, Any] = dict(config.get('agent', {}))
    harness_cfg: Optional[Dict[str, Any]] = config.get('harness')

    instructions = _read_optional_text(root / 'instructions.md')
    tools = _discover_tools(root / 'tools')
    skills = _discover_skills(root / 'skills')

    harness = _build_harness(harness_cfg, skills)

    kwargs: Dict[str, Any] = {
        'id': agent_cfg.pop('id', None) or root.name,
        'model': model,
    }
    if tools:
        kwargs['tools'] = tools
    if instructions:
        kwargs['system_prompt'] = instructions
    if harness is not None:
        kwargs['harness'] = harness
    # Pass through any remaining simple [agent] keys (prompt_env, policies,
    # generation_config, ...) that Agent.__init__ accepts.
    kwargs.update(agent_cfg)

    return Agent(**kwargs)


def _build_harness(harness_cfg: Optional[Dict[str, Any]], skills: List[Any]) -> Optional[Any]:
    """Compose a HarnessConfig from the [harness] table + discovered skills."""
    from .harness import HarnessConfig, SkillRef

    if harness_cfg is None and not skills:
        return None

    cfg_dict: Dict[str, Any] = dict(harness_cfg or {})
    # Skills are a harness feature; if a directory ships skills it implies the
    # harness is on unless the author explicitly disabled it.
    if skills:
        cfg_dict.setdefault('enabled', True)

    harness = HarnessConfig(**cfg_dict)
    for skill in skills:
        harness.skills.append(SkillRef(name=skill.name, skill=skill))
    return harness


def _discover_tools(tools_dir: Path) -> List[ToolSpec]:
    """Load every ``tools/*.py`` module and collect its ``tool`` export."""
    specs: List[ToolSpec] = []
    for module, py in _iter_modules(tools_dir, prefix='vel_agent_tool'):
        export = getattr(module, 'tool', None)
        if export is None:
            export = getattr(module, py.stem, None)
        if export is None:
            raise ValueError(
                f"{py}: expected a module-level `tool` (a ToolSpec or a "
                f"callable) or a callable named '{py.stem}'"
            )
        if isinstance(export, ToolSpec):
            specs.append(export)
        elif callable(export):
            specs.append(ToolSpec.from_function(export))
        else:
            raise ValueError(f"{py}: `tool` must be a ToolSpec or a callable")
    return specs


def _discover_skills(skills_dir: Path) -> List[Any]:
    """Load every ``skills/*.py`` module and collect its ``skill`` export."""
    skills: List[Any] = []
    from .harness import Skill

    for module, py in _iter_modules(skills_dir, prefix='vel_agent_skill'):
        export = getattr(module, 'skill', None)
        if export is None:
            raise ValueError(f"{py}: expected a module-level `skill` (a Skill)")
        if not isinstance(export, Skill):
            raise ValueError(f"{py}: `skill` must be a vel.harness.Skill instance")
        skills.append(export)
    return skills


def _iter_modules(directory: Path, *, prefix: str):
    """Yield ``(module, path)`` for each non-underscore ``*.py`` in a directory."""
    if not directory.is_dir():
        return
    for py in sorted(directory.glob('*.py')):
        if py.name.startswith('_'):
            continue
        yield _import_from_path(py, f"{prefix}_{py.stem}"), py


def _import_from_path(py: Path, module_name: str):
    """Import a standalone .py file under a synthetic, unique module name."""
    spec = importlib.util.spec_from_file_location(module_name, py)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {py}")
    module = importlib.util.module_from_spec(spec)
    # Register so dataclasses / typing in the module resolve correctly.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _read_optional_text(path: Path) -> Optional[str]:
    if path.is_file():
        return path.read_text(encoding='utf-8').strip()
    return None
