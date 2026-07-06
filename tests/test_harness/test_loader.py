"""Feature 3a: agent-as-directory loader (load_agent)."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from vel import Agent, load_agent


def _write_agent_dir(root: Path) -> Path:
    (root / 'tools').mkdir(parents=True)
    (root / 'skills').mkdir(parents=True)
    (root / 'agent.toml').write_text(textwrap.dedent("""
        [model]
        provider = "openai"
        model = "gpt-4o"

        [agent]
        id = "dir-agent"

        [harness]
        enabled = true

        [harness.budget]
        max_steps = 9

        [harness.approval]
        require_for_tools = ["delete_file"]
    """).strip())
    (root / 'instructions.md').write_text("You are a directory-defined agent.")
    (root / 'tools' / 'echo.py').write_text(textwrap.dedent("""
        from vel import ToolSpec

        async def echo(text: str = "") -> dict:
            return {"echo": text}

        tool = ToolSpec.from_function(echo)
    """).strip())
    (root / 'skills' / 'greeter.py').write_text(textwrap.dedent("""
        from vel.harness import Skill

        skill = Skill(name="greeter", instructions="Always greet warmly.")
    """).strip())
    return root


def test_load_agent_builds_full_agent(tmp_path):
    root = _write_agent_dir(tmp_path / 'my_agent')
    agent = load_agent(root)

    assert isinstance(agent, Agent)
    assert agent.id == 'dir-agent'
    assert 'echo' in agent._tool_names

    hc = agent.harness_config
    assert hc is not None and hc.enabled is True
    assert hc.budget.max_steps == 9
    assert hc.approval.require_for_tools == ['delete_file']
    assert [s.name for s in hc.skills] == ['greeter']
    assert hc.skills[0].skill is not None


def test_missing_toml_raises(tmp_path):
    (tmp_path / 'empty').mkdir()
    with pytest.raises(FileNotFoundError):
        load_agent(tmp_path / 'empty')


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_agent(tmp_path / 'nope')


def test_model_required(tmp_path):
    root = tmp_path / 'bad'
    root.mkdir()
    (root / 'agent.toml').write_text('[agent]\nid = "x"\n')
    with pytest.raises(ValueError):
        load_agent(root)


def test_tool_must_export_symbol(tmp_path):
    root = tmp_path / 'a'
    (root / 'tools').mkdir(parents=True)
    (root / 'agent.toml').write_text('[model]\nmodel = "gpt-4o"\n')
    (root / 'tools' / 'broken.py').write_text('x = 1\n')
    with pytest.raises(ValueError):
        load_agent(root)


def test_directory_without_harness_or_skills(tmp_path):
    root = tmp_path / 'plain'
    (root / 'tools').mkdir(parents=True)
    (root / 'agent.toml').write_text('[model]\nprovider = "openai"\nmodel = "gpt-4o"\n')
    (root / 'tools' / 'echo.py').write_text(
        'from vel import ToolSpec\n'
        'async def echo(text: str = "") -> dict:\n    return {"echo": text}\n'
        'tool = ToolSpec.from_function(echo)\n'
    )
    agent = load_agent(root)
    assert agent.id == 'plain'          # falls back to directory name
    assert agent.harness_config is None  # no [harness], no skills
