"""Tiny zero-dependency .env loader for the examples.

Loads ``examples/.env`` (git-ignored) into ``os.environ`` without requiring
python-dotenv. Existing environment variables win, so real env always overrides
the file. Import and call :func:`load_env` at the top of an example script.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str | None = None) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (no overwrite).

    Args:
        path: .env file path. Defaults to ``examples/.env`` next to this file.
    """
    env_path = Path(path) if path else Path(__file__).with_name(".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:  # real env vars take precedence
            os.environ[key] = value
