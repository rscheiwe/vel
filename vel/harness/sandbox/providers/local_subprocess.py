from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from vel.harness.config import SandboxConfig
from vel.harness.sandbox.base import ExecResult


class LocalSubprocessSandboxProvider:
    """Explicitly gated local subprocess sandbox for development and tests.

    This is not isolation. It runs commands on the host in a temp directory and
    requires ``provider_options={'unsafe_local': True}`` to avoid accidental use.
    """

    _sessions: Dict[str, LocalSubprocessSandboxSession] = {}
    _roots: Dict[str, str] = {}

    async def create(self, config: SandboxConfig) -> 'LocalSubprocessSandboxSession':
        if not config.provider_options.get('unsafe_local'):
            raise ValueError("local subprocess sandbox requires provider_options.unsafe_local=True")
        root = Path(config.provider_options.get('root') or tempfile.mkdtemp(prefix='vel-sandbox-'))
        root.mkdir(parents=True, exist_ok=True)
        root_key = str(root.resolve())
        if config.lifecycle in ('per_session', 'persistent') and root_key in self._roots:
            return self._sessions[self._roots[root_key]]
        session = LocalSubprocessSandboxSession(
            id=str(uuid.uuid4()),
            root=root,
            workdir=config.workdir,
            env=config.env,
            owns_root='root' not in config.provider_options,
        )
        self._sessions[session.id] = session
        if config.lifecycle in ('per_session', 'persistent'):
            self._roots[root_key] = session.id
        return session

    async def connect(self, sandbox_ref: str) -> 'LocalSubprocessSandboxSession':
        if sandbox_ref not in self._sessions:
            raise KeyError(f"Unknown local sandbox session: {sandbox_ref}")
        return self._sessions[sandbox_ref]


class LocalSubprocessSandboxSession:
    def __init__(
        self,
        *,
        id: str,
        root: Path,
        workdir: str = '/workspace',
        env: Optional[Dict[str, str]] = None,
        owns_root: bool = True,
    ):
        self.id = id
        self.root = root.resolve()
        self.env = env or {}
        self.owns_root = owns_root
        self.workdir = self._resolve_from_root(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)

    async def exec(self, cmd: str, *, timeout: int) -> ExecResult:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            cwd=str(self.workdir),
            env=None if not self.env else {**os.environ, **self.env},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ExecResult(stdout='', stderr=f'timeout after {timeout}s', exit_code=124)
        return ExecResult(
            stdout=stdout.decode('utf-8', errors='replace'),
            stderr=stderr.decode('utf-8', errors='replace'),
            exit_code=proc.returncode or 0,
        )

    async def read_file(self, path: str) -> bytes:
        return self._resolve(path).read_bytes()

    async def write_file(self, path: str, data: bytes) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    async def list_dir(self, path: str) -> List[str]:
        target = self._resolve(path)
        return sorted(child.name for child in target.iterdir())

    async def close(self) -> None:
        if self.owns_root and self.root.exists():
            shutil.rmtree(self.root)
        LocalSubprocessSandboxProvider._sessions.pop(self.id, None)
        for root, session_id in list(LocalSubprocessSandboxProvider._roots.items()):
            if session_id == self.id:
                LocalSubprocessSandboxProvider._roots.pop(root, None)

    def _resolve(self, path: str) -> Path:
        raw = Path(path)
        if raw.is_absolute():
            target = (self.root / Path(*raw.parts[1:])).resolve()
        else:
            target = (self.workdir / raw).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError(f"Path escapes sandbox root: {path}")
        return target

    def _resolve_from_root(self, path: str) -> Path:
        raw = Path(path)
        target = (self.root / Path(*raw.parts[1:])).resolve() if raw.is_absolute() else (self.root / raw).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError(f"Path escapes sandbox root: {path}")
        return target


__all__ = ['LocalSubprocessSandboxProvider', 'LocalSubprocessSandboxSession']
