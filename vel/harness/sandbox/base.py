from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

from vel.harness.config import SandboxConfig


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int


class SandboxProvider(Protocol):
    async def create(self, config: SandboxConfig) -> 'SandboxSession': ...
    async def connect(self, sandbox_ref: str) -> 'SandboxSession': ...


class SandboxSession(Protocol):
    id: str

    async def exec(self, cmd: str, *, timeout: int) -> ExecResult: ...
    async def read_file(self, path: str) -> bytes: ...
    async def write_file(self, path: str, data: bytes) -> None: ...
    async def list_dir(self, path: str) -> List[str]: ...
    async def close(self) -> None: ...


__all__ = ['ExecResult', 'SandboxProvider', 'SandboxSession']
