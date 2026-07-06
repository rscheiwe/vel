from __future__ import annotations

from typing import Any, List, Optional

from vel.harness.config import SandboxConfig
from vel.harness.sandbox.base import ExecResult


class E2BSandboxProvider:
    """E2B cloud sandbox adapter.

    The SDK is imported lazily so Vel can expose harness APIs without requiring
    the optional ``e2b`` package unless this provider is selected.
    """

    async def create(self, config: SandboxConfig) -> 'E2BSandboxSession':
        AsyncSandbox = _load_async_sandbox()
        kwargs = dict(config.provider_options)
        if config.image and 'template' not in kwargs and 'template_id' not in kwargs:
            kwargs['template'] = config.image
        if config.env and 'envs' not in kwargs:
            kwargs['envs'] = config.env
        sandbox = await AsyncSandbox.create(**kwargs)
        return E2BSandboxSession(sandbox)

    async def connect(self, sandbox_ref: str) -> 'E2BSandboxSession':
        AsyncSandbox = _load_async_sandbox()
        sandbox = await AsyncSandbox.connect(sandbox_ref)
        return E2BSandboxSession(sandbox)


class E2BSandboxSession:
    def __init__(self, sandbox: Any):
        self._sandbox = sandbox
        self.id = (
            getattr(sandbox, 'sandbox_id', None)
            or getattr(sandbox, 'sandboxId', None)
            or getattr(sandbox, 'id', '')
        )

    async def exec(self, cmd: str, *, timeout: int) -> ExecResult:
        try:
            result = await self._sandbox.commands.run(cmd, timeout=timeout)
        except TypeError:
            result = await self._sandbox.commands.run(cmd, request_timeout=timeout)
        return ExecResult(
            stdout=getattr(result, 'stdout', '') or '',
            stderr=getattr(result, 'stderr', '') or '',
            exit_code=(
                getattr(result, 'exit_code', None)
                if getattr(result, 'exit_code', None) is not None
                else getattr(result, 'exitCode', getattr(result, 'code', 0))
            ),
        )

    async def read_file(self, path: str) -> bytes:
        try:
            data = await self._sandbox.files.read(path, format='bytes')
        except TypeError:
            data = await self._sandbox.files.read(path)
        if isinstance(data, str):
            return data.encode('utf-8')
        return bytes(data)

    async def write_file(self, path: str, data: bytes) -> None:
        await self._sandbox.files.write(path, data)

    async def list_dir(self, path: str) -> List[str]:
        entries = await self._sandbox.files.list(path)
        return sorted(_entry_name(entry) for entry in entries)

    async def close(self) -> None:
        close = getattr(self._sandbox, 'kill', None) or getattr(self._sandbox, 'close', None)
        if close is not None:
            result = close()
            if hasattr(result, '__await__'):
                await result


def _load_async_sandbox():
    try:
        from e2b import AsyncSandbox
    except ImportError as exc:
        raise ImportError(
            "E2B sandbox provider requires the optional 'sandbox-e2b' extra: "
            "pip install 'vel-ai[sandbox-e2b]'"
        ) from exc
    return AsyncSandbox


def _entry_name(entry: Any) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get('name') or entry.get('path') or str(entry)
    return getattr(entry, 'name', getattr(entry, 'path', str(entry)))


__all__ = ['E2BSandboxProvider', 'E2BSandboxSession']
