from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Iterable, List, Optional

from vel.events import FileEvent
from vel.tools import ToolSpec

from .base import SandboxSession


def build_sandbox_tools(
    session: SandboxSession,
    tools: Iterable[str],
    *,
    timeout_seconds: int = 300,
) -> List[ToolSpec]:
    requested = set(tools)
    specs: List[ToolSpec] = []

    if 'read' in requested:
        specs.append(_read_tool(session))
    if 'write' in requested:
        specs.append(_write_tool(session))
    if 'edit' in requested:
        specs.append(_edit_tool(session))
    if 'list' in requested:
        specs.append(_list_tool(session))
    if 'bash' in requested:
        specs.append(_bash_tool(session, timeout_seconds))
    if 'python' in requested:
        specs.append(_python_tool(session, timeout_seconds))

    return specs


def _read_tool(session: SandboxSession) -> ToolSpec:
    async def sandbox_read(
        path: str,
        expose_file: bool = False,
        name: Optional[str] = None,
        mime_type: Optional[str] = None,
    ):
        """Read a file from the sandbox."""
        data = await session.read_file(path)
        encoded = base64.b64encode(data).decode('ascii')
        if expose_file:
            yield FileEvent(
                content=encoded,
                name=name or Path(path).name,
                mime_type=mime_type or mimetypes.guess_type(path)[0] or 'application/octet-stream',
            ).to_dict()
        try:
            output = {'path': path, 'content': data.decode('utf-8'), 'encoding': 'utf-8'}
        except UnicodeDecodeError:
            output = {
                'path': path,
                'content': encoded,
                'encoding': 'base64',
            }
        yield {'type': 'tool-output', 'output': output}

    return ToolSpec.from_function(
        sandbox_read,
        name='sandbox_read',
        category='sandbox',
        tags=['sandbox', 'filesystem', 'read'],
    )


def _write_tool(session: SandboxSession) -> ToolSpec:
    async def sandbox_write(path: str, content: str, encoding: str = 'utf-8') -> dict:
        """Write a file in the sandbox."""
        data = _decode_content(content, encoding)
        await session.write_file(path, data)
        return {'path': path, 'bytes': len(data)}

    return ToolSpec.from_function(
        sandbox_write,
        name='sandbox_write',
        category='sandbox',
        tags=['sandbox', 'filesystem', 'write'],
        requires_confirmation=True,
    )


def _edit_tool(session: SandboxSession) -> ToolSpec:
    async def sandbox_edit(path: str, old: str, new: str, count: int = 1) -> dict:
        """Replace text in a sandbox file."""
        data = await session.read_file(path)
        text = data.decode('utf-8')
        if old not in text:
            return {'path': path, 'replacements': 0, 'changed': False}
        replacements = text.count(old) if count < 0 else min(text.count(old), count)
        updated = text.replace(old, new, count if count >= 0 else -1)
        await session.write_file(path, updated.encode('utf-8'))
        return {'path': path, 'replacements': replacements, 'changed': replacements > 0}

    return ToolSpec.from_function(
        sandbox_edit,
        name='sandbox_edit',
        category='sandbox',
        tags=['sandbox', 'filesystem', 'edit'],
        requires_confirmation=True,
    )


def _list_tool(session: SandboxSession) -> ToolSpec:
    async def sandbox_list(path: str = '.') -> dict:
        """List a directory in the sandbox."""
        return {'path': path, 'entries': await session.list_dir(path)}

    return ToolSpec.from_function(
        sandbox_list,
        name='sandbox_list',
        category='sandbox',
        tags=['sandbox', 'filesystem', 'list'],
    )


def _bash_tool(session: SandboxSession, default_timeout: int) -> ToolSpec:
    async def sandbox_bash(cmd: str, timeout: Optional[int] = None) -> dict:
        """Execute a shell command in the sandbox."""
        result = await session.exec(cmd, timeout=timeout or default_timeout)
        return {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'exit_code': result.exit_code,
        }

    return ToolSpec.from_function(
        sandbox_bash,
        name='sandbox_bash',
        category='sandbox',
        tags=['sandbox', 'exec', 'bash'],
        requires_confirmation=True,
    )


def _python_tool(session: SandboxSession, default_timeout: int) -> ToolSpec:
    async def sandbox_python(code: str, timeout: Optional[int] = None) -> dict:
        """Execute Python code in the sandbox."""
        script_path = '.vel_sandbox_script.py'
        await session.write_file(script_path, code.encode('utf-8'))
        result = await session.exec(f'python3 {script_path}', timeout=timeout or default_timeout)
        return {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'exit_code': result.exit_code,
        }

    return ToolSpec.from_function(
        sandbox_python,
        name='sandbox_python',
        category='sandbox',
        tags=['sandbox', 'exec', 'python'],
        requires_confirmation=True,
    )


def _decode_content(content: str, encoding: str) -> bytes:
    if encoding == 'utf-8':
        return content.encode('utf-8')
    if encoding == 'base64':
        return base64.b64decode(content)
    raise ValueError(f"Unsupported encoding: {encoding}")


__all__ = ['build_sandbox_tools']
