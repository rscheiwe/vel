"""Channel adapter seam — drive a harness run over any transport.

A :class:`Channel` turns an inbound platform message into a harness run via
:class:`~vel.harness.runner.RunManager`, streams the run's events back out
(rendered by :func:`format_event`), and resolves durable approvals inline. This
generalizes the FastAPI reference host (``examples/harness/fastapi_server.py``)
into a reusable base with concrete adapters (CLI, Slack).

Vel stays transport-free: this is a thin, optional convenience over the existing
``RunManager`` API — no new runtime behavior.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..approvals import ApprovalDecision
from ..runner import RunManager


def format_event(event: Dict[str, Any]) -> Optional[str]:
    """Render a single stream event as a human-readable line.

    Returns ``None`` for events a chat surface should ignore. ``text-delta`` is
    intentionally not handled here — the driver buffers deltas into one message.
    """
    etype = event.get('type', '')
    if etype == 'tool-input-available':
        name = event.get('toolName') or event.get('tool_name') or 'tool'
        return f"→ calling `{name}`"
    if etype == 'tool-output-available':
        return "  ✓ tool result"
    data = event.get('data') if isinstance(event.get('data'), dict) else {}
    data = data or {}
    if etype == 'data-harness-approval-required':
        return f"⏸ approval required: `{data.get('tool_name')}`"
    if etype == 'data-harness-suspended':
        return "⏸ run suspended — awaiting approval"
    if etype == 'data-harness-resumed':
        return "▶ resumed"
    if etype == 'data-harness-recovered':
        return f"↻ recovered (skipped {data.get('skipped_tools', 0)} completed tool call(s))"
    if etype == 'data-harness-budget-exhausted':
        return f"■ budget exhausted: {data.get('reason', '')}"
    if etype == 'error':
        return f"⚠ error: {event.get('error') or event.get('errorText') or ''}"
    return None


class Channel(ABC):
    """Transport adapter over a :class:`RunManager`.

    Concrete channels implement :meth:`send` (outbound) and, if the platform
    resolves approvals synchronously, :meth:`request_approval`. The shared driver
    (:meth:`handle` / :meth:`_pump`) starts a run, streams and renders its
    events, and resolves durable approvals before continuing.
    """

    def __init__(
        self,
        agent: Any,
        *,
        run_manager: Optional[RunManager] = None,
        harness: Optional[Any] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.agent = agent
        self.rm = run_manager or RunManager()
        self.harness = harness
        self.session_id = session_id

    @abstractmethod
    async def send(self, text: str) -> None:
        """Deliver an outbound message to the platform."""

    async def request_approval(
        self, pending: List[Dict[str, Any]]
    ) -> Optional[List[ApprovalDecision]]:
        """Resolve pending approval-required events.

        Return a list of decisions to resume immediately, or ``None`` to leave
        the run suspended (for platforms that resolve asynchronously via a
        callback/interaction endpoint). Default: leave suspended.
        """
        return None

    async def handle(self, text: str, *, session_id: Optional[str] = None) -> str:
        """Run one inbound message to completion; returns its run id."""
        sid = session_id or self.session_id
        run_id = await self.rm.start(
            self.agent, input={'message': text}, session_id=sid, harness=self.harness
        )
        await self._pump(run_id)
        return run_id

    async def _pump(self, run_id: str, cursor: int = 0) -> None:
        """Stream a run, render events, and resolve approvals inline."""
        pending: List[Dict[str, Any]] = []
        buffer: List[str] = []

        async def flush() -> None:
            if buffer:
                await self.send(''.join(buffer))
                buffer.clear()

        async for event in self.rm.stream(run_id, cursor=cursor):
            got = event.get('_cursor')
            if got is not None:
                cursor = got
            etype = event.get('type', '')
            if etype == 'text-delta':
                buffer.append(event.get('delta', ''))
                continue
            note = format_event(event)
            if note is not None:
                await flush()
                await self.send(note)
            if etype == 'data-harness-approval-required':
                pending.append(event)
        await flush()

        # If the run suspended for approval and this channel resolves
        # synchronously, decide and continue from where the stream left off.
        if pending and await self.rm.get_status(run_id) == 'suspended':
            decisions = await self.request_approval(pending)
            if decisions:
                await self.rm.resume(run_id, decisions)
                await self._pump(run_id, cursor=cursor)


__all__ = ['Channel', 'format_event']
