"""CLIChannel — a zero-dependency terminal channel over RunManager.

Runs entirely offline (no external credentials): prints streamed output and
resolves durable approvals with a ``y/N`` prompt on stdin. Ideal for trying a
harness agent end-to-end, including the suspend→approve→resume round-trip.
"""
from __future__ import annotations

import asyncio
from typing import Any, List, Optional

from ..approvals import ApprovalDecision
from .base import Channel


class CLIChannel(Channel):
    """Interactive terminal channel. Call :meth:`repl` for a chat loop."""

    def __init__(
        self,
        agent: Any,
        *,
        run_manager: Optional[Any] = None,
        harness: Optional[Any] = None,
        session_id: str = 'cli',
        prompt: str = 'you> ',
    ) -> None:
        super().__init__(
            agent, run_manager=run_manager, harness=harness, session_id=session_id
        )
        self._prompt = prompt

    async def send(self, text: str) -> None:
        if text:
            print(text)

    async def request_approval(self, pending: List[dict]) -> Optional[List[ApprovalDecision]]:
        decisions: List[ApprovalDecision] = []
        for event in pending:
            data = event.get('data', {}) or {}
            answer = (
                await asyncio.to_thread(
                    input, f"Approve `{data.get('tool_name')}`? [y/N] "
                )
            ).strip().lower()
            decisions.append(
                ApprovalDecision(
                    approval_id=data.get('approval_id'),
                    decision='approve' if answer in ('y', 'yes') else 'reject',
                )
            )
        return decisions

    async def repl(self) -> None:
        """Read messages from stdin until EOF or ``/quit``."""
        print("Vel CLI channel — type a message, /quit to exit.")
        while True:
            try:
                line = await asyncio.to_thread(input, self._prompt)
            except (EOFError, KeyboardInterrupt):
                print()
                break
            line = line.strip()
            if not line:
                continue
            if line in ('/quit', '/exit'):
                break
            await self.handle(line)


__all__ = ['CLIChannel']
