"""SlackChannel — a Slack adapter over RunManager.

Outbound replies go through ``chat.postMessage``; durable approvals are rendered
as Block Kit buttons and resolved asynchronously by the app's interactivity
endpoint (:meth:`SlackChannel.submit_interaction`). Like the FastAPI reference
host, wiring the Slack Events API and Interactivity HTTP routes to this adapter
is the application's job — Vel stays transport-free.

``slack_sdk`` is soft-imported: an absent SDK only errors when a ``SlackChannel``
is constructed, mirroring the lazy sandbox-provider pattern. Install with
``pip install 'vel-ai[channels-slack]'``.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from ..approvals import ApprovalDecision
from .base import Channel


class SlackChannel(Channel):
    """Slack Web API channel. See :meth:`handle_message` / :meth:`submit_interaction`."""

    def __init__(
        self,
        agent: Any,
        *,
        token: Optional[str] = None,
        channel: Optional[str] = None,
        run_manager: Optional[Any] = None,
        harness: Optional[Any] = None,
        session_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            agent, run_manager=run_manager, harness=harness, session_id=session_id
        )
        try:
            from slack_sdk.web.async_client import AsyncWebClient
        except ImportError as exc:  # soft dependency
            raise ImportError(
                "SlackChannel requires slack_sdk — install with "
                "pip install 'vel-ai[channels-slack]'"
            ) from exc
        self._client = AsyncWebClient(token=token or os.environ.get('SLACK_BOT_TOKEN'))
        self._channel = channel

    async def send(self, text: str) -> None:
        if text and self._channel:
            await self._client.chat_postMessage(channel=self._channel, text=text)

    async def handle_message(
        self, *, channel: str, text: str, session_id: Optional[str] = None
    ) -> str:
        """Entry point for a Slack message event → one agent turn.

        Route your Events API handler here. ``session_id`` defaults to the Slack
        channel so a channel is a continuous conversation.
        """
        self._channel = channel
        return await self.handle(text, session_id=session_id or channel)

    async def request_approval(
        self, pending: List[Dict[str, Any]]
    ) -> Optional[List[ApprovalDecision]]:
        # Post interactive approval prompts; decisions arrive later via
        # submit_interaction(). Leaving suspended (return None) is correct here.
        for event in pending:
            data = event.get('data', {}) or {}
            await self._client.chat_postMessage(
                channel=self._channel,
                text=f"Approve tool `{data.get('tool_name')}`?",
                blocks=_approval_blocks(data),
            )
        return None

    async def submit_interaction(self, payload: Dict[str, Any]) -> None:
        """Handle a Block Kit button click → resume the suspended run.

        Route your Interactivity endpoint here with the parsed Slack payload.
        The button ``value`` carries ``{run_id, approval_id, decision}``.
        """
        action = (payload.get('actions') or [{}])[0]
        value = json.loads(action.get('value', '{}'))
        decision = ApprovalDecision(
            approval_id=value['approval_id'], decision=value['decision']
        )
        self._channel = (payload.get('channel') or {}).get('id') or self._channel
        await self.rm.resume(value['run_id'], [decision])
        # Re-pump so the continued run streams back into the channel.
        await self._pump(value['run_id'])


def _approval_blocks(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Block Kit approve/reject buttons encoding the run + approval id."""
    base = {'run_id': data.get('run_id'), 'approval_id': data.get('approval_id')}
    return [
        {
            'type': 'section',
            'text': {'type': 'mrkdwn', 'text': f"Approve `{data.get('tool_name')}`?"},
        },
        {
            'type': 'actions',
            'elements': [
                {
                    'type': 'button',
                    'text': {'type': 'plain_text', 'text': 'Approve'},
                    'style': 'primary',
                    'action_id': 'vel_approve',
                    'value': json.dumps({**base, 'decision': 'approve'}),
                },
                {
                    'type': 'button',
                    'text': {'type': 'plain_text', 'text': 'Reject'},
                    'style': 'danger',
                    'action_id': 'vel_reject',
                    'value': json.dumps({**base, 'decision': 'reject'}),
                },
            ],
        },
    ]


__all__ = ['SlackChannel']
