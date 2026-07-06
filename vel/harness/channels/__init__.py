"""Channel adapters — drive a harness run over a transport (CLI, Slack, …).

Optional convenience over :class:`~vel.harness.runner.RunManager`. ``SlackChannel``
soft-imports ``slack_sdk`` only when constructed, so importing this package never
requires channel extras.
"""
from __future__ import annotations

from .base import Channel, format_event
from .cli import CLIChannel
from .slack import SlackChannel

__all__ = ['Channel', 'format_event', 'CLIChannel', 'SlackChannel']
