"""
RLM Configuration

Configuration dataclass for RLM (Recursive Language Model) middleware.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RlmConfig:
    """
    Configuration for RLM (Recursive Language Model) middleware.

    RLM enables agents to handle very long contexts through recursive reasoning,
    bounded iteration, scratchpad notes, and context probing tools.

    Args:
        enabled: Whether RLM is enabled (default: False)
        depth: Maximum recursion depth for rlm_call (0=no recursion, 1=one level, 2=two levels)
        control_model: Model config for control/planning iterations (cheap, fast model)
            Format: {'provider': 'openai', 'model': 'gpt-4o-mini'} or None to use agent's model
        writer_model: Model config for final synthesis (strong model)
            Format: {'provider': 'openai', 'model': 'gpt-4o'} or None to use control_model
        notes_cap: Maximum number of notes in scratchpad (older notes summarized/dropped)
        notes_window: Number of recent notes to show in context (for budget control)
        budgets: Budget limits
            - max_steps_root: Max tool calls in root execution
            - max_steps_child: Max tool calls in child rlm_call
            - max_tokens_total: Max total tokens across all LLM calls
            - max_cost_usd: Max cost in USD (approximate)
        tools: Tool configuration
            - allow_exec: Enable python_exec tool (default: False, security risk)
            - probe_max_bytes: Max bytes returned by context_probe (default: 4096)
        stream_events: Whether to emit RLM-specific stream events (default: True)
    """
    enabled: bool = False
    depth: int = 1
    control_model: Optional[Dict[str, Any]] = None
    writer_model: Optional[Dict[str, Any]] = None
    notes_cap: int = 200
    notes_window: int = 40
    budgets: Dict[str, Any] = field(default_factory=lambda: {
        'max_steps_root': 12,
        'max_steps_child': 8,
        'max_tokens_total': 120000,
        'max_cost_usd': 0.50
    })
    tools: Dict[str, Any] = field(default_factory=lambda: {
        'allow_exec': False,
        'probe_max_bytes': 4096
    })
    stream_events: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.depth < 0 or self.depth > 2:
            raise ValueError(f"depth must be 0-2, got {self.depth}")

        if self.notes_cap < 1:
            raise ValueError(f"notes_cap must be >= 1, got {self.notes_cap}")

        if self.notes_window < 1:
            raise ValueError(f"notes_window must be >= 1, got {self.notes_window}")

        # Validate budgets
        if 'max_steps_root' not in self.budgets or self.budgets['max_steps_root'] < 1:
            raise ValueError("budgets.max_steps_root must be >= 1")

        if 'max_steps_child' not in self.budgets or self.budgets['max_steps_child'] < 1:
            raise ValueError("budgets.max_steps_child must be >= 1")

        if 'max_tokens_total' not in self.budgets or self.budgets['max_tokens_total'] < 1:
            raise ValueError("budgets.max_tokens_total must be >= 1")

        if 'max_cost_usd' not in self.budgets or self.budgets['max_cost_usd'] <= 0:
            raise ValueError("budgets.max_cost_usd must be > 0")

        # Validate tools
        if 'probe_max_bytes' not in self.tools or self.tools['probe_max_bytes'] < 1:
            raise ValueError("tools.probe_max_bytes must be >= 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'enabled': self.enabled,
            'depth': self.depth,
            'control_model': self.control_model,
            'writer_model': self.writer_model,
            'notes_cap': self.notes_cap,
            'notes_window': self.notes_window,
            'budgets': self.budgets,
            'tools': self.tools,
            'stream_events': self.stream_events
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RlmConfig:
        """Create from dictionary."""
        return cls(**data)
