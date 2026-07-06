from __future__ import annotations

import time
from typing import Any, Dict, Optional

from .config import HarnessBudgetConfig


class BudgetExhausted(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class HarnessBudget:
    def __init__(
        self,
        max_steps: Optional[int] = 100,
        max_tokens: Optional[int] = None,
        max_cost_usd: Optional[float] = None,
        max_wallclock_seconds: Optional[int] = None,
        *,
        started_at: Optional[float] = None,
        steps: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
    ):
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.max_cost_usd = max_cost_usd
        self.max_wallclock_seconds = max_wallclock_seconds
        self.started_at = started_at or time.time()
        self.steps = steps
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.tokens = prompt_tokens + completion_tokens
        self.cost_usd = cost_usd

    @classmethod
    def from_config(
        cls,
        config: Optional[HarnessBudgetConfig],
        restore: Optional[Dict[str, Any]] = None,
    ) -> 'HarnessBudget':
        config = config or HarnessBudgetConfig()
        restore = restore or {}
        return cls(
            max_steps=config.max_steps,
            max_tokens=config.max_tokens,
            max_cost_usd=config.max_cost_usd,
            max_wallclock_seconds=config.max_wallclock_seconds,
            started_at=restore.get('started_at'),
            steps=restore.get('steps', 0),
            prompt_tokens=restore.get('prompt_tokens', 0),
            completion_tokens=restore.get('completion_tokens', 0),
            cost_usd=restore.get('cost_usd', restore.get('cost', 0.0)),
        )

    def bump_step(self) -> None:
        self.steps += 1

    def bump_tokens(self, prompt: int = 0, completion: int = 0) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.tokens = self.prompt_tokens + self.completion_tokens

    def bump_cost(self, amount: float) -> None:
        self.cost_usd += amount

    def bump(self, response_or_usage: Dict[str, Any]) -> None:
        usage = response_or_usage.get('usage', response_or_usage)
        if not isinstance(usage, dict):
            return

        prompt = usage.get(
            'prompt_tokens',
            usage.get('input_tokens', usage.get('promptTokens', usage.get('inputTokens', 0))),
        )
        completion = usage.get(
            'completion_tokens',
            usage.get('output_tokens', usage.get('completionTokens', usage.get('outputTokens', 0))),
        )
        total = usage.get('total_tokens', usage.get('totalTokens'))

        if prompt or completion:
            self.bump_tokens(int(prompt or 0), int(completion or 0))
        elif total:
            self.tokens += int(total)

        cost = usage.get('cost_usd', usage.get('costUsd', usage.get('cost')))
        if cost is not None:
            self.cost_usd += float(cost)
        else:
            self.cost_usd = self._estimate_cost(self.prompt_tokens, self.completion_tokens)

    def check(self) -> None:
        exhausted, reason = self.exhausted()
        if exhausted:
            raise BudgetExhausted(reason or 'budget exhausted')

    def exhausted(self) -> tuple[bool, Optional[str]]:
        if self.max_steps is not None and self.steps >= self.max_steps:
            return True, f"steps exhausted ({self.steps}/{self.max_steps})"
        if self.max_tokens is not None and self.tokens >= self.max_tokens:
            return True, f"tokens exhausted ({self.tokens}/{self.max_tokens})"
        if self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd:
            return True, f"cost exhausted (${self.cost_usd:.4f}/${self.max_cost_usd:.2f})"
        if self.max_wallclock_seconds is not None and self.elapsed_seconds >= self.max_wallclock_seconds:
            return True, f"wallclock exhausted ({self.elapsed_seconds:.1f}s/{self.max_wallclock_seconds}s)"
        return False, None

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, time.time() - self.started_at)

    def remaining_steps(self) -> Optional[int]:
        if self.max_steps is None:
            return None
        return max(0, self.max_steps - self.steps)

    def remaining_tokens(self) -> Optional[int]:
        if self.max_tokens is None:
            return None
        return max(0, self.max_tokens - self.tokens)

    def remaining_cost_usd(self) -> Optional[float]:
        if self.max_cost_usd is None:
            return None
        return max(0.0, self.max_cost_usd - self.cost_usd)

    def remaining_wallclock_seconds(self) -> Optional[float]:
        if self.max_wallclock_seconds is None:
            return None
        return max(0.0, self.max_wallclock_seconds - self.elapsed_seconds)

    def to_event_budget(self) -> Dict[str, Any]:
        return {
            'steps': self.steps,
            'tokens': self.tokens,
            'cost': round(self.cost_usd, 4),
        }

    def to_dict(self) -> Dict[str, Any]:
        is_exhausted, reason = self.exhausted()
        return {
            'started_at': self.started_at,
            'elapsed_seconds': self.elapsed_seconds,
            'steps': self.steps,
            'max_steps': self.max_steps,
            'tokens': self.tokens,
            'max_tokens': self.max_tokens,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'cost_usd': round(self.cost_usd, 6),
            'max_cost_usd': self.max_cost_usd,
            'max_wallclock_seconds': self.max_wallclock_seconds,
            'exhausted': is_exhausted,
            'exhausted_reason': reason,
            'remaining_steps': self.remaining_steps(),
            'remaining_tokens': self.remaining_tokens(),
            'remaining_cost_usd': self.remaining_cost_usd(),
            'remaining_wallclock_seconds': self.remaining_wallclock_seconds(),
        }

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        prompt_cost = (prompt_tokens / 1_000_000) * 1.0
        completion_cost = (completion_tokens / 1_000_000) * 3.0
        return prompt_cost + completion_cost

    def __repr__(self) -> str:
        return (
            f"HarnessBudget(steps={self.steps}/{self.max_steps}, "
            f"tokens={self.tokens}/{self.max_tokens}, "
            f"cost=${self.cost_usd:.4f}/{self.max_cost_usd})"
        )


__all__ = ['BudgetExhausted', 'HarnessBudget']
