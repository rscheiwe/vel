"""
Budget tracking for RLM

Tracks resource usage (steps, tokens, cost) and enforces limits.
"""
from __future__ import annotations
from typing import Any, Dict, Optional


class Budget:
    """
    Budget tracker for RLM execution.

    Tracks steps (tool calls), tokens (prompt + completion), and estimated cost.
    Enforces hard limits and provides exhaustion checks.
    """

    def __init__(
        self,
        max_steps: int,
        max_tokens: int,
        max_cost: float,
        depth: int = 0
    ):
        """
        Initialize budget tracker.

        Args:
            max_steps: Maximum tool call steps
            max_tokens: Maximum total tokens (prompt + completion)
            max_cost: Maximum cost in USD
            depth: Current recursion depth
        """
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.max_cost = max_cost
        self.depth = depth

        # Current counters
        self.steps = 0
        self.tokens = 0
        self.cost = 0.0

        # Detailed tracking
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def bump_step(self):
        """Increment step counter."""
        self.steps += 1

    def bump_tokens(self, prompt: int, completion: int):
        """
        Increment token counters.

        Args:
            prompt: Prompt tokens
            completion: Completion tokens
        """
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.tokens = self.prompt_tokens + self.completion_tokens

    def bump_cost(self, amount: float):
        """
        Increment cost.

        Args:
            amount: Cost in USD
        """
        self.cost += amount

    def bump(self, response: Dict[str, Any]):
        """
        Bump counters from LLM response.

        Extracts usage from response and updates counters.
        Handles OpenAI, Anthropic, and Gemini response formats.

        Args:
            response: LLM response with usage info
        """
        # Extract usage info (provider-agnostic)
        usage = response.get('usage', {})

        if usage:
            prompt = usage.get('prompt_tokens', usage.get('input_tokens', 0))
            completion = usage.get('completion_tokens', usage.get('output_tokens', 0))
            self.bump_tokens(prompt, completion)

        # Estimate cost (rough approximations for common models)
        # In production, use provider-specific pricing
        estimated_cost = self._estimate_cost(self.prompt_tokens, self.completion_tokens)
        self.cost = estimated_cost

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Estimate cost from token counts.

        Uses rough approximations:
        - gpt-4o: $2.50 / 1M prompt, $10 / 1M completion
        - gpt-4o-mini: $0.15 / 1M prompt, $0.60 / 1M completion
        - Average: $1 / 1M prompt, $3 / 1M completion

        Args:
            prompt_tokens: Prompt tokens
            completion_tokens: Completion tokens

        Returns:
            Estimated cost in USD
        """
        # Use conservative mid-range pricing
        prompt_cost_per_1m = 1.0  # $1 / 1M tokens
        completion_cost_per_1m = 3.0  # $3 / 1M tokens

        prompt_cost = (prompt_tokens / 1_000_000) * prompt_cost_per_1m
        completion_cost = (completion_tokens / 1_000_000) * completion_cost_per_1m

        return prompt_cost + completion_cost

    def exhausted(self) -> tuple[bool, Optional[str]]:
        """
        Check if budget is exhausted.

        Returns:
            (is_exhausted, reason)
        """
        if self.steps >= self.max_steps:
            return True, f"steps exhausted ({self.steps}/{self.max_steps})"

        if self.tokens >= self.max_tokens:
            return True, f"tokens exhausted ({self.tokens}/{self.max_tokens})"

        if self.cost >= self.max_cost:
            return True, f"cost exhausted (${self.cost:.4f}/${self.max_cost:.2f})"

        return False, None

    def remaining_steps(self) -> int:
        """Return remaining steps."""
        return max(0, self.max_steps - self.steps)

    def remaining_tokens(self) -> int:
        """Return remaining tokens."""
        return max(0, self.max_tokens - self.tokens)

    def remaining_cost(self) -> float:
        """Return remaining cost budget."""
        return max(0.0, self.max_cost - self.cost)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        is_exhausted, reason = self.exhausted()
        return {
            'depth': self.depth,
            'steps': self.steps,
            'max_steps': self.max_steps,
            'tokens': self.tokens,
            'max_tokens': self.max_tokens,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'cost': round(self.cost, 4),
            'max_cost': self.max_cost,
            'exhausted': is_exhausted,
            'exhausted_reason': reason,
            'remaining_steps': self.remaining_steps(),
            'remaining_tokens': self.remaining_tokens(),
            'remaining_cost': round(self.remaining_cost(), 4)
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Budget(steps={self.steps}/{self.max_steps}, "
            f"tokens={self.tokens}/{self.max_tokens}, "
            f"cost=${self.cost:.4f}/${self.max_cost:.2f})"
        )
