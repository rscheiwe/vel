"""Configuration for Extended Thinking."""

from dataclasses import dataclass, asdict
from typing import Literal, Optional, Dict, Any


@dataclass
class ThinkingConfig:
    """
    Configuration for extended thinking mode.

    Extended thinking enables standard models to perform multi-pass reasoning
    through a Reflection Controller pattern (Analyze -> Critique -> Refine -> Conclude).

    Example:
        ```python
        from vel import Agent
        from vel.thinking import ThinkingConfig

        agent = Agent(
            id='deep-thinker',
            model={'provider': 'openai', 'model': 'gpt-4o'},
            thinking=ThinkingConfig(
                mode='reflection',
                max_refinements=3,
                confidence_threshold=0.85
            )
        )
        ```
    """

    mode: Literal['reflection', 'none'] = 'none'
    """Thinking mode: 'reflection' for multi-pass reasoning, 'none' for standard execution."""

    # Display controls
    show_analysis: bool = True
    """Include analysis step content in reasoning events."""

    show_critiques: bool = True
    """Include critique content in reasoning events."""

    show_refinements: bool = True
    """Include refinement content in reasoning events."""

    stream_thinking: bool = True
    """Stream reasoning-delta tokens in real-time (vs batch per step)."""

    # Adaptive iteration controls
    max_refinements: int = 3
    """Maximum refine iterations (1-5). Prevents runaway costs."""

    confidence_threshold: float = 0.8
    """Stop early if confidence >= this (0-1). Default 0.8 = 80%."""

    thinking_timeout: float = 120.0
    """Maximum seconds for entire thinking process."""

    # Tool support
    thinking_tools: bool = True
    """Allow tool calls during thinking steps (analyze, critique, refine). CONCLUDE never uses tools."""

    max_tool_rounds_per_phase: int = 3
    """Maximum tool calls per thinking phase to prevent infinite loops."""

    # Model override
    thinking_model: Optional[Dict[str, Any]] = None
    """
    Optional model config for thinking steps. Use a cheaper/faster model for thinking,
    and reserve the main agent model for final answer.

    Example:
        ThinkingConfig(
            mode='reflection',
            thinking_model={'provider': 'openai', 'model': 'gpt-4o-mini'}
        )
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def __post_init__(self):
        """Validate configuration."""
        if self.max_refinements < 1:
            self.max_refinements = 1
        elif self.max_refinements > 5:
            self.max_refinements = 5

        if self.confidence_threshold < 0:
            self.confidence_threshold = 0
        elif self.confidence_threshold > 1:
            self.confidence_threshold = 1

        if self.thinking_timeout < 10:
            self.thinking_timeout = 10
        elif self.thinking_timeout > 600:
            self.thinking_timeout = 600
