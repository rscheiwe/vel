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

    routing: Literal['always', 'auto', 'never'] = 'always'
    """When reflection mode is enabled, decide whether to always think, auto-route, or never think."""

    router_model: Optional[Dict[str, Any]] = None
    """Optional model config for the auto-routing classifier."""

    router_confidence_threshold: float = 0.8
    """Minimum router confidence required before auto-routing can select reflection."""

    effort: Literal['low', 'medium', 'high', 'extra', 'max'] = 'high'
    """Requested thinking effort. Higher effort increases refinements/confidence targets."""

    emit_summaries_only: bool = True
    """Prefer UI-safe summary/progress reasoning events rather than raw hidden chain-of-thought."""

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

    convergence_threshold: float = 0.98
    """Stop refining when a refinement is at least this similar (0-1) to the
    previous one — no-progress / convergence detection (2026 convention favors
    verifiable stops over pure self-assessment)."""

    verify: Any = 'none'
    """Termination verifier: 'none' (model self-reported confidence), 'judge'
    (external LLM-as-judge via vel.memory.judge.LLMJudge), or a callable
    ``(question, reasoning) -> float`` in [0,1]. External verification is the
    industry-convention-preferred stop signal over self-assessment."""

    verify_model: Optional[Dict[str, Any]] = None
    """Optional model/JudgeConfig kwargs for the 'judge' verifier."""

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
        if self.routing not in {'always', 'auto', 'never'}:
            self.routing = 'always'

        if self.effort not in {'low', 'medium', 'high', 'extra', 'max'}:
            self.effort = 'high'

        if self.router_confidence_threshold < 0:
            self.router_confidence_threshold = 0
        elif self.router_confidence_threshold > 1:
            self.router_confidence_threshold = 1

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
