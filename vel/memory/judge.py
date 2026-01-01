# vel/memory/judge.py
"""
LLM-as-Judge: Automatic evaluation of trajectory success/failure.

Part of ReasoningBank Phase 2 - evaluates agent trajectories to:
1. Determine success/failure
2. Extract failure patterns for anti-patterns
3. Provide confidence scores for downstream learning
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Literal, Callable
from enum import Enum
import json
import re
from time import time
import asyncio


class JudgeOutcome(str, Enum):
    """Binary outcome following ReasoningBank paper convention."""
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"  # Evaluation itself failed


@dataclass
class JudgeResult:
    """
    Result of LLM-as-Judge evaluation for a single trajectory.

    Follows ReasoningBank paper's binary classification approach
    with additional metadata for confidence and failure analysis.
    """
    trajectory_id: str
    outcome: JudgeOutcome
    confidence: float = 0.5  # Model's confidence in judgment (0.0-1.0)
    failure_notes: List[str] = field(default_factory=list)  # Anti-patterns
    reasoning: str = ""  # LLM's explanation
    usage: Optional[Dict[str, int]] = None  # Token usage for cost tracking
    model: str = ""  # Model used
    latency_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage/transmission."""
        return {
            "trajectory_id": self.trajectory_id,
            "outcome": self.outcome.value,
            "confidence": self.confidence,
            "failure_notes": self.failure_notes,
            "reasoning": self.reasoning,
            "usage": self.usage,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "error": self.error
        }


@dataclass
class JudgeConfig:
    """Configuration for LLM-as-Judge evaluator."""
    provider: str = "openai"
    model: Optional[str] = None  # Will use provider-specific default
    api_key: Optional[str] = None  # If None, uses env var
    temperature: float = 0.0  # Deterministic per ReasoningBank paper
    max_tokens: int = 1024
    timeout: float = 60.0
    prompt_template: Optional[str] = None  # Custom prompt (uses default if None)
    batch_size: int = 5  # Max concurrent evaluations
    max_retries: int = 2

    # Provider-specific default models (cost-efficient)
    PROVIDER_DEFAULTS = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-haiku-20240307",
        "google": "gemini-1.5-flash"
    }

    def get_model(self) -> str:
        """Get model, falling back to provider-specific default."""
        if self.model:
            return self.model
        return self.PROVIDER_DEFAULTS.get(self.provider, "gpt-4o-mini")


# Default evaluation prompt (following ReasoningBank paper Figure 9)
DEFAULT_JUDGE_PROMPT = """<evaluation_task>
You are evaluating whether an AI agent successfully completed a task.

<task_query>
{query}
</task_query>

<agent_trajectory>
{trajectory}
</agent_trajectory>

<final_output>
{final_output}
</final_output>

<instructions>
Analyze the trajectory and determine if the agent successfully resolved the query.

Consider:
1. Did the agent understand the user's intent correctly?
2. Were the actions taken appropriate and efficient?
3. Does the final output correctly address the original query?
4. Were there any errors, loops, or wasted steps?

You MUST respond with ONLY a JSON object in this exact format:
{{
    "outcome": "success" or "failure",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of your judgment",
    "failure_notes": ["anti-pattern 1", "anti-pattern 2"] (only if outcome is failure, otherwise empty array)
}}
</instructions>
</evaluation_task>"""


class LLMJudge:
    """
    LLM-as-Judge evaluator for trajectory success/failure.

    Uses Vel's existing provider infrastructure for LLM calls.
    Based on ReasoningBank paper's binary classification approach.

    Usage:
        ```python
        judge = LLMJudge(JudgeConfig(provider="openai", model="gpt-4o-mini"))

        # Evaluate a trajectory
        result = await judge.evaluate(trajectory)

        if result.outcome == JudgeOutcome.SUCCESS:
            print("Task completed successfully")
        else:
            print(f"Task failed: {result.failure_notes}")
        ```
    """

    def __init__(
        self,
        config: Optional[JudgeConfig] = None,
        llm_fn: Optional[Callable] = None,
    ):
        """
        Initialize LLM-as-Judge.

        Args:
            config: Judge configuration
            llm_fn: Optional async function for LLM calls.
                    Signature: async (messages: List[Dict], model: str, **kwargs) -> str
                    If not provided, will attempt to use Vel's provider infrastructure.
        """
        self.config = config or JudgeConfig()
        self._llm_fn = llm_fn
        self._provider = None

    def _get_prompt_template(self) -> str:
        """Get prompt template (custom or default)."""
        return self.config.prompt_template or DEFAULT_JUDGE_PROMPT

    def _format_trajectory(self, trajectory: Any) -> str:
        """Format trajectory for prompt insertion."""
        # Handle Trajectory dataclass
        if hasattr(trajectory, 'messages'):
            messages = trajectory.messages
        elif isinstance(trajectory, dict):
            messages = trajectory.get('messages', [])
        else:
            return str(trajectory)

        lines = []
        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')

            # Handle multi-part content
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        parts.append(part.get('text', str(part)))
                    else:
                        parts.append(str(part))
                content = "\n".join(parts)

            # Truncate long content
            if len(content) > 500:
                content = content[:500] + "..."

            lines.append(f"[Step {i+1}] {role.upper()}: {content}")

        # Add tool calls if available
        tool_calls = getattr(trajectory, 'tool_calls', None) or (
            trajectory.get('tool_calls', []) if isinstance(trajectory, dict) else []
        )

        for tc in tool_calls:
            if hasattr(tc, 'tool_name'):
                name = tc.tool_name
                inp = tc.input
                out = tc.output
                err = tc.error
            else:
                name = tc.get('tool_name', tc.get('name', 'unknown'))
                inp = tc.get('input', {})
                out = tc.get('output')
                err = tc.get('error')

            status = "ERROR" if err else "OK"
            lines.append(f"[Tool] {name}({json.dumps(inp)[:100]}) -> {status}")

        return "\n".join(lines)

    def _build_prompt(self, trajectory: Any) -> str:
        """Build evaluation prompt from trajectory."""
        template = self._get_prompt_template()

        # Extract query/input
        if hasattr(trajectory, 'input_message'):
            query = trajectory.input_message
        elif isinstance(trajectory, dict):
            query = trajectory.get('input_message', trajectory.get('query', ''))
        else:
            query = str(trajectory)

        # Extract final output
        if hasattr(trajectory, 'final_answer'):
            final_output = trajectory.final_answer or "(no final answer)"
        elif isinstance(trajectory, dict):
            final_output = trajectory.get('final_answer', trajectory.get('answer', ''))
        else:
            final_output = "(no final answer)"

        # Format trajectory
        formatted_trajectory = self._format_trajectory(trajectory)

        return template.format(
            query=query,
            trajectory=formatted_trajectory,
            final_output=final_output
        )

    def _parse_response(self, response: str, trajectory_id: str) -> JudgeResult:
        """Parse LLM response to JudgeResult."""
        # Try to extract JSON from response
        try:
            # Look for JSON block
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")

            outcome_str = data.get('outcome', 'failure').lower()
            if outcome_str == 'success':
                outcome = JudgeOutcome.SUCCESS
            else:
                outcome = JudgeOutcome.FAILURE

            return JudgeResult(
                trajectory_id=trajectory_id,
                outcome=outcome,
                confidence=float(data.get('confidence', 0.5)),
                failure_notes=data.get('failure_notes', []),
                reasoning=data.get('reasoning', ''),
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Fallback: try to infer from text
            response_lower = response.lower()

            if 'success' in response_lower and 'fail' not in response_lower:
                outcome = JudgeOutcome.SUCCESS
            elif 'fail' in response_lower or 'error' in response_lower:
                outcome = JudgeOutcome.FAILURE
            else:
                outcome = JudgeOutcome.ERROR

            return JudgeResult(
                trajectory_id=trajectory_id,
                outcome=outcome,
                confidence=0.3,  # Low confidence for fallback
                reasoning=f"Parse fallback: {response[:200]}",
                error=f"Failed to parse response: {e}"
            )

    async def _call_llm(self, prompt: str) -> tuple[str, Optional[Dict[str, int]]]:
        """
        Call LLM with prompt.

        Returns:
            Tuple of (response_text, usage_dict)
        """
        if self._llm_fn:
            # Use provided LLM function
            result = await self._llm_fn(
                [{"role": "user", "content": prompt}],
                self.config.get_model()
            )
            return result, None

        # Try to use Vel's provider infrastructure
        if self._provider is None:
            try:
                from vel.providers import ProviderRegistry
                registry = ProviderRegistry()
                self._provider = registry.get(self.config.provider)
            except ImportError:
                pass

        if self._provider:
            # Use Vel provider
            result = await self._provider.generate(
                messages=[{"role": "user", "content": prompt}],
                model=self.config.get_model(),
                tools=[]
            )
            return result.get('answer', ''), result.get('usage')

        # Fallback: direct API call (OpenAI)
        try:
            import openai
            import os

            client = openai.AsyncOpenAI(
                api_key=self.config.api_key or os.getenv('OPENAI_API_KEY')
            )

            response = await client.chat.completions.create(
                model=self.config.get_model(),
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return response.choices[0].message.content or "", usage

        except Exception as e:
            raise RuntimeError(f"No LLM provider available: {e}")

    async def evaluate(
        self,
        trajectory: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> JudgeResult:
        """
        Evaluate a single trajectory.

        Args:
            trajectory: Trajectory instance or dict with trajectory data
            context: Optional additional context for evaluation

        Returns:
            JudgeResult with evaluation outcome
        """
        # Get trajectory ID
        if hasattr(trajectory, 'run_id'):
            trajectory_id = trajectory.run_id
        elif isinstance(trajectory, dict):
            trajectory_id = trajectory.get('run_id', trajectory.get('id', 'unknown'))
        else:
            trajectory_id = str(id(trajectory))

        start_time = time()

        try:
            # Build prompt
            prompt = self._build_prompt(trajectory)

            # Call LLM with retries
            response = None
            usage = None
            last_error = None

            for attempt in range(self.config.max_retries + 1):
                try:
                    response, usage = await asyncio.wait_for(
                        self._call_llm(prompt),
                        timeout=self.config.timeout
                    )
                    break
                except asyncio.TimeoutError:
                    last_error = "Timeout"
                except Exception as e:
                    last_error = str(e)

            if response is None:
                return JudgeResult(
                    trajectory_id=trajectory_id,
                    outcome=JudgeOutcome.ERROR,
                    error=f"LLM call failed: {last_error}",
                    latency_ms=(time() - start_time) * 1000,
                    model=self.config.get_model()
                )

            # Parse response
            result = self._parse_response(response, trajectory_id)
            result.usage = usage
            result.model = self.config.get_model()
            result.latency_ms = (time() - start_time) * 1000

            return result

        except Exception as e:
            return JudgeResult(
                trajectory_id=trajectory_id,
                outcome=JudgeOutcome.ERROR,
                error=str(e),
                latency_ms=(time() - start_time) * 1000,
                model=self.config.get_model()
            )

    async def evaluate_batch(
        self,
        trajectories: List[Any],
        context: Optional[Dict[str, Any]] = None
    ) -> List[JudgeResult]:
        """
        Evaluate multiple trajectories with concurrency control.

        Args:
            trajectories: List of trajectories to evaluate
            context: Optional additional context

        Returns:
            List of JudgeResults in same order as input
        """
        semaphore = asyncio.Semaphore(self.config.batch_size)

        async def evaluate_with_limit(traj):
            async with semaphore:
                return await self.evaluate(traj, context)

        tasks = [evaluate_with_limit(t) for t in trajectories]
        return await asyncio.gather(*tasks)


# Convenience function for Bayesian confidence update
def update_confidence_bayesian(
    current: float,
    success: bool,
    success_multiplier: float = 1.20,
    failure_multiplier: float = 0.85,
    max_confidence: float = 0.95,
    min_confidence: float = 0.05
) -> float:
    """
    Bayesian-style confidence update from claude-flow.

    - Success: confidence *= 1.20 (capped at 95%)
    - Failure: confidence *= 0.85 (floored at 5%)

    Args:
        current: Current confidence value (0.0-1.0)
        success: Whether the outcome was successful
        success_multiplier: Multiplier for success (default 1.20)
        failure_multiplier: Multiplier for failure (default 0.85)
        max_confidence: Maximum confidence cap (default 0.95)
        min_confidence: Minimum confidence floor (default 0.05)

    Returns:
        Updated confidence value
    """
    if success:
        new_confidence = current * success_multiplier
    else:
        new_confidence = current * failure_multiplier

    return max(min_confidence, min(max_confidence, new_confidence))
