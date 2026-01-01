# vel/memory/strategy_extractor.py
"""
StrategyExtractor: Distills generalizable strategies from successful trajectories.

Part of ReasoningBank Phase 2 - enables automatic learning by:
1. Analyzing successful execution patterns
2. Extracting one-sentence generalizable strategies
3. Deduplicating against existing strategies
4. Computing initial confidence scores
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
import json
import re
from time import time
import asyncio

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore


@dataclass
class ExtractedStrategy:
    """
    Result from StrategyExtractor.extract() call.

    Contains the distilled strategy plus metadata for deduplication
    and quality assessment.
    """
    strategy_text: str           # The generalizable strategy (1 sentence)
    signature: Dict[str, Any]    # Task signature for retrieval
    initial_confidence: float    # Starting confidence (0.0-1.0)
    anti_patterns: List[str]     # Patterns to avoid (max 3)
    reasoning: str               # Why this strategy was extracted
    source_trajectory_id: str    # Evidence reference (run_id)
    embedding: Optional[Any] = None  # Pre-computed for dedup (np.ndarray)


# Default extraction prompt
DEFAULT_EXTRACTION_PROMPT = """<strategy_extraction>
You are an expert at analyzing AI agent reasoning patterns.

Your task is to extract a GENERALIZABLE reasoning strategy from this successful execution.

## Task Signature
Intent: {intent}
Domain: {domain}
Risk Level: {risk}

## User Query
{query}

## Execution Trajectory
{trajectory}

## Final Answer
{final_answer}

## Constraints
- Strategy must be ONE clear sentence
- Must be applicable to SIMILAR tasks (not specific to this exact case)
- Must be actionable (describes HOW to think, not WHAT to think)
- Must capture the key insight that made this execution successful

## Anti-Patterns
Identify 1-3 things to AVOID based on potential failure modes or inefficiencies.

## Response Format (JSON only)
{{
  "strategy_text": "One sentence describing the reasoning approach",
  "anti_patterns": ["Pattern to avoid 1", "Pattern to avoid 2"],
  "reasoning": "Brief explanation of why this strategy is valuable and generalizable"
}}

IMPORTANT:
- Strategy must be generalizable (not "Use API key abc123")
- Focus on reasoning approach, not specific actions
- Anti-patterns should be cautionary warnings
- If no good strategy can be extracted, respond with {{"skip": true, "reason": "..."}}
</strategy_extraction>"""


class StrategyExtractor:
    """
    Distills generalizable strategies from successful trajectories.

    Key responsibilities:
    1. Analyze successful trajectory patterns
    2. Extract one-sentence generalizable strategies
    3. Deduplicate against existing strategies
    4. Queue for async background processing

    Design principles:
    - Bayesian confidence: Initial confidence based on trajectory quality
    - Deduplication: Embedding-based similarity check before insert
    - Non-blocking: Can be used with WorkQueue for async processing

    Usage:
        ```python
        from vel.memory import ReasoningBankStore, Embeddings
        from vel.memory.strategy_extractor import StrategyExtractor

        # Initialize
        extractor = StrategyExtractor(
            reasoning_bank_store=rb_store,
            model_config={"provider": "openai", "model": "gpt-4o-mini"}
        )

        # Extract from successful trajectory
        strategy = await extractor.extract(trajectory, judge_result)

        if strategy:
            print(f"Extracted: {strategy.strategy_text}")
        ```
    """

    def __init__(
        self,
        reasoning_bank_store: Any,  # ReasoningBankStore
        model_config: Dict[str, Any],
        extraction_prompt: Optional[str] = None,
        similarity_threshold: float = 0.85,
        min_trajectory_steps: int = 2,
        max_strategies_per_run: int = 1,
        llm_fn: Optional[Callable] = None,
    ):
        """
        Args:
            reasoning_bank_store: ReasoningBankStore for persistence
            model_config: LLM config (e.g., {'provider': 'openai', 'model': 'gpt-4o-mini'})
            extraction_prompt: Custom extraction prompt (uses default if None)
            similarity_threshold: Embedding similarity for deduplication (0.0-1.0)
            min_trajectory_steps: Minimum steps in trajectory to consider extraction
            max_strategies_per_run: Max strategies to extract per trajectory
            llm_fn: Optional async function for LLM calls
        """
        self.store = reasoning_bank_store
        self.model_config = model_config
        self.extraction_prompt = extraction_prompt or DEFAULT_EXTRACTION_PROMPT
        self.similarity_threshold = similarity_threshold
        self.min_trajectory_steps = min_trajectory_steps
        self.max_strategies_per_run = max_strategies_per_run
        self._llm_fn = llm_fn
        self._provider = None

    def _format_trajectory(self, trajectory: Any) -> str:
        """Format trajectory for prompt insertion."""
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

            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        parts.append(part.get('text', str(part)))
                    else:
                        parts.append(str(part))
                content = "\n".join(parts)

            if len(content) > 300:
                content = content[:300] + "..."

            lines.append(f"[{role.upper()}]: {content}")

        return "\n".join(lines)

    def _build_prompt(self, trajectory: Any) -> str:
        """Build extraction prompt from trajectory."""
        # Extract signature
        if hasattr(trajectory, 'signature'):
            sig = trajectory.signature
        elif isinstance(trajectory, dict):
            sig = trajectory.get('signature', {})
        else:
            sig = {}

        intent = sig.get('intent', 'general')
        domain = sig.get('domain', 'general')
        risk = sig.get('risk', 'medium')

        # Extract query
        if hasattr(trajectory, 'input_message'):
            query = trajectory.input_message
        elif isinstance(trajectory, dict):
            query = trajectory.get('input_message', trajectory.get('query', ''))
        else:
            query = str(trajectory)

        # Extract final answer
        if hasattr(trajectory, 'final_answer'):
            final_answer = trajectory.final_answer or "(no final answer)"
        elif isinstance(trajectory, dict):
            final_answer = trajectory.get('final_answer', '')
        else:
            final_answer = "(no final answer)"

        formatted_trajectory = self._format_trajectory(trajectory)

        return self.extraction_prompt.format(
            intent=intent,
            domain=domain,
            risk=risk,
            query=query,
            trajectory=formatted_trajectory,
            final_answer=final_answer
        )

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt."""
        if self._llm_fn:
            return await self._llm_fn(
                [{"role": "user", "content": prompt}],
                self.model_config.get('model', 'gpt-4o-mini')
            )

        # Try to use Vel's provider infrastructure
        if self._provider is None:
            try:
                from vel.providers import ProviderRegistry
                registry = ProviderRegistry()
                provider_name = self.model_config.get('provider', 'openai')
                self._provider = registry.get(provider_name)
            except ImportError:
                pass

        if self._provider:
            result = await self._provider.generate(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_config.get('model', 'gpt-4o-mini'),
                tools=[]
            )
            return result.get('answer', '')

        # Fallback: direct OpenAI call
        try:
            import openai
            import os

            client = openai.AsyncOpenAI(
                api_key=self.model_config.get('api_key') or os.getenv('OPENAI_API_KEY')
            )

            response = await client.chat.completions.create(
                model=self.model_config.get('model', 'gpt-4o-mini'),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            raise RuntimeError(f"No LLM provider available: {e}")

    def _parse_response(
        self,
        response: str,
        trajectory: Any,
        judge_result: Optional[Any]
    ) -> Optional[ExtractedStrategy]:
        """Parse LLM response to ExtractedStrategy."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                return None

            # Check for skip signal
            if data.get('skip'):
                return None

            strategy_text = data.get('strategy_text', '').strip()
            if not strategy_text or len(strategy_text) < 10:
                return None

            # Get signature
            if hasattr(trajectory, 'signature'):
                signature = trajectory.signature
            elif isinstance(trajectory, dict):
                signature = trajectory.get('signature', {})
            else:
                signature = {}

            # Get trajectory ID
            if hasattr(trajectory, 'run_id'):
                traj_id = trajectory.run_id
            elif isinstance(trajectory, dict):
                traj_id = trajectory.get('run_id', trajectory.get('id', 'unknown'))
            else:
                traj_id = str(id(trajectory))

            # Calculate initial confidence
            initial_confidence = self._calculate_initial_confidence(trajectory, judge_result)

            return ExtractedStrategy(
                strategy_text=strategy_text,
                signature=signature,
                initial_confidence=initial_confidence,
                anti_patterns=data.get('anti_patterns', [])[:3],
                reasoning=data.get('reasoning', ''),
                source_trajectory_id=traj_id,
            )

        except (json.JSONDecodeError, ValueError, KeyError):
            return None

    def _calculate_initial_confidence(
        self,
        trajectory: Any,
        judge_result: Optional[Any]
    ) -> float:
        """
        Calculate initial confidence using Bayesian approach.

        Base: 0.5 (neutral prior)

        Adjustments:
        - Judge confidence: +(judge.confidence - 0.5) * 0.3
        - Trajectory length: +0.1 if 3-8 steps (optimal range)
        - Tool diversity: +0.05 per unique tool (max 0.15)
        - Error-free: +0.1 if no error field

        Capped to [0.35, 0.75] range for new strategies.
        """
        base = 0.5

        # Judge confidence boost
        if judge_result:
            judge_conf = getattr(judge_result, 'confidence', None)
            if judge_conf is None and isinstance(judge_result, dict):
                judge_conf = judge_result.get('confidence')
            if judge_conf is not None:
                base += (float(judge_conf) - 0.5) * 0.3

        # Trajectory length bonus
        if hasattr(trajectory, 'messages'):
            step_count = len(trajectory.messages) // 2
        elif isinstance(trajectory, dict):
            step_count = len(trajectory.get('messages', [])) // 2
        else:
            step_count = 0

        if 3 <= step_count <= 8:
            base += 0.1

        # Tool diversity bonus
        tool_calls = None
        if hasattr(trajectory, 'tool_calls'):
            tool_calls = trajectory.tool_calls
        elif isinstance(trajectory, dict):
            tool_calls = trajectory.get('tool_calls', [])

        if tool_calls:
            unique_tools = set()
            for tc in tool_calls:
                if hasattr(tc, 'tool_name'):
                    unique_tools.add(tc.tool_name)
                elif isinstance(tc, dict):
                    unique_tools.add(tc.get('tool_name', tc.get('name', '')))
            base += min(len(unique_tools) * 0.05, 0.15)

        # Error-free bonus
        error = None
        if hasattr(trajectory, 'error'):
            error = trajectory.error
        elif isinstance(trajectory, dict):
            error = trajectory.get('error')

        if not error:
            base += 0.1

        return max(0.35, min(0.75, base))

    def _compute_embedding(self, signature: Dict[str, Any], strategy_text: str) -> Optional[Any]:
        """Compute embedding for strategy using store's embedder."""
        if not hasattr(self.store, 'emb') or self.store.emb is None:
            return None

        try:
            text = self.store._embed_text(signature, strategy_text)
            return self.store.emb.encode([text])[0]
        except Exception:
            return None

    def is_duplicate(
        self,
        new_strategy: ExtractedStrategy,
        existing_strategies: Optional[List[Any]] = None
    ) -> bool:
        """
        Check if strategy is semantically duplicate of existing.

        Two-phase deduplication:
        1. Fast embedding similarity check (cosine > threshold)
        2. Word overlap verification (> 50%)

        Args:
            new_strategy: Strategy to check
            existing_strategies: List of existing StrategyItems (fetches if None)

        Returns:
            True if duplicate, False if unique
        """
        if np is None:
            # No numpy, can't check embeddings
            return False

        # Compute embedding for new strategy
        new_emb = new_strategy.embedding
        if new_emb is None:
            new_emb = self._compute_embedding(new_strategy.signature, new_strategy.strategy_text)

        if new_emb is None:
            return False

        # Get existing strategies if not provided
        if existing_strategies is None:
            existing_strategies = self.store.retrieve(
                new_strategy.signature, k=50, min_conf=0.0
            )

        if not existing_strategies:
            return False

        # Phase 1: Embedding similarity
        candidates = []
        for existing in existing_strategies:
            # Get existing embedding from DB
            try:
                row = self.store.db.execute(
                    "SELECT embedding FROM rb_embeddings WHERE strategy_id = ?",
                    (existing.id,)
                ).fetchone()

                if not row:
                    continue

                existing_emb = np.frombuffer(row["embedding"], dtype=np.float32)
                existing_emb = existing_emb / (np.linalg.norm(existing_emb) + 1e-8)

                similarity = float(np.dot(new_emb, existing_emb))
                if similarity >= self.similarity_threshold:
                    candidates.append((existing, similarity))
            except Exception:
                continue

        if not candidates:
            return False

        # Phase 2: Word overlap verification
        new_words = set(new_strategy.strategy_text.lower().split())

        for existing, sim in candidates:
            existing_words = set(existing.strategy_text.lower().split())
            overlap = len(new_words & existing_words) / max(len(new_words), 1)

            # High embedding similarity + moderate word overlap = duplicate
            if overlap > 0.5:
                return True

        return False

    async def extract(
        self,
        trajectory: Any,
        judge_result: Optional[Any] = None
    ) -> Optional[ExtractedStrategy]:
        """
        Extract a strategy from a successful trajectory.

        Args:
            trajectory: Trajectory instance with messages, tool_calls, etc.
            judge_result: Optional JudgeResult for additional context

        Returns:
            ExtractedStrategy if extraction successful and non-duplicate, None otherwise
        """
        # Check minimum trajectory length
        if hasattr(trajectory, 'messages'):
            step_count = len(trajectory.messages)
        elif isinstance(trajectory, dict):
            step_count = len(trajectory.get('messages', []))
        else:
            step_count = 0

        if step_count < self.min_trajectory_steps:
            return None

        try:
            prompt = self._build_prompt(trajectory)
            response = await self._call_llm(prompt)
            strategy = self._parse_response(response, trajectory, judge_result)

            if strategy is None:
                return None

            # Compute embedding for deduplication
            strategy.embedding = self._compute_embedding(strategy.signature, strategy.strategy_text)

            # Check for duplicates
            if self.is_duplicate(strategy):
                return None

            return strategy

        except Exception:
            return None

    async def extract_and_store(
        self,
        trajectory: Any,
        judge_result: Optional[Any] = None
    ) -> Optional[int]:
        """
        Extract strategy and store if valid (non-duplicate).

        Args:
            trajectory: Trajectory to extract from
            judge_result: Optional judge result

        Returns:
            Strategy ID if stored, None if skipped (duplicate or low quality)
        """
        strategy = await self.extract(trajectory, judge_result)

        if strategy is None:
            return None

        try:
            strategy_id = self.store.upsert_strategy(
                signature=strategy.signature,
                strategy_text=strategy.strategy_text,
                anti_patterns=strategy.anti_patterns,
                evidence_refs=[strategy.source_trajectory_id],
                confidence=strategy.initial_confidence
            )
            return strategy_id
        except Exception:
            return None
