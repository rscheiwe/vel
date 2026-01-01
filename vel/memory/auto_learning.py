# vel/memory/auto_learning.py
"""
AutoLearningManager: Orchestrates ReasoningBank Phase 2 auto-learning.

Manages the complete learning cycle:
1. TrajectoryStore records agent executions
2. EvaluationWorker evaluates trajectories with LLM-as-Judge
3. ExtractionWorker distills strategies from successful runs
4. ConsolidationWorker maintains memory health
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class AutoLearningConfig:
    """
    Configuration for auto-learning system.

    All settings are optional with sensible defaults.
    """
    # Enable/disable
    enabled: bool = False

    # LLM configuration for Judge and Extractor
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: Optional[str] = None

    # Evaluation settings
    evaluation_interval_seconds: int = 60  # How often to check for unevaluated trajectories
    evaluation_batch_size: int = 5

    # Extraction settings
    extraction_interval_seconds: int = 120  # How often to check for unextracted trajectories
    extraction_similarity_threshold: float = 0.85

    # Consolidation settings
    consolidation_interval_seconds: int = 21600  # 6 hours
    consolidation_similarity_threshold: float = 0.85
    max_strategies: int = 1000
    min_confidence_threshold: float = 0.20
    confidence_decay_rate: float = 0.02


class EvaluationWorker:
    """
    Background worker that evaluates trajectories with LLM-as-Judge.

    Lifecycle:
    1. Poll for trajectories with finished_at IS NOT NULL AND evaluated = 0
    2. For each trajectory, call LLMJudge.evaluate()
    3. Mark trajectory with evaluation result
    4. Update strategy confidence if strategies were used
    5. Sleep and repeat
    """

    def __init__(
        self,
        trajectory_store: Any,
        judge: Any,
        reasoning_bank: Optional[Any] = None,
        batch_size: int = 5,
        interval_seconds: int = 60
    ):
        self.trajectory_store = trajectory_store
        self.judge = judge
        self.rb = reasoning_bank
        self.batch_size = batch_size
        self.interval_seconds = interval_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start background worker."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("EvaluationWorker started")

    async def stop(self):
        """Stop background worker gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("EvaluationWorker stopped")

    async def _run_loop(self):
        """Main evaluation loop."""
        while self._running:
            try:
                await self._process_batch()
            except Exception as e:
                logger.error(f"EvaluationWorker error: {e}")

            await asyncio.sleep(self.interval_seconds)

    async def _process_batch(self):
        """Process a batch of unevaluated trajectories."""
        trajectories = self.trajectory_store.get_unevaluated_trajectories(self.batch_size)

        if not trajectories:
            return

        logger.debug(f"Evaluating {len(trajectories)} trajectories")

        results = await self.judge.evaluate_batch(trajectories)

        for traj, result in zip(trajectories, results):
            try:
                # Mark trajectory as evaluated
                from vel.memory.judge import JudgeOutcome
                success = result.outcome == JudgeOutcome.SUCCESS

                self.trajectory_store.mark_evaluated(
                    trajectory_id=traj.id,
                    success=success,
                    confidence=result.confidence,
                    notes=result.reasoning
                )

                # Update strategy confidence if strategies were used
                if self.rb and traj.strategies_used:
                    self.rb.mark_outcome(
                        traj.strategies_used,
                        success=success,
                        fail_notes=result.failure_notes
                    )

                logger.debug(f"Evaluated trajectory {traj.run_id}: {result.outcome.value}")

            except Exception as e:
                logger.error(f"Failed to process evaluation for {traj.run_id}: {e}")


class ExtractionWorker:
    """
    Background worker that extracts strategies from successful trajectories.

    Lifecycle:
    1. Poll for trajectories with evaluated=True, success=True, strategies_extracted=False
    2. For each trajectory, call StrategyExtractor.extract_and_store()
    3. Mark trajectory as strategies_extracted=True
    4. Sleep and repeat
    """

    def __init__(
        self,
        trajectory_store: Any,
        strategy_extractor: Any,
        batch_size: int = 10,
        interval_seconds: int = 120
    ):
        self.trajectory_store = trajectory_store
        self.extractor = strategy_extractor
        self.batch_size = batch_size
        self.interval_seconds = interval_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start background worker."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("ExtractionWorker started")

    async def stop(self):
        """Stop background worker gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("ExtractionWorker stopped")

    async def _run_loop(self):
        """Main extraction loop."""
        while self._running:
            try:
                await self._process_batch()
            except Exception as e:
                logger.error(f"ExtractionWorker error: {e}")

            await asyncio.sleep(self.interval_seconds)

    async def _process_batch(self):
        """Process a batch of successful trajectories."""
        trajectories = self.trajectory_store.get_successful_unextracted(self.batch_size)

        if not trajectories:
            return

        logger.debug(f"Extracting from {len(trajectories)} trajectories")

        for traj in trajectories:
            try:
                # Build a minimal judge result from stored evaluation
                judge_result = {
                    'confidence': traj.evaluation_confidence or 0.5,
                    'reasoning': traj.evaluation_notes or ''
                }

                strategy_id = await self.extractor.extract_and_store(traj, judge_result)

                if strategy_id:
                    logger.info(f"Extracted strategy {strategy_id} from trajectory {traj.run_id}")

            except Exception as e:
                logger.error(f"Failed to extract from trajectory {traj.run_id}: {e}")

            finally:
                # Mark as processed regardless of outcome
                self.trajectory_store.mark_strategies_extracted(traj.id)


class ConsolidationWorker:
    """
    Background worker that periodically consolidates ReasoningBank.

    Runs less frequently than evaluation/extraction (e.g., every 6 hours).
    """

    def __init__(
        self,
        consolidator: Any,
        interval_seconds: int = 21600  # 6 hours
    ):
        self.consolidator = consolidator
        self.interval_seconds = interval_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start background worker."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("ConsolidationWorker started")

    async def stop(self):
        """Stop background worker gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("ConsolidationWorker stopped")

    async def _run_loop(self):
        """Main consolidation loop."""
        while self._running:
            try:
                result = self.consolidator.consolidate()
                logger.info(
                    f"Consolidation complete: merged={result.strategies_merged}, "
                    f"pruned={result.strategies_pruned}, "
                    f"decayed={result.strategies_decayed}, "
                    f"before={result.total_strategies_before}, "
                    f"after={result.total_strategies_after}"
                )
            except Exception as e:
                logger.error(f"ConsolidationWorker error: {e}")

            await asyncio.sleep(self.interval_seconds)


class AutoLearningManager:
    """
    Orchestrates all Phase 2 auto-learning components.

    Usage:
        ```python
        from vel.memory.auto_learning import AutoLearningManager, AutoLearningConfig

        config = AutoLearningConfig(
            enabled=True,
            llm_provider="openai",
            llm_model="gpt-4o-mini"
        )

        manager = AutoLearningManager(
            config=config,
            trajectory_store=traj_store,
            reasoning_bank_store=rb_store
        )

        # Start all workers
        await manager.start()

        # ... agent runs ...

        # Stop gracefully
        await manager.stop()
        ```
    """

    def __init__(
        self,
        config: AutoLearningConfig,
        trajectory_store: Any,
        reasoning_bank_store: Any,
        reasoning_bank: Optional[Any] = None,
    ):
        """
        Initialize AutoLearningManager.

        Args:
            config: Auto-learning configuration
            trajectory_store: TrajectoryStore instance
            reasoning_bank_store: ReasoningBankStore instance
            reasoning_bank: Optional ReasoningBank wrapper (for confidence updates)
        """
        self.config = config
        self.trajectory_store = trajectory_store
        self.rb_store = reasoning_bank_store
        self.rb = reasoning_bank

        # Workers (initialized on start)
        self._evaluation_worker: Optional[EvaluationWorker] = None
        self._extraction_worker: Optional[ExtractionWorker] = None
        self._consolidation_worker: Optional[ConsolidationWorker] = None

        self._running = False

    async def start(self) -> None:
        """Start all background workers."""
        if not self.config.enabled:
            logger.info("AutoLearning is disabled, not starting workers")
            return

        if self._running:
            return

        logger.info("Starting AutoLearningManager...")

        # Create Judge
        from vel.memory.judge import LLMJudge, JudgeConfig
        judge_config = JudgeConfig(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
            api_key=self.config.llm_api_key,
        )
        judge = LLMJudge(config=judge_config)

        # Create StrategyExtractor
        from vel.memory.strategy_extractor import StrategyExtractor
        extractor = StrategyExtractor(
            reasoning_bank_store=self.rb_store,
            model_config={
                "provider": self.config.llm_provider,
                "model": self.config.llm_model,
                "api_key": self.config.llm_api_key,
            },
            similarity_threshold=self.config.extraction_similarity_threshold,
        )

        # Create MemoryConsolidator
        from vel.memory.memory_consolidator import MemoryConsolidator
        consolidator = MemoryConsolidator(
            reasoning_bank_store=self.rb_store,
            similarity_threshold=self.config.consolidation_similarity_threshold,
            max_strategies=self.config.max_strategies,
            min_confidence=self.config.min_confidence_threshold,
            decay_rate=self.config.confidence_decay_rate,
        )

        # Create workers
        self._evaluation_worker = EvaluationWorker(
            trajectory_store=self.trajectory_store,
            judge=judge,
            reasoning_bank=self.rb,
            batch_size=self.config.evaluation_batch_size,
            interval_seconds=self.config.evaluation_interval_seconds,
        )

        self._extraction_worker = ExtractionWorker(
            trajectory_store=self.trajectory_store,
            strategy_extractor=extractor,
            batch_size=10,
            interval_seconds=self.config.extraction_interval_seconds,
        )

        self._consolidation_worker = ConsolidationWorker(
            consolidator=consolidator,
            interval_seconds=self.config.consolidation_interval_seconds,
        )

        # Start all workers
        await self._evaluation_worker.start()
        await self._extraction_worker.start()
        await self._consolidation_worker.start()

        self._running = True
        logger.info("AutoLearningManager started")

    async def stop(self) -> None:
        """Stop all background workers gracefully."""
        if not self._running:
            return

        logger.info("Stopping AutoLearningManager...")

        if self._evaluation_worker:
            await self._evaluation_worker.stop()

        if self._extraction_worker:
            await self._extraction_worker.stop()

        if self._consolidation_worker:
            await self._consolidation_worker.stop()

        self._running = False
        logger.info("AutoLearningManager stopped")

    @property
    def is_running(self) -> bool:
        """Check if manager is running."""
        return self._running


# Example seed strategies for research/critical thinking
EXAMPLE_SEED_STRATEGIES = [
    # Research Patterns
    {
        "signature": {"intent": "research", "domain": "general", "risk": "low"},
        "strategy_text": "Before synthesizing conclusions, gather evidence from at least 3 independent sources to triangulate accuracy.",
        "anti_patterns": ["Relying on single source", "Confirmation bias", "Ignoring contradictory evidence"],
        "confidence": 0.7
    },
    {
        "signature": {"intent": "research", "domain": "technical", "risk": "medium"},
        "strategy_text": "Verify technical claims by checking official documentation, then cross-reference with community implementations.",
        "anti_patterns": ["Trusting outdated docs", "Ignoring version differences", "Assuming examples are production-ready"],
        "confidence": 0.7
    },
    {
        "signature": {"intent": "analysis", "domain": "data", "risk": "medium"},
        "strategy_text": "State assumptions explicitly before analysis, then validate each assumption with data before proceeding.",
        "anti_patterns": ["Hidden assumptions", "Correlation implies causation", "Cherry-picking data points"],
        "confidence": 0.7
    },

    # Critical Thinking Patterns
    {
        "signature": {"intent": "evaluation", "domain": "general", "risk": "low"},
        "strategy_text": "Apply steel-man reasoning: articulate the strongest version of opposing viewpoints before critiquing them.",
        "anti_patterns": ["Strawman arguments", "Dismissing without understanding", "Ad hominem attacks"],
        "confidence": 0.7
    },
    {
        "signature": {"intent": "decision", "domain": "general", "risk": "high"},
        "strategy_text": "For high-stakes decisions, enumerate second-order consequences and identify potential failure modes before committing.",
        "anti_patterns": ["First-order thinking only", "Ignoring edge cases", "Overconfidence in predictions"],
        "confidence": 0.7
    },
    {
        "signature": {"intent": "problem_solving", "domain": "general", "risk": "medium"},
        "strategy_text": "Decompose complex problems into independent sub-problems, solve each separately, then verify the composition.",
        "anti_patterns": ["Attempting to solve everything at once", "Ignoring dependencies", "Skipping verification"],
        "confidence": 0.7
    },
    {
        "signature": {"intent": "hypothesis", "domain": "scientific", "risk": "medium"},
        "strategy_text": "Generate multiple competing hypotheses, then design tests that could falsify each before investing in validation.",
        "anti_patterns": ["Single hypothesis fixation", "Confirmation-only tests", "Ignoring null results"],
        "confidence": 0.7
    },

    # Planning Patterns
    {
        "signature": {"intent": "planning", "domain": "project", "risk": "medium"},
        "strategy_text": "Identify the critical path and highest-risk items first; front-load uncertainty resolution before committing resources.",
        "anti_patterns": ["Starting with easy tasks", "Ignoring dependencies", "Optimistic scheduling"],
        "confidence": 0.7
    },
    {
        "signature": {"intent": "planning", "domain": "api", "risk": "low"},
        "strategy_text": "Design API contracts and interfaces before implementation; validate with stakeholders before coding.",
        "anti_patterns": ["Implementation-first design", "Skipping contract review", "Assuming requirements are stable"],
        "confidence": 0.7
    },

    # Debugging/Investigation Patterns
    {
        "signature": {"intent": "debugging", "domain": "software", "risk": "low"},
        "strategy_text": "Reproduce the issue with minimal steps first, then bisect to isolate the root cause before attempting fixes.",
        "anti_patterns": ["Guessing at fixes", "Changing multiple things at once", "Not reproducing first"],
        "confidence": 0.7
    },
    {
        "signature": {"intent": "investigation", "domain": "incident", "risk": "high"},
        "strategy_text": "Preserve evidence and establish timeline before taking corrective action; document observations before interpretations.",
        "anti_patterns": ["Destroying evidence", "Jumping to conclusions", "Mixing facts with speculation"],
        "confidence": 0.7
    }
]


def populate_seed_strategies(reasoning_bank_store: Any, strategies: Optional[List[Dict]] = None) -> int:
    """
    Populate ReasoningBank with seed strategies.

    Args:
        reasoning_bank_store: ReasoningBankStore instance
        strategies: List of strategy dicts (uses EXAMPLE_SEED_STRATEGIES if None)

    Returns:
        Number of strategies added
    """
    seeds = strategies or EXAMPLE_SEED_STRATEGIES
    count = 0

    for seed in seeds:
        try:
            reasoning_bank_store.upsert_strategy(
                signature=seed["signature"],
                strategy_text=seed["strategy_text"],
                anti_patterns=seed.get("anti_patterns", []),
                evidence_refs=seed.get("evidence_refs", []),
                confidence=seed.get("confidence", 0.7)
            )
            count += 1
        except Exception as e:
            logger.warning(f"Failed to add seed strategy: {e}")

    logger.info(f"Populated {count} seed strategies")
    return count
