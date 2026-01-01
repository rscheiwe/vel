"""
Memory system for Vel agents.

Provides optional runtime-owned memory without LLM tool calls:
- FactStore: Namespaced key-value store for long-term facts
- ReasoningBank: Strategic memory with embeddings

Phase 2 (Auto-Learning):
- TrajectoryStore: Records agent execution traces
- LLMJudge: Automatic success/failure evaluation
- StrategyExtractor: Distills strategies from successful runs
- MemoryConsolidator: Maintains memory health
- AutoLearningManager: Orchestrates all components
"""
from __future__ import annotations
import warnings

__all__ = [
    # Phase 1: Core Memory
    'FactStore',
    'ReasoningBank',
    'ReasoningBankStore',
    'Embeddings',
    'StrategyItem',

    # Phase 2: Trajectory Recording
    'TrajectoryStore',
    'Trajectory',
    'ToolCallRecord',

    # Phase 2: Evaluation
    'LLMJudge',
    'JudgeConfig',
    'JudgeResult',
    'JudgeOutcome',
    'update_confidence_bayesian',

    # Phase 2: Strategy Extraction
    'StrategyExtractor',
    'ExtractedStrategy',

    # Phase 2: Memory Consolidation
    'MemoryConsolidator',
    'ConsolidationResult',

    # Phase 2: Auto-Learning
    'AutoLearningManager',
    'AutoLearningConfig',
    'EvaluationWorker',
    'ExtractionWorker',
    'ConsolidationWorker',
    'EXAMPLE_SEED_STRATEGIES',
    'populate_seed_strategies',

    # Deprecated (backwards compatibility)
    'EpisodicStore'
]

# Phase 1: Core Memory
try:
    from .fact_store import FactStore
    # Backwards compatibility - will be removed in v2.0
    EpisodicStore = FactStore
except ImportError:
    FactStore = None
    EpisodicStore = None

try:
    from .strategy_reasoningbank import (
        ReasoningBank,
        ReasoningBankStore,
        Embeddings,
        StrategyItem,
    )
except ImportError:
    ReasoningBank = None
    ReasoningBankStore = None
    Embeddings = None
    StrategyItem = None

# Phase 2: Trajectory Recording
try:
    from .trajectory_store import (
        TrajectoryStore,
        Trajectory,
        ToolCallRecord,
    )
except ImportError:
    TrajectoryStore = None
    Trajectory = None
    ToolCallRecord = None

# Phase 2: Evaluation
try:
    from .judge import (
        LLMJudge,
        JudgeConfig,
        JudgeResult,
        JudgeOutcome,
        update_confidence_bayesian,
    )
except ImportError:
    LLMJudge = None
    JudgeConfig = None
    JudgeResult = None
    JudgeOutcome = None
    update_confidence_bayesian = None

# Phase 2: Strategy Extraction
try:
    from .strategy_extractor import (
        StrategyExtractor,
        ExtractedStrategy,
    )
except ImportError:
    StrategyExtractor = None
    ExtractedStrategy = None

# Phase 2: Memory Consolidation
try:
    from .memory_consolidator import (
        MemoryConsolidator,
        ConsolidationResult,
    )
except ImportError:
    MemoryConsolidator = None
    ConsolidationResult = None

# Phase 2: Auto-Learning
try:
    from .auto_learning import (
        AutoLearningManager,
        AutoLearningConfig,
        EvaluationWorker,
        ExtractionWorker,
        ConsolidationWorker,
        EXAMPLE_SEED_STRATEGIES,
        populate_seed_strategies,
    )
except ImportError:
    AutoLearningManager = None
    AutoLearningConfig = None
    EvaluationWorker = None
    ExtractionWorker = None
    ConsolidationWorker = None
    EXAMPLE_SEED_STRATEGIES = None
    populate_seed_strategies = None
