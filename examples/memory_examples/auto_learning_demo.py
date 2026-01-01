#!/usr/bin/env python3
"""
ReasoningBank Phase 2 — Auto-Learning Demo

This comprehensive example demonstrates the full auto-learning pipeline:

1. TrajectoryStore: Records agent execution traces
2. LLM-as-Judge: Automatic success/failure evaluation
3. StrategyExtractor: Distills strategies from successful runs
4. MemoryConsolidator: Maintains memory health
5. AutoLearningManager: Orchestrates background workers

The learning cycle:
    Agent Run → Record Trajectory → Evaluate → Extract Strategy → Consolidate

Requirements:
    - OPENAI_API_KEY environment variable (for LLM-as-Judge and agent)
    - numpy (for embeddings)

Usage:
    python examples/memory_examples/auto_learning_demo.py
"""

import asyncio
import os
import hashlib
from typing import List
from datetime import datetime

import numpy as np

from vel import Agent
from vel.core import ContextManager, MemoryConfig
from vel.memory import (
    # Phase 1
    ReasoningBankStore,
    Embeddings,
    # Phase 2
    TrajectoryStore,
    LLMJudge,
    JudgeConfig,
    JudgeOutcome,
    StrategyExtractor,
    MemoryConsolidator,
    AutoLearningManager,
    AutoLearningConfig,
    EXAMPLE_SEED_STRATEGIES,
    populate_seed_strategies,
    update_confidence_bayesian,
)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_subsection(title: str):
    """Print a subsection header."""
    print(f"\n--- {title} ---\n")


def encode_embeddings(texts: List[str]) -> np.ndarray:
    """
    Hash-based embeddings for demonstration.

    In production, use sentence-transformers or OpenAI embeddings:

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, normalize_embeddings=True)
    """
    out = []
    for t in texts:
        h = hashlib.sha256(t.encode()).digest()
        v = np.frombuffer(h, dtype=np.uint8).astype(np.float32)[:256]
        v = (v - v.mean()) / (v.std() + 1e-8)
        out.append(v)
    return np.vstack(out)


async def demo_trajectory_store():
    """Demonstrate TrajectoryStore functionality."""
    print_section("1. TrajectoryStore Demo")

    print("TrajectoryStore records agent execution traces for later analysis.")
    print("This enables automatic learning from agent behavior.\n")

    # Create trajectory store
    store = TrajectoryStore(".vel/demo_trajectories.db")

    # Start a trajectory
    run_id = f"demo-run-{datetime.now().strftime('%H%M%S')}"
    signature = {"intent": "planning", "domain": "api", "risk": "low"}

    traj_id = store.start_trajectory(
        run_id=run_id,
        signature=signature,
        input_message="Help me design a REST API for a todo app",
        agent_id="demo-agent",
        session_id="demo-session"
    )
    print(f"✓ Started trajectory: {run_id} (ID: {traj_id})")

    # Simulate tool calls
    store.record_tool_call(
        run_id=run_id,
        step_index=0,
        tool_name="search_docs",
        input={"query": "REST API best practices"},
        output={"results": ["Use proper HTTP methods", "Implement pagination"]},
        duration_ms=150
    )
    print("✓ Recorded tool call: search_docs")

    store.record_tool_call(
        run_id=run_id,
        step_index=1,
        tool_name="generate_schema",
        input={"resource": "todo"},
        output={"schema": {"id": "int", "title": "str", "completed": "bool"}},
        duration_ms=50
    )
    print("✓ Recorded tool call: generate_schema")

    # Finish trajectory
    messages = [
        {"role": "user", "content": "Help me design a REST API for a todo app"},
        {"role": "assistant", "content": "I'll design a REST API with CRUD endpoints..."}
    ]

    store.finish_trajectory(
        run_id=run_id,
        messages=messages,
        final_answer="Here's a complete REST API design with endpoints for /todos...",
        strategies_used=[1, 2],  # IDs of strategies that were used
        step_count=3
    )
    print("✓ Finished trajectory recording")

    # Retrieve and display
    traj = store.get_trajectory(run_id)
    print(f"\nTrajectory Details:")
    print(f"  Run ID: {traj.run_id}")
    print(f"  Agent ID: {traj.agent_id}")
    print(f"  Signature: {traj.signature}")
    print(f"  Tool Calls: {len(traj.tool_calls)}")
    print(f"  Duration: {traj.duration_ms}ms")
    print(f"  Evaluated: {traj.evaluated}")

    # Show statistics
    stats = store.get_statistics(time_range_days=1)
    print(f"\nStatistics (last 24h):")
    print(f"  Total trajectories: {stats['total']}")
    print(f"  Pending evaluation: {stats['pending_evaluation']}")

    return store, run_id


async def demo_llm_judge(trajectory_store, run_id):
    """Demonstrate LLM-as-Judge functionality."""
    print_section("2. LLM-as-Judge Demo")

    print("LLM-as-Judge automatically evaluates trajectory success/failure.")
    print("Uses gpt-4o-mini by default for cost efficiency (~$0.0003/eval).\n")

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set - using mock evaluation")

        # Mock evaluation for demo
        trajectory_store.mark_evaluated(
            trajectory_id=1,
            success=True,
            confidence=0.85,
            notes="Mock evaluation: Task completed successfully"
        )
        print("✓ Mock evaluation completed")
        return

    # Create judge
    judge = LLMJudge(
        config=JudgeConfig(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.0,  # Deterministic
            max_tokens=512,
        )
    )

    # Get trajectory
    traj = trajectory_store.get_trajectory(run_id)

    print(f"Evaluating trajectory: {run_id}")
    print(f"  Input: {traj.input_message[:50]}...")
    print(f"  Final answer: {traj.final_answer[:50]}...")
    print("\nCalling LLM-as-Judge...", end=" ", flush=True)

    # Evaluate
    result = await judge.evaluate(traj)

    print(f"Done! ({result.latency_ms:.0f}ms)")
    print(f"\nEvaluation Result:")
    print(f"  Outcome: {result.outcome.value}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Reasoning: {result.reasoning}")
    if result.failure_notes:
        print(f"  Failure Notes: {result.failure_notes}")

    # Mark trajectory as evaluated
    trajectory_store.mark_evaluated(
        trajectory_id=traj.id,
        success=(result.outcome == JudgeOutcome.SUCCESS),
        confidence=result.confidence,
        notes=result.reasoning
    )
    print("\n✓ Trajectory marked as evaluated")

    return result


async def demo_strategy_extractor():
    """Demonstrate StrategyExtractor functionality."""
    print_section("3. StrategyExtractor Demo")

    print("StrategyExtractor distills generalizable strategies from successful runs.")
    print("It uses deduplication to prevent strategy bloat.\n")

    # Create ReasoningBank store
    emb = Embeddings(encode_embeddings)
    rb_store = ReasoningBankStore(".vel/demo_strategies.db", emb)

    # Create extractor
    extractor = StrategyExtractor(
        reasoning_bank_store=rb_store,
        model_config={"provider": "openai", "model": "gpt-4o-mini"},
        similarity_threshold=0.85,
        min_trajectory_steps=2,
        max_strategies_per_run=1
    )

    # Create a mock successful trajectory
    trajectory = {
        "run_id": "successful-run-001",
        "signature": {"intent": "planning", "domain": "api", "risk": "low"},
        "input_message": "Help me design a REST API for user authentication",
        "messages": [
            {"role": "user", "content": "Help me design a REST API for user authentication"},
            {"role": "assistant", "content": "I'll design a secure auth API..."},
            {"role": "user", "content": "What about refresh tokens?"},
            {"role": "assistant", "content": "Great question! For refresh tokens..."}
        ],
        "tool_calls": [
            {"tool_name": "search_docs", "input": {"query": "OAuth2"}, "output": {"results": []}},
            {"tool_name": "validate_schema", "input": {"schema": {}}, "output": {"valid": True}}
        ],
        "final_answer": "Here's a complete auth API design with JWT tokens and refresh mechanism...",
        "error": None
    }

    judge_result = {"confidence": 0.9, "reasoning": "Comprehensive and secure design"}

    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set - using mock extraction\n")

        # Mock extraction
        mock_strategy = {
            "strategy_text": "When designing authentication APIs, always include token refresh mechanisms to improve security without sacrificing user experience.",
            "anti_patterns": ["Storing tokens in localStorage", "Long-lived access tokens"],
            "reasoning": "This approach balances security with usability."
        }

        print("Mock Extracted Strategy:")
        print(f"  Text: {mock_strategy['strategy_text']}")
        print(f"  Anti-patterns: {mock_strategy['anti_patterns']}")

        # Add to store
        sid = rb_store.upsert_strategy(
            signature={"intent": "planning", "domain": "api", "risk": "low"},
            strategy_text=mock_strategy["strategy_text"],
            anti_patterns=mock_strategy["anti_patterns"],
            evidence_refs=["successful-run-001"],
            confidence=0.6
        )
        print(f"\n✓ Strategy stored with ID: {sid}")
        return rb_store

    print("Extracting strategy from trajectory...")

    # Extract strategy
    strategy = await extractor.extract(trajectory, judge_result)

    if strategy:
        print(f"\nExtracted Strategy:")
        print(f"  Text: {strategy.strategy_text}")
        print(f"  Anti-patterns: {strategy.anti_patterns}")
        print(f"  Initial confidence: {strategy.initial_confidence:.2f}")
        print(f"  Source: {strategy.source_trajectory_id}")

        # Store it
        strategy_id = await extractor.extract_and_store(trajectory, judge_result)
        if strategy_id:
            print(f"\n✓ Strategy stored with ID: {strategy_id}")
        else:
            print("\n✓ Strategy was duplicate - not stored")
    else:
        print("No strategy extracted (trajectory may be too short or no useful pattern)")

    return rb_store


def demo_bayesian_confidence():
    """Demonstrate Bayesian confidence updates."""
    print_section("4. Bayesian Confidence Updates Demo")

    print("Vel uses Bayesian-style multiplicative confidence updates:")
    print("  - Success: confidence × 1.20 (capped at 95%)")
    print("  - Failure: confidence × 0.85 (floored at 5%)")
    print("\nThis provides exponential convergence to reliable strategies.\n")

    # Simulate confidence evolution
    print("Scenario: Strategy starts at 0.5 confidence\n")

    conf = 0.5
    events = [
        (True, "First use - success"),
        (True, "Second use - success"),
        (False, "Third use - failure"),
        (True, "Fourth use - success"),
        (True, "Fifth use - success"),
        (True, "Sixth use - success"),
    ]

    print(f"{'Event':<30} {'Outcome':<10} {'Confidence':<12}")
    print("-" * 52)
    print(f"{'Initial':<30} {'-':<10} {conf:.4f}")

    for success, description in events:
        conf = update_confidence_bayesian(conf, success)
        outcome = "SUCCESS" if success else "FAILURE"
        print(f"{description:<30} {outcome:<10} {conf:.4f}")

    print(f"\nFinal confidence: {conf:.4f}")
    print("Note: Confidence naturally caps at 0.95 after multiple successes")


def demo_memory_consolidator(rb_store):
    """Demonstrate MemoryConsolidator functionality."""
    print_section("5. MemoryConsolidator Demo")

    print("MemoryConsolidator maintains memory health by:")
    print("  1. Merging similar strategies (prevents fragmentation)")
    print("  2. Pruning low-confidence strategies (prevents bloat)")
    print("  3. Applying confidence decay to unused strategies")
    print("  4. Enforcing maximum strategy count\n")

    # Add some test strategies
    print("Adding test strategies with varying confidence...\n")

    test_strategies = [
        ("Always validate user input before processing", 0.8),
        ("Always validate input data before processing", 0.7),  # Similar to above
        ("Use pagination for list endpoints", 0.6),
        ("Implement rate limiting for public APIs", 0.5),
        ("Old unused strategy", 0.15),  # Will be pruned
    ]

    for text, conf in test_strategies:
        rb_store.upsert_strategy(
            signature={"intent": "general"},
            strategy_text=text,
            confidence=conf
        )
        print(f"  Added: '{text[:40]}...' (conf: {conf})")

    # Create consolidator
    consolidator = MemoryConsolidator(
        reasoning_bank_store=rb_store,
        similarity_threshold=0.85,
        max_strategies=100,
        min_confidence=0.20,
        decay_rate=0.05
    )

    # Run consolidation (dry run first)
    print("\nDry run consolidation (preview only):")
    preview = consolidator.consolidate(dry_run=True)
    print(f"  Would merge: {preview.strategies_merged} strategies")
    print(f"  Would prune: {preview.strategies_pruned} strategies")
    print(f"  Before: {preview.total_strategies_before} → After: {preview.total_strategies_after}")

    # Run actual consolidation
    print("\nRunning actual consolidation...")
    result = consolidator.consolidate()
    print(f"  Merged: {result.strategies_merged}")
    print(f"  Pruned: {result.strategies_pruned}")
    print(f"  Decayed: {result.strategies_decayed}")
    print(f"  Final count: {result.total_strategies_after}")


def demo_seed_strategies():
    """Demonstrate seed strategies for research/critical thinking."""
    print_section("6. Example Seed Strategies")

    print("Vel includes example seed strategies for common patterns.")
    print("These are templates - you should customize for your use case.\n")

    print(f"Total example strategies: {len(EXAMPLE_SEED_STRATEGIES)}\n")

    # Group by intent
    by_intent = {}
    for s in EXAMPLE_SEED_STRATEGIES:
        intent = s["signature"]["intent"]
        if intent not in by_intent:
            by_intent[intent] = []
        by_intent[intent].append(s)

    for intent, strategies in by_intent.items():
        print(f"{intent.upper()} ({len(strategies)} strategies):")
        for s in strategies[:2]:  # Show first 2 per category
            print(f"  • {s['strategy_text'][:60]}...")
            if s["anti_patterns"]:
                print(f"    Anti-patterns: {', '.join(s['anti_patterns'][:2])}")
        if len(strategies) > 2:
            print(f"  ... and {len(strategies) - 2} more")
        print()


async def demo_auto_learning_manager():
    """Demonstrate AutoLearningManager (brief overview)."""
    print_section("7. AutoLearningManager Overview")

    print("AutoLearningManager orchestrates all Phase 2 components:")
    print("  - EvaluationWorker: Evaluates trajectories with LLM-as-Judge")
    print("  - ExtractionWorker: Extracts strategies from successful runs")
    print("  - ConsolidationWorker: Periodic memory maintenance\n")

    print("Configuration options:")
    config = AutoLearningConfig(
        enabled=True,
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        evaluation_interval_seconds=60,
        extraction_interval_seconds=120,
        consolidation_interval_seconds=21600,  # 6 hours
        max_strategies=1000,
        min_confidence_threshold=0.20
    )

    print(f"  LLM: {config.llm_provider}/{config.llm_model}")
    print(f"  Evaluation interval: {config.evaluation_interval_seconds}s")
    print(f"  Extraction interval: {config.extraction_interval_seconds}s")
    print(f"  Consolidation interval: {config.consolidation_interval_seconds}s (6h)")
    print(f"  Max strategies: {config.max_strategies}")
    print(f"  Min confidence: {config.min_confidence_threshold}")

    print("\nUsage (async):")
    print("""
    manager = AutoLearningManager(
        config=config,
        trajectory_store=traj_store,
        reasoning_bank_store=rb_store,
        reasoning_bank=rb
    )

    await manager.start()   # Start background workers
    # ... agent runs ...
    await manager.stop()    # Graceful shutdown
    """)


async def main():
    """Run all demos."""
    print_section("ReasoningBank Phase 2: Auto-Learning Demo")

    print("This demo showcases the complete auto-learning pipeline.")
    print("Phase 2 enables agents to automatically learn from experience.\n")

    print("Components demonstrated:")
    print("  1. TrajectoryStore - Records execution traces")
    print("  2. LLM-as-Judge - Automatic evaluation")
    print("  3. StrategyExtractor - Strategy distillation")
    print("  4. Bayesian Confidence - Smart updates")
    print("  5. MemoryConsolidator - Memory maintenance")
    print("  6. Seed Strategies - Example patterns")
    print("  7. AutoLearningManager - Orchestration")

    # Run demos
    traj_store, run_id = await demo_trajectory_store()
    await demo_llm_judge(traj_store, run_id)
    rb_store = await demo_strategy_extractor()
    demo_bayesian_confidence()
    demo_memory_consolidator(rb_store)
    demo_seed_strategies()
    await demo_auto_learning_manager()

    # Summary
    print_section("Summary")

    print("✓ Phase 2 Auto-Learning Demo Complete!\n")

    print("The learning cycle in production:")
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                        Agent Execution                           │
    │  User Query → Agent Run → Tool Calls → Final Answer             │
    └────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                     TrajectoryStore                              │
    │  Records: messages, tool calls, timing, strategies used          │
    └────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                      LLM-as-Judge                                │
    │  Evaluates: success/failure, confidence, failure notes          │
    └────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼ (if successful)
    ┌─────────────────────────────────────────────────────────────────┐
    │                    StrategyExtractor                             │
    │  Distills: generalizable strategy, anti-patterns                 │
    │  Deduplicates: embedding similarity check                        │
    └────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    ReasoningBank                                 │
    │  Stores: strategy text, signature, confidence, evidence          │
    │  Updates: Bayesian confidence (×1.2 success / ×0.85 failure)    │
    └────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼ (periodically)
    ┌─────────────────────────────────────────────────────────────────┐
    │                   MemoryConsolidator                             │
    │  Merges: similar strategies                                      │
    │  Prunes: low-confidence strategies                               │
    │  Decays: unused strategies                                       │
    └─────────────────────────────────────────────────────────────────┘
    """)

    print("Next steps:")
    print("  1. Set OPENAI_API_KEY for full LLM-powered evaluation")
    print("  2. Integrate with your Agent using AutoLearningManager")
    print("  3. Customize seed strategies for your domain")
    print("  4. Monitor strategy evolution over time")
    print("\nSee docs/Memory/reasoningbank-phase2-roadmap.md for full documentation")


if __name__ == "__main__":
    asyncio.run(main())
