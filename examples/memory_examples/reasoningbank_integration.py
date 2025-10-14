#!/usr/bin/env python3
"""
ReasoningBank — Full E2E Demo with Streaming

This example demonstrates:
1. Setting up ReasoningBank with embeddings
2. Seeding initial strategies
3. Retrieving strategy advice before agent run
4. Running agent with streaming output
5. Updating confidence scores based on outcomes
"""

import asyncio
import os
from typing import Any, Dict, List
from datetime import datetime
import numpy as np
import hashlib

from agents import Agent
from agents.core import ContextManager, MemoryConfig


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# Minimal deterministic embedding function (for demo purposes)
def encode_embeddings(texts: List[str]) -> np.ndarray:
    """
    Hash-based embeddings for demonstration.

    In production, use sentence-transformers or OpenAI embeddings.
    """
    out = []
    for t in texts:
        h = hashlib.sha256(t.encode()).digest()
        v = np.frombuffer(h, dtype=np.uint8).astype(np.float32)[:256]
        v = (v - v.mean()) / (v.std() + 1e-8)
        out.append(v)
    return np.vstack(out)


async def main():
    print_section("ReasoningBank Memory Demo")

    # Ensure API key is set
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  Please set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable")
        print("   export ANTHROPIC_API_KEY=sk-ant-...")
        print("   or")
        print("   export OPENAI_API_KEY=sk-...")
        return

    # Choose provider based on available API key
    if os.environ.get("ANTHROPIC_API_KEY"):
        model_config = {"provider": "anthropic", "model": "claude-sonnet-4"}
        print("Using Anthropic Claude Sonnet 4")
    else:
        model_config = {"provider": "openai", "model": "gpt-4o"}
        print("Using OpenAI GPT-4o")

    # ========================================================================
    # Step 1: Configure ReasoningBank
    # ========================================================================
    print_section("Step 1: Configure ReasoningBank")

    mem = MemoryConfig(
        mode="reasoning",        # Enable ReasoningBank
        db_path=".vel/demo.db",      # SQLite database path
        rb_top_k=3,                  # Retrieve top 3 strategies
        embeddings_fn=encode_embeddings  # Embedding function
    )
    print(f"✓ Memory configured: mode={mem.mode}, db={mem.db_path}")
    print(f"✓ Using hash-based embeddings (for demo; use sentence-transformers in prod)")

    # Initialize context manager with memory
    ctx = ContextManager(max_history=10)
    ctx.set_memory_config(mem)
    print("✓ Context manager initialized with ReasoningBank")

    # ========================================================================
    # Step 2: Seed Initial Strategies
    # ========================================================================
    print_section("Step 2: Seed Initial Strategies")

    rb = ctx._adapters.get("rb")
    if not rb:
        print("❌ ReasoningBank not available")
        return

    # Check if strategies already exist
    existing = rb.store.retrieve({"intent": "planning"}, k=10, min_conf=0.0)

    if len(existing) == 0:
        print("No existing strategies found. Seeding initial strategies...\n")

        # Seed some initial strategies for different task types
        initial_strategies = [
            {
                "signature": {"intent": "planning", "domain": "api", "risk": "low"},
                "strategy_text": "Break the project into clear phases: setup, core features, testing, deployment",
                "anti_patterns": ["Start coding without a plan", "Skip dependency analysis"],
                "confidence": 0.7
            },
            {
                "signature": {"intent": "planning", "domain": "api", "risk": "medium"},
                "strategy_text": "Always clarify the user's requirements before proposing architecture",
                "anti_patterns": ["Assume requirements", "Over-engineer solutions"],
                "confidence": 0.8
            },
            {
                "signature": {"intent": "debugging", "domain": "backend", "risk": "high"},
                "strategy_text": "Start by examining error messages and logs before modifying code",
                "anti_patterns": ["Make random changes", "Skip error analysis"],
                "confidence": 0.75
            },
            {
                "signature": {"intent": "refactoring", "domain": "python", "risk": "low"},
                "strategy_text": "Run existing tests before and after refactoring to ensure correctness",
                "anti_patterns": ["Refactor without tests", "Change too much at once"],
                "confidence": 0.65
            }
        ]

        for strategy in initial_strategies:
            sid = rb.store.upsert_strategy(
                signature=strategy["signature"],
                strategy_text=strategy["strategy_text"],
                anti_patterns=strategy["anti_patterns"],
                confidence=strategy["confidence"]
            )
            print(f"✓ Seeded strategy {sid}: {strategy['strategy_text'][:60]}...")
    else:
        print(f"Found {len(existing)} existing strategies. Using existing memory.\n")

    # ========================================================================
    # Step 3: Retrieve Strategy Advice
    # ========================================================================
    print_section("Step 3: Retrieve Strategy Advice")

    # Define task signature
    signature = {
        "intent": "planning",
        "domain": "api",
        "risk": "low"
    }
    print(f"Task signature: {signature}\n")

    # Get strategy advice
    advice = ctx.prepare_for_run(signature)

    if advice:
        print("Retrieved strategy advice:")
        print("-" * 60)
        print(advice)
        print("-" * 60)
    else:
        print("No relevant strategies found (this is normal if DB is empty)")
        advice = ""

    # ========================================================================
    # Step 4: Run Agent with Strategy Advice (Streaming)
    # ========================================================================
    print_section("Step 4: Run Agent with Streaming Output")

    # Create agent
    agent = Agent(
        id="reasoningbank-demo:v1",
        model=model_config,
        tools=[],
        policies={"max_steps": 5}
    )

    # User's question
    user_message = "Help me plan a FastAPI project for a URL shortener service. What architecture should I use?"

    print(f"User: {user_message}\n")

    # Build full message with strategy advice
    system_context = ""
    if advice:
        system_context = f"""
{advice}

Please use the above strategies to guide your response.
"""

    full_message = f"{system_context}\nUSER REQUEST:\n{user_message}"

    print("Assistant: ", end="", flush=True)

    # Stream response
    session_id = "session-reasoningbank-demo"
    response_text = ""

    try:
        async for event in agent.run_stream({"message": full_message}, session_id=session_id):
            if event.get("type") == "text-delta":
                delta = event.get("delta", "")
                print(delta, end="", flush=True)
                response_text += delta
            elif event.get("type") == "error":
                print(f"\n❌ Error: {event.get('error')}")
                break

        print("\n")

    except Exception as e:
        print(f"\n❌ Error during agent run: {e}")
        return

    # ========================================================================
    # Step 5: Update Confidence Based on Outcome
    # ========================================================================
    print_section("Step 5: Update Strategy Confidence")

    # In real usage, you'd evaluate if the response was good
    # For demo, we'll assume success
    run_success = True

    if run_success:
        print("✓ Run succeeded - updating confidence scores for retrieved strategies")
        ctx.finalize_outcome(run_success=True, fail_notes=[])
    else:
        print("✗ Run failed - decreasing confidence and adding anti-patterns")
        ctx.finalize_outcome(
            run_success=False,
            fail_notes=["Response was too vague", "Missed key requirements"]
        )

    # ========================================================================
    # Step 6: Show Updated Strategies
    # ========================================================================
    print_section("Step 6: View Updated Strategies")

    # Retrieve strategies again to show updated confidence
    updated_strategies = rb.store.retrieve(signature, k=5, min_conf=0.0)

    print(f"Strategies for signature: {signature}\n")
    for i, strategy in enumerate(updated_strategies, 1):
        print(f"{i}. {strategy.strategy_text}")
        print(f"   Confidence: {strategy.confidence:.2f}")
        if strategy.anti_patterns:
            print(f"   Anti-patterns: {', '.join(strategy.anti_patterns[:2])}")
        print()

    # ========================================================================
    # Demonstrate Different Task Types
    # ========================================================================
    print_section("Demonstration: Different Task Types")

    print("Showing strategies for different task signatures:\n")

    test_signatures = [
        {"intent": "debugging", "domain": "backend", "risk": "high"},
        {"intent": "refactoring", "domain": "python", "risk": "low"},
    ]

    for sig in test_signatures:
        print(f"Signature: {sig}")
        strats = rb.store.retrieve(sig, k=2, min_conf=0.0)
        if strats:
            for s in strats:
                print(f"  → {s.strategy_text} (confidence: {s.confidence:.2f})")
        else:
            print("  → No strategies found")
        print()

    # ========================================================================
    # Summary
    # ========================================================================
    print_section("Summary")

    print("✓ ReasoningBank Demo Complete!")
    print("\nWhat we demonstrated:")
    print("  1. Configured ReasoningBank with embeddings")
    print("  2. Seeded initial strategies (Phase 1 - manual)")
    print("  3. Retrieved relevant strategy advice via embedding similarity")
    print("  4. Ran agent with streaming output using strategy advice")
    print("  5. Updated confidence scores based on outcome")
    print("  6. Showed how strategies evolve over time")
    print("\nMemory persists in:", mem.db_path)
    print("\nKey Insight:")
    print("  - Phase 1: You manually add strategies (as shown here)")
    print("  - Phase 2: Strategies learned automatically from trajectories")
    print("  - See docs/Memory/reasoningbank-phase2-roadmap.md for Phase 2 plans")
    print("\nTry running this again to see confidence scores update! 🎉")


if __name__ == "__main__":
    asyncio.run(main())
