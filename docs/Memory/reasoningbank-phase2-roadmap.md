---
layout: default
title: ReasoningBank Phase 2 Roadmap
parent: Memory System
nav_order: 6
---

# ReasoningBank Phase 2: Full Academic Implementation

This document outlines how to extend Vel's current ReasoningBank implementation to match the full capabilities described in the Google Research paper ([arxiv:2509.25140](https://arxiv.org/abs/2509.25140)).

**Status:** Roadmap for future development
**Current Phase:** Phase 1 (Infrastructure complete)
**Target Phase:** Phase 2 (Automatic self-evolution)

---

## Current State vs. Target State

### Phase 1: Current Implementation ✅

**What Vel Has Now:**

| Component | Status | Description |
|-----------|--------|-------------|
| Storage Infrastructure | ✅ Complete | SQLite with `rb_strategies` and `rb_embeddings` tables |
| Embedding-Based Retrieval | ✅ Complete | Cosine similarity search with hybrid scoring |
| Confidence Scoring | ✅ Complete | Incremental updates (±0.1 per outcome) |
| Anti-Pattern Storage | ✅ Complete | Accumulates failure notes |
| Pre-Run Advice Injection | ✅ Complete | `prepare_for_run(signature)` returns formatted advice |
| Post-Run Confidence Update | ✅ Complete | `finalize_outcome(success, fail_notes)` updates scores |
| Runtime-Owned Memory | ✅ Complete | No LLM tool calls, bounded latency |

**What's Manual:**
- ❌ Strategy creation (user must insert)
- ❌ Success/failure evaluation (user provides boolean)
- ❌ Anti-pattern generation (user provides strings)
- ❌ Trajectory analysis (not implemented)

---

### Phase 2: Target Implementation (Google Paper)

**What We Need to Add:**

| Component | Priority | Complexity | Description |
|-----------|----------|------------|-------------|
| Trajectory Storage | High | Medium | Store raw agent trajectories for analysis |
| LLM-as-Judge | High | Medium | Automatic success/failure evaluation |
| Strategy Distillation | High | High | Extract strategies from trajectories automatically |
| Anti-Pattern Extraction | Medium | Medium | Generate "what to avoid" from failures |
| Memory Consolidation | Medium | Medium | Deduplicate and merge similar strategies |
| Parallel Scaling | Low | High | Generate multiple trajectories per query |
| Sequential Scaling | Low | Medium | Iterative refinement within single trajectory |

---

## Architecture Overview

### Current Phase 1 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Agent Runtime                        │
└────────────┬────────────────────────────────────────────┘
             │
      ┌──────┴────────┐
      │ ContextManager │
      └──────┬────────┘
             │
   ┌─────────┴─────────┐
   │  prepare_for_run  │ (retrieval)
   │  finalize_outcome │ (confidence update)
   └─────────┬─────────┘
             │
   ┌─────────┴─────────┐
   │  ReasoningBank    │
   └─────────┬─────────┘
             │
   ┌─────────┴─────────┐
   │ ReasoningBankStore│ (SQLite + embeddings)
   └───────────────────┘

USER PROVIDES:
  - Strategy items
  - Success boolean
  - Fail notes
```

### Target Phase 2 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Agent Runtime                        │
└────────────┬────────────────────────────────────────────┘
             │
      ┌──────┴────────┐
      │ ContextManager │
      └──────┬────────┘
             │
   ┌─────────┴──────────────┐
   │  prepare_for_run       │ (retrieval)
   │  finalize_outcome      │ (confidence update)
   │  record_trajectory     │ ← NEW
   └─────────┬──────────────┘
             │
   ┌─────────┴─────────────────────┐
   │      ReasoningBank            │
   └─────────┬─────────────────────┘
             │
   ┌─────────┴─────────────────────┐
   │  ReasoningBankStore           │
   │  + TrajectoryStore       ← NEW│
   └─────────┬─────────────────────┘
             │
   ┌─────────┴─────────────────────┐
   │  Strategy Learning Pipeline   │ ← NEW
   │                               │
   │  1. LLMJudge                  │
   │  2. StrategyExtractor         │
   │  3. AntiPatternGenerator      │
   │  4. MemoryConsolidator        │
   └───────────────────────────────┘

AUTOMATIC:
  - Trajectory recording
  - Success evaluation
  - Strategy extraction
  - Anti-pattern generation
  - Memory consolidation
```

---

## Implementation Plan

### Milestone 1: Trajectory Storage

**Goal:** Capture and store raw agent trajectories for analysis.

#### 1.1 Database Schema

Add new table for trajectories:

```sql
CREATE TABLE rb_trajectories (
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL,
  session_id TEXT,
  signature_json TEXT NOT NULL,
  messages TEXT NOT NULL,              -- JSON array of messages
  tool_calls TEXT DEFAULT '[]',        -- JSON array of tool calls
  final_answer TEXT,
  error TEXT,
  created_at REAL DEFAULT (strftime('%s','now')),
  evaluated BOOLEAN DEFAULT 0,         -- Has LLM-as-Judge run?
  success BOOLEAN,                     -- Judge's verdict
  strategies_extracted BOOLEAN DEFAULT 0,
  UNIQUE(run_id)
);

CREATE INDEX idx_traj_eval ON rb_trajectories(evaluated, success);
CREATE INDEX idx_traj_extracted ON rb_trajectories(strategies_extracted);
```

#### 1.2 Implementation

Create `agents/memory/trajectory_store.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path
import sqlite3, json
from time import time

@dataclass
class Trajectory:
    id: Optional[int]
    run_id: str
    session_id: Optional[str]
    signature: Dict[str, Any]
    messages: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]
    final_answer: Optional[str]
    error: Optional[str]
    evaluated: bool
    success: Optional[bool]
    strategies_extracted: bool

class TrajectoryStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self.db.execute("PRAGMA journal_mode=WAL;")
        self.db.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        # Schema from above
        pass

    def record_trajectory(
        self,
        run_id: str,
        signature: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
        final_answer: Optional[str] = None,
        error: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> int:
        """Store a trajectory for later analysis."""
        self.db.execute("""
            INSERT INTO rb_trajectories(run_id, session_id, signature_json, messages, tool_calls, final_answer, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                messages=excluded.messages,
                tool_calls=excluded.tool_calls,
                final_answer=excluded.final_answer,
                error=excluded.error
        """, (
            run_id,
            session_id,
            json.dumps(signature),
            json.dumps(messages),
            json.dumps(tool_calls),
            final_answer,
            error
        ))
        self.db.commit()
        return self.db.execute("SELECT id FROM rb_trajectories WHERE run_id=?", (run_id,)).fetchone()["id"]

    def get_unevaluated_trajectories(self, limit: int = 100) -> List[Trajectory]:
        """Get trajectories that haven't been evaluated by LLM-as-Judge."""
        rows = self.db.execute("""
            SELECT * FROM rb_trajectories
            WHERE evaluated = 0
            ORDER BY created_at ASC
            LIMIT ?
        """, (limit,)).fetchall()
        return [self._row_to_trajectory(r) for r in rows]

    def get_successful_unevaluated(self, limit: int = 100) -> List[Trajectory]:
        """Get successful trajectories that haven't had strategies extracted."""
        rows = self.db.execute("""
            SELECT * FROM rb_trajectories
            WHERE evaluated = 1 AND success = 1 AND strategies_extracted = 0
            ORDER BY created_at ASC
            LIMIT ?
        """, (limit,)).fetchall()
        return [self._row_to_trajectory(r) for r in rows]

    def mark_evaluated(self, trajectory_id: int, success: bool):
        """Mark trajectory as evaluated."""
        self.db.execute("""
            UPDATE rb_trajectories
            SET evaluated=1, success=?
            WHERE id=?
        """, (1 if success else 0, trajectory_id))
        self.db.commit()

    def mark_strategies_extracted(self, trajectory_id: int):
        """Mark that strategies have been extracted from this trajectory."""
        self.db.execute("""
            UPDATE rb_trajectories
            SET strategies_extracted=1
            WHERE id=?
        """, (trajectory_id,))
        self.db.commit()

    def _row_to_trajectory(self, row) -> Trajectory:
        return Trajectory(
            id=row["id"],
            run_id=row["run_id"],
            session_id=row["session_id"],
            signature=json.loads(row["signature_json"]),
            messages=json.loads(row["messages"]),
            tool_calls=json.loads(row["tool_calls"]),
            final_answer=row["final_answer"],
            error=row["error"],
            evaluated=bool(row["evaluated"]),
            success=bool(row["success"]) if row["success"] is not None else None,
            strategies_extracted=bool(row["strategies_extracted"])
        )
```

#### 1.3 Integration with ContextManager

Update `agents/core/context.py`:

```python
class ContextManager:
    # ... existing code ...

    def record_trajectory(
        self,
        run_id: str,
        signature: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
        final_answer: Optional[str] = None,
        error: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Record trajectory for later analysis (Phase 2 feature).
        No-op if trajectory storage is not enabled.
        """
        traj_store = self._adapters.get("trajectory")
        if traj_store:
            traj_store.record_trajectory(
                run_id, signature, messages, tool_calls,
                final_answer, error, session_id
            )
```

Update `build_memory_adapters()` to include trajectory store when mode includes "reasoningbank".

---

### Milestone 2: LLM-as-Judge

**Goal:** Automatically evaluate whether a trajectory was successful.

#### 2.1 Judge Implementation

Create `agents/memory/llm_judge.py`:

```python
from __future__ import annotations
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class JudgeResult:
    success: bool
    reasoning: str
    confidence: float  # 0.0-1.0

class LLMJudge:
    """
    Evaluates trajectories for success/failure.

    Based on the ReasoningBank paper's LLM-as-Judge approach.
    """

    def __init__(self, model_config: Dict[str, Any]):
        """
        Args:
            model_config: {'provider': 'anthropic', 'model': 'claude-sonnet-4'}
        """
        self.model_config = model_config

    async def evaluate_trajectory(
        self,
        signature: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
        final_answer: Optional[str],
        error: Optional[str]
    ) -> JudgeResult:
        """
        Evaluate if the trajectory successfully completed its task.

        Returns:
            JudgeResult with success boolean, reasoning, and confidence
        """
        from agents import Agent

        # Format trajectory for analysis
        trajectory_text = self._format_trajectory(
            signature, messages, tool_calls, final_answer, error
        )

        # Create judge agent
        judge = Agent(
            id='llm-judge:v1',
            model=self.model_config,
            tools=[]
        )

        prompt = self._build_judge_prompt(trajectory_text)

        # Get judgment
        result = await judge.run({"message": prompt})

        # Parse result
        return self._parse_judge_response(result)

    def _format_trajectory(
        self,
        signature: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
        final_answer: Optional[str],
        error: Optional[str]
    ) -> str:
        """Format trajectory for judge analysis."""
        lines = [
            "=== TASK SIGNATURE ===",
            f"Intent: {signature.get('intent', 'unknown')}",
            f"Domain: {signature.get('domain', 'unknown')}",
            f"Risk: {signature.get('risk', 'unknown')}",
            "",
            "=== TRAJECTORY ===",
        ]

        for i, msg in enumerate(messages, 1):
            role = msg.get('role', 'unknown')
            content = str(msg.get('content', ''))[:500]  # Truncate long messages
            lines.append(f"[{i}] {role}: {content}")

        if tool_calls:
            lines.append("")
            lines.append("=== TOOL CALLS ===")
            for i, tool in enumerate(tool_calls, 1):
                lines.append(f"[{i}] {tool.get('name', 'unknown')}: {tool.get('args', {})}")

        if final_answer:
            lines.append("")
            lines.append("=== FINAL ANSWER ===")
            lines.append(final_answer)

        if error:
            lines.append("")
            lines.append("=== ERROR ===")
            lines.append(error)

        return "\n".join(lines)

    def _build_judge_prompt(self, trajectory_text: str) -> str:
        """Build prompt for LLM judge."""
        return f"""You are an expert evaluator analyzing an AI agent's task execution.

Your job is to determine whether the agent successfully completed its intended task.

Criteria for SUCCESS:
- Agent understood and addressed the core task
- Final answer is relevant and helpful
- No critical errors or failures
- Reasoning was sound

Criteria for FAILURE:
- Agent misunderstood the task
- Got stuck in loops or errors
- Produced irrelevant or incorrect output
- Critical errors occurred

Analyze the trajectory below and respond in JSON format:

{{
  "success": true/false,
  "reasoning": "Brief explanation of why this succeeded or failed",
  "confidence": 0.0-1.0
}}

{trajectory_text}

Respond ONLY with the JSON object, no other text."""

    def _parse_judge_response(self, response: str) -> JudgeResult:
        """Parse judge's JSON response."""
        import json
        import re

        # Extract JSON from response (might have markdown code blocks)
        json_match = re.search(r'\{[^\}]+\}', response, re.DOTALL)
        if not json_match:
            # Default to failure if can't parse
            return JudgeResult(success=False, reasoning="Parse error", confidence=0.5)

        try:
            data = json.loads(json_match.group(0))
            return JudgeResult(
                success=bool(data.get("success", False)),
                reasoning=str(data.get("reasoning", "")),
                confidence=float(data.get("confidence", 0.5))
            )
        except Exception:
            return JudgeResult(success=False, reasoning="Parse error", confidence=0.5)
```

#### 2.2 Background Evaluation Worker

Create `agents/memory/evaluation_worker.py`:

```python
from __future__ import annotations
import asyncio
from typing import Optional
from .trajectory_store import TrajectoryStore
from .llm_judge import LLMJudge

class EvaluationWorker:
    """
    Background worker that continuously evaluates unevaluated trajectories.
    """

    def __init__(
        self,
        trajectory_store: TrajectoryStore,
        judge: LLMJudge,
        batch_size: int = 10,
        interval_seconds: int = 60
    ):
        self.trajectory_store = trajectory_store
        self.judge = judge
        self.batch_size = batch_size
        self.interval_seconds = interval_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the background evaluation worker."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self):
        """Stop the background evaluation worker."""
        self._running = False
        if self._task:
            await self._task

    async def _run_loop(self):
        """Main evaluation loop."""
        while self._running:
            try:
                await self._evaluate_batch()
            except Exception as e:
                # Log error but keep running
                print(f"Evaluation worker error: {e}")

            # Wait before next batch
            await asyncio.sleep(self.interval_seconds)

    async def _evaluate_batch(self):
        """Evaluate a batch of trajectories."""
        trajectories = self.trajectory_store.get_unevaluated_trajectories(self.batch_size)

        for traj in trajectories:
            try:
                # Evaluate with LLM judge
                result = await self.judge.evaluate_trajectory(
                    signature=traj.signature,
                    messages=traj.messages,
                    tool_calls=traj.tool_calls,
                    final_answer=traj.final_answer,
                    error=traj.error
                )

                # Mark as evaluated
                self.trajectory_store.mark_evaluated(traj.id, result.success)

            except Exception as e:
                print(f"Error evaluating trajectory {traj.id}: {e}")
                # Mark as evaluated (failure) to avoid retry loop
                self.trajectory_store.mark_evaluated(traj.id, False)
```

---

### Milestone 3: Strategy Distillation

**Goal:** Automatically extract strategies from successful trajectories.

#### 3.1 Strategy Extractor Implementation

Create `agents/memory/strategy_extractor.py`:

```python
from __future__ import annotations
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class ExtractedStrategy:
    strategy_text: str
    anti_patterns: List[str]
    confidence: float
    reasoning: str

class StrategyExtractor:
    """
    Extracts generalizable strategies from successful trajectories.

    Based on the ReasoningBank paper's strategy distillation approach.
    """

    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config

    async def extract_strategy(
        self,
        signature: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
        final_answer: str
    ) -> Optional[ExtractedStrategy]:
        """
        Extract a generalizable strategy from a successful trajectory.

        Returns:
            ExtractedStrategy if one can be extracted, None otherwise
        """
        from agents import Agent

        # Format trajectory
        trajectory_text = self._format_trajectory(
            signature, messages, tool_calls, final_answer
        )

        # Create extractor agent
        extractor = Agent(
            id='strategy-extractor:v1',
            model=self.model_config,
            tools=[]
        )

        prompt = self._build_extraction_prompt(trajectory_text)

        # Extract strategy
        result = await extractor.run({"message": prompt})

        # Parse result
        return self._parse_extraction_response(result)

    def _format_trajectory(
        self,
        signature: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
        final_answer: str
    ) -> str:
        """Format trajectory for strategy extraction."""
        lines = [
            "=== TASK CONTEXT ===",
            f"Intent: {signature.get('intent', 'unknown')}",
            f"Domain: {signature.get('domain', 'unknown')}",
            "",
            "=== EXECUTION STEPS ===",
        ]

        for i, msg in enumerate(messages, 1):
            role = msg.get('role', 'unknown')
            content = str(msg.get('content', ''))[:300]
            lines.append(f"Step {i} [{role}]: {content}")

        if tool_calls:
            lines.append("")
            lines.append("=== TOOLS USED ===")
            for tool in tool_calls:
                lines.append(f"- {tool.get('name', 'unknown')}")

        lines.append("")
        lines.append("=== OUTCOME ===")
        lines.append(final_answer[:500])

        return "\n".join(lines)

    def _build_extraction_prompt(self, trajectory_text: str) -> str:
        """Build prompt for strategy extraction."""
        return f"""You are an expert at analyzing AI agent reasoning patterns.

Your task is to extract a GENERALIZABLE reasoning strategy from this successful execution.

The strategy should be:
- One clear sentence
- Applicable to similar tasks (not specific to this exact case)
- Actionable (describes HOW to think, not WHAT to think)

Also identify 1-3 anti-patterns (things to AVOID) based on potential failure modes.

Analyze the trajectory and respond in JSON format:

{{
  "strategy_text": "One sentence describing the reasoning approach",
  "anti_patterns": ["Pattern to avoid 1", "Pattern to avoid 2"],
  "confidence": 0.0-1.0,
  "reasoning": "Why this strategy worked"
}}

{trajectory_text}

Respond ONLY with the JSON object, no other text.

IMPORTANT:
- Strategy must be generalizable (not "Use API key abc123")
- Focus on reasoning approach, not specific actions
- Anti-patterns should be cautionary, not just negations"""

    def _parse_extraction_response(self, response: str) -> Optional[ExtractedStrategy]:
        """Parse extractor's JSON response."""
        import json
        import re

        json_match = re.search(r'\{[^\}]+\}', response, re.DOTALL)
        if not json_match:
            return None

        try:
            data = json.loads(json_match.group(0))

            # Validate strategy quality
            strategy_text = data.get("strategy_text", "").strip()
            if len(strategy_text) < 10 or len(strategy_text) > 200:
                return None  # Too short or too long

            return ExtractedStrategy(
                strategy_text=strategy_text,
                anti_patterns=data.get("anti_patterns", [])[:3],  # Max 3
                confidence=float(data.get("confidence", 0.6)),
                reasoning=str(data.get("reasoning", ""))
            )
        except Exception:
            return None
```

#### 3.2 Strategy Extraction Worker

Create `agents/memory/extraction_worker.py`:

```python
from __future__ import annotations
import asyncio
from typing import Optional
from .trajectory_store import TrajectoryStore
from .strategy_reasoningbank import ReasoningBankStore
from .strategy_extractor import StrategyExtractor

class ExtractionWorker:
    """
    Background worker that extracts strategies from successful trajectories.
    """

    def __init__(
        self,
        trajectory_store: TrajectoryStore,
        reasoning_bank: ReasoningBankStore,
        extractor: StrategyExtractor,
        batch_size: int = 10,
        interval_seconds: int = 120
    ):
        self.trajectory_store = trajectory_store
        self.reasoning_bank = reasoning_bank
        self.extractor = extractor
        self.batch_size = batch_size
        self.interval_seconds = interval_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the background extraction worker."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self):
        """Stop the background extraction worker."""
        self._running = False
        if self._task:
            await self._task

    async def _run_loop(self):
        """Main extraction loop."""
        while self._running:
            try:
                await self._extract_batch()
            except Exception as e:
                print(f"Extraction worker error: {e}")

            await asyncio.sleep(self.interval_seconds)

    async def _extract_batch(self):
        """Extract strategies from a batch of successful trajectories."""
        trajectories = self.trajectory_store.get_successful_unevaluated(self.batch_size)

        for traj in trajectories:
            try:
                # Extract strategy
                strategy = await self.extractor.extract_strategy(
                    signature=traj.signature,
                    messages=traj.messages,
                    tool_calls=traj.tool_calls,
                    final_answer=traj.final_answer or ""
                )

                if strategy:
                    # Check for similar existing strategies
                    existing = self.reasoning_bank.retrieve(traj.signature, k=5, min_conf=0.0)

                    # Simple deduplication: skip if very similar strategy exists
                    if not self._is_duplicate(strategy.strategy_text, existing):
                        # Add to ReasoningBank
                        self.reasoning_bank.upsert_strategy(
                            signature=traj.signature,
                            strategy_text=strategy.strategy_text,
                            anti_patterns=strategy.anti_patterns,
                            evidence_refs=[traj.run_id],
                            confidence=strategy.confidence
                        )

                # Mark as extracted
                self.trajectory_store.mark_strategies_extracted(traj.id)

            except Exception as e:
                print(f"Error extracting from trajectory {traj.id}: {e}")
                # Mark as extracted to avoid retry loop
                self.trajectory_store.mark_strategies_extracted(traj.id)

    def _is_duplicate(self, new_strategy: str, existing_strategies) -> bool:
        """
        Simple duplicate check using string similarity.

        In production, use more sophisticated similarity (embeddings, etc.)
        """
        new_lower = new_strategy.lower()

        for existing in existing_strategies:
            existing_lower = existing.strategy_text.lower()

            # Very simple: check if >70% of words overlap
            new_words = set(new_lower.split())
            existing_words = set(existing_lower.split())

            if len(new_words & existing_words) / max(len(new_words), 1) > 0.7:
                return True

        return False
```

---

### Milestone 4: Memory Consolidation

**Goal:** Merge similar strategies to prevent redundancy.

#### 4.1 Strategy Consolidator

Create `agents/memory/consolidator.py`:

```python
from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from .strategy_reasoningbank import ReasoningBankStore, StrategyItem

class MemoryConsolidator:
    """
    Merges similar strategies to prevent memory bloat.

    Based on the ReasoningBank paper's consolidation approach.
    """

    def __init__(
        self,
        reasoning_bank: ReasoningBankStore,
        similarity_threshold: float = 0.85
    ):
        self.reasoning_bank = reasoning_bank
        self.similarity_threshold = similarity_threshold

    def consolidate_strategies(self, signature: Dict[str, Any]) -> int:
        """
        Find and merge similar strategies for a given signature.

        Returns:
            Number of strategies merged
        """
        # Get all strategies for this signature
        strategies = self.reasoning_bank.retrieve(signature, k=100, min_conf=0.0)

        if len(strategies) < 2:
            return 0

        # Build similarity matrix
        embeddings = self._get_embeddings(strategies)
        similarity_matrix = self._compute_similarity_matrix(embeddings)

        # Find clusters of similar strategies
        clusters = self._find_clusters(similarity_matrix, self.similarity_threshold)

        # Merge each cluster
        merged_count = 0
        for cluster in clusters:
            if len(cluster) > 1:
                self._merge_cluster([strategies[i] for i in cluster])
                merged_count += len(cluster) - 1

        return merged_count

    def _get_embeddings(self, strategies: List[StrategyItem]) -> np.ndarray:
        """Get embeddings for all strategies."""
        # In real implementation, retrieve from database
        # For now, re-encode (inefficient but simple)
        texts = [s.strategy_text for s in strategies]
        return self.reasoning_bank.emb.encode(texts)

    def _compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity."""
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)

        # Compute similarity
        return normalized @ normalized.T

    def _find_clusters(
        self,
        similarity_matrix: np.ndarray,
        threshold: float
    ) -> List[List[int]]:
        """Find clusters of similar strategies using simple threshold."""
        n = similarity_matrix.shape[0]
        visited = set()
        clusters = []

        for i in range(n):
            if i in visited:
                continue

            # Start new cluster
            cluster = [i]
            visited.add(i)

            # Find all similar strategies
            for j in range(i + 1, n):
                if j not in visited and similarity_matrix[i, j] >= threshold:
                    cluster.append(j)
                    visited.add(j)

            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    def _merge_cluster(self, strategies: List[StrategyItem]):
        """
        Merge a cluster of similar strategies into one.

        Strategy:
        - Keep the highest confidence strategy as base
        - Merge anti-patterns from all
        - Merge evidence refs from all
        - Update confidence to average
        """
        # Sort by confidence
        sorted_strategies = sorted(strategies, key=lambda s: s.confidence, reverse=True)
        base = sorted_strategies[0]

        # Merge anti-patterns
        all_anti_patterns = set()
        for s in strategies:
            all_anti_patterns.update(s.anti_patterns)

        # Merge evidence refs
        all_evidence = set()
        for s in strategies:
            all_evidence.update(s.evidence_refs)

        # Average confidence
        avg_confidence = sum(s.confidence for s in strategies) / len(strategies)

        # Update base strategy
        self.reasoning_bank.db.execute("""
            UPDATE rb_strategies
            SET anti_patterns = ?,
                evidence_refs = ?,
                confidence = ?
            WHERE id = ?
        """, (
            json.dumps(list(all_anti_patterns)),
            json.dumps(list(all_evidence)),
            avg_confidence,
            base.id
        ))

        # Delete merged strategies
        for s in strategies[1:]:
            self.reasoning_bank.db.execute("""
                DELETE FROM rb_strategies WHERE id = ?
            """, (s.id,))
            self.reasoning_bank.db.execute("""
                DELETE FROM rb_embeddings WHERE strategy_id = ?
            """, (s.id,))

        self.reasoning_bank.db.commit()
```

---

### Milestone 5: Configuration and Integration

**Goal:** Make Phase 2 features opt-in and configurable.

#### 5.1 Extended Memory Config

Update `agents/core/context.py`:

```python
@dataclass
class MemoryConfig:
    """
    Memory configuration.

    Phase 1 fields (existing):
        mode: "none" | "facts" | "reasoning" | "all"
        db_path: SQLite file path
        rb_top_k: top-k strategies to retrieve
        embeddings_fn: embedding function

    Phase 2 fields (new):
        enable_auto_learning: Enable automatic strategy learning
        judge_model: Model config for LLM-as-Judge
        extractor_model: Model config for strategy extraction
        eval_interval_seconds: How often to run evaluation worker
        extraction_interval_seconds: How often to run extraction worker
        consolidation_interval_seconds: How often to consolidate
        min_confidence_threshold: Prune strategies below this confidence
    """
    # Phase 1 (existing)
    mode: str = "none"
    db_path: str = ".vel/vel.db"
    rb_top_k: int = 5
    embeddings_fn: Optional[Callable[[List[str]], "object"]] = None

    # Phase 2 (new)
    enable_auto_learning: bool = False
    judge_model: Optional[Dict[str, Any]] = None
    extractor_model: Optional[Dict[str, Any]] = None
    eval_interval_seconds: int = 60
    extraction_interval_seconds: int = 120
    consolidation_interval_seconds: int = 3600
    min_confidence_threshold: float = 0.3
```

#### 5.2 Auto-Learning Manager

Create `agents/memory/auto_learning.py`:

```python
from __future__ import annotations
from typing import Optional
from .trajectory_store import TrajectoryStore
from .llm_judge import LLMJudge
from .strategy_extractor import StrategyExtractor
from .evaluation_worker import EvaluationWorker
from .extraction_worker import ExtractionWorker
from .consolidator import MemoryConsolidator
from ..core.context import MemoryConfig

class AutoLearningManager:
    """
    Manages all Phase 2 automatic learning components.
    """

    def __init__(
        self,
        config: MemoryConfig,
        trajectory_store: TrajectoryStore,
        reasoning_bank_store
    ):
        self.config = config
        self.trajectory_store = trajectory_store
        self.reasoning_bank = reasoning_bank_store

        # Initialize components
        self.judge = LLMJudge(config.judge_model or {"provider": "anthropic", "model": "claude-sonnet-4"})
        self.extractor = StrategyExtractor(config.extractor_model or {"provider": "anthropic", "model": "claude-sonnet-4"})
        self.consolidator = MemoryConsolidator(reasoning_bank_store)

        # Workers
        self.eval_worker = EvaluationWorker(
            trajectory_store,
            self.judge,
            interval_seconds=config.eval_interval_seconds
        )
        self.extraction_worker = ExtractionWorker(
            trajectory_store,
            reasoning_bank_store,
            self.extractor,
            interval_seconds=config.extraction_interval_seconds
        )

    async def start(self):
        """Start all background workers."""
        await self.eval_worker.start()
        await self.extraction_worker.start()

    async def stop(self):
        """Stop all background workers."""
        await self.eval_worker.stop()
        await self.extraction_worker.stop()
```

#### 5.3 Environment Variables

Add to `.env.example`:

```bash
# Memory Phase 2: Auto-Learning (optional)
VEL_ENABLE_AUTO_LEARNING=false
VEL_JUDGE_MODEL=anthropic:claude-sonnet-4
VEL_EXTRACTOR_MODEL=anthropic:claude-sonnet-4
VEL_EVAL_INTERVAL=60
VEL_EXTRACTION_INTERVAL=120
```

---

## Implementation Phases

### Phase 2.1: Foundation (Weeks 1-2)
- [ ] Implement `TrajectoryStore`
- [ ] Add trajectory recording to `ContextManager`
- [ ] Update `build_memory_adapters()` to include trajectory store
- [ ] Write tests for trajectory storage
- [ ] Update documentation

**Deliverable:** Trajectories are recorded automatically

### Phase 2.2: Evaluation (Weeks 3-4)
- [ ] Implement `LLMJudge`
- [ ] Implement `EvaluationWorker`
- [ ] Add configuration for judge model
- [ ] Write tests for judge accuracy
- [ ] Add metrics/logging for evaluation

**Deliverable:** Trajectories are automatically evaluated

### Phase 2.3: Extraction (Weeks 5-6)
- [ ] Implement `StrategyExtractor`
- [ ] Implement `ExtractionWorker`
- [ ] Add deduplication logic
- [ ] Write tests for extraction quality
- [ ] Add metrics/logging for extraction

**Deliverable:** Strategies are automatically extracted

### Phase 2.4: Consolidation (Weeks 7-8)
- [ ] Implement `MemoryConsolidator`
- [ ] Add periodic consolidation job
- [ ] Add confidence decay mechanism
- [ ] Implement pruning of low-confidence strategies
- [ ] Write tests for consolidation

**Deliverable:** Memory is automatically maintained

### Phase 2.5: Integration (Weeks 9-10)
- [ ] Implement `AutoLearningManager`
- [ ] Update `MemoryConfig` with Phase 2 fields
- [ ] Add opt-in configuration
- [ ] Comprehensive integration tests
- [ ] Performance benchmarks
- [ ] Update all documentation

**Deliverable:** Full Phase 2 system operational

---

## Testing Strategy

### Unit Tests

```python
# Test trajectory storage
def test_trajectory_store_record()
def test_trajectory_store_retrieval()

# Test LLM judge
async def test_judge_success_case()
async def test_judge_failure_case()

# Test strategy extraction
async def test_extract_valid_strategy()
async def test_extract_rejects_poor_quality()

# Test consolidation
def test_find_similar_strategies()
def test_merge_strategies()
```

### Integration Tests

```python
# End-to-end auto-learning
async def test_full_learning_pipeline()
async def test_trajectory_to_strategy()
async def test_confidence_updates_over_time()
```

### Quality Metrics

Track these metrics in production:
- Judge accuracy (compare to human evaluation)
- Strategy quality (user ratings)
- Deduplication effectiveness (clusters found)
- Memory growth rate (strategies per day)
- Retrieval relevance (top-K accuracy)

---

## Backwards Compatibility

### Ensuring Phase 1 Continues to Work

```python
# Phase 1 usage (manual) - still works
mem = MemoryConfig(mode="reasoning", embeddings_fn=encode)
ctx = ContextManager()
ctx.set_memory_config(mem)

# Manually add strategies
rb = ctx._adapters.get("rb")
rb.store.upsert_strategy(...)

# Phase 2 usage (automatic) - opt-in
mem = MemoryConfig(
    mode="reasoning",
    embeddings_fn=encode,
    enable_auto_learning=True,  # NEW: opt-in
    judge_model={"provider": "anthropic", "model": "claude-sonnet-4"},
    extractor_model={"provider": "anthropic", "model": "claude-sonnet-4"}
)
```

All Phase 1 features remain unchanged. Phase 2 is **100% opt-in**.

---

## Performance Considerations

### Cost Analysis

Phase 2 adds LLM calls for evaluation and extraction:

| Operation | LLM Calls per Trajectory | Approx. Cost (Claude Sonnet 4) |
|-----------|--------------------------|--------------------------------|
| Judge evaluation | 1 | ~$0.01 |
| Strategy extraction | 1 (only for successful) | ~$0.02 |
| **Total per successful trajectory** | 2 | ~$0.03 |

With 100 runs/day:
- Cost: ~$3/day
- Strategies learned: ~50/day (assuming 50% success rate)

### Latency

All Phase 2 operations happen asynchronously:
- Agent execution: No added latency
- Trajectory recording: <1ms (async)
- Evaluation: Background worker (no impact)
- Extraction: Background worker (no impact)

---

## Alternative Approaches

### Option 1: On-Demand Learning

Instead of background workers, learn on-demand:

```python
async def learn_from_run(run_id, ctx):
    """Manually trigger learning for a specific run."""
    manager = ctx._adapters.get("auto_learning")
    await manager.process_run(run_id)
```

**Pros:** More control, lower cost
**Cons:** Manual trigger required

### Option 2: Batch Learning

Run learning once per day on all trajectories:

```python
# Cron job
python -m agents.memory.batch_learn --db .vel/vel.db
```

**Pros:** Lower cost, easier to monitor
**Cons:** Not real-time, delayed learning

### Option 3: Hybrid Approach

Combine background workers with manual triggers:

```python
# Background workers for critical cases
if signature.get("risk") == "high":
    await manager.process_immediately(run_id)
else:
    # Background worker will pick it up later
    pass
```

---

## Success Metrics

Track these to measure Phase 2 effectiveness:

1. **Learning Rate:** Strategies added per day
2. **Quality Score:** User ratings of generated strategies
3. **Coverage:** % of signature space with strategies
4. **Reuse Rate:** How often strategies are retrieved
5. **Confidence Evolution:** Average confidence over time
6. **Deduplication Rate:** Clusters merged per week

---

## References

- **Google ReasoningBank Paper:** [arxiv:2509.25140](https://arxiv.org/abs/2509.25140)
- **Vel Phase 1 Implementation:** `agents/memory/strategy_reasoningbank.py`
- **claude-flow Implementation:** [github.com/ruvnet/claude-flow](https://github.com/ruvnet/claude-flow)

---

## Questions for Discussion

Before implementing, consider:

1. **Judge Model:** Use fast/cheap model (Haiku) or accurate/expensive (Sonnet)?
2. **Extraction Frequency:** Real-time vs batched vs on-demand?
3. **Cost Controls:** Max strategies per day? Budget limits?
4. **Quality Controls:** Human-in-the-loop review? Approval workflows?
5. **Privacy:** Trajectory storage retention policy?

---

## Conclusion

Phase 2 transforms Vel's ReasoningBank from a **manual infrastructure** into a **self-evolving memory system**. By adding automatic trajectory recording, LLM-based evaluation, and strategy extraction, agents can learn from experience without human intervention.

The implementation is designed to be:
- ✅ **Opt-in** (Phase 1 unaffected)
- ✅ **Async** (no latency impact)
- ✅ **Modular** (can enable parts independently)
- ✅ **Cost-aware** (configurable intervals)
- ✅ **Backwards compatible** (existing code works)

**Estimated Timeline:** 10 weeks for full Phase 2 implementation
**Estimated Cost:** ~$3/day per 100 agent runs (with Claude Sonnet 4)

Ready to implement when priorities align with automatic agent learning goals.
