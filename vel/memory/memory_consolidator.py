# vel/memory/memory_consolidator.py
"""
MemoryConsolidator: Maintains ReasoningBank memory health.

Part of ReasoningBank Phase 2 - prevents unbounded memory growth by:
1. Merging similar strategies (prevent fragmentation)
2. Pruning low-confidence strategies (prevent bloat)
3. Applying confidence decay for unused strategies
4. Enforcing maximum strategy count
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Set, Tuple
import json
from time import time

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore


@dataclass
class ConsolidationResult:
    """Result from MemoryConsolidator.consolidate() call."""
    strategies_merged: int       # Number of strategies merged
    strategies_pruned: int       # Number of low-confidence strategies removed
    strategies_decayed: int      # Number of strategies with decayed confidence
    clusters_found: int          # Number of similarity clusters detected
    total_strategies_before: int
    total_strategies_after: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/storage."""
        return {
            "strategies_merged": self.strategies_merged,
            "strategies_pruned": self.strategies_pruned,
            "strategies_decayed": self.strategies_decayed,
            "clusters_found": self.clusters_found,
            "total_strategies_before": self.total_strategies_before,
            "total_strategies_after": self.total_strategies_after,
        }


class MemoryConsolidator:
    """
    Maintains ReasoningBank memory health.

    Responsibilities:
    1. Merge similar strategies (prevent fragmentation)
    2. Prune low-confidence strategies (prevent bloat)
    3. Apply confidence decay for unused strategies
    4. Enforce maximum strategy count

    Design principles:
    - Bounded memory: Cap total strategies to prevent unbounded growth
    - Usage-weighted: Preserve frequently-used strategies
    - Gradual decay: Unused strategies lose confidence over time
    - Non-destructive: Preserves information during merges

    Usage:
        ```python
        consolidator = MemoryConsolidator(
            reasoning_bank_store=rb_store,
            max_strategies=1000,
            min_confidence=0.20
        )

        # Run consolidation
        result = consolidator.consolidate()
        print(f"Merged: {result.strategies_merged}, Pruned: {result.strategies_pruned}")

        # Dry run to preview changes
        preview = consolidator.consolidate(dry_run=True)
        ```
    """

    def __init__(
        self,
        reasoning_bank_store: Any,  # ReasoningBankStore
        similarity_threshold: float = 0.85,
        max_strategies: int = 1000,
        min_confidence: float = 0.20,
        decay_rate: float = 0.02,
        decay_interval_days: int = 30
    ):
        """
        Args:
            reasoning_bank_store: ReasoningBankStore for persistence
            similarity_threshold: Embedding similarity for merging (0.0-1.0)
            max_strategies: Maximum strategies to retain (prune excess)
            min_confidence: Prune strategies below this confidence
            decay_rate: Confidence decay per interval for unused strategies
            decay_interval_days: Days between decay applications
        """
        self.store = reasoning_bank_store
        self.similarity_threshold = similarity_threshold
        self.max_strategies = max_strategies
        self.min_confidence = min_confidence
        self.decay_rate = decay_rate
        self.decay_interval_seconds = decay_interval_days * 24 * 60 * 60

    def _get_all_strategies(self) -> List[Any]:
        """Get all strategies from the database."""
        rows = self.store.db.execute("""
            SELECT s.*, e.embedding, e.dim
            FROM rb_strategies s
            LEFT JOIN rb_embeddings e ON e.strategy_id = s.id
            ORDER BY s.confidence DESC
        """).fetchall()

        strategies = []
        for r in rows:
            from vel.memory.strategy_reasoningbank import StrategyItem
            strategies.append(StrategyItem(
                id=r["id"],
                signature_json=r["signature_json"],
                strategy_text=r["strategy_text"],
                anti_patterns=json.loads(r["anti_patterns"] or "[]"),
                evidence_refs=json.loads(r["evidence_refs"] or "[]"),
                confidence=float(r["confidence"]),
            ))
        return strategies

    def _get_embeddings(self) -> Dict[int, Any]:
        """Get all embeddings keyed by strategy_id."""
        if np is None:
            return {}

        rows = self.store.db.execute(
            "SELECT strategy_id, embedding FROM rb_embeddings"
        ).fetchall()

        embeddings = {}
        for r in rows:
            emb = np.frombuffer(r["embedding"], dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            embeddings[r["strategy_id"]] = emb

        return embeddings

    def _find_clusters(
        self,
        strategies: List[Any],
        embeddings: Dict[int, Any]
    ) -> List[List[Any]]:
        """
        Find clusters of similar strategies using greedy clustering.

        Algorithm:
        1. Compute pairwise similarity for strategies with embeddings
        2. Use Union-Find to group connected components above threshold
        3. Return clusters with size > 1

        Args:
            strategies: List of StrategyItems
            embeddings: Dict of strategy_id -> embedding vector

        Returns:
            List of clusters, each cluster is list of StrategyItems
        """
        if np is None or len(strategies) < 2:
            return []

        # Filter to strategies with embeddings
        indexed = []
        for s in strategies:
            if s.id in embeddings:
                indexed.append((s, embeddings[s.id]))

        if len(indexed) < 2:
            return []

        n = len(indexed)

        # Union-Find for clustering
        parent = list(range(n))

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Connect strategies above threshold
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(np.dot(indexed[i][1], indexed[j][1]))
                if sim >= self.similarity_threshold:
                    union(i, j)

        # Group by cluster
        clusters_map: Dict[int, List[Any]] = {}
        for i in range(n):
            root = find(i)
            if root not in clusters_map:
                clusters_map[root] = []
            clusters_map[root].append(indexed[i][0])

        # Return only clusters with multiple members
        return [cluster for cluster in clusters_map.values() if len(cluster) > 1]

    def _merge_cluster(self, cluster: List[Any]) -> int:
        """
        Merge a cluster into a single strategy.

        Strategy selection: Keep highest-confidence strategy as base
        Merging:
        - anti_patterns: Union of all (deduplicated)
        - evidence_refs: Union of all
        - confidence: Weighted average by evidence count

        Args:
            cluster: List of StrategyItems to merge

        Returns:
            Number of strategies removed (cluster size - 1)
        """
        if len(cluster) < 2:
            return 0

        # Sort by confidence (descending)
        sorted_cluster = sorted(cluster, key=lambda s: s.confidence, reverse=True)
        base = sorted_cluster[0]

        # Collect merged data
        all_anti_patterns: Set[str] = set()
        all_evidence_refs: Set[str] = set()
        total_evidence = 0
        weighted_confidence_sum = 0.0

        for strategy in cluster:
            all_anti_patterns.update(strategy.anti_patterns or [])
            all_evidence_refs.update(strategy.evidence_refs or [])

            # Weight by evidence count (more evidence = more reliable)
            evidence_count = max(len(strategy.evidence_refs or []), 1)
            weighted_confidence_sum += strategy.confidence * evidence_count
            total_evidence += evidence_count

        # Calculate weighted average confidence
        merged_confidence = weighted_confidence_sum / total_evidence if total_evidence > 0 else base.confidence

        # Bayesian update: More evidence = higher confidence cap
        # Cap increases from 0.8 (1 evidence) to 0.95 (10+ evidence)
        evidence_bonus = min(total_evidence / 10, 1.0) * 0.15
        confidence_cap = 0.80 + evidence_bonus
        merged_confidence = min(merged_confidence, confidence_cap)

        # Update base strategy in database
        self.store.db.execute("""
            UPDATE rb_strategies
            SET anti_patterns = ?,
                evidence_refs = ?,
                confidence = ?,
                updated_at = ?
            WHERE id = ?
        """, (
            json.dumps(list(all_anti_patterns)),
            json.dumps(list(all_evidence_refs)),
            merged_confidence,
            time(),
            base.id
        ))

        # Delete merged strategies (not the base)
        removed = 0
        for strategy in sorted_cluster[1:]:
            self.store.db.execute(
                "DELETE FROM rb_strategies WHERE id = ?", (strategy.id,)
            )
            self.store.db.execute(
                "DELETE FROM rb_embeddings WHERE strategy_id = ?", (strategy.id,)
            )
            removed += 1

        self.store.db.commit()
        return removed

    def merge_similar_strategies(self, strategies: Optional[List[Any]] = None) -> int:
        """
        Find and merge similar strategies.

        Args:
            strategies: List of strategies (fetches if None)

        Returns:
            Number of strategies merged (removed)
        """
        if strategies is None:
            strategies = self._get_all_strategies()

        embeddings = self._get_embeddings()
        clusters = self._find_clusters(strategies, embeddings)

        total_merged = 0
        for cluster in clusters:
            merged = self._merge_cluster(cluster)
            total_merged += merged

        return total_merged

    def apply_confidence_decay(self, strategies: Optional[List[Any]] = None) -> int:
        """
        Reduce confidence for strategies not used recently.

        Uses updated_at timestamp to determine recency.
        Decay formula: new_confidence = old_confidence * (1 - decay_rate)

        Args:
            strategies: List of strategies (fetches if None)

        Returns:
            Number of strategies decayed
        """
        now = time()
        cutoff = now - self.decay_interval_seconds

        # Get strategies not updated recently
        rows = self.store.db.execute("""
            SELECT id, confidence FROM rb_strategies
            WHERE updated_at < ?
        """, (cutoff,)).fetchall()

        decayed = 0
        for r in rows:
            old_conf = float(r["confidence"])
            new_conf = old_conf * (1 - self.decay_rate)

            # Don't decay below minimum
            if new_conf < self.min_confidence:
                continue

            self.store.db.execute("""
                UPDATE rb_strategies
                SET confidence = ?, updated_at = ?
                WHERE id = ?
            """, (new_conf, now, r["id"]))
            decayed += 1

        self.store.db.commit()
        return decayed

    def prune_low_confidence(self, min_confidence: Optional[float] = None) -> int:
        """
        Remove strategies below confidence threshold.

        Args:
            min_confidence: Threshold (uses instance default if None)

        Returns:
            Number of strategies pruned
        """
        threshold = min_confidence if min_confidence is not None else self.min_confidence

        # Get IDs to delete
        rows = self.store.db.execute(
            "SELECT id FROM rb_strategies WHERE confidence < ?",
            (threshold,)
        ).fetchall()

        # Delete embeddings first (foreign key)
        for r in rows:
            self.store.db.execute(
                "DELETE FROM rb_embeddings WHERE strategy_id = ?", (r["id"],)
            )

        # Delete strategies
        cur = self.store.db.execute(
            "DELETE FROM rb_strategies WHERE confidence < ?",
            (threshold,)
        )

        pruned = cur.rowcount
        self.store.db.commit()
        return pruned

    def enforce_strategy_cap(self, max_strategies: Optional[int] = None) -> int:
        """
        Enforce maximum strategy count by removing lowest-confidence.

        Args:
            max_strategies: Cap (uses instance default if None)

        Returns:
            Number of strategies removed to meet cap
        """
        cap = max_strategies if max_strategies is not None else self.max_strategies

        # Get current count
        count = self.store.db.execute(
            "SELECT COUNT(*) FROM rb_strategies"
        ).fetchone()[0]

        if count <= cap:
            return 0

        excess = count - cap

        # Get IDs of lowest-confidence strategies
        rows = self.store.db.execute("""
            SELECT id FROM rb_strategies
            ORDER BY confidence ASC
            LIMIT ?
        """, (excess,)).fetchall()

        # Delete embeddings first
        for r in rows:
            self.store.db.execute(
                "DELETE FROM rb_embeddings WHERE strategy_id = ?", (r["id"],)
            )

        # Delete strategies
        ids = [r["id"] for r in rows]
        placeholders = ",".join("?" * len(ids))
        self.store.db.execute(
            f"DELETE FROM rb_strategies WHERE id IN ({placeholders})",
            ids
        )

        self.store.db.commit()
        return excess

    def consolidate(
        self,
        signature_filter: Optional[Dict[str, Any]] = None,
        dry_run: bool = False
    ) -> ConsolidationResult:
        """
        Run full consolidation cycle:
        1. Find and merge similar strategies
        2. Apply confidence decay to unused strategies
        3. Prune low-confidence strategies
        4. Enforce strategy cap

        Args:
            signature_filter: Optional filter (not yet implemented)
            dry_run: If True, report what would happen without changes

        Returns:
            ConsolidationResult with metrics
        """
        strategies = self._get_all_strategies()
        total_before = len(strategies)

        if dry_run:
            # Calculate what would happen without making changes
            embeddings = self._get_embeddings()
            clusters = self._find_clusters(strategies, embeddings)
            would_merge = sum(len(c) - 1 for c in clusters)

            would_prune = sum(1 for s in strategies if s.confidence < self.min_confidence)

            remaining = total_before - would_merge
            excess = max(0, remaining - would_prune - self.max_strategies)

            return ConsolidationResult(
                strategies_merged=would_merge,
                strategies_pruned=would_prune + excess,
                strategies_decayed=0,  # Can't predict decay without checking timestamps
                clusters_found=len(clusters),
                total_strategies_before=total_before,
                total_strategies_after=total_before - would_merge - would_prune - excess
            )

        # 1. Merge similar strategies
        merged = self.merge_similar_strategies(strategies)

        # Refresh strategies after merge
        strategies = self._get_all_strategies()
        embeddings = self._get_embeddings()
        clusters_found = len(self._find_clusters(strategies, embeddings))

        # 2. Apply confidence decay
        decayed = self.apply_confidence_decay()

        # 3. Prune low confidence
        pruned = self.prune_low_confidence()

        # 4. Enforce strategy cap
        capped = self.enforce_strategy_cap()

        # Get final count
        final_strategies = self._get_all_strategies()

        return ConsolidationResult(
            strategies_merged=merged,
            strategies_pruned=pruned + capped,
            strategies_decayed=decayed,
            clusters_found=clusters_found,
            total_strategies_before=total_before,
            total_strategies_after=len(final_strategies)
        )
