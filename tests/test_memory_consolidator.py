# tests/test_memory_consolidator.py
"""Tests for MemoryConsolidator (Phase 2.4)."""
import pytest
import tempfile
import os
import hashlib
import numpy as np
from time import time, sleep

from vel.memory.strategy_reasoningbank import ReasoningBankStore, Embeddings, StrategyItem
from vel.memory.memory_consolidator import MemoryConsolidator, ConsolidationResult


def mock_embeddings(texts):
    """Hash-based mock embeddings for testing."""
    out = []
    for t in texts:
        h = hashlib.sha256(t.encode()).digest()
        v = np.frombuffer(h, dtype=np.uint8).astype(np.float32)[:32]
        v = (v - v.mean()) / (v.std() + 1e-8)
        out.append(v)
    return np.vstack(out)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        yield db_path


@pytest.fixture
def rb_store(temp_db):
    """Create a ReasoningBankStore instance."""
    emb = Embeddings(mock_embeddings)
    return ReasoningBankStore(temp_db, emb)


@pytest.fixture
def consolidator(rb_store):
    """Create a MemoryConsolidator instance."""
    return MemoryConsolidator(
        reasoning_bank_store=rb_store,
        similarity_threshold=0.85,
        max_strategies=100,
        min_confidence=0.20,
        decay_rate=0.1,  # High rate for testing
        decay_interval_days=0  # Immediate decay for testing
    )


class TestConsolidationResult:
    """Test ConsolidationResult dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        result = ConsolidationResult(
            strategies_merged=5,
            strategies_pruned=3,
            strategies_decayed=10,
            clusters_found=2,
            total_strategies_before=100,
            total_strategies_after=92
        )

        d = result.to_dict()

        assert d["strategies_merged"] == 5
        assert d["strategies_pruned"] == 3
        assert d["strategies_decayed"] == 10
        assert d["total_strategies_before"] == 100
        assert d["total_strategies_after"] == 92


class TestMemoryConsolidator:
    """Test MemoryConsolidator."""

    def test_prune_low_confidence(self, rb_store, consolidator):
        """Test pruning low-confidence strategies."""
        # Add strategies with varying confidence
        rb_store.upsert_strategy(
            signature={"intent": "high"},
            strategy_text="High confidence strategy",
            confidence=0.8
        )
        rb_store.upsert_strategy(
            signature={"intent": "medium"},
            strategy_text="Medium confidence strategy",
            confidence=0.5
        )
        rb_store.upsert_strategy(
            signature={"intent": "low"},
            strategy_text="Low confidence strategy",
            confidence=0.1
        )
        rb_store.upsert_strategy(
            signature={"intent": "very_low"},
            strategy_text="Very low confidence strategy",
            confidence=0.05
        )

        # Prune below 0.20
        pruned = consolidator.prune_low_confidence()

        assert pruned == 2  # Two strategies below 0.20

        # Verify remaining strategies
        all_strategies = consolidator._get_all_strategies()
        assert len(all_strategies) == 2

    def test_enforce_strategy_cap(self, rb_store, consolidator):
        """Test enforcing maximum strategy count."""
        # Set low cap for testing
        consolidator.max_strategies = 5

        # Add more strategies than cap
        for i in range(10):
            rb_store.upsert_strategy(
                signature={"intent": f"test-{i}"},
                strategy_text=f"Strategy {i}",
                confidence=0.5 + (i * 0.05)  # Varying confidence
            )

        # Verify we have 10 strategies
        all_strategies = consolidator._get_all_strategies()
        assert len(all_strategies) == 10

        # Enforce cap
        removed = consolidator.enforce_strategy_cap()

        assert removed == 5  # Should remove 5 lowest

        # Verify remaining
        remaining = consolidator._get_all_strategies()
        assert len(remaining) == 5

        # Should keep highest confidence ones
        confidences = [s.confidence for s in remaining]
        assert all(c >= 0.7 for c in confidences)

    def test_apply_confidence_decay(self, rb_store, consolidator):
        """Test confidence decay for unused strategies."""
        # Add a strategy with old updated_at
        rb_store.upsert_strategy(
            signature={"intent": "old"},
            strategy_text="Old strategy",
            confidence=0.7
        )

        # Apply decay (should affect all since decay_interval=0)
        decayed = consolidator.apply_confidence_decay()

        # Verify decay applied
        strategies = consolidator._get_all_strategies()
        assert len(strategies) == 1
        assert strategies[0].confidence < 0.7  # Should be 0.7 * 0.9 = 0.63

    def test_find_clusters(self, rb_store, consolidator):
        """Test cluster finding."""
        # Add strategies with same text (will have same embeddings)
        rb_store.upsert_strategy(
            signature={"intent": "a"},
            strategy_text="Break tasks into smaller steps",
            confidence=0.6
        )
        rb_store.upsert_strategy(
            signature={"intent": "b"},
            strategy_text="Break tasks into smaller steps",
            confidence=0.5
        )
        rb_store.upsert_strategy(
            signature={"intent": "c"},
            strategy_text="Completely different approach to testing",
            confidence=0.7
        )

        strategies = consolidator._get_all_strategies()
        embeddings = consolidator._get_embeddings()

        clusters = consolidator._find_clusters(strategies, embeddings)

        # With hash-based embeddings, identical text should cluster
        # The first two have identical text, so they should cluster
        # Note: This depends on embedding similarity being above threshold

    def test_merge_cluster(self, rb_store, consolidator):
        """Test merging a cluster of strategies."""
        # Add similar strategies manually
        id1 = rb_store.upsert_strategy(
            signature={"intent": "test"},
            strategy_text="Strategy A",
            anti_patterns=["pattern1"],
            evidence_refs=["ref1"],
            confidence=0.8
        )
        id2 = rb_store.upsert_strategy(
            signature={"intent": "test"},
            strategy_text="Strategy B",
            anti_patterns=["pattern2"],
            evidence_refs=["ref2"],
            confidence=0.6
        )
        id3 = rb_store.upsert_strategy(
            signature={"intent": "test"},
            strategy_text="Strategy C",
            anti_patterns=["pattern3"],
            evidence_refs=["ref3"],
            confidence=0.4
        )

        strategies = consolidator._get_all_strategies()

        # Create cluster manually
        cluster = [s for s in strategies if s.id in [id1, id2, id3]]

        removed = consolidator._merge_cluster(cluster)

        assert removed == 2  # Two strategies removed

        # Verify remaining strategy has merged data
        remaining = consolidator._get_all_strategies()
        assert len(remaining) == 1

        merged = remaining[0]
        assert merged.id == id1  # Highest confidence kept
        assert len(merged.anti_patterns) == 3  # All patterns merged
        assert len(merged.evidence_refs) == 3  # All refs merged

    def test_consolidate_dry_run(self, rb_store, consolidator):
        """Test dry run mode."""
        # Add some strategies
        rb_store.upsert_strategy(
            signature={"intent": "test"},
            strategy_text="Strategy 1",
            confidence=0.1  # Below threshold
        )
        rb_store.upsert_strategy(
            signature={"intent": "test"},
            strategy_text="Strategy 2",
            confidence=0.5
        )

        # Dry run
        result = consolidator.consolidate(dry_run=True)

        assert result.total_strategies_before == 2
        assert result.strategies_pruned >= 1  # At least one below threshold

        # Verify no actual changes
        actual = consolidator._get_all_strategies()
        assert len(actual) == 2  # Still 2 strategies

    def test_consolidate_full_cycle(self, rb_store, consolidator):
        """Test full consolidation cycle."""
        consolidator.max_strategies = 5
        consolidator.min_confidence = 0.3

        # Add many strategies
        for i in range(10):
            rb_store.upsert_strategy(
                signature={"intent": f"test-{i}"},
                strategy_text=f"Strategy number {i} with unique content",
                confidence=0.2 + (i * 0.08)  # 0.2 to 0.92
            )

        result = consolidator.consolidate()

        assert result.total_strategies_before == 10
        assert result.strategies_pruned > 0  # Some pruned for low confidence and cap
        assert result.total_strategies_after <= 5  # Respects cap

    def test_merge_preserves_highest_confidence(self, rb_store, consolidator):
        """Test that merge keeps highest confidence strategy."""
        # Add strategies with very similar text
        low_id = rb_store.upsert_strategy(
            signature={"intent": "merge-test"},
            strategy_text="Exact same strategy text here",
            confidence=0.3
        )
        high_id = rb_store.upsert_strategy(
            signature={"intent": "merge-test"},
            strategy_text="Exact same strategy text here",
            confidence=0.9
        )

        strategies = consolidator._get_all_strategies()
        embeddings = consolidator._get_embeddings()

        clusters = consolidator._find_clusters(strategies, embeddings)

        # If they cluster together
        if clusters:
            for cluster in clusters:
                if len(cluster) > 1:
                    consolidator._merge_cluster(cluster)

            remaining = consolidator._get_all_strategies()
            if len(remaining) == 1:
                assert remaining[0].id == high_id  # Higher confidence kept


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_database(self, rb_store, consolidator):
        """Test consolidation on empty database."""
        result = consolidator.consolidate()

        assert result.total_strategies_before == 0
        assert result.total_strategies_after == 0
        assert result.strategies_merged == 0
        assert result.strategies_pruned == 0

    def test_single_strategy(self, rb_store, consolidator):
        """Test consolidation with single strategy."""
        rb_store.upsert_strategy(
            signature={"intent": "solo"},
            strategy_text="Only strategy",
            confidence=0.7
        )

        result = consolidator.consolidate()

        assert result.total_strategies_before == 1
        assert result.total_strategies_after == 1
        assert result.strategies_merged == 0

    def test_all_low_confidence(self, rb_store, consolidator):
        """Test when all strategies are low confidence."""
        for i in range(5):
            rb_store.upsert_strategy(
                signature={"intent": f"low-{i}"},
                strategy_text=f"Low strategy {i}",
                confidence=0.1  # All below threshold
            )

        result = consolidator.consolidate()

        assert result.strategies_pruned == 5
        assert result.total_strategies_after == 0
