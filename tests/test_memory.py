"""
Tests for memory system (FactStore and ReasoningBank).
"""
import pytest
import tempfile
import os
import numpy as np
from pathlib import Path

from vel.memory.fact_store import FactStore
from vel.memory.strategy_reasoningbank import (
    ReasoningBank,
    ReasoningBankStore,
    Embeddings,
    StrategyItem,
)


# --- Fixtures ---

@pytest.fixture
def temp_db():
    """Create temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    try:
        os.unlink(db_path)
    except FileNotFoundError:
        pass


@pytest.fixture
def fact_store(temp_db):
    """Create FactStore instance."""
    return FactStore(temp_db)


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings function."""
    def encode(texts):
        """Deterministic hash-based embeddings for testing."""
        import hashlib
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            v = np.frombuffer(h, dtype=np.uint8).astype(np.float32)[:128]
            v = (v - v.mean()) / (v.std() + 1e-8)
            out.append(v)
        return np.vstack(out)
    return Embeddings(encode)


@pytest.fixture
def rb_store(temp_db, mock_embeddings):
    """Create ReasoningBankStore instance."""
    return ReasoningBankStore(db_path=temp_db, emb=mock_embeddings)


@pytest.fixture
def reasoning_bank(rb_store):
    """Create ReasoningBank instance."""
    return ReasoningBank(rb_store)


# --- FactStore Tests ---

class TestFactStore:
    def test_put_and_get(self, fact_store):
        """Test basic put/get operations."""
        fact_store.put("user:alice", "theme", "dark")
        value = fact_store.get("user:alice", "theme")
        assert value == "dark"

    def test_get_nonexistent(self, fact_store):
        """Test getting non-existent key returns None."""
        value = fact_store.get("user:alice", "missing")
        assert value is None

    def test_update_existing(self, fact_store):
        """Test updating existing key."""
        fact_store.put("user:alice", "theme", "dark")
        fact_store.put("user:alice", "theme", "light")
        value = fact_store.get("user:alice", "theme")
        assert value == "light"

    def test_list_namespace(self, fact_store):
        """Test listing all keys in namespace."""
        fact_store.put("user:alice", "theme", "dark")
        fact_store.put("user:alice", "lang", "en")
        items = fact_store.list("user:alice")
        assert len(items) == 2
        keys = {item["key"] for item in items}
        assert keys == {"theme", "lang"}

    def test_namespace_isolation(self, fact_store):
        """Test that namespaces are isolated."""
        fact_store.put("user:alice", "theme", "dark")
        fact_store.put("user:bob", "theme", "light")

        alice_theme = fact_store.get("user:alice", "theme")
        bob_theme = fact_store.get("user:bob", "theme")

        assert alice_theme == "dark"
        assert bob_theme == "light"

    def test_search_prefix(self, fact_store):
        """Test prefix search."""
        fact_store.put("user:alice", "pref_theme", "dark")
        fact_store.put("user:alice", "pref_lang", "en")
        fact_store.put("user:alice", "other", "value")

        results = fact_store.search("user:alice", "pref_")
        assert len(results) == 2
        keys = {item["key"] for item in results}
        assert keys == {"pref_theme", "pref_lang"}

    def test_delete(self, fact_store):
        """Test deleting a key."""
        fact_store.put("user:alice", "theme", "dark")
        fact_store.delete("user:alice", "theme")
        value = fact_store.get("user:alice", "theme")
        assert value is None

    def test_complex_values(self, fact_store):
        """Test storing complex JSON values."""
        data = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "bool": True,
        }
        fact_store.put("user:alice", "complex", data)
        retrieved = fact_store.get("user:alice", "complex")
        assert retrieved == data


# --- ReasoningBank Tests ---

class TestReasoningBank:
    def test_upsert_and_retrieve(self, rb_store):
        """Test basic upsert and retrieval."""
        signature = {"intent": "planning", "domain": "api"}
        strategy_id = rb_store.upsert_strategy(
            signature=signature,
            strategy_text="Clarify the user goal first",
            confidence=0.7
        )
        assert strategy_id > 0

        # Retrieve
        items = rb_store.retrieve(signature, k=1)
        assert len(items) == 1
        assert items[0].strategy_text == "Clarify the user goal first"
        assert items[0].confidence == 0.7

    def test_retrieve_by_similarity(self, rb_store):
        """Test retrieval by embedding similarity."""
        # Insert strategies with different signatures
        rb_store.upsert_strategy(
            {"intent": "planning", "domain": "api"},
            "Clarify the user goal",
            confidence=0.8
        )
        rb_store.upsert_strategy(
            {"intent": "debugging", "domain": "database"},
            "Check logs first",
            confidence=0.9
        )

        # Retrieve for similar signature
        items = rb_store.retrieve(
            {"intent": "planning", "domain": "web"},
            k=2
        )

        assert len(items) == 2
        # Both strategies should be returned (order depends on embeddings)
        texts = {item.strategy_text for item in items}
        assert "Clarify the user goal" in texts
        assert "Check logs first" in texts

    def test_confidence_filtering(self, rb_store):
        """Test minimum confidence filtering."""
        rb_store.upsert_strategy(
            {"intent": "test"},
            "High confidence strategy",
            confidence=0.8
        )
        rb_store.upsert_strategy(
            {"intent": "test"},
            "Low confidence strategy",
            confidence=0.2
        )

        # Retrieve with min_conf=0.5
        items = rb_store.retrieve({"intent": "test"}, k=10, min_conf=0.5)
        assert len(items) == 1
        assert items[0].strategy_text == "High confidence strategy"

    def test_update_confidence(self, rb_store):
        """Test confidence updates."""
        signature = {"intent": "test"}
        sid = rb_store.upsert_strategy(signature, "Test strategy", confidence=0.5)

        # Update confidence on success
        rb_store.update_confidence(sid, success=True, alpha=0.1)
        items = rb_store.retrieve(signature, k=1)
        assert items[0].confidence == pytest.approx(0.6, abs=0.01)

        # Update confidence on failure
        rb_store.update_confidence(sid, success=False, alpha=0.1)
        items = rb_store.retrieve(signature, k=1)
        assert items[0].confidence == pytest.approx(0.5, abs=0.01)

    def test_add_anti_patterns(self, rb_store):
        """Test adding anti-patterns."""
        signature = {"intent": "test"}
        sid = rb_store.upsert_strategy(
            signature,
            "Test strategy",
            anti_patterns=["avoid X"]
        )

        # Add more anti-patterns
        rb_store.add_anti_patterns(sid, ["avoid Y", "avoid Z"])

        items = rb_store.retrieve(signature, k=1)
        anti = set(items[0].anti_patterns)
        assert anti == {"avoid X", "avoid Y", "avoid Z"}

    def test_reasoning_bank_integration(self, reasoning_bank, rb_store):
        """Test ReasoningBank wrapper class."""
        signature = {"intent": "planning"}

        # Add strategy via store
        rb_store.upsert_strategy(signature, "Plan carefully", confidence=0.7)

        # Get advice via ReasoningBank
        advice = reasoning_bank.get_advice(signature, k=1)
        assert len(advice) == 1
        assert advice[0].strategy_text == "Plan carefully"

    def test_mark_outcome_success(self, reasoning_bank, rb_store):
        """Test marking outcome as success."""
        signature = {"intent": "test"}
        sid = rb_store.upsert_strategy(signature, "Test", confidence=0.5)

        # Mark success
        reasoning_bank.mark_outcome([sid], success=True)

        items = rb_store.retrieve(signature, k=1)
        assert items[0].confidence > 0.5

    def test_mark_outcome_failure(self, reasoning_bank, rb_store):
        """Test marking outcome as failure with notes."""
        signature = {"intent": "test"}
        sid = rb_store.upsert_strategy(signature, "Test", confidence=0.5)

        # Mark failure with notes
        reasoning_bank.mark_outcome(
            [sid],
            success=False,
            fail_notes=["error: timeout"]
        )

        items = rb_store.retrieve(signature, k=1)
        assert items[0].confidence < 0.5
        assert "error: timeout" in items[0].anti_patterns

    def test_top_k_retrieval(self, rb_store):
        """Test top-K retrieval limits."""
        signature = {"intent": "test"}

        # Add 10 strategies
        for i in range(10):
            rb_store.upsert_strategy(
                signature,
                f"Strategy {i}",
                confidence=0.5 + i * 0.01
            )

        # Retrieve top 3
        items = rb_store.retrieve(signature, k=3)
        assert len(items) == 3

    def test_evidence_refs(self, rb_store):
        """Test storing evidence references."""
        signature = {"intent": "test"}
        sid = rb_store.upsert_strategy(
            signature,
            "Test strategy",
            evidence_refs=["run-123", "run-456"]
        )

        items = rb_store.retrieve(signature, k=1)
        assert set(items[0].evidence_refs) == {"run-123", "run-456"}


# --- Integration Tests ---

class TestMemoryIntegration:
    def test_fact_store_and_rb_same_db(self, temp_db, mock_embeddings):
        """Test FactStore and ReasoningBank using same database."""
        # Create both stores on same DB
        fact_store = FactStore(temp_db)
        rb_store = ReasoningBankStore(temp_db, mock_embeddings)

        # Use fact store
        fact_store.put("user:alice", "theme", "dark")

        # Use reasoning bank
        sid = rb_store.upsert_strategy(
            {"intent": "test"},
            "Test strategy"
        )

        # Both should work
        assert fact_store.get("user:alice", "theme") == "dark"
        items = rb_store.retrieve({"intent": "test"}, k=1)
        assert len(items) == 1

    def test_db_directory_creation(self):
        """Test that database directory is created if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "subdir", "test.db")

            # Directory doesn't exist yet
            assert not os.path.exists(os.path.dirname(db_path))

            # Creating store should create directory
            store = FactStore(db_path)
            assert os.path.exists(db_path)

            # Should be usable
            store.put("test", "key", "value")
            assert store.get("test", "key") == "value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
