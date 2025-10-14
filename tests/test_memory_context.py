"""
Integration tests for memory-enabled ContextManager.
"""
import pytest
import tempfile
import os
import numpy as np

from vel.core import ContextManager, MemoryConfig, load_memory_config_from_env


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
def mock_embeddings_fn():
    """Mock embeddings function for testing."""
    def encode(texts):
        import hashlib
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            v = np.frombuffer(h, dtype=np.uint8).astype(np.float32)[:128]
            v = (v - v.mean()) / (v.std() + 1e-8)
            out.append(v)
        return np.vstack(out)
    return encode


# --- ContextManager Memory Tests ---

class TestContextManagerMemory:
    def test_context_manager_without_memory(self):
        """Test ContextManager works without memory (backwards compatible)."""
        ctx = ContextManager()
        ctx.set_input("run-1", {"message": "Hello"})

        messages = ctx.messages_for_llm("run-1")
        assert len(messages) == 1
        assert messages[0]["content"] == "Hello"

    def test_fact_store_mode(self, temp_db):
        """Test ContextManager with fact store only."""
        ctx = ContextManager()
        mem_cfg = MemoryConfig(mode="facts", db_path=temp_db)
        ctx.set_memory_config(mem_cfg)

        # Use fact store
        ctx.fact_put("user:alice", "theme", "dark")
        assert ctx.fact_get("user:alice", "theme") == "dark"

        # List items
        items = ctx.fact_list("user:alice")
        assert len(items) == 1
        assert items[0]["key"] == "theme"

    def test_memory_mode_none(self, temp_db):
        """Test that mode='none' disables memory."""
        ctx = ContextManager()
        mem_cfg = MemoryConfig(mode="none", db_path=temp_db)
        ctx.set_memory_config(mem_cfg)

        # Fact operations should be no-ops
        ctx.fact_put("user:alice", "theme", "dark")
        assert ctx.fact_get("user:alice", "theme") is None
        assert ctx.fact_list("user:alice") == []

    def test_reasoningbank_mode(self, temp_db, mock_embeddings_fn):
        """Test ContextManager with ReasoningBank."""
        ctx = ContextManager()
        mem_cfg = MemoryConfig(
            mode="reasoning",
            db_path=temp_db,
            rb_top_k=3,
            embeddings_fn=mock_embeddings_fn
        )
        ctx.set_memory_config(mem_cfg)

        # Manually insert a strategy via the adapter
        rb = ctx._adapters.get("rb")
        assert rb is not None

        signature = {"intent": "planning", "domain": "api"}
        rb.store.upsert_strategy(
            signature,
            "Clarify user intent first",
            confidence=0.8
        )

        # Prepare for run should retrieve advice
        advice = ctx.prepare_for_run(signature)
        assert "Clarify user intent first" in advice
        assert "Strategy Advice:" in advice

    def test_reasoningbank_finalize_outcome(self, temp_db, mock_embeddings_fn):
        """Test ReasoningBank outcome tracking."""
        ctx = ContextManager()
        mem_cfg = MemoryConfig(
            mode="reasoning",
            db_path=temp_db,
            rb_top_k=3,
            embeddings_fn=mock_embeddings_fn
        )
        ctx.set_memory_config(mem_cfg)

        rb = ctx._adapters.get("rb")
        signature = {"intent": "test"}
        sid = rb.store.upsert_strategy(signature, "Test strategy", confidence=0.5)

        # Prepare for run (stores strategy IDs internally)
        ctx.prepare_for_run(signature)

        # Finalize with success
        ctx.finalize_outcome(run_success=True)

        # Confidence should increase
        items = rb.store.retrieve(signature, k=1)
        assert items[0].confidence > 0.5

    def test_reasoningbank_failure_notes(self, temp_db, mock_embeddings_fn):
        """Test ReasoningBank records failure notes."""
        ctx = ContextManager()
        mem_cfg = MemoryConfig(
            mode="reasoning",
            db_path=temp_db,
            rb_top_k=3,
            embeddings_fn=mock_embeddings_fn
        )
        ctx.set_memory_config(mem_cfg)

        rb = ctx._adapters.get("rb")
        signature = {"intent": "test"}
        sid = rb.store.upsert_strategy(signature, "Test strategy", confidence=0.5)

        # Prepare and fail
        ctx.prepare_for_run(signature)
        ctx.finalize_outcome(run_success=False, fail_notes=["timeout"])

        # Check anti-patterns
        items = rb.store.retrieve(signature, k=1)
        assert "timeout" in items[0].anti_patterns

    def test_both_memory_modes(self, temp_db, mock_embeddings_fn):
        """Test using both fact store and ReasoningBank together."""
        ctx = ContextManager()
        mem_cfg = MemoryConfig(
            mode="all",
            db_path=temp_db,
            rb_top_k=3,
            embeddings_fn=mock_embeddings_fn
        )
        ctx.set_memory_config(mem_cfg)

        # Use fact store
        ctx.fact_put("user:alice", "theme", "dark")
        assert ctx.fact_get("user:alice", "theme") == "dark"

        # Use ReasoningBank
        rb = ctx._adapters.get("rb")
        signature = {"intent": "test"}
        rb.store.upsert_strategy(signature, "Test", confidence=0.7)

        advice = ctx.prepare_for_run(signature)
        assert "Test" in advice

        # Both should work
        assert ctx.fact_get("user:alice", "theme") == "dark"

    def test_prepare_for_run_empty(self, temp_db, mock_embeddings_fn):
        """Test prepare_for_run with no matching strategies."""
        ctx = ContextManager()
        mem_cfg = MemoryConfig(
            mode="reasoning",
            db_path=temp_db,
            rb_top_k=3,
            embeddings_fn=mock_embeddings_fn
        )
        ctx.set_memory_config(mem_cfg)

        # No strategies in DB
        advice = ctx.prepare_for_run({"intent": "unknown"})
        assert advice == ""

    def test_memory_config_from_env(self, monkeypatch, temp_db):
        """Test loading MemoryConfig from environment variables."""
        monkeypatch.setenv("VEL_MEMORY_MODE", "facts")
        monkeypatch.setenv("VEL_MEMORY_DB", temp_db)
        monkeypatch.setenv("VEL_RB_TOP_K", "3")

        cfg = load_memory_config_from_env()
        assert cfg.mode == "facts"
        assert cfg.db_path == temp_db
        assert cfg.rb_top_k == 3

    def test_memory_config_env_defaults(self):
        """Test MemoryConfig defaults when env vars not set."""
        cfg = load_memory_config_from_env()
        assert cfg.mode == "none"
        assert cfg.db_path == ".vel/vel.db"
        assert cfg.rb_top_k == 5

    def test_reasoningbank_without_embeddings_fn(self, temp_db):
        """Test that ReasoningBank gracefully handles missing embeddings_fn."""
        ctx = ContextManager()
        mem_cfg = MemoryConfig(
            mode="reasoning",
            db_path=temp_db,
            embeddings_fn=None  # Missing!
        )
        ctx.set_memory_config(mem_cfg)

        # RB adapter should be None
        rb = ctx._adapters.get("rb")
        assert rb is None

        # prepare_for_run should return empty string
        advice = ctx.prepare_for_run({"intent": "test"})
        assert advice == ""

    def test_multiple_strategies_advice_format(self, temp_db, mock_embeddings_fn):
        """Test advice formatting with multiple strategies."""
        ctx = ContextManager()
        mem_cfg = MemoryConfig(
            mode="reasoning",
            db_path=temp_db,
            rb_top_k=3,
            embeddings_fn=mock_embeddings_fn
        )
        ctx.set_memory_config(mem_cfg)

        rb = ctx._adapters.get("rb")
        signature = {"intent": "planning"}

        # Insert multiple strategies
        rb.store.upsert_strategy(
            signature,
            "Clarify intent",
            anti_patterns=["skip validation"],
            confidence=0.9
        )
        rb.store.upsert_strategy(
            signature,
            "Break into steps",
            anti_patterns=["rush"],
            confidence=0.8
        )

        advice = ctx.prepare_for_run(signature)

        # Should have both strategies (order may vary with hash embeddings)
        assert "Clarify intent" in advice
        assert "Break into steps" in advice
        assert "Avoid: skip validation" in advice
        assert "Avoid: rush" in advice
        assert "Strategy Advice:" in advice

    def test_finalize_without_prepare(self, temp_db, mock_embeddings_fn):
        """Test finalize_outcome without prepare_for_run (should be no-op)."""
        ctx = ContextManager()
        mem_cfg = MemoryConfig(
            mode="reasoning",
            db_path=temp_db,
            embeddings_fn=mock_embeddings_fn
        )
        ctx.set_memory_config(mem_cfg)

        # Finalize without prepare (no strategies retrieved)
        ctx.finalize_outcome(run_success=True)

        # Should not crash


class TestMemoryConfigOptions:
    def test_memory_config_defaults(self):
        """Test MemoryConfig default values."""
        cfg = MemoryConfig()
        assert cfg.mode == "none"
        assert cfg.db_path == ".vel/vel.db"
        assert cfg.rb_top_k == 5
        assert cfg.embeddings_fn is None

    def test_memory_config_custom(self, mock_embeddings_fn):
        """Test MemoryConfig with custom values."""
        cfg = MemoryConfig(
            mode="all",
            db_path="/tmp/test.db",
            rb_top_k=10,
            embeddings_fn=mock_embeddings_fn
        )
        assert cfg.mode == "all"
        assert cfg.db_path == "/tmp/test.db"
        assert cfg.rb_top_k == 10
        assert cfg.embeddings_fn is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
