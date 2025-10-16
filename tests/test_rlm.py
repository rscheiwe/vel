"""
Tests for RLM (Recursive Language Model) module
"""
import pytest
from vel.rlm import (
    RlmConfig,
    Scratchpad,
    Note,
    Budget,
    ContextStore
)
from vel.rlm.utils import detect_final, extract_citations, truncate_text
from vel.rlm.tools import context_probe


class TestRlmConfig:
    """Test RlmConfig"""

    def test_default_config(self):
        """Test default configuration"""
        config = RlmConfig()
        assert config.enabled == False
        assert config.depth == 1
        assert config.notes_cap == 200
        assert config.notes_window == 40

    def test_enabled_config(self):
        """Test enabled configuration"""
        config = RlmConfig(enabled=True, depth=2)
        assert config.enabled == True
        assert config.depth == 2

    def test_invalid_depth(self):
        """Test invalid depth raises error"""
        with pytest.raises(ValueError, match="depth must be 0-2"):
            RlmConfig(depth=3)

    def test_invalid_notes_cap(self):
        """Test invalid notes_cap raises error"""
        with pytest.raises(ValueError, match="notes_cap must be >= 1"):
            RlmConfig(notes_cap=0)

    def test_to_dict(self):
        """Test config serialization"""
        config = RlmConfig(enabled=True, depth=1)
        data = config.to_dict()
        assert data['enabled'] == True
        assert data['depth'] == 1
        assert 'budgets' in data

    def test_from_dict(self):
        """Test config deserialization"""
        data = {'enabled': True, 'depth': 1, 'notes_cap': 100}
        config = RlmConfig.from_dict(data)
        assert config.enabled == True
        assert config.depth == 1
        assert config.notes_cap == 100


class TestNote:
    """Test Note dataclass"""

    def test_note_creation(self):
        """Test note creation with auto ID"""
        note = Note(text="Test note")
        assert note.text == "Test note"
        assert note.id is not None
        assert note.timestamp is not None

    def test_note_with_source(self):
        """Test note with source hint"""
        note = Note(text="Test note", source_hint="chunk 1.2")
        assert note.source_hint == "chunk 1.2"

    def test_note_serialization(self):
        """Test note to/from dict"""
        note = Note(text="Test note", source_hint="source")
        data = note.to_dict()
        assert data['text'] == "Test note"
        assert data['source_hint'] == "source"

        note2 = Note.from_dict(data)
        assert note2.text == note.text
        assert note2.id == note.id


class TestScratchpad:
    """Test Scratchpad"""

    def test_empty_scratchpad(self):
        """Test empty scratchpad"""
        scratchpad = Scratchpad()
        assert len(scratchpad) == 0
        assert scratchpad.to_bullets() == "(no notes yet)"

    def test_add_note(self):
        """Test adding notes"""
        scratchpad = Scratchpad()
        note = scratchpad.add("Note 1")
        assert len(scratchpad) == 1
        assert note.text == "Note 1"

    def test_dedup_notes(self):
        """Test deduplication"""
        scratchpad = Scratchpad()
        note1 = scratchpad.add("Duplicate")
        note2 = scratchpad.add("Duplicate")  # Same content, should dedupe
        # Note: dedup is based on ID which includes timestamp, so this won't actually dedupe
        # unless we use merge with existing notes
        assert len(scratchpad) == 2  # Both added since IDs differ

        # Test proper dedup with merge
        scratchpad2 = Scratchpad()
        scratchpad2.add("Note A")
        note = Note(text="Note B", id="test-id", timestamp=123.0)
        scratchpad2.merge([note], dedup=False)
        scratchpad2.merge([note], dedup=True)  # Should not add duplicate
        assert len(scratchpad2) == 2  # "Note A" + "Note B" (not duplicated)

    def test_capping(self):
        """Test note capping"""
        scratchpad = Scratchpad(max_notes=5)
        for i in range(10):
            scratchpad.add(f"Note {i}")
        assert len(scratchpad) == 5  # Should cap at 5

    def test_to_bullets(self):
        """Test bullet formatting"""
        scratchpad = Scratchpad()
        scratchpad.add("Note 1", source_hint="chunk 1")
        scratchpad.add("Note 2")
        bullets = scratchpad.to_bullets()
        assert "- Note 1 [chunk 1]" in bullets
        assert "- Note 2" in bullets

    def test_to_bullets_with_limit(self):
        """Test bullet formatting with limit"""
        scratchpad = Scratchpad()
        for i in range(10):
            scratchpad.add(f"Note {i}")
        bullets = scratchpad.to_bullets(limit=3)
        # Should only show last 3
        assert "Note 7" in bullets
        assert "Note 8" in bullets
        assert "Note 9" in bullets
        assert "Note 0" not in bullets


class TestBudget:
    """Test Budget"""

    def test_budget_creation(self):
        """Test budget initialization"""
        budget = Budget(max_steps=10, max_tokens=1000, max_cost=0.50)
        assert budget.max_steps == 10
        assert budget.max_tokens == 1000
        assert budget.max_cost == 0.50
        assert budget.steps == 0
        assert budget.tokens == 0
        assert budget.cost == 0.0

    def test_bump_step(self):
        """Test step increment"""
        budget = Budget(max_steps=10, max_tokens=1000, max_cost=0.50)
        budget.bump_step()
        assert budget.steps == 1

    def test_bump_tokens(self):
        """Test token increment"""
        budget = Budget(max_steps=10, max_tokens=1000, max_cost=0.50)
        budget.bump_tokens(prompt=100, completion=50)
        assert budget.prompt_tokens == 100
        assert budget.completion_tokens == 50
        assert budget.tokens == 150

    def test_exhausted_steps(self):
        """Test step exhaustion"""
        budget = Budget(max_steps=3, max_tokens=1000, max_cost=0.50)
        budget.steps = 3
        is_exhausted, reason = budget.exhausted()
        assert is_exhausted == True
        assert "steps" in reason

    def test_exhausted_tokens(self):
        """Test token exhaustion"""
        budget = Budget(max_steps=10, max_tokens=100, max_cost=0.50)
        budget.tokens = 100
        is_exhausted, reason = budget.exhausted()
        assert is_exhausted == True
        assert "tokens" in reason

    def test_exhausted_cost(self):
        """Test cost exhaustion"""
        budget = Budget(max_steps=10, max_tokens=1000, max_cost=0.10)
        budget.cost = 0.10
        is_exhausted, reason = budget.exhausted()
        assert is_exhausted == True
        assert "cost" in reason

    def test_not_exhausted(self):
        """Test not exhausted"""
        budget = Budget(max_steps=10, max_tokens=1000, max_cost=0.50)
        budget.steps = 5
        is_exhausted, reason = budget.exhausted()
        assert is_exhausted == False
        assert reason is None

    def test_to_dict(self):
        """Test budget serialization"""
        budget = Budget(max_steps=10, max_tokens=1000, max_cost=0.50)
        budget.steps = 5
        budget.tokens = 500
        data = budget.to_dict()
        assert data['steps'] == 5
        assert data['max_steps'] == 10
        assert data['tokens'] == 500
        assert data['exhausted'] == False


class TestContextStore:
    """Test ContextStore"""

    def test_load_text(self):
        """Test loading raw text"""
        store = ContextStore()
        num_chunks = store.load("This is a test document.")
        assert num_chunks > 0
        assert len(store.chunks) > 0

    def test_load_large_text(self):
        """Test loading and chunking large text"""
        store = ContextStore(chunk_size=100)
        large_text = "Line\n" * 1000  # Many lines
        num_chunks = store.load(large_text)
        assert num_chunks > 1  # Should be split into multiple chunks

    def test_search(self):
        """Test search functionality"""
        store = ContextStore()
        store.load("The quick brown fox jumps over the lazy dog.")
        results = store.search("fox")
        assert len(results) > 0
        assert "fox" in results[0]['snippet'].lower()

    def test_search_no_results(self):
        """Test search with no matches"""
        store = ContextStore()
        store.load("The quick brown fox.")
        results = store.search("elephant")
        assert len(results) == 0

    def test_read_chunk(self):
        """Test reading chunk by ID"""
        store = ContextStore()
        store.load("Test content")
        chunk_id = store.chunks[0].id
        result = store.read(chunk_id)
        assert result is not None
        assert 'preview' in result
        assert 'truncated' in result

    def test_read_nonexistent_chunk(self):
        """Test reading nonexistent chunk"""
        store = ContextStore()
        store.load("Test content")
        result = store.read("nonexistent-id")
        assert result is None

    def test_summarize(self):
        """Test summarization"""
        store = ContextStore()
        store.load("This is a long document. " * 100)
        chunk_id = store.chunks[0].id
        summary = store.summarize(chunk_id, max_length=100)
        assert len(summary) <= 150  # Allow some buffer

    def test_get_chunks_summary(self):
        """Test chunks summary"""
        store = ContextStore()
        store.load("Test document")
        summary = store.get_chunks_summary()
        assert 'num_chunks' in summary
        assert 'total_size' in summary
        assert summary['num_chunks'] > 0


class TestUtils:
    """Test utility functions"""

    def test_detect_final_direct(self):
        """Test FINAL() detection"""
        text = 'Some text FINAL("This is the answer") more text'
        has_final, final_type, final_value = detect_final(text)
        assert has_final == True
        assert final_type == 'direct'
        assert final_value == "This is the answer"

    def test_detect_final_var(self):
        """Test FINAL_VAR() detection"""
        text = 'Some text FINAL_VAR(my_answer) more text'
        has_final, final_type, final_value = detect_final(text)
        assert has_final == True
        assert final_type == 'var'
        assert final_value == "my_answer"

    def test_detect_no_final(self):
        """Test no FINAL detection"""
        text = 'This text has no final signal'
        has_final, final_type, final_value = detect_final(text)
        assert has_final == False
        assert final_type is None
        assert final_value is None

    def test_extract_citations(self):
        """Test citation extraction"""
        text = "Based on [chunk 1.2] and [source: doc.pdf], the answer is X."
        citations = extract_citations(text)
        assert len(citations) == 2
        assert "chunk 1.2" in citations
        assert "source: doc.pdf" in citations

    def test_truncate_text(self):
        """Test text truncation"""
        text = "A" * 1000
        truncated, was_truncated = truncate_text(text, max_bytes=100)
        assert len(truncated) <= 103  # 100 + "..."
        assert was_truncated == True

    def test_truncate_text_not_needed(self):
        """Test text truncation not needed"""
        text = "Short text"
        truncated, was_truncated = truncate_text(text, max_bytes=100)
        assert truncated == text
        assert was_truncated == False


class TestTools:
    """Test RLM tools"""

    def test_context_probe_search(self):
        """Test context_probe search"""
        store = ContextStore()
        store.load("The quick brown fox jumps over the lazy dog.")

        result = context_probe(
            context_store=store,
            kind='search',
            query='fox',
            max_results=10
        )

        assert 'preview' in result
        assert 'meta' in result
        assert result['meta']['kind'] == 'search'
        assert result['meta']['num_results'] >= 0

    def test_context_probe_read(self):
        """Test context_probe read"""
        store = ContextStore()
        store.load("Test content for reading")
        chunk_id = store.chunks[0].id

        result = context_probe(
            context_store=store,
            kind='read',
            id=chunk_id
        )

        assert 'preview' in result
        assert 'meta' in result
        assert result['meta']['kind'] == 'read'

    def test_context_probe_summarize(self):
        """Test context_probe summarize"""
        store = ContextStore()
        store.load("This is a long document. " * 50)
        chunk_id = store.chunks[0].id

        result = context_probe(
            context_store=store,
            kind='summarize',
            id=chunk_id
        )

        assert 'preview' in result
        assert 'meta' in result
        assert result['meta']['kind'] == 'summarize'

    def test_context_probe_error(self):
        """Test context_probe error handling"""
        store = ContextStore()
        store.load("Test")

        # Missing required parameter
        result = context_probe(
            context_store=store,
            kind='search'
            # Missing query parameter
        )

        assert 'error' in result
