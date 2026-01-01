"""
Tests for the Scratchpad feature.

Tests cover:
- ScratchpadConfig validation and defaults
- ScratchpadEntry hash computation and serialization
- Scratchpad core operations (write, read, clear)
- Specialized operations (plan, findings, observations, reasoning, errors)
- Eviction strategy
- Summary generation
- Search functionality
- Checkpointing
- Thread safety
- Tool binding
"""

import pytest
import threading
from datetime import datetime, timezone
from typing import List

from vel.tools.scratchpad import (
    Scratchpad,
    ScratchpadConfig,
    ScratchpadEntry,
    EntryType,
    get_scratchpad_tools,
)


# =============================================================================
# ScratchpadConfig Tests
# =============================================================================

class TestScratchpadConfig:
    """Tests for ScratchpadConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ScratchpadConfig()
        assert config.max_entries == 100
        assert config.max_content_length == 50000
        assert config.summary_max_chars == 500
        assert config.include_search is True
        assert config.include_checkpoint is True

    def test_validation_bounds_min(self):
        """Test that config values are clamped to minimum bounds."""
        config = ScratchpadConfig(
            max_entries=5,  # Below min of 10
            max_content_length=500,  # Below min of 1000
            summary_max_chars=50,  # Below min of 100
        )
        assert config.max_entries == 10
        assert config.max_content_length == 1000
        assert config.summary_max_chars == 100

    def test_validation_bounds_max(self):
        """Test that config values are clamped to maximum bounds."""
        config = ScratchpadConfig(
            max_entries=1000,  # Above max of 500
            max_content_length=200000,  # Above max of 100000
            summary_max_chars=5000,  # Above max of 2000
        )
        assert config.max_entries == 500
        assert config.max_content_length == 100000
        assert config.summary_max_chars == 2000

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = ScratchpadConfig(max_entries=50)
        d = config.to_dict()
        assert d['max_entries'] == 50
        assert 'max_content_length' in d
        assert 'summary_max_chars' in d


# =============================================================================
# ScratchpadEntry Tests
# =============================================================================

class TestScratchpadEntry:
    """Tests for ScratchpadEntry dataclass."""

    def test_hash_computation(self):
        """Test that content hash is computed correctly."""
        entry = ScratchpadEntry(
            key='test',
            content='Hello, World!',
            entry_type=EntryType.NOTE,
            created_at=datetime.now(timezone.utc),
        )
        assert len(entry.content_hash) == 16  # Truncated SHA-256

    def test_hash_different_content(self):
        """Test that different content produces different hashes."""
        entry1 = ScratchpadEntry(
            key='test1',
            content='Hello',
            entry_type=EntryType.NOTE,
            created_at=datetime.now(timezone.utc),
        )
        entry2 = ScratchpadEntry(
            key='test2',
            content='World',
            entry_type=EntryType.NOTE,
            created_at=datetime.now(timezone.utc),
        )
        assert entry1.content_hash != entry2.content_hash

    def test_to_dict_from_dict_roundtrip(self):
        """Test serialization/deserialization roundtrip."""
        original = ScratchpadEntry(
            key='test_key',
            content='Test content',
            entry_type=EntryType.FINDING,
            created_at=datetime.now(timezone.utc),
            metadata={'source': 'api_docs', 'confidence': 0.9},
        )
        d = original.to_dict()
        restored = ScratchpadEntry.from_dict(d)

        assert restored.key == original.key
        assert restored.content == original.content
        assert restored.entry_type == original.entry_type
        assert restored.metadata == original.metadata
        assert restored.content_hash == original.content_hash


# =============================================================================
# Scratchpad Core Operations Tests
# =============================================================================

class TestScratchpadCore:
    """Tests for core scratchpad operations."""

    def test_write_and_read_single(self):
        """Test basic write and read operations."""
        sp = Scratchpad()
        result = sp.write('my_key', 'my content', EntryType.NOTE)

        assert 'my_key' in result
        content = sp.read('my_key')
        assert 'my content' in content

    def test_write_overwrites_same_key(self):
        """Test that writing to same key overwrites."""
        sp = Scratchpad()
        sp.write('key1', 'first content')
        sp.write('key1', 'second content')

        content = sp.read('key1')
        assert 'second content' in content
        assert 'first content' not in content

    def test_read_all_formatted(self):
        """Test reading all entries returns formatted output."""
        sp = Scratchpad()
        sp.write('note1', 'First note')
        sp.write('note2', 'Second note')

        all_content = sp.read()
        assert 'SCRATCHPAD CONTENTS' in all_content
        assert 'First note' in all_content
        assert 'Second note' in all_content

    def test_read_nonexistent_key(self):
        """Test reading nonexistent key returns helpful message."""
        sp = Scratchpad()
        result = sp.read('nonexistent')
        assert 'not found' in result.lower()

    def test_read_empty_scratchpad(self):
        """Test reading empty scratchpad."""
        sp = Scratchpad()
        result = sp.read()
        assert 'empty' in result.lower()

    def test_content_length_truncation(self):
        """Test that content exceeding max length is truncated."""
        config = ScratchpadConfig(max_content_length=1000)
        sp = Scratchpad(config)

        long_content = 'x' * 2000
        sp.write('long', long_content)

        # Entry should be stored but truncated
        entry = sp._entries.get('long')
        assert entry is not None
        assert len(entry.content) == 1000

    def test_clear_preserves_persistent(self):
        """Test that clear preserves plan and error entries."""
        sp = Scratchpad()
        sp.set_plan('My execution plan')
        sp.log_error('An error occurred')
        sp.write('temp_note', 'Temporary data')

        result = sp.clear(preserve_persistent=True)
        assert '1' in result  # Only 1 entry cleared

        # Plan and error should still exist
        assert sp._entries.get('_plan') is not None
        assert any(e.entry_type == EntryType.ERROR for e in sp._entries.values())
        # Note should be gone
        assert sp._entries.get('temp_note') is None

    def test_clear_all(self):
        """Test clearing all entries including persistent."""
        sp = Scratchpad()
        sp.set_plan('My plan')
        sp.write('note', 'A note')

        result = sp.clear(preserve_persistent=False)

        assert len(sp._entries) == 0


# =============================================================================
# Specialized Operations Tests
# =============================================================================

class TestScratchpadSpecialized:
    """Tests for specialized write operations."""

    def test_set_plan_overwrites(self):
        """Test that set_plan overwrites existing plan."""
        sp = Scratchpad()
        sp.set_plan('Plan version 1')
        sp.set_plan('Plan version 2')

        content = sp.read('_plan')
        assert 'version 2' in content
        assert 'version 1' not in content

    def test_add_finding_auto_numbers(self):
        """Test that findings are auto-numbered."""
        sp = Scratchpad()
        result1 = sp.add_finding('First finding')
        result2 = sp.add_finding('Second finding')
        result3 = sp.add_finding('Third finding')

        assert '#1' in result1
        assert '#2' in result2
        assert '#3' in result3

    def test_add_finding_with_source(self):
        """Test finding with source attribution."""
        sp = Scratchpad()
        sp.add_finding('OAuth required', source='api_docs')

        all_content = sp.read()
        assert 'api_docs' in all_content

    def test_add_observation_with_tool(self):
        """Test observation with tool name."""
        sp = Scratchpad()
        sp.add_observation('Found 5 results', tool_name='search_tool')

        all_content = sp.read()
        assert 'search_tool' in all_content
        assert '5 results' in all_content

    def test_add_reasoning_auto_increment(self):
        """Test reasoning steps auto-increment."""
        sp = Scratchpad()
        sp.add_reasoning('First thought')
        sp.add_reasoning('Second thought')

        all_content = sp.read()
        assert 'step 1' in all_content
        assert 'step 2' in all_content

    def test_add_reasoning_explicit_step(self):
        """Test reasoning with explicit step number."""
        sp = Scratchpad()
        sp.add_reasoning('Later thought', step=5)

        all_content = sp.read()
        assert 'step 5' in all_content

    def test_log_error_marked_persistent(self):
        """Test that errors are marked as persistent."""
        sp = Scratchpad()
        sp.log_error('Critical error', context='during API call')

        # Error should have persistent metadata
        errors = [e for e in sp._entries.values() if e.entry_type == EntryType.ERROR]
        assert len(errors) == 1
        assert errors[0].metadata.get('persistent') is True


# =============================================================================
# Eviction Tests
# =============================================================================

class TestScratchpadEviction:
    """Tests for entry eviction strategy."""

    def test_eviction_when_over_limit(self):
        """Test that oldest entries are evicted when over limit."""
        config = ScratchpadConfig(max_entries=10)
        sp = Scratchpad(config)

        # Add 15 entries
        for i in range(15):
            sp.write(f'note_{i}', f'Content {i}')

        # Should have max 10 entries
        assert len(sp._entries) <= 10

    def test_eviction_protects_errors(self):
        """Test that error entries are never evicted."""
        config = ScratchpadConfig(max_entries=10)
        sp = Scratchpad(config)

        # Add an error first
        sp.log_error('Important error')

        # Add many notes to trigger eviction
        for i in range(15):
            sp.write(f'note_{i}', f'Content {i}')

        # Error should still exist
        errors = [e for e in sp._entries.values() if e.entry_type == EntryType.ERROR]
        assert len(errors) == 1

    def test_eviction_protects_plan(self):
        """Test that plan entries are never evicted."""
        config = ScratchpadConfig(max_entries=10)
        sp = Scratchpad(config)

        # Add a plan first
        sp.set_plan('Important plan')

        # Add many notes
        for i in range(15):
            sp.write(f'note_{i}', f'Content {i}')

        # Plan should still exist
        assert sp._entries.get('_plan') is not None

    def test_eviction_protects_checkpoints(self):
        """Test that checkpoint entries are never evicted."""
        config = ScratchpadConfig(max_entries=10)
        sp = Scratchpad(config)

        # Add a checkpoint first
        sp.checkpoint('important_state')

        # Add many notes
        for i in range(15):
            sp.write(f'note_{i}', f'Content {i}')

        # Checkpoint should still exist
        checkpoints = [e for e in sp._entries.values() if e.entry_type == EntryType.CHECKPOINT]
        assert len(checkpoints) >= 1


# =============================================================================
# Summary Tests
# =============================================================================

class TestScratchpadSummary:
    """Tests for summary generation."""

    def test_empty_scratchpad_returns_empty(self):
        """Test that empty scratchpad returns empty summary."""
        sp = Scratchpad()
        summary = sp.get_summary()
        assert summary == ""

    def test_summary_includes_errors_first(self):
        """Test that errors appear in summary with priority."""
        sp = Scratchpad()
        sp.write('note', 'A regular note')
        sp.log_error('Critical error occurred')

        summary = sp.get_summary()
        # Errors should be mentioned
        assert 'error' in summary.lower() or 'Error' in summary

    def test_summary_includes_plan(self):
        """Test that plan appears in summary."""
        sp = Scratchpad()
        sp.set_plan('Step 1: Do this\nStep 2: Do that')

        summary = sp.get_summary()
        assert 'Plan' in summary or 'Step 1' in summary

    def test_summary_includes_findings(self):
        """Test that findings appear in summary."""
        sp = Scratchpad()
        sp.add_finding('Important discovery', source='research')

        summary = sp.get_summary()
        assert 'Finding' in summary or 'discovery' in summary

    def test_summary_respects_max_chars(self):
        """Test that summary respects max_chars limit."""
        sp = Scratchpad()
        sp.set_plan('A' * 1000)  # Long plan
        for i in range(10):
            sp.add_finding(f'Finding {i} with long content ' * 5)

        summary = sp.get_summary(max_chars=200)
        assert len(summary) <= 200


# =============================================================================
# Search Tests
# =============================================================================

class TestScratchpadSearch:
    """Tests for search functionality."""

    def test_search_case_insensitive(self):
        """Test that search is case-insensitive."""
        sp = Scratchpad()
        sp.write('key1', 'Hello World')
        sp.write('key2', 'goodbye world')

        results = sp.search('WORLD')
        assert len(results) == 2

    def test_search_by_entry_type(self):
        """Test filtering search by entry type."""
        sp = Scratchpad()
        sp.add_finding('Finding about OAuth')
        sp.write('note', 'Note about OAuth')

        results = sp.search('OAuth', entry_types=[EntryType.FINDING])
        assert len(results) == 1
        assert results[0].entry_type == EntryType.FINDING

    def test_search_no_results(self):
        """Test search with no matches."""
        sp = Scratchpad()
        sp.write('key', 'Some content')

        results = sp.search('nonexistent')
        assert len(results) == 0

    def test_search_in_key_and_content(self):
        """Test that search looks in both key and content."""
        sp = Scratchpad()
        sp.write('oauth_config', 'Some configuration')
        sp.write('other_key', 'oauth settings here')

        results = sp.search('oauth')
        assert len(results) == 2


# =============================================================================
# Checkpoint Tests
# =============================================================================

class TestScratchpadCheckpoint:
    """Tests for checkpointing functionality."""

    def test_create_checkpoint(self):
        """Test creating a checkpoint."""
        sp = Scratchpad()
        sp.write('note', 'Some data')
        result = sp.checkpoint('state_v1')

        assert 'state_v1' in result
        assert len(sp._checkpoints) == 1

    def test_list_checkpoints_ordered(self):
        """Test that checkpoints are listed newest first."""
        sp = Scratchpad()
        sp.checkpoint('first')
        sp.checkpoint('second')
        sp.checkpoint('third')

        checkpoints = sp.list_checkpoints()
        assert len(checkpoints) == 3
        assert checkpoints[0]['label'] == 'third'  # Newest first

    def test_checkpoint_stores_state(self):
        """Test that checkpoint stores full state."""
        sp = Scratchpad()
        sp.write('note', 'Data before checkpoint')
        sp.checkpoint('snapshot')

        # Verify checkpoint contains state
        assert len(sp._checkpoints) == 1
        assert 'state' in sp._checkpoints[0]
        assert 'entries' in sp._checkpoints[0]['state']


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestScratchpadThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_writes_no_corruption(self):
        """Test that concurrent writes don't corrupt state."""
        sp = Scratchpad()
        errors: List[Exception] = []

        def writer(thread_id: int):
            try:
                for i in range(50):
                    sp.write(f'thread_{thread_id}_entry_{i}', f'Content from thread {thread_id}')
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should have occurred
        assert len(errors) == 0

        # All entries should be present (or evicted properly)
        # With eviction, we should have at most max_entries
        assert len(sp._entries) <= sp._config.max_entries

    def test_concurrent_reads_consistent(self):
        """Test that concurrent reads are consistent."""
        sp = Scratchpad()
        sp.write('key', 'value')
        errors: List[Exception] = []

        def reader():
            try:
                for _ in range(100):
                    sp.read('key')
                    sp.read()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# Serialization Tests
# =============================================================================

class TestScratchpadSerialization:
    """Tests for serialization."""

    def test_to_dict_from_dict_roundtrip(self):
        """Test full scratchpad serialization roundtrip."""
        sp1 = Scratchpad()
        sp1.set_plan('My plan')
        sp1.add_finding('Finding 1', source='docs')
        sp1.add_finding('Finding 2')
        sp1.write('note', 'A note')

        # Serialize
        d = sp1.to_dict()

        # Deserialize
        sp2 = Scratchpad.from_dict(d)

        # Verify
        assert len(sp2._entries) == len(sp1._entries)
        assert sp2._finding_counter == sp1._finding_counter
        assert sp2._entries.get('_plan') is not None


# =============================================================================
# Tool Binding Tests
# =============================================================================

class TestScratchpadTools:
    """Tests for tool binding."""

    def test_get_scratchpad_tools_default_count(self):
        """Test that default config returns expected number of tools."""
        sp = Scratchpad()
        tools = get_scratchpad_tools(sp)

        # Default: 5 core + 1 search + 2 checkpoint = 8 tools
        assert len(tools) == 8

    def test_get_scratchpad_tools_without_search(self):
        """Test tools without search feature."""
        config = ScratchpadConfig(include_search=False)
        sp = Scratchpad(config)
        tools = get_scratchpad_tools(sp)

        # 5 core + 2 checkpoint = 7 tools
        assert len(tools) == 7
        assert not any(t.name == 'search_scratchpad' for t in tools)

    def test_get_scratchpad_tools_without_checkpoint(self):
        """Test tools without checkpoint features."""
        config = ScratchpadConfig(include_checkpoint=False)
        sp = Scratchpad(config)
        tools = get_scratchpad_tools(sp)

        # 5 core + 1 search = 6 tools
        assert len(tools) == 6
        assert not any('checkpoint' in t.name for t in tools)

    def test_tools_share_scratchpad_instance(self):
        """Test that all tools operate on the same scratchpad."""
        sp = Scratchpad()
        tools = get_scratchpad_tools(sp)

        # Find tools by name
        write_tool = next(t for t in tools if t.name == 'write_to_scratchpad')
        read_tool = next(t for t in tools if t.name == 'read_from_scratchpad')

        # Write using tool
        write_tool._handler(key='test_key', content='test_content')

        # Read using tool - should see the written content
        result = read_tool._handler(key='test_key')
        assert 'test_content' in result

    def test_tool_schemas_have_required_fields(self):
        """Test that tool schemas have required fields."""
        sp = Scratchpad()
        tools = get_scratchpad_tools(sp)

        for tool in tools:
            assert tool.name is not None
            assert tool.description is not None
            assert tool.input_schema is not None


# =============================================================================
# Stats Tests
# =============================================================================

class TestScratchpadStats:
    """Tests for statistics."""

    def test_get_stats(self):
        """Test getting scratchpad statistics."""
        sp = Scratchpad()
        sp.set_plan('A plan')
        sp.add_finding('Finding 1')
        sp.add_finding('Finding 2')
        sp.write('note', 'A note')

        stats = sp.get_stats()

        assert stats.total_entries == 4
        assert stats.entries_by_type.get('plan') == 1
        assert stats.entries_by_type.get('finding') == 2
        assert stats.entries_by_type.get('note') == 1
        assert stats.oldest_entry is not None
        assert stats.newest_entry is not None
