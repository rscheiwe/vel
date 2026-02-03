"""
Unit tests for Vel Harness Compatibility features.

Tests for:
- Direct system_prompt injection (Feature 1)
- Direct message injection and stateless mode (Feature 2)
- Event metadata for orchestration (Feature 4)
- ToolSpec enhancements (Feature 5)
"""
import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from vel.core.context import ContextManager
from vel.tools.registry import ToolSpec
from vel.events import EventMetadata, add_metadata


# ================================================================================
# Feature 1: Direct System Prompt Tests
# ================================================================================

class TestDirectSystemPrompt:
    """Tests for direct system_prompt injection"""

    def test_string_system_prompt_stored(self):
        """Test that string system_prompt is stored correctly"""
        from vel import Agent

        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'},
            system_prompt='You are a helpful assistant.'
        )
        assert agent._system_prompt == 'You are a helpful assistant.'

    def test_callable_system_prompt_stored(self):
        """Test that callable system_prompt is stored correctly"""
        from vel import Agent

        def build_prompt(ctx):
            return f"You are {ctx.get('role', 'an assistant')}."

        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'},
            system_prompt=build_prompt
        )
        assert callable(agent._system_prompt)

    def test_get_system_prompt_returns_string(self):
        """Test get_system_prompt returns string prompt directly"""
        from vel import Agent

        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'},
            system_prompt='Direct prompt'
        )
        result = agent.get_system_prompt()
        assert result == 'Direct prompt'

    def test_get_system_prompt_calls_callable(self):
        """Test get_system_prompt invokes callable with context"""
        from vel import Agent

        def build_prompt(ctx):
            role = ctx.get('role', 'default')
            return f"You are {role}."

        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'},
            system_prompt=build_prompt
        )

        # Without context
        result = agent.get_system_prompt()
        assert result == 'You are default.'

        # With context
        result = agent.get_system_prompt(context={'role': 'a data analyst'})
        assert result == 'You are a data analyst.'

    def test_system_prompt_priority_over_template(self):
        """Test that system_prompt takes priority over prompt template"""
        from vel import Agent
        from vel.prompts import PromptTemplate

        template = PromptTemplate(
            id='test:v1',
            system='Template prompt',
            variables={}
        )

        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'},
            system_prompt='Direct prompt',  # Should take priority
            prompt=template
        )

        result = agent.get_system_prompt()
        assert result == 'Direct prompt'

    def test_backwards_compat_no_system_prompt(self):
        """Test that agents without system_prompt still work"""
        from vel import Agent

        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )
        assert agent._system_prompt is None
        # get_system_prompt should return None or template result
        result = agent.get_system_prompt()
        # Without template, should be None
        assert result is None


# ================================================================================
# Feature 2: Direct Message Injection Tests
# ================================================================================

class TestDirectMessageInjection:
    """Tests for direct message injection and stateless mode"""

    def test_context_manager_set_input_with_messages(self):
        """Test that set_input handles messages array correctly"""
        ctx = ContextManager()
        messages = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!'},
            {'role': 'user', 'content': 'How are you?'}
        ]

        ctx.set_input('run-1', {'messages': messages})
        result = ctx.messages_for_llm('run-1')

        assert len(result) == 3
        assert result[0]['content'] == 'Hello'
        assert result[2]['content'] == 'How are you?'

    def test_context_manager_messages_plus_message(self):
        """Test that messages + message appends correctly"""
        ctx = ContextManager()
        messages = [
            {'role': 'user', 'content': 'First message'}
        ]

        ctx.set_input('run-1', {
            'messages': messages,
            'message': 'Follow up'
        })
        result = ctx.messages_for_llm('run-1')

        assert len(result) == 2
        assert result[0]['content'] == 'First message'
        assert result[1]['content'] == 'Follow up'

    def test_stateless_mode_does_not_mutate_session(self):
        """Test that stateless=True doesn't mutate session"""
        ctx = ContextManager()

        # Create session with some history
        ctx._by_session['sess-1'] = [
            {'role': 'user', 'content': 'Old message'}
        ]

        # Call set_input with stateless=True
        ctx.set_input('run-1', {'message': 'New message'}, session_id='sess-1', stateless=True)

        # Session should NOT have the new message (copy was made)
        assert len(ctx._by_session['sess-1']) == 1
        assert ctx._by_session['sess-1'][0]['content'] == 'Old message'

        # But run should have both
        run_messages = ctx._by_run['run-1']
        assert len(run_messages) == 2
        assert run_messages[1]['content'] == 'New message'

    def test_stateful_mode_mutates_session(self):
        """Test that stateless=False (default) mutates session"""
        ctx = ContextManager()

        # Create session
        ctx._by_session['sess-1'] = [
            {'role': 'user', 'content': 'Old message'}
        ]

        # Call set_input with stateless=False (default)
        ctx.set_input('run-1', {'message': 'New message'}, session_id='sess-1', stateless=False)

        # Session SHOULD have the new message
        assert len(ctx._by_session['sess-1']) == 2
        assert ctx._by_session['sess-1'][1]['content'] == 'New message'


# ================================================================================
# Feature 4: Event Metadata Tests
# ================================================================================

class TestEventMetadata:
    """Tests for EventMetadata and add_metadata"""

    def test_event_metadata_creation(self):
        """Test EventMetadata dataclass creation"""
        metadata = EventMetadata(
            node_id='agent-1',
            run_id='run-123',
            step=5
        )
        assert metadata.node_id == 'agent-1'
        assert metadata.run_id == 'run-123'
        assert metadata.step == 5
        assert metadata.timestamp is not None

    def test_event_metadata_to_dict(self):
        """Test EventMetadata.to_dict excludes None values"""
        metadata = EventMetadata(node_id='agent-1')
        result = metadata.to_dict()

        assert 'node_id' in result
        assert result['node_id'] == 'agent-1'
        assert 'timestamp' in result
        # run_id and step should not be in dict since they're None
        # Actually step defaults to None, so should be excluded
        assert 'run_id' not in result or result.get('run_id') is not None

    def test_add_metadata_includes_metadata(self):
        """Test add_metadata adds metadata field to event"""
        metadata = EventMetadata(node_id='agent-1', step=3)
        event = {'type': 'text-delta', 'delta': 'Hello'}

        result = add_metadata(event, metadata)

        assert 'metadata' in result
        assert result['metadata']['node_id'] == 'agent-1'
        assert result['metadata']['step'] == 3
        assert result['type'] == 'text-delta'
        assert result['delta'] == 'Hello'

    def test_add_metadata_returns_original_if_empty(self):
        """Test add_metadata returns original event if metadata is empty"""
        metadata = EventMetadata()  # All None except timestamp
        event = {'type': 'text-delta', 'delta': 'Hello'}

        result = add_metadata(event, metadata)

        # Should still have metadata because timestamp is always set
        assert 'metadata' in result
        assert 'timestamp' in result['metadata']


# ================================================================================
# Feature 5: ToolSpec Enhancement Tests
# ================================================================================

class TestToolSpecEnhancements:
    """Tests for ToolSpec new fields"""

    def test_is_async_inferred_from_async_function(self):
        """Test is_async is inferred from async function"""
        async def async_tool(x: str) -> dict:
            return {'result': x}

        tool = ToolSpec.from_function(async_tool)
        assert tool.is_async is True

    def test_is_async_inferred_from_sync_function(self):
        """Test is_async is inferred from sync function"""
        def sync_tool(x: str) -> dict:
            return {'result': x}

        tool = ToolSpec.from_function(sync_tool)
        assert tool.is_async is False

    def test_is_async_explicit_override(self):
        """Test is_async can be explicitly overridden"""
        def sync_tool(x: str) -> dict:
            return {'result': x}

        tool = ToolSpec.from_function(sync_tool, is_async=True)
        assert tool.is_async is True

    def test_category_field(self):
        """Test category field is set correctly"""
        def my_tool(x: str) -> dict:
            return {'result': x}

        tool = ToolSpec.from_function(my_tool, category='data')
        assert tool.category == 'data'

    def test_tags_field(self):
        """Test tags field is set correctly"""
        def my_tool(x: str) -> dict:
            return {'result': x}

        tool = ToolSpec.from_function(my_tool, tags=['api', 'network'])
        assert tool.tags == ['api', 'network']

    def test_tags_default_empty_list(self):
        """Test tags defaults to empty list"""
        def my_tool(x: str) -> dict:
            return {'result': x}

        tool = ToolSpec.from_function(my_tool)
        assert tool.tags == []

    def test_requires_confirmation_field(self):
        """Test requires_confirmation field is set correctly"""
        def destructive_tool(path: str) -> dict:
            return {'deleted': path}

        tool = ToolSpec.from_function(destructive_tool, requires_confirmation=True)
        assert tool.requires_confirmation is True

    def test_requires_confirmation_default_false(self):
        """Test requires_confirmation defaults to False"""
        def safe_tool(x: str) -> dict:
            return {'result': x}

        tool = ToolSpec.from_function(safe_tool)
        assert tool.requires_confirmation is False

    def test_all_fields_together(self):
        """Test all new fields can be set together"""
        async def complex_tool(data: dict) -> dict:
            return {'processed': True}

        tool = ToolSpec.from_function(
            complex_tool,
            is_async=True,
            category='processing',
            tags=['async', 'heavy', 'io'],
            requires_confirmation=True
        )

        assert tool.is_async is True
        assert tool.category == 'processing'
        assert tool.tags == ['async', 'heavy', 'io']
        assert tool.requires_confirmation is True

    def test_backwards_compat_no_new_fields(self):
        """Test that tools work without new fields"""
        def simple_tool(x: str) -> str:
            return x.upper()

        # Should work exactly as before
        tool = ToolSpec.from_function(simple_tool)

        assert tool.name == 'simple_tool'
        assert tool.is_async is False
        assert tool.category is None
        assert tool.tags == []
        assert tool.requires_confirmation is False


# ================================================================================
# Integration Tests
# ================================================================================

class TestHarnessIntegration:
    """Integration tests for harness compatibility features"""

    def test_agent_with_all_new_features(self):
        """Test agent can be created with all new features"""
        from vel import Agent

        def dynamic_prompt(ctx):
            skill = ctx.get('skill', '')
            return f"You are an assistant.\n\n{skill}"

        def get_data(query: str) -> dict:
            """Get data based on query."""
            return {'result': query}

        tool = ToolSpec.from_function(
            get_data,
            category='data',
            tags=['query'],
            requires_confirmation=False
        )

        # Should not raise
        agent = Agent(
            id='harness-test',
            model={'provider': 'openai', 'model': 'gpt-4o'},
            system_prompt=dynamic_prompt,
            tools=[tool]
        )

        # Verify prompt works
        prompt = agent.get_system_prompt(context={'skill': 'Data analysis'})
        assert 'Data analysis' in prompt
