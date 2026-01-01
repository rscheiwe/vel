"""
Tests for Agent.as_tool() method - exposing agents as tools for orchestration.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from vel import Agent, ToolSpec


class TestAsToolBasics:
    """Basic as_tool() functionality tests"""

    def test_as_tool_returns_toolspec(self):
        """as_tool() should return a ToolSpec instance"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )
        tool = agent.as_tool()

        assert isinstance(tool, ToolSpec)

    def test_as_tool_default_name_from_agent_id(self):
        """Tool name should default to sanitized agent ID"""
        agent = Agent(
            id='my-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )
        tool = agent.as_tool()

        assert tool.name == 'my_agent'

    def test_as_tool_custom_name(self):
        """Tool name can be overridden"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )
        tool = agent.as_tool(name='custom_tool')

        assert tool.name == 'custom_tool'

    def test_as_tool_default_description(self):
        """Description should default to 'Run the {id} agent'"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )
        tool = agent.as_tool()

        assert tool.description == 'Run the test-agent agent'

    def test_as_tool_custom_description(self):
        """Description can be overridden"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )
        tool = agent.as_tool(description='Custom description for orchestrator')

        assert tool.description == 'Custom description for orchestrator'


class TestNameSanitization:
    """Tests for tool name sanitization"""

    def test_sanitize_colons(self):
        """Colons should be replaced with underscores"""
        agent = Agent(
            id='agent:v1',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )
        tool = agent.as_tool()

        assert tool.name == 'agent_v1'

    def test_sanitize_dashes(self):
        """Dashes should be replaced with underscores"""
        agent = Agent(
            id='my-cool-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )
        tool = agent.as_tool()

        assert tool.name == 'my_cool_agent'

    def test_sanitize_dots(self):
        """Dots should be replaced with underscores"""
        agent = Agent(
            id='agent.v1.2',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )
        tool = agent.as_tool()

        assert tool.name == 'agent_v1_2'

    def test_sanitize_mixed_characters(self):
        """Multiple special characters should all be sanitized"""
        agent = Agent(
            id='research-expert:v1.0',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )
        tool = agent.as_tool()

        assert tool.name == 'research_expert_v1_0'

    def test_sanitize_custom_name(self):
        """Custom names should also be sanitized"""
        agent = Agent(
            id='agent',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )
        tool = agent.as_tool(name='my-custom:name.here')

        assert tool.name == 'my_custom_name_here'


class TestInputOutputSchemas:
    """Tests for custom input/output schemas"""

    def test_default_input_schema(self):
        """Default input schema should have 'message' property"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )
        tool = agent.as_tool()

        assert tool.input_schema['type'] == 'object'
        assert 'message' in tool.input_schema['properties']
        assert tool.input_schema['properties']['message']['type'] == 'string'
        assert 'message' in tool.input_schema['required']

    def test_default_output_schema_is_empty(self):
        """Default output schema should be empty (flexible)"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )
        tool = agent.as_tool()

        assert tool.output_schema == {}

    def test_custom_input_schema(self):
        """Custom input schema should be used when provided"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )
        custom_schema = {
            'type': 'object',
            'properties': {
                'query': {'type': 'string'},
                'max_results': {'type': 'integer'}
            },
            'required': ['query']
        }
        tool = agent.as_tool(input_schema=custom_schema)

        assert tool.input_schema == custom_schema

    def test_custom_output_schema(self):
        """Custom output schema should be used when provided"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )
        custom_schema = {
            'type': 'object',
            'properties': {
                'results': {'type': 'array'},
                'total': {'type': 'integer'}
            }
        }
        tool = agent.as_tool(output_schema=custom_schema)

        assert tool.output_schema == custom_schema


class TestContextPassthrough:
    """Tests for context passthrough behavior"""

    @pytest.mark.asyncio
    async def test_pass_context_true_merges_contexts(self):
        """With pass_context=True, parent ctx should be merged into agent's tool_context"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'},
            tool_context={'agent_key': 'agent_value'}
        )
        tool = agent.as_tool(pass_context=True)

        captured_context = None

        async def mock_run(input_dict):
            nonlocal captured_context
            captured_context = agent.tool_context
            return 'result'

        with patch.object(agent, 'run', side_effect=mock_run):
            parent_ctx = {'parent_key': 'parent_value', 'user_id': 'alice'}
            await tool.run({'message': 'test'}, ctx=parent_ctx)

        # Context should have both agent and parent keys
        assert captured_context['agent_key'] == 'agent_value'
        assert captured_context['parent_key'] == 'parent_value'
        assert captured_context['user_id'] == 'alice'

    @pytest.mark.asyncio
    async def test_pass_context_true_parent_overrides_agent(self):
        """Parent context should override agent context for same keys"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'},
            tool_context={'shared_key': 'agent_value'}
        )
        tool = agent.as_tool(pass_context=True)

        captured_context = None

        async def mock_run(input_dict):
            nonlocal captured_context
            captured_context = agent.tool_context
            return 'result'

        with patch.object(agent, 'run', side_effect=mock_run):
            parent_ctx = {'shared_key': 'parent_value'}
            await tool.run({'message': 'test'}, ctx=parent_ctx)

        # Parent value should override agent value
        assert captured_context['shared_key'] == 'parent_value'

    @pytest.mark.asyncio
    async def test_pass_context_false_isolates_agent(self):
        """With pass_context=False, parent ctx should not be passed"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'},
            tool_context={'agent_key': 'agent_value'}
        )
        tool = agent.as_tool(pass_context=False)

        captured_context = None

        async def mock_run(input_dict):
            nonlocal captured_context
            captured_context = agent.tool_context
            return 'result'

        with patch.object(agent, 'run', side_effect=mock_run):
            parent_ctx = {'parent_key': 'parent_value'}
            await tool.run({'message': 'test'}, ctx=parent_ctx)

        # Context should only have agent keys
        assert captured_context['agent_key'] == 'agent_value'
        assert 'parent_key' not in captured_context

    @pytest.mark.asyncio
    async def test_context_restored_after_execution(self):
        """Agent's original tool_context should be restored after execution"""
        original_context = {'original': 'value'}
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'},
            tool_context=original_context.copy()
        )
        tool = agent.as_tool(pass_context=True)

        with patch.object(agent, 'run', return_value='result'):
            parent_ctx = {'parent_key': 'parent_value'}
            await tool.run({'message': 'test'}, ctx=parent_ctx)

        # Original context should be restored
        assert agent.tool_context == original_context
        assert 'parent_key' not in agent.tool_context


class TestErrorHandling:
    """Tests for error handling in as_tool()"""

    @pytest.mark.asyncio
    async def test_error_returns_structured_response(self):
        """Errors should return {success: False, error: ..., error_type: ...}"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )
        tool = agent.as_tool()

        with patch.object(agent, 'run', side_effect=ValueError('Something went wrong')):
            result = await tool.run({'message': 'test'}, ctx={})

        assert result['success'] is False
        assert result['error'] == 'Something went wrong'
        assert result['error_type'] == 'ValueError'

    @pytest.mark.asyncio
    async def test_error_preserves_exception_type(self):
        """Error type should match the actual exception class"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )
        tool = agent.as_tool()

        with patch.object(agent, 'run', side_effect=RuntimeError('Runtime issue')):
            result = await tool.run({'message': 'test'}, ctx={})

        assert result['error_type'] == 'RuntimeError'

    @pytest.mark.asyncio
    async def test_context_restored_on_error(self):
        """Context should be restored even when an error occurs"""
        original_context = {'original': 'value'}
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'},
            tool_context=original_context.copy()
        )
        tool = agent.as_tool(pass_context=True)

        with patch.object(agent, 'run', side_effect=ValueError('Error')):
            await tool.run({'message': 'test'}, ctx={'parent': 'value'})

        # Original context should be restored despite error
        assert agent.tool_context == original_context


class TestResultFormatting:
    """Tests for result formatting from sub-agent"""

    @pytest.mark.asyncio
    async def test_string_result_wrapped_in_response(self):
        """String results should be wrapped in {'response': ...}"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )
        tool = agent.as_tool()

        with patch.object(agent, 'run', return_value='Hello world'):
            result = await tool.run({'message': 'test'}, ctx={})

        assert result == {'response': 'Hello world'}

    @pytest.mark.asyncio
    async def test_dict_result_passed_through(self):
        """Dict results should be passed through as-is"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )
        tool = agent.as_tool()

        expected = {'data': 'value', 'count': 42}
        with patch.object(agent, 'run', return_value=expected):
            result = await tool.run({'message': 'test'}, ctx={})

        assert result == expected

    @pytest.mark.asyncio
    async def test_pydantic_model_dump(self):
        """Pydantic models should be converted via model_dump()"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )
        tool = agent.as_tool()

        mock_model = MagicMock()
        mock_model.model_dump.return_value = {'field': 'value'}

        with patch.object(agent, 'run', return_value=mock_model):
            result = await tool.run({'message': 'test'}, ctx={})

        assert result == {'field': 'value'}
        mock_model.model_dump.assert_called_once()

    @pytest.mark.asyncio
    async def test_legacy_pydantic_dict(self):
        """Legacy Pydantic v1 models should use .dict()"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )
        tool = agent.as_tool()

        mock_model = MagicMock(spec=['dict'])  # Only has .dict(), not .model_dump()
        mock_model.dict.return_value = {'field': 'legacy_value'}

        with patch.object(agent, 'run', return_value=mock_model):
            result = await tool.run({'message': 'test'}, ctx={})

        assert result == {'field': 'legacy_value'}

    @pytest.mark.asyncio
    async def test_other_types_stringified(self):
        """Other types should be stringified and wrapped in response"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )
        tool = agent.as_tool()

        with patch.object(agent, 'run', return_value=12345):
            result = await tool.run({'message': 'test'}, ctx={})

        assert result == {'response': '12345'}


class TestInputExtraction:
    """Tests for message extraction from input"""

    @pytest.mark.asyncio
    async def test_default_schema_extracts_message(self):
        """Default schema should extract 'message' key"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )
        tool = agent.as_tool()

        captured_input = None

        async def mock_run(input_dict):
            nonlocal captured_input
            captured_input = input_dict
            return 'result'

        with patch.object(agent, 'run', side_effect=mock_run):
            await tool.run({'message': 'Hello agent'}, ctx={})

        assert captured_input == {'message': 'Hello agent'}

    @pytest.mark.asyncio
    async def test_default_schema_fallback_to_query(self):
        """Default schema should fallback to 'query' key"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )
        tool = agent.as_tool()

        captured_input = None

        async def mock_run(input_dict):
            nonlocal captured_input
            captured_input = input_dict
            return 'result'

        with patch.object(agent, 'run', side_effect=mock_run):
            await tool.run({'query': 'Search query'}, ctx={})

        assert captured_input == {'message': 'Search query'}

    @pytest.mark.asyncio
    async def test_custom_schema_with_message_key(self):
        """Custom schema with 'message' key should use it directly"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )
        custom_schema = {
            'type': 'object',
            'properties': {
                'message': {'type': 'string'},
                'extra': {'type': 'string'}
            }
        }
        tool = agent.as_tool(input_schema=custom_schema)

        captured_input = None

        async def mock_run(input_dict):
            nonlocal captured_input
            captured_input = input_dict
            return 'result'

        with patch.object(agent, 'run', side_effect=mock_run):
            await tool.run({'message': 'Direct message', 'extra': 'ignored'}, ctx={})

        assert captured_input == {'message': 'Direct message'}

    @pytest.mark.asyncio
    async def test_custom_schema_without_message_jsonifies_input(self):
        """Custom schema without 'message' should JSON stringify entire input"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'}
        )
        custom_schema = {
            'type': 'object',
            'properties': {
                'query': {'type': 'string'},
                'max_results': {'type': 'integer'}
            }
        }
        tool = agent.as_tool(input_schema=custom_schema)

        captured_input = None

        async def mock_run(input_dict):
            nonlocal captured_input
            captured_input = input_dict
            return 'result'

        with patch.object(agent, 'run', side_effect=mock_run):
            await tool.run({'query': 'search', 'max_results': 10}, ctx={})

        # Input should be JSON stringified
        import json
        expected_message = json.dumps({'query': 'search', 'max_results': 10})
        assert captured_input == {'message': expected_message}
