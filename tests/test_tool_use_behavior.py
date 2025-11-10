"""
Tests for tool use behavior (stop_on_first_tool)
"""

import pytest
from vel import Agent, ToolSpec
from vel.tools import ToolRegistry


@pytest.fixture
def mock_tool():
    """Create a mock tool for testing"""
    async def handler(input: dict, ctx: dict) -> dict:
        return {'result': f"Processed: {input.get('value', 'default')}"}

    tool = ToolSpec(
        name='test_tool',
        input_schema={
            'type': 'object',
            'properties': {'value': {'type': 'string'}},
        },
        output_schema={
            'type': 'object',
            'properties': {'result': {'type': 'string'}},
            'required': ['result']
        },
        handler=handler
    )

    # Register tool
    registry = ToolRegistry.default()
    registry.register(tool)

    yield tool

    # Cleanup
    if 'test_tool' in registry._tools:
        del registry._tools['test_tool']


@pytest.fixture
def mock_tool_2():
    """Create a second mock tool for testing"""
    async def handler(input: dict, ctx: dict) -> dict:
        return {'status': 'completed'}

    tool = ToolSpec(
        name='test_tool_2',
        input_schema={
            'type': 'object',
            'properties': {'action': {'type': 'string'}},
        },
        output_schema={
            'type': 'object',
            'properties': {'status': {'type': 'string'}},
            'required': ['status']
        },
        handler=handler
    )

    # Register tool
    registry = ToolRegistry.default()
    registry.register(tool)

    yield tool

    # Cleanup
    if 'test_tool_2' in registry._tools:
        del registry._tools['test_tool_2']


class TestToolUseBehavior:
    """Tests for stop_on_first_tool policies"""

    def test_should_stop_after_tool_global_true(self, mock_tool):
        """Test global stop_on_first_tool=True"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'},
            tools=['test_tool'],
            policies={'stop_on_first_tool': True}
        )

        assert agent.should_stop_after_tool('test_tool') is True

    def test_should_stop_after_tool_global_false(self, mock_tool):
        """Test global stop_on_first_tool=False"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'},
            tools=['test_tool'],
            policies={'stop_on_first_tool': False}
        )

        assert agent.should_stop_after_tool('test_tool') is False

    def test_should_stop_after_tool_default(self, mock_tool):
        """Test default behavior (no policy set)"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'},
            tools=['test_tool'],
            # No policies - should default to False
        )

        assert agent.should_stop_after_tool('test_tool') is False

    def test_should_stop_after_tool_per_tool_override(self, mock_tool, mock_tool_2):
        """Test per-tool behavior overrides global setting"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'},
            tools=['test_tool', 'test_tool_2'],
            policies={
                'stop_on_first_tool': False,  # Global default
                'tool_behavior': {
                    'test_tool': {'stop_on_first_use': True},  # Override for test_tool
                }
            }
        )

        # test_tool should stop (per-tool override)
        assert agent.should_stop_after_tool('test_tool') is True

        # test_tool_2 should continue (global default)
        assert agent.should_stop_after_tool('test_tool_2') is False

    def test_should_stop_after_tool_per_tool_explicit_false(self, mock_tool):
        """Test per-tool behavior with explicit False"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'},
            tools=['test_tool'],
            policies={
                'stop_on_first_tool': True,  # Global: stop
                'tool_behavior': {
                    'test_tool': {'stop_on_first_use': False}  # Override: don't stop
                }
            }
        )

        # Per-tool False should override global True
        assert agent.should_stop_after_tool('test_tool') is False

    def test_should_stop_after_tool_unknown_tool(self):
        """Test behavior for unknown tool (no registration needed)"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'},
            policies={'stop_on_first_tool': True}
        )

        # Should use global setting for unknown tool
        assert agent.should_stop_after_tool('unknown_tool') is True

    def test_policies_structure(self, mock_tool):
        """Test that policies structure is correctly parsed"""
        agent = Agent(
            id='test-agent',
            model={'provider': 'openai', 'model': 'gpt-4o'},
            tools=['test_tool'],
            policies={
                'max_steps': 10,
                'stop_on_first_tool': True,
                'tool_behavior': {
                    'test_tool': {'stop_on_first_use': False}
                },
                'retry': {'attempts': 3}
            }
        )

        # Verify policies are stored correctly
        assert agent.policies['max_steps'] == 10
        assert agent.policies['stop_on_first_tool'] is True
        assert agent.policies['tool_behavior']['test_tool']['stop_on_first_use'] is False
        assert agent.policies['retry']['attempts'] == 3

        # Verify behavior resolution works correctly
        assert agent.should_stop_after_tool('test_tool') is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
