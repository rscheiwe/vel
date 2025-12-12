"""Tests for Extended Thinking (ReflectionController)."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncGenerator, Dict, Any

from vel.thinking import ThinkingConfig, ReflectionController
from vel.thinking.controller import ThinkingPhase, ThinkingState
from vel.events import (
    ReasoningStartEvent, ReasoningDeltaEvent, ReasoningEndEvent,
    TextStartEvent, TextDeltaEvent, TextEndEvent,
    ToolInputAvailableEvent, ToolOutputAvailableEvent
)


class TestThinkingConfig:
    """Test ThinkingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ThinkingConfig()
        assert config.mode == 'none'
        assert config.show_analysis is True
        assert config.show_critiques is True
        assert config.show_refinements is True
        assert config.stream_thinking is True
        assert config.max_refinements == 3
        assert config.confidence_threshold == 0.8
        assert config.thinking_tools is True
        assert config.thinking_model is None

    def test_reflection_mode(self):
        """Test reflection mode configuration."""
        config = ThinkingConfig(
            mode='reflection',
            max_refinements=2,
            confidence_threshold=0.9
        )
        assert config.mode == 'reflection'
        assert config.max_refinements == 2
        assert config.confidence_threshold == 0.9

    def test_validation_max_refinements(self):
        """Test max_refinements validation bounds."""
        # Below minimum
        config = ThinkingConfig(max_refinements=0)
        assert config.max_refinements == 1

        # Above maximum
        config = ThinkingConfig(max_refinements=10)
        assert config.max_refinements == 5

    def test_validation_confidence_threshold(self):
        """Test confidence_threshold validation bounds."""
        # Below minimum
        config = ThinkingConfig(confidence_threshold=-0.5)
        assert config.confidence_threshold == 0

        # Above maximum
        config = ThinkingConfig(confidence_threshold=1.5)
        assert config.confidence_threshold == 1

    def test_thinking_model_override(self):
        """Test thinking_model configuration."""
        config = ThinkingConfig(
            mode='reflection',
            thinking_model={'provider': 'openai', 'model': 'gpt-4o-mini'}
        )
        assert config.thinking_model['provider'] == 'openai'
        assert config.thinking_model['model'] == 'gpt-4o-mini'

    def test_to_dict(self):
        """Test serialization to dict."""
        config = ThinkingConfig(mode='reflection')
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d['mode'] == 'reflection'
        assert 'max_refinements' in d


class TestThinkingState:
    """Test ThinkingState dataclass."""

    def test_default_state(self):
        """Test default state values."""
        state = ThinkingState(question='What is 2+2?')
        assert state.question == 'What is 2+2?'
        assert state.analysis == ''
        assert state.critiques == ''
        assert state.refined == ''
        assert state.confidence == 0.0
        assert state.iteration == 0
        assert state.context == []

    def test_state_updates(self):
        """Test state can be updated."""
        state = ThinkingState(question='Test')
        state.analysis = 'Analysis content'
        state.confidence = 0.75
        state.iteration = 2

        assert state.analysis == 'Analysis content'
        assert state.confidence == 0.75
        assert state.iteration == 2


class TestReflectionController:
    """Test ReflectionController."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock provider."""
        provider = MagicMock()
        return provider

    @pytest.fixture
    def config(self):
        """Create test config."""
        return ThinkingConfig(
            mode='reflection',
            max_refinements=2,
            confidence_threshold=0.8,
            thinking_tools=False  # Disable tools for simpler tests
        )

    @pytest.fixture
    def controller(self, mock_provider, config):
        """Create controller instance."""
        return ReflectionController(
            provider=mock_provider,
            model='test-model',
            config=config,
            tools=None,
            tool_executor=None
        )

    def test_confidence_extraction(self, controller):
        """Test confidence extraction from response text."""
        # Percentage format
        assert controller._extract_confidence('Confidence: 85%') == 0.85
        assert controller._extract_confidence('confidence: 90%') == 0.90

        # Decimal format
        assert controller._extract_confidence('Confidence: 0.75') == 0.75

        # Inline formats
        assert controller._extract_confidence('[75%] sure about this') == 0.75
        assert controller._extract_confidence('(80% confident) in the result') == 0.80

        # Not found - default
        assert controller._extract_confidence('No confidence mentioned') == 0.6

    def test_refinement_extraction(self, controller):
        """Test refinement content extraction."""
        response = """This is the refined analysis.
It addresses the critiques.

Confidence: 85%"""

        refined = controller._extract_refinement(response)
        assert 'This is the refined analysis' in refined
        assert 'Confidence: 85%' not in refined

    def test_should_show_phase(self, controller):
        """Test phase visibility logic."""
        # Analysis shown by default
        assert controller._should_show_phase(ThinkingPhase.ANALYZE) is True

        # Critique shown by default
        assert controller._should_show_phase(ThinkingPhase.CRITIQUE) is True

        # Refine shown by default
        assert controller._should_show_phase(ThinkingPhase.REFINE) is True

        # Conclude never shown (goes to text, not reasoning)
        assert controller._should_show_phase(ThinkingPhase.CONCLUDE) is False

    def test_should_show_phase_disabled(self, mock_provider):
        """Test phase visibility when disabled."""
        config = ThinkingConfig(
            mode='reflection',
            show_analysis=False,
            show_critiques=False,
            show_refinements=False
        )
        controller = ReflectionController(
            provider=mock_provider,
            model='test-model',
            config=config,
            tools=None,
            tool_executor=None
        )

        assert controller._should_show_phase(ThinkingPhase.ANALYZE) is False
        assert controller._should_show_phase(ThinkingPhase.CRITIQUE) is False
        assert controller._should_show_phase(ThinkingPhase.REFINE) is False

    def test_stage_event(self, controller):
        """Test stage event generation."""
        state = ThinkingState(question='Test')
        state.iteration = 1
        state.confidence = 0.65

        # Analyze stage
        event = controller._stage_event(ThinkingPhase.ANALYZE, step=1, state=state)
        assert event['type'] == 'data-thinking-stage'
        assert event['data']['stage'] == 'analyzing'
        assert event['data']['step'] == 1
        assert event['transient'] is True

        # Refine stage includes iteration and confidence
        event = controller._stage_event(ThinkingPhase.REFINE, step=3, state=state)
        assert event['data']['stage'] == 'refining'
        assert event['data']['iteration'] == 1
        assert event['data']['confidence'] == 0.65

    def test_build_phase_messages(self, controller):
        """Test message building for different phases."""
        state = ThinkingState(question='What is Python?')
        state.analysis = 'Python is a programming language...'
        state.critiques = 'Consider mentioning its use cases...'

        # Analyze
        msgs = controller._build_phase_messages(ThinkingPhase.ANALYZE, state)
        assert len(msgs) == 1
        assert 'What is Python?' in msgs[0]['content']

        # Critique
        msgs = controller._build_phase_messages(ThinkingPhase.CRITIQUE, state)
        assert 'Python is a programming language' in msgs[0]['content']

        # Refine
        msgs = controller._build_phase_messages(ThinkingPhase.REFINE, state)
        assert 'What is Python?' in msgs[0]['content']
        assert 'Consider mentioning' in msgs[0]['content']

    def test_update_state(self, controller):
        """Test state updates for different phases."""
        state = ThinkingState(question='Test')

        # Update analysis
        controller._update_state(ThinkingPhase.ANALYZE, state, 'Analysis content')
        assert state.analysis == 'Analysis content'

        # Update critiques
        controller._update_state(ThinkingPhase.CRITIQUE, state, 'Critique content')
        assert state.critiques == 'Critique content'

        # Update refine (with confidence)
        controller._update_state(
            ThinkingPhase.REFINE,
            state,
            'Refined content\n\nConfidence: 75%'
        )
        assert 'Refined content' in state.refined
        assert state.confidence == 0.75


class TestReflectionControllerIntegration:
    """Integration tests for ReflectionController with mocked LLM."""

    @pytest.fixture
    def mock_stream_events(self):
        """Create mock stream events generator."""
        async def make_stream(content: str):
            """Generate mock text-delta events."""
            yield MagicMock(type='text-start', block_id='test')
            for i in range(0, len(content), 10):
                chunk = content[i:i+10]
                event = MagicMock(type='text-delta', delta=chunk)
                yield event
            yield MagicMock(type='text-end', block_id='test')
            yield MagicMock(type='finish-message', finish_reason='stop')

        return make_stream

    @pytest.mark.asyncio
    async def test_full_flow_high_confidence(self, mock_stream_events):
        """Test full reflection flow with high confidence (early exit)."""
        mock_provider = MagicMock()

        # Mock responses for each phase
        responses = [
            'This is my analysis of the problem...',  # Analyze
            'The analysis has one minor gap...',       # Critique
            'Addressing the gap...\n\nConfidence: 95%',  # Refine (high confidence)
            'The final answer is...'                   # Conclude
        ]
        response_idx = 0

        async def mock_stream(*args, **kwargs):
            nonlocal response_idx
            content = responses[response_idx]
            response_idx += 1
            async for event in mock_stream_events(content):
                yield event

        mock_provider.stream = mock_stream

        config = ThinkingConfig(
            mode='reflection',
            max_refinements=3,
            confidence_threshold=0.9,
            thinking_tools=False
        )

        controller = ReflectionController(
            provider=mock_provider,
            model='test-model',
            config=config,
            tools=None,
            tool_executor=None
        )

        events = []
        async for event in controller.run('What is 2+2?'):
            events.append(event)

        # Verify event sequence
        event_types = [e.get('type') for e in events]

        # Should have reasoning-start at beginning
        assert 'reasoning-start' in event_types

        # Should have reasoning-delta events
        reasoning_deltas = [e for e in events if e.get('type') == 'reasoning-delta']
        assert len(reasoning_deltas) > 0

        # Should have reasoning-end
        assert 'reasoning-end' in event_types

        # Should have text events for final answer
        assert 'text-start' in event_types
        assert 'text-delta' in event_types
        assert 'text-end' in event_types

        # Should have completion metadata
        assert 'data-thinking-complete' in event_types

        # Find completion event and verify
        complete_event = next(e for e in events if e.get('type') == 'data-thinking-complete')
        assert complete_event['data']['final_confidence'] >= 0.9
        assert complete_event['data']['iterations'] == 1  # High confidence, single iteration

    @pytest.mark.asyncio
    async def test_adaptive_refinement(self, mock_stream_events):
        """Test adaptive refinement loop with low confidence."""
        mock_provider = MagicMock()

        # Responses with increasing confidence
        responses = [
            'Initial analysis...',
            'First critique...',
            'First refinement...\nConfidence: 50%',  # Low confidence
            'Second critique...',
            'Better refinement...\nConfidence: 85%',  # High enough
            'Final answer...'
        ]
        response_idx = 0

        async def mock_stream(*args, **kwargs):
            nonlocal response_idx
            content = responses[min(response_idx, len(responses)-1)]
            response_idx += 1
            async for event in mock_stream_events(content):
                yield event

        mock_provider.stream = mock_stream

        config = ThinkingConfig(
            mode='reflection',
            max_refinements=3,
            confidence_threshold=0.8,
            thinking_tools=False
        )

        controller = ReflectionController(
            provider=mock_provider,
            model='test-model',
            config=config,
            tools=None,
            tool_executor=None
        )

        events = []
        async for event in controller.run('Complex question'):
            events.append(event)

        # Find completion event
        complete_event = next(e for e in events if e.get('type') == 'data-thinking-complete')

        # Should have multiple iterations (low confidence first, then high)
        assert complete_event['data']['iterations'] >= 2


class TestContextManagerWithReasoning:
    """Test ContextManager multi-part message support."""

    def test_append_assistant_with_reasoning(self):
        """Test storing reasoning + answer message."""
        from vel.core import ContextManager

        ctx = ContextManager()
        run_id = 'test-run'

        # Set initial input
        ctx.set_input(run_id, {'message': 'What is Python?'})

        # Append reasoning + answer
        ctx.append_assistant_with_reasoning(
            run_id,
            reasoning='[Analysis]\nPython is...\n[Refinement]\nConsidering...',
            answer='Python is a high-level programming language...',
            metadata={'steps': 4, 'iterations': 2, 'final_confidence': 0.9}
        )

        # Get messages
        messages = ctx.messages_for_llm(run_id)
        assert len(messages) == 2  # User + assistant

        # Check assistant message structure
        assistant_msg = messages[1]
        assert assistant_msg['role'] == 'assistant'
        assert isinstance(assistant_msg['content'], list)
        assert len(assistant_msg['content']) == 2

        # Check parts
        reasoning_part = assistant_msg['content'][0]
        assert reasoning_part['type'] == 'reasoning'
        assert '[Analysis]' in reasoning_part['text']

        text_part = assistant_msg['content'][1]
        assert text_part['type'] == 'text'
        assert 'Python is a high-level' in text_part['text']

        # Check metadata
        assert assistant_msg['thinking_metadata']['final_confidence'] == 0.9
