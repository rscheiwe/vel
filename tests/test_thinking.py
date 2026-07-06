"""Tests for Extended Thinking (ReflectionController)."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncGenerator, Dict, Any

from vel import Agent
from vel.providers import BaseProvider
from vel.thinking import ThinkingConfig, ReflectionController, route_thinking
from vel.thinking.controller import ThinkingPhase, ThinkingState
from vel.events import (
    ReasoningStartEvent, ReasoningDeltaEvent, ReasoningEndEvent,
    TextStartEvent, TextDeltaEvent, TextEndEvent,
    ToolInputAvailableEvent, ToolOutputAvailableEvent,
    FinishMessageEvent,
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
        assert config.routing == 'always'
        assert config.router_model is None
        assert config.router_confidence_threshold == 0.8
        assert config.effort == 'high'
        assert config.emit_summaries_only is True

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

    def test_validation_router_confidence_threshold(self):
        """Test router_confidence_threshold validation bounds."""
        config = ThinkingConfig(router_confidence_threshold=-0.5)
        assert config.router_confidence_threshold == 0

        config = ThinkingConfig(router_confidence_threshold=1.5)
        assert config.router_confidence_threshold == 1

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
        assert d['routing'] == 'always'
        assert d['effort'] == 'high'

    def test_routing_and_effort_validation(self):
        """Invalid routing/effort values fall back to safe defaults."""
        config = ThinkingConfig(mode='reflection', routing='sometimes', effort='huge')  # type: ignore[arg-type]
        assert config.routing == 'always'
        assert config.effort == 'high'


class FakeProvider(BaseProvider):
    """Deterministic test provider."""

    name = 'fake'

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def stream(self, messages, model, tools, generation_config=None):
        self.calls.append({
            'messages': messages,
            'model': model,
            'tools': tools,
            'generation_config': generation_config,
        })
        content = self.responses.pop(0) if self.responses else ''
        yield TextStartEvent(block_id='test')
        if content:
            yield TextDeltaEvent(block_id='test', delta=content)
        yield TextEndEvent(block_id='test')
        yield FinishMessageEvent(finish_reason='stop')

    async def generate(self, messages, model, tools, generation_config=None):
        return {'done': True, 'answer': self.responses.pop(0) if self.responses else ''}


class TestThinkingRouter:
    """Test automatic thinking routing."""

    @pytest.mark.asyncio
    async def test_route_thinking_reflection(self):
        provider = FakeProvider([
            '{"mode":"reflection","confidence":0.95,"category":"tradeoff_analysis","reason":"complex tradeoff"}'
        ])

        decision = await route_thinking(
            provider=provider,
            model='router-model',
            message='Compare three ontology designs and recommend one.',
            effort='extra',
        )

        assert decision.mode == 'reflection'
        assert decision.reason == 'complex tradeoff'
        assert decision.effort == 'extra'
        assert decision.confidence == 0.95
        assert decision.category == 'tradeoff_analysis'
        assert decision.raw_mode == 'reflection'

    @pytest.mark.asyncio
    async def test_route_thinking_low_confidence_reflection_downgrades_to_direct(self):
        provider = FakeProvider([
            '{"mode":"reflection","confidence":0.4,"category":"ambiguous","reason":"maybe complex"}'
        ])

        decision = await route_thinking(
            provider=provider,
            model='router-model',
            message='Could you look at this?',
            effort='high',
            confidence_threshold=0.8,
        )

        assert decision.mode == 'direct'
        assert decision.raw_mode == 'reflection'
        assert decision.confidence == 0.4
        assert decision.category == 'ambiguous'

    @pytest.mark.asyncio
    async def test_route_thinking_acknowledgement_stays_direct(self):
        provider = FakeProvider([
            '{"mode":"direct","confidence":0.98,"category":"acknowledgement","reason":"thanks only"}'
        ])

        decision = await route_thinking(
            provider=provider,
            model='router-model',
            message='thanks',
            effort='high',
        )

        assert decision.mode == 'direct'
        assert decision.confidence == 0.98
        assert decision.category == 'acknowledgement'

    @pytest.mark.asyncio
    async def test_route_thinking_effort_does_not_bias_mode_prompt(self):
        provider = FakeProvider([
            '{"mode":"direct","confidence":0.98,"category":"acknowledgement","reason":"thanks only"}'
        ])

        await route_thinking(
            provider=provider,
            model='router-model',
            message='thanks',
            effort='max',
        )

        system_prompt = provider.calls[0]['messages'][0]['content']
        assert 'must not make a simple request more likely to reflect' in system_prompt
        assert 'A thank-you after a complex answer is still direct' in system_prompt

    @pytest.mark.asyncio
    async def test_route_thinking_defaults_to_direct_on_bad_payload(self):
        provider = FakeProvider(['not json'])

        decision = await route_thinking(
            provider=provider,
            model='router-model',
            message='What is 2 + 2?',
            effort='medium',
        )

        assert decision.mode == 'direct'
        assert decision.effort == 'medium'


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
    def controller(self, config):
        """Create controller instance (agent-based engine)."""
        agent = Agent(id='t', model={'provider': 'fake', 'model': 'test-model'})
        return ReflectionController(agent=agent, config=config)

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

        # Conclude streams as text (final answer), not hidden reasoning
        assert controller._should_show_phase(ThinkingPhase.CONCLUDE) is True

    def test_should_show_phase_disabled(self):
        """Test phase visibility when disabled."""
        config = ThinkingConfig(
            mode='reflection',
            show_analysis=False,
            show_critiques=False,
            show_refinements=False
        )
        agent = Agent(id='t', model={'provider': 'fake', 'model': 'test-model'})
        controller = ReflectionController(agent=agent, config=config)

        assert controller._should_show_phase(ThinkingPhase.ANALYZE) is False
        assert controller._should_show_phase(ThinkingPhase.CRITIQUE) is False
        assert controller._should_show_phase(ThinkingPhase.REFINE) is False

    def test_stage_event(self, controller):
        """Test stage event generation."""
        controller.state = ThinkingState(question='Test')
        controller.state.iteration = 1
        controller.state.confidence = 0.65

        # Analyze stage
        event = controller._stage_event(ThinkingPhase.ANALYZE, step=1)
        assert event['type'] == 'data-thinking-stage'
        assert event['data']['stage'] == 'analyzing'
        assert event['data']['step'] == 1
        assert event['transient'] is True

        # Refine stage includes iteration and confidence
        event = controller._stage_event(ThinkingPhase.REFINE, step=3)
        assert event['data']['stage'] == 'refining'
        assert event['data']['iteration'] == 1
        assert event['data']['confidence'] == 0.65


class TestReflectionControllerIntegration:
    """Integration: reflection over the shared loop machinery (agent-based)."""

    def _agent(self, responses):
        agent = Agent(id='t', model={'provider': 'fake', 'model': 'test-model'})
        agent.providers.register(FakeProvider(responses))
        return agent

    @pytest.mark.asyncio
    async def test_full_flow_high_confidence(self):
        """Full reflection flow with high confidence (early exit)."""
        agent = self._agent([
            'This is my analysis of the problem...',      # Analyze
            'The analysis has one minor gap...',          # Critique
            'Addressing the gap...\n\nConfidence: 95%',   # Refine (high confidence)
            'The final answer is...',                     # Conclude
        ])
        config = ThinkingConfig(mode='reflection', max_refinements=3,
                                confidence_threshold=0.9, thinking_tools=False)
        controller = ReflectionController(agent=agent, config=config)

        events = [e async for e in controller.run('What is 2+2?')]
        types = [e.get('type') for e in events]

        # Per-phase reasoning blocks (analyze/critique/refine stream as reasoning).
        assert 'reasoning-start' in types
        assert any(e.get('type') == 'reasoning-delta' for e in events)
        assert 'reasoning-end' in types
        # Conclude streams as the final answer text.
        assert 'text-start' in types and 'text-delta' in types and 'text-end' in types
        assert 'data-thinking-complete' in types

        complete = next(e for e in events if e.get('type') == 'data-thinking-complete')
        assert complete['data']['final_confidence'] >= 0.9
        assert complete['data']['iterations'] == 1

    @pytest.mark.asyncio
    async def test_adaptive_refinement(self):
        """Adaptive refinement loop: low confidence triggers a second pass."""
        agent = self._agent([
            'Initial analysis...',
            'First critique...',
            'First refinement...\nConfidence: 50%',   # low -> re-critique
            'Second critique...',
            'Better refinement...\nConfidence: 85%',  # high enough -> stop
            'Final answer...',
        ])
        config = ThinkingConfig(mode='reflection', max_refinements=3,
                                confidence_threshold=0.8, thinking_tools=False)
        controller = ReflectionController(agent=agent, config=config)

        events = [e async for e in controller.run('Complex question')]
        complete = next(e for e in events if e.get('type') == 'data-thinking-complete')
        assert complete['data']['iterations'] >= 2

    @pytest.mark.asyncio
    async def test_convergence_stops_refinement(self):
        """No-progress convergence: identical refinements stop the loop early
        even when self-reported confidence stays low."""
        # Same refinement text repeated -> converges; low confidence would
        # otherwise run to max_refinements=5.
        same = 'The same refinement.\nConfidence: 10%'
        agent = self._agent(['analysis', 'critique', same, 'critique2', same,
                             same, same, same, same, same])
        config = ThinkingConfig(mode='reflection', max_refinements=5,
                                confidence_threshold=0.9, thinking_tools=False)
        controller = ReflectionController(agent=agent, config=config)
        events = [e async for e in controller.run('q')]
        complete = next(e for e in events if e.get('type') == 'data-thinking-complete')
        assert complete['data']['iterations'] < 5
        assert any(e.get('type') == 'data-thinking-verify' and e['data'].get('converged')
                   for e in events)

    @pytest.mark.asyncio
    async def test_custom_verifier_drives_termination(self):
        """A custom verify callable replaces self-reported confidence as the
        stop signal."""
        calls = {'n': 0}

        def verifier(question, reasoning):
            calls['n'] += 1
            return 0.95  # immediately 'good enough' -> stop after one refine

        agent = self._agent(['analysis', 'critique', 'refine (no confidence line)',
                             'critique2', 'more', 'answer'])
        config = ThinkingConfig(mode='reflection', max_refinements=5,
                                confidence_threshold=0.9, thinking_tools=False,
                                verify=verifier)
        controller = ReflectionController(agent=agent, config=config)
        events = [e async for e in controller.run('q')]
        complete = next(e for e in events if e.get('type') == 'data-thinking-complete')
        assert calls['n'] >= 1
        assert complete['data']['iterations'] == 1  # verifier said done immediately
        verify_ev = next(e for e in events if e.get('type') == 'data-thinking-verify')
        assert verify_ev['data']['method'] == 'callable'
        assert verify_ev['data']['confidence'] == 0.95

    @pytest.mark.asyncio
    async def test_composes_with_harness_budget(self):
        """Thinking + Harness compose: the harness token budget bounds the
        refine loop even when confidence stays low (no more mutual exclusion)."""
        from vel.harness import HarnessBudgetConfig

        class UsageProvider(BaseProvider):
            name = 'fake'

            def __init__(self, responses):
                self._r = list(responses)

            async def stream(self, messages, model, tools, generation_config=None):
                c = self._r.pop(0) if self._r else 'Confidence: 10%'
                yield TextStartEvent(block_id='b')
                yield TextDeltaEvent(block_id='b', delta=c)
                yield TextEndEvent(block_id='b')
                yield ResponseMetadataEvent(usage={'totalTokens': 100})
                yield FinishMessageEvent(finish_reason='stop')

            async def generate(self, messages, model, tools, generation_config=None):
                return {}

        agent = Agent(id='t', model={'provider': 'fake', 'model': 'test-model'})
        # Always low confidence -> would refine to max_refinements=5 without a budget.
        agent.providers.register(UsageProvider(['low\nConfidence: 10%'] * 20))
        config = ThinkingConfig(mode='reflection', max_refinements=5,
                                confidence_threshold=0.9, thinking_tools=False)
        # Tiny token budget: exhausts after a couple of phases -> refine stops early.
        controller = ReflectionController(
            agent=agent, config=config, budget=HarnessBudgetConfig(max_tokens=250)
        )
        events = [e async for e in controller.run('q')]
        complete = next(e for e in events if e.get('type') == 'data-thinking-complete')
        assert complete['data']['iterations'] < 5  # budget cut it short


@pytest.mark.asyncio
async def test_thinking_and_harness_compose_no_warning(caplog):
    """run_stream with both thinking and harness set no longer warns about
    non-composition and completes a reasoning answer."""
    provider = FakeProvider([
        'Analysis.', 'Critique.', 'Refined.\nConfidence: 95%', 'Final answer.',
    ])
    agent = Agent(id='both', model={'provider': 'fake', 'model': 'm'})
    agent.providers.register(provider)
    events = [e async for e in agent.run_stream(
        {'message': 'q'},
        thinking=ThinkingConfig(mode='reflection', confidence_threshold=0.9, thinking_tools=False),
        harness={'enabled': True, 'budget': {'max_wallclock_seconds': 60}},
    )]
    types = [e.get('type') for e in events]
    assert 'text-delta' in types and 'data-thinking-complete' in types
    assert 'not composed' not in caplog.text


class TestAgentExtendedThinking:
    """Agent-level thinking stream envelope tests."""

    @pytest.mark.asyncio
    async def test_reflection_stream_uses_step_envelope(self):
        provider = FakeProvider([
            'Analysis summary.',
            'Critique summary.',
            'Refined summary.\nConfidence: 95%',
            'Final answer.',
        ])
        agent = Agent(
            id='thinking-agent',
            model={'provider': 'fake', 'model': 'answer-model'},
        )
        agent.providers.register(provider)

        events = []
        async for event in agent.run_stream(
            {'message': 'Compare subtype and role modeling.'},
            thinking=ThinkingConfig(mode='reflection', thinking_tools=False),
        ):
            events.append(event)

        types = [event.get('type') for event in events]
        assert types[0] == 'start'
        assert types[1] == 'start-step'
        assert 'reasoning-start' in types
        assert 'reasoning-end' in types
        assert types.index('reasoning-start') < types.index('reasoning-end')
        text_start_index = types.index('text-start')
        reasoning_end_index = types.index('reasoning-end')
        assert reasoning_end_index < text_start_index
        assert 'finish-step' in types[reasoning_end_index:text_start_index]
        assert 'start-step' in types[reasoning_end_index:text_start_index]
        assert types[-1] == 'finish'
        assert any(event.get('type') == 'text-delta' and event.get('delta') == 'Final answer.' for event in events)

    @pytest.mark.asyncio
    async def test_auto_router_direct_uses_standard_stream(self):
        provider = FakeProvider([
            '{"mode":"direct","confidence":0.99,"category":"simple_factual","reason":"simple"}',
            'Direct answer.',
        ])
        agent = Agent(
            id='auto-agent',
            model={'provider': 'fake', 'model': 'answer-model'},
        )
        agent.providers.register(provider)

        events = []
        async for event in agent.run_stream(
            {'message': 'What is 2 + 2?'},
            thinking=ThinkingConfig(mode='reflection', routing='auto', thinking_tools=False),
        ):
            events.append(event)

        types = [event.get('type') for event in events]
        assert 'reasoning-start' not in types
        assert any(event.get('type') == 'text-delta' and event.get('delta') == 'Direct answer.' for event in events)


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
