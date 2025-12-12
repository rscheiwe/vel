"""
ReflectionController - Multi-pass reasoning orchestrator.

Implements the Reflection pattern: Analyze -> Critique -> Refine (adaptive) -> Conclude
All phases stream events compatible with Vercel AI SDK V5.
"""

from __future__ import annotations
import asyncio
import json
import re
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

from ..events import (
    DataEvent,
    ReasoningStartEvent,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    TextStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    ToolInputAvailableEvent,
    ToolOutputAvailableEvent,
)

from .prompts import (
    ANALYZE_PROMPT,
    CRITIQUE_PROMPT,
    REFINE_PROMPT,
    CONCLUDE_PROMPT,
)

if TYPE_CHECKING:
    from .config import ThinkingConfig
    from ..providers.base import BaseProvider


class ThinkingPhase(Enum):
    """Thinking phases."""
    ANALYZE = 'analyzing'
    CRITIQUE = 'critiquing'
    REFINE = 'refining'
    CONCLUDE = 'concluding'


@dataclass
class ThinkingState:
    """Internal state for reflection loop."""
    question: str
    context: List[Dict[str, Any]] = None
    analysis: str = ""
    critiques: str = ""
    refined: str = ""
    confidence: float = 0.0
    iteration: int = 0
    total_tokens: int = 0

    def __post_init__(self):
        if self.context is None:
            self.context = []


class ReflectionController:
    """
    Multi-pass reasoning through reflection.

    Flow: Analyze -> Critique -> Refine (adaptive loop) -> Conclude

    All phases stream reasoning-delta events as tokens arrive.
    Tool calls during thinking are supported and emit standard tool events.
    """

    def __init__(
        self,
        provider: 'BaseProvider',
        model: str,
        config: 'ThinkingConfig',
        tools: Optional[Dict[str, Any]] = None,
        tool_executor: Optional[Callable] = None,
    ):
        """
        Initialize ReflectionController.

        Args:
            provider: LLM provider instance
            model: Model name/identifier
            config: ThinkingConfig with iteration and display settings
            tools: Tool schemas (if thinking_tools enabled)
            tool_executor: Async function to execute tools: (name, args) -> result
        """
        self.provider = provider
        self.model = model
        self.config = config
        self.tools = tools if config.thinking_tools else None
        self.tool_executor = tool_executor

    async def run(
        self,
        question: str,
        context: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute reflection and stream events.

        Args:
            question: The user's question/input
            context: Optional conversation context

        Yields:
            Stream protocol events (dict form)
        """
        reasoning_id = str(uuid.uuid4())
        state = ThinkingState(question=question, context=context or [])
        step = 0

        try:
            async with asyncio.timeout(self.config.thinking_timeout):
                # --- Start Reasoning Block ---
                yield ReasoningStartEvent(block_id=reasoning_id).to_dict()

                # --- Phase 1: ANALYZE ---
                step += 1
                yield self._stage_event(ThinkingPhase.ANALYZE, step, state)

                async for event in self._execute_phase(
                    ThinkingPhase.ANALYZE, state, reasoning_id
                ):
                    yield event

                # --- Phase 2: CRITIQUE ---
                step += 1
                yield self._stage_event(ThinkingPhase.CRITIQUE, step, state)

                async for event in self._execute_phase(
                    ThinkingPhase.CRITIQUE, state, reasoning_id
                ):
                    yield event

                # --- Phase 3+: Adaptive REFINE Loop ---
                while (
                    state.confidence < self.config.confidence_threshold
                    and state.iteration < self.config.max_refinements
                ):
                    state.iteration += 1
                    step += 1

                    yield self._stage_event(ThinkingPhase.REFINE, step, state)

                    async for event in self._execute_phase(
                        ThinkingPhase.REFINE, state, reasoning_id
                    ):
                        yield event

                    # Re-critique if not confident enough
                    if (
                        state.confidence < self.config.confidence_threshold
                        and state.iteration < self.config.max_refinements
                    ):
                        step += 1
                        yield self._stage_event(ThinkingPhase.CRITIQUE, step, state)

                        async for event in self._execute_phase(
                            ThinkingPhase.CRITIQUE, state, reasoning_id
                        ):
                            yield event

                # --- Phase Final: CONCLUDE ---
                step += 1
                yield self._stage_event(ThinkingPhase.CONCLUDE, step, state)

                # End reasoning before final answer
                yield ReasoningEndEvent(block_id=reasoning_id).to_dict()

                # Stream final answer as text
                async for event in self._execute_conclude(state):
                    yield event

        except asyncio.TimeoutError:
            # Graceful degradation on timeout
            yield ReasoningDeltaEvent(
                block_id=reasoning_id,
                delta="\n\n[Thinking timeout - providing best effort answer]\n"
            ).to_dict()
            yield ReasoningEndEvent(block_id=reasoning_id).to_dict()

            # Provide timeout response
            async for event in self._timeout_response(state):
                yield event

        except Exception as e:
            # Emit error in reasoning and provide fallback
            yield ReasoningDeltaEvent(
                block_id=reasoning_id,
                delta=f"\n\n[Thinking error: {str(e)}]\n"
            ).to_dict()
            yield ReasoningEndEvent(block_id=reasoning_id).to_dict()

            # Provide error response
            text_id = str(uuid.uuid4())
            yield TextStartEvent(block_id=text_id).to_dict()
            yield TextDeltaEvent(
                block_id=text_id,
                delta=f"I encountered an error during reasoning. Let me provide a direct answer.\n\n{state.refined or state.analysis or 'Unable to process the question.'}"
            ).to_dict()
            yield TextEndEvent(block_id=text_id).to_dict()

        # --- Emit Completion Metadata ---
        yield DataEvent(
            type='data-thinking-complete',
            data={
                'steps': step,
                'iterations': state.iteration,
                'final_confidence': state.confidence,
                'thinking_tokens': state.total_tokens,
                'thinking_model': self.model
            },
            transient=False
        ).to_dict()

    async def _execute_phase(
        self,
        phase: ThinkingPhase,
        state: ThinkingState,
        reasoning_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute a single thinking phase with streaming + optional tools.

        Args:
            phase: The thinking phase to execute
            state: Current thinking state
            reasoning_id: Block ID for reasoning events

        Yields:
            Stream events (reasoning-delta, tool events)
        """
        # Build phase-specific messages
        messages = self._build_phase_messages(phase, state)

        # Determine if tools available (CONCLUDE never uses tools)
        phase_tools = self.tools if phase != ThinkingPhase.CONCLUDE else None

        # Execute with tool handling loop
        content_parts = []
        tool_round = 0

        while tool_round <= self.config.max_tool_rounds_per_phase:
            # Stream from provider
            tool_call = None
            tool_call_id = None
            tool_name = None
            tool_args = {}

            async for event in self.provider.stream(
                messages=messages,
                model=self.model,
                tools=phase_tools or {},
                generation_config={'temperature': 0.7}
            ):
                event_type = event.type if hasattr(event, 'type') else event.get('type')

                if event_type == 'text-delta':
                    # Convert text-delta to reasoning-delta
                    delta = event.delta if hasattr(event, 'delta') else event.get('delta', '')
                    content_parts.append(delta)

                    if self._should_show_phase(phase):
                        yield ReasoningDeltaEvent(
                            block_id=reasoning_id,
                            delta=delta
                        ).to_dict()

                elif event_type == 'tool-input-available':
                    # Tool call detected
                    tool_call = event
                    tool_call_id = event.tool_call_id if hasattr(event, 'tool_call_id') else event.get('toolCallId')
                    tool_name = event.tool_name if hasattr(event, 'tool_name') else event.get('toolName')
                    tool_args = event.input if hasattr(event, 'input') else event.get('input', {})

                elif event_type == 'response-metadata':
                    # Track token usage
                    usage = event.usage if hasattr(event, 'usage') else event.get('usage')
                    if usage:
                        state.total_tokens += usage.get('totalTokens', 0)

            # Handle tool call if detected
            if tool_call and self.tool_executor and tool_round < self.config.max_tool_rounds_per_phase:
                tool_round += 1

                # Emit tool input event
                yield ToolInputAvailableEvent(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    input=tool_args
                ).to_dict()

                # Execute tool
                try:
                    result = await self.tool_executor(tool_name, tool_args)
                except Exception as e:
                    result = {'error': str(e)}

                # Emit tool output event
                yield ToolOutputAvailableEvent(
                    tool_call_id=tool_call_id,
                    output=result
                ).to_dict()

                # Emit reasoning delta showing tool result
                if self._should_show_phase(phase):
                    tool_delta = f"\n[Tool: {tool_name}] {json.dumps(result)[:500]}\n"
                    content_parts.append(tool_delta)
                    yield ReasoningDeltaEvent(
                        block_id=reasoning_id,
                        delta=tool_delta
                    ).to_dict()

                # Add tool result to messages and continue
                messages.append({
                    'role': 'assistant',
                    'content': f'I will use the {tool_name} tool.'
                })
                messages.append({
                    'role': 'user',
                    'content': f'Tool {tool_name} returned: {json.dumps(result)}'
                })
                continue  # Continue to next tool round

            # No tool call or max rounds reached - exit loop
            break

        # Update state based on phase
        content = ''.join(content_parts)
        self._update_state(phase, state, content)

        # Add section separator in reasoning output
        if self._should_show_phase(phase):
            yield ReasoningDeltaEvent(
                block_id=reasoning_id,
                delta='\n\n'
            ).to_dict()

    async def _execute_conclude(
        self,
        state: ThinkingState
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute conclude phase - stream final answer as text.

        Args:
            state: Current thinking state

        Yields:
            text-start, text-delta, text-end events
        """
        # Build conclude prompt
        best_reasoning = state.refined if state.refined else state.analysis
        prompt = CONCLUDE_PROMPT.format(
            question=state.question,
            reasoning=best_reasoning,
            confidence=state.confidence
        )

        messages = [{'role': 'user', 'content': prompt}]

        # Stream final answer (no tools in conclude)
        text_id = str(uuid.uuid4())
        yield TextStartEvent(block_id=text_id).to_dict()

        async for event in self.provider.stream(
            messages=messages,
            model=self.model,
            tools={},
            generation_config={'temperature': 0.7}
        ):
            event_type = event.type if hasattr(event, 'type') else event.get('type')

            if event_type == 'text-delta':
                delta = event.delta if hasattr(event, 'delta') else event.get('delta', '')
                yield TextDeltaEvent(block_id=text_id, delta=delta).to_dict()

            elif event_type == 'response-metadata':
                usage = event.usage if hasattr(event, 'usage') else event.get('usage')
                if usage:
                    state.total_tokens += usage.get('totalTokens', 0)

        yield TextEndEvent(block_id=text_id).to_dict()

    async def _timeout_response(
        self,
        state: ThinkingState
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate response when thinking times out."""
        text_id = str(uuid.uuid4())
        yield TextStartEvent(block_id=text_id).to_dict()

        # Use best available reasoning
        best = state.refined or state.analysis
        if best:
            response = f"Based on my analysis so far:\n\n{best[:2000]}"
        else:
            response = "I wasn't able to complete my analysis in time. Could you please rephrase or simplify your question?"

        yield TextDeltaEvent(block_id=text_id, delta=response).to_dict()
        yield TextEndEvent(block_id=text_id).to_dict()

    def _build_phase_messages(
        self,
        phase: ThinkingPhase,
        state: ThinkingState
    ) -> List[Dict[str, Any]]:
        """Build messages for a specific thinking phase."""
        if phase == ThinkingPhase.ANALYZE:
            prompt = ANALYZE_PROMPT.format(question=state.question)

        elif phase == ThinkingPhase.CRITIQUE:
            # Critique the current best reasoning
            content_to_critique = state.refined if state.refined else state.analysis
            prompt = CRITIQUE_PROMPT.format(content=content_to_critique)

        elif phase == ThinkingPhase.REFINE:
            # Refine based on current reasoning and critiques
            content = state.refined if state.refined else state.analysis
            prompt = REFINE_PROMPT.format(
                question=state.question,
                content=content,
                critiques=state.critiques
            )

        elif phase == ThinkingPhase.CONCLUDE:
            best_reasoning = state.refined if state.refined else state.analysis
            prompt = CONCLUDE_PROMPT.format(
                question=state.question,
                reasoning=best_reasoning,
                confidence=state.confidence
            )
        else:
            prompt = state.question

        return [{'role': 'user', 'content': prompt}]

    def _update_state(
        self,
        phase: ThinkingPhase,
        state: ThinkingState,
        content: str
    ):
        """Update thinking state based on phase output."""
        if phase == ThinkingPhase.ANALYZE:
            state.analysis = content

        elif phase == ThinkingPhase.CRITIQUE:
            state.critiques = content

        elif phase == ThinkingPhase.REFINE:
            # Extract confidence and refinement
            state.confidence = self._extract_confidence(content)
            state.refined = self._extract_refinement(content)

    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from refine response."""
        patterns = [
            r'[Cc]onfidence[:\s]+(\d+)\s*%',
            r'[Cc]onfidence[:\s]+(\d+\.?\d*)',
            r'\[(\d+)%\]',
            r'\((\d+)%\s*confident\)',
            r'(\d+)%\s*confidence',
            r'confidence.*?(\d+)\s*%',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                # Normalize to 0-1 if percentage
                return value / 100 if value > 1 else value

        # Default to moderate confidence if not found
        return 0.6

    def _extract_refinement(self, response: str) -> str:
        """Extract refinement content (strip confidence line if present)."""
        # Remove confidence lines at the end
        cleaned = re.sub(
            r'\n*[Cc]onfidence[:\s]+\d+\.?\d*\s*%?\s*$',
            '',
            response,
            flags=re.MULTILINE
        )
        return cleaned.strip()

    def _should_show_phase(self, phase: ThinkingPhase) -> bool:
        """Check if phase content should be shown in reasoning."""
        if not self.config.stream_thinking:
            return False

        if phase == ThinkingPhase.ANALYZE:
            return self.config.show_analysis
        elif phase == ThinkingPhase.CRITIQUE:
            return self.config.show_critiques
        elif phase == ThinkingPhase.REFINE:
            return self.config.show_refinements
        elif phase == ThinkingPhase.CONCLUDE:
            return False  # Conclude goes to text, not reasoning

        return True

    def _stage_event(
        self,
        phase: ThinkingPhase,
        step: int,
        state: ThinkingState
    ) -> Dict[str, Any]:
        """Create a transient stage event for UI progress."""
        data = {
            'stage': phase.value,
            'step': step
        }

        if phase == ThinkingPhase.REFINE:
            data['iteration'] = state.iteration
            data['confidence'] = state.confidence

        return DataEvent(
            type='data-thinking-stage',
            data=data,
            transient=True
        ).to_dict()
