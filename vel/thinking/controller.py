"""
ReflectionController - Multi-pass reasoning over the shared loop machinery.

Implements the Reflection pattern (Analyze -> Critique -> Refine (adaptive) ->
Conclude) as real turns in a scratch conversation context, driven by the agent's
shared ``_stream_llm_call`` (turn atom) and ``_run_tool_calls`` (real tool round).

This is the unified engine: reflection no longer reimplements streaming, tool
execution, or synthesis — it reuses the same primitives the base loop uses, so
tool calls during thinking get real approval/guardrails/observability, and every
phase is a nested ``THINKING`` span under one parent trace (no more disconnected
traces). Each phase streams its own ``reasoning-*`` block (Conclude streams
``text-*``), giving per-phase switchovers in reasoning-aware UIs for free.
"""

from __future__ import annotations
import asyncio
import difflib
import json
import re
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
)

from ..events import (
    DataEvent,
    TextStartEvent,
    TextDeltaEvent,
    TextEndEvent,
)

if TYPE_CHECKING:
    from .config import ThinkingConfig
    from ..agent import Agent


class ThinkingPhase(Enum):
    """Thinking phases."""
    ANALYZE = 'analyzing'
    CRITIQUE = 'critiquing'
    REFINE = 'refining'
    CONCLUDE = 'concluding'


# Conversational phase instructions. The scratch context accumulates the
# question + each phase's turn, so instructions reference "above" rather than
# re-injecting content (a real reasoning conversation, not one-shot prompts).
ANALYZE_INSTRUCTION = (
    "Analyze this problem step by step. Break down what is being asked, the key "
    "considerations, and your initial reasoning. Do not give a final answer yet."
)
CRITIQUE_INSTRUCTION = (
    "Critically review your reasoning above. Identify weaknesses, gaps, wrong "
    "assumptions, or missing considerations. Be specific."
)
REFINE_INSTRUCTION = (
    "Revise and improve your reasoning to address the critique. Be concrete. "
    "End your response with a line 'Confidence: X%' estimating how confident you "
    "are in the reasoning (0-100)."
)
CONCLUDE_INSTRUCTION = (
    "Now give the final answer to the original question, clearly and directly, "
    "based on your refined reasoning."
)


@dataclass
class ThinkingState:
    """Internal state for the reflection loop."""
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
    """Multi-pass reasoning driven over the agent's shared turn/tool primitives."""

    def __init__(
        self,
        agent: 'Agent',
        config: 'ThinkingConfig',
        budget: Optional[Any] = None,
        harness_controller: Optional[Any] = None,
    ):
        """Initialize.

        Args:
            agent: The owning Agent — provides ``_stream_llm_call``,
                ``_run_tool_calls``, the context manager, tool schemas, and the
                model config.
            config: ThinkingConfig with iteration/display/tool settings.
            budget: Optional ``HarnessBudgetConfig`` — when Extended Thinking
                composes with Harness Mode, the refine loop is bounded by its
                ``max_wallclock_seconds`` / ``max_tokens`` in addition to
                ``max_refinements``/``confidence_threshold``.
            harness_controller: Optional ``HarnessController`` — when present with
                durable approval, gated tools called during a thinking phase
                suspend the whole reflection run durably (checkpoint at the phase
                boundary) and resume on the human decision.
        """
        self.agent = agent
        self.config = config
        self.budget = budget
        self.harness_controller = harness_controller
        self.state: Optional[ThinkingState] = None
        self._thinking_model = (
            config.thinking_model.get('model') if config.thinking_model else None
        )
        # Durable-approval wiring (reflection suspend/resume).
        self._durable = bool(
            harness_controller is not None
            and getattr(harness_controller, '_durable_approvals', False)
        )
        self._gate = harness_controller._gate if self._durable else None
        self._checkpoints = harness_controller._checkpoints if self._durable else None
        self._config_hash = harness_controller.config_hash if self._durable else ''
        self._approved_tools: set = set()
        self._denied_tools: set = set()
        self._run_id: Optional[str] = None
        self._cursor: Optional[Dict[str, Any]] = None

    def _budget_exhausted(self) -> bool:
        """True if a composed harness token budget is spent (stops refining)."""
        if self.budget is None or self.state is None:
            return False
        max_tokens = getattr(self.budget, 'max_tokens', None)
        return bool(max_tokens and self.state.total_tokens >= max_tokens)

    def _effective_timeout(self) -> float:
        """Wallclock ceiling — the harness budget's, else the thinking timeout."""
        if self.budget is not None:
            wall = getattr(self.budget, 'max_wallclock_seconds', None)
            if wall:
                return float(wall)
        return self.config.thinking_timeout

    # ------------------------------------------------------- durable approval
    async def _gate_or_suspend(self, tool_calls: List[Dict[str, Any]], scratch_id: str):
        """Resolve a phase's tool calls against the durable approval gate.

        Returns a sync ``approval_resolver`` (tool_call_id -> bool) for
        :meth:`Agent._run_tool_calls` — approved/ungated tools run, denied tools
        get a denial result. If any gated tool has no decision yet (keyed by tool
        name so a resume re-run matches), suspends the whole reflection run."""
        from ..harness.exceptions import SuspendRun

        decisions: Dict[str, bool] = {}
        undecided = []
        for tc in tool_calls:
            tool = self.agent._get_tool(tc['tool_name'])
            name = tc['tool_name']
            if not self._gate.requires_approval(tool, name):
                decisions[tc['tool_call_id']] = True
            elif name in self._approved_tools:
                decisions[tc['tool_call_id']] = True
            elif name in self._denied_tools:
                decisions[tc['tool_call_id']] = False
            else:
                undecided.append(tc)
        if not undecided:
            return lambda tcid: decisions.get(tcid, True)
        requests = [
            self._gate.build_request(
                run_id=self._run_id,
                tool_call_id=tc['tool_call_id'],
                tool_name=tc['tool_name'],
                args=tc.get('input', {}),
                reason=f"Tool '{tc['tool_name']}' requires approval (reflection)",
            )
            for tc in undecided
        ]
        await self._gate.open(requests)
        # Suspend from the phase-boundary checkpoint (saved before this phase
        # mutated state/scratch) so resume re-runs the phase cleanly — no double
        # state mutation (e.g. iteration increments once).
        ckpt = self._checkpoints.load(self._run_id)
        if ckpt is None:
            ckpt = self._save_reflection_checkpoint(
                scratch_id, self._cursor, status='suspended', pending_approvals=requests
            )
        else:
            ckpt.status = 'suspended'
            ckpt.pending_approvals = requests
            self._checkpoints.save(ckpt)
        raise SuspendRun(ckpt, requests)

    def _serialize_reflection(self, scratch_id: str, cursor: Dict[str, Any]) -> Dict[str, Any]:
        s = self.state
        return {
            'scratch_id': scratch_id,
            'cursor': {
                'phase': cursor.get('phase'),
                'step': cursor.get('step', 0),
                'prev_refined': cursor.get('prev_refined'),
            },
            'approved_tools': sorted(self._approved_tools),
            'denied_tools': sorted(self._denied_tools),
            'state': {
                'question': s.question, 'analysis': s.analysis, 'critiques': s.critiques,
                'refined': s.refined, 'confidence': s.confidence,
                'iteration': s.iteration, 'total_tokens': s.total_tokens,
            },
        }

    def _save_reflection_checkpoint(self, scratch_id, cursor, *, status, pending_approvals=None):
        from ..harness.checkpoint import RunCheckpoint

        messages = list(self.agent.ctxmgr._by_run.get(scratch_id, []))
        ckpt = RunCheckpoint(
            run_id=self._run_id,
            agent_id=self.agent.id,
            session_id=None,
            status=status,
            step=cursor.get('step', 0),
            messages=messages,
            pending_approvals=pending_approvals or [],
            reflection=self._serialize_reflection(scratch_id, cursor),
            config_hash=self._config_hash,
        )
        self._checkpoints.save(ckpt)
        return ckpt

    def restore(self, ckpt: Any) -> Dict[str, Any]:
        """Rehydrate reflection state + scratch context from a checkpoint; return
        the cursor to continue ``_drive_phases`` from."""
        refl = ckpt.reflection or {}
        st = refl.get('state', {})
        self.state = ThinkingState(question=st.get('question', ''))
        self.state.analysis = st.get('analysis', '')
        self.state.critiques = st.get('critiques', '')
        self.state.refined = st.get('refined', '')
        self.state.confidence = st.get('confidence', 0.0)
        self.state.iteration = st.get('iteration', 0)
        self.state.total_tokens = st.get('total_tokens', 0)
        self._approved_tools = set(refl.get('approved_tools', []))
        self._denied_tools = set(refl.get('denied_tools', []))
        scratch_id = refl.get('scratch_id')
        self.agent.ctxmgr._by_run[scratch_id] = list(ckpt.messages)
        cursor = dict(refl.get('cursor', {'phase': 'analyze', 'step': 0, 'prev_refined': None}))
        self._cursor = cursor
        return cursor

    # ------------------------------------------------------------------ run
    async def run(
        self,
        question: str,
        context: Optional[List[Dict[str, Any]]] = None,
        *,
        trace_ctx: Optional[Any] = None,
        observer: Optional[Any] = None,
        session_id: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute reflection and stream events.

        Phases run in a scratch context so ``_run_tool_calls`` works verbatim;
        the scratch context is discarded at the end (the caller persists the
        reasoning + final answer into the real run). When durable, a gated tool
        during a phase raises ``SuspendRun`` (checkpoint already persisted).
        """
        from ..harness.exceptions import SuspendRun
        self.state = ThinkingState(question=question, context=context or [])
        ctx = self.agent.ctxmgr
        self._run_id = run_id or parent_run_id or uuid.uuid4().hex
        scratch_id = f"{parent_run_id or self._run_id}::think"
        if context:
            ctx.set_input(scratch_id, {'messages': list(context), 'message': question})
        else:
            ctx.set_input(scratch_id, {'message': question})

        cursor = {'phase': 'analyze', 'step': 0, 'prev_refined': None}
        self._cursor = cursor
        try:
            async with asyncio.timeout(self._effective_timeout()):
                async for ev in self._drive_phases(
                    cursor, scratch_id,
                    trace_ctx=trace_ctx, observer=observer, session_id=session_id,
                ):
                    yield ev
        except SuspendRun:  # gated tool mid-phase -> propagate to the caller
            ctx._by_run.pop(scratch_id, None)
            raise
        except asyncio.TimeoutError:
            async for ev in self._timeout_response():
                yield ev
        except Exception:  # graceful degradation
            text_id = str(uuid.uuid4())
            yield TextStartEvent(block_id=text_id).to_dict()
            fallback = self.state.refined or self.state.analysis or "Unable to process the question."
            yield TextDeltaEvent(
                block_id=text_id,
                delta=f"I encountered an error during reasoning.\n\n{fallback}",
            ).to_dict()
            yield TextEndEvent(block_id=text_id).to_dict()
        finally:
            # Discard the scratch reasoning context.
            ctx._by_run.pop(scratch_id, None)

        yield DataEvent(
            type='data-thinking-complete',
            data={
                'steps': cursor['step'],
                'iterations': self.state.iteration,
                'final_confidence': self.state.confidence,
                'thinking_tokens': self.state.total_tokens,
                'thinking_model': self._thinking_model or self.agent.model_cfg.get('model'),
            },
            transient=False,
        ).to_dict()

    # ------------------------------------------------------- phase machine
    async def _drive_phases(
        self,
        cursor: Dict[str, Any],
        scratch_id: str,
        *,
        trace_ctx: Optional[Any],
        observer: Optional[Any],
        session_id: Optional[str],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run the reflection phase state machine from ``cursor['phase']`` to
        done: analyze -> critique -> refine <-> critique -> conclude. Expressed
        as a resumable machine (``cursor`` = {phase, step, prev_refined}, mutated
        in place) so a suspended run can continue from the cursor. Behavior is
        identical to the previous straight-line loop for the non-suspend case."""
        self._cursor = cursor
        while cursor['phase'] != 'done':
            # Persist a running reflection checkpoint at each phase boundary
            # (before the phase mutates the scratch context) so a suspend during
            # the phase resumes by re-running it cleanly.
            if self._durable:
                self._save_reflection_checkpoint(scratch_id, cursor, status='running')
            phase = cursor['phase']
            cursor['step'] += 1
            step = cursor['step']
            h: Dict[str, Any] = {}

            if phase == 'analyze':
                yield self._stage_event(ThinkingPhase.ANALYZE, step)
                async for ev in self._run_phase(
                    ThinkingPhase.ANALYZE, scratch_id, ANALYZE_INSTRUCTION,
                    emit_as='reasoning', holder=h,
                    trace_ctx=trace_ctx, observer=observer, session_id=session_id,
                ):
                    yield ev
                self.state.analysis = h.get('text', '')
                cursor['phase'] = 'critique'

            elif phase == 'critique':
                yield self._stage_event(ThinkingPhase.CRITIQUE, step)
                async for ev in self._run_phase(
                    ThinkingPhase.CRITIQUE, scratch_id, CRITIQUE_INSTRUCTION,
                    emit_as='reasoning', holder=h,
                    trace_ctx=trace_ctx, observer=observer, session_id=session_id,
                ):
                    yield ev
                self.state.critiques = h.get('text', '')
                cursor['phase'] = (
                    'refine'
                    if self.state.iteration < self.config.max_refinements and not self._budget_exhausted()
                    else 'conclude'
                )

            elif phase == 'refine':
                self.state.iteration += 1
                yield self._stage_event(ThinkingPhase.REFINE, step)
                async for ev in self._run_phase(
                    ThinkingPhase.REFINE, scratch_id, REFINE_INSTRUCTION,
                    emit_as='reasoning', holder=h,
                    trace_ctx=trace_ctx, observer=observer, session_id=session_id,
                ):
                    yield ev
                refine_text = h.get('text', '')
                new_refined = self._extract_refinement(refine_text)
                converged = (
                    cursor['prev_refined'] is not None
                    and difflib.SequenceMatcher(None, cursor['prev_refined'], new_refined).ratio()
                    >= self.config.convergence_threshold
                )
                cursor['prev_refined'] = new_refined
                self.state.refined = new_refined
                score = await self._verify(refine_text, trace_ctx, observer)
                self.state.confidence = score
                yield self._verify_event(score, converged)
                if (converged or score >= self.config.confidence_threshold
                        or self.state.iteration >= self.config.max_refinements
                        or self._budget_exhausted()):
                    cursor['phase'] = 'conclude'
                else:
                    cursor['phase'] = 'critique'

            elif phase == 'conclude':
                yield self._stage_event(ThinkingPhase.CONCLUDE, step)
                async for ev in self._run_phase(
                    ThinkingPhase.CONCLUDE, scratch_id, CONCLUDE_INSTRUCTION,
                    emit_as='text', holder=h,
                    trace_ctx=trace_ctx, observer=observer, session_id=session_id,
                ):
                    yield ev
                cursor['phase'] = 'done'
            else:
                cursor['phase'] = 'done'

    # -------------------------------------------------------------- one phase
    async def _run_phase(
        self,
        phase: ThinkingPhase,
        scratch_id: str,
        instruction: str,
        *,
        emit_as: str,
        holder: Dict[str, Any],
        trace_ctx: Optional[Any],
        observer: Optional[Any],
        session_id: Optional[str],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run one phase as a turn (or bounded tool sub-loop) in the scratch
        context. Streams events; the phase's final text lands in ``holder['text']``."""
        from ..integrations.base import SpanKind

        # The scratch reasoning context is always isolated (run-based). Never
        # thread the real session_id into it, or the phase scaffolding would
        # pollute the user's conversation history.
        session_id = None
        ctx = self.agent.ctxmgr
        ctx.append(scratch_id, {'role': 'user', 'content': instruction}, session_id)

        tools = None
        if self.config.thinking_tools and phase != ThinkingPhase.CONCLUDE:
            tools = self.agent._get_tool_schemas()

        show = self._should_show_phase(phase)
        text = ''
        rounds = 0
        while True:
            phase_span = None
            if trace_ctx and observer:
                phase_span = observer.start_span(
                    trace_ctx, phase.value, SpanKind.THINKING, input={'phase': phase.value}
                )

            messages = ctx.messages_for_llm(scratch_id, session_id)
            out: Dict[str, Any] = {}
            async for ev in self.agent._stream_llm_call(
                messages,
                out=out,
                tools=tools,
                emit_as=emit_as,
                model=self._thinking_model,
                trace_ctx=trace_ctx,
                current_step_ctx=phase_span,
                observer=observer,
            ):
                # Hidden phases still stream tool events, just not reasoning text.
                if not show and str(ev.get('type', '')).startswith('reasoning'):
                    continue
                yield ev

            if phase_span and observer:
                observer.end_span(phase_span, output=(out.get('text') or '')[:500])

            text = out.get('text', '')
            usage = out.get('usage') or {}
            self.state.total_tokens += usage.get('totalTokens') or usage.get('total_tokens') or 0
            tool_calls = out.get('tool_calls') or []

            if tool_calls and tools and rounds < self.config.max_tool_rounds_per_phase:
                rounds += 1
                # Durable approval: a gated tool the human hasn't approved for
                # this reflection run suspends the whole run (checkpoint already
                # at the phase boundary; resume re-runs the phase with the
                # decision known, keyed by tool name). Returns a resolver so
                # denied tools get a denial result instead of running.
                approval_resolver = None
                if self._durable:
                    approval_resolver = await self._gate_or_suspend(tool_calls, scratch_id)
                # Append the assistant tool_calls message before the real tool
                # round (mirrors _step_loop) so _run_tool_calls appends results
                # against a well-formed context.
                ctx.append(scratch_id, {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': [
                        {
                            'id': tc['tool_call_id'],
                            'type': 'function',
                            'function': {
                                'name': tc['tool_name'],
                                'arguments': json.dumps(tc['input']) if isinstance(tc['input'], dict) else str(tc['input']),
                            },
                        }
                        for tc in tool_calls
                    ],
                }, session_id)
                async for ev in self.agent._run_tool_calls(
                    tool_calls,
                    scratch_id,
                    session_id,
                    steps=rounds,
                    trace_ctx=trace_ctx,
                    current_step_ctx=phase_span,
                    observer=observer,
                    approval_resolver=approval_resolver,
                ):
                    yield ev
                continue  # re-call the LLM with tool results now in context

            # No (more) tools — commit the phase's assistant turn and finish.
            if text:
                ctx.append_assistant_message(scratch_id, text, session_id)
            break

        holder['text'] = text

    # -------------------------------------------------------------- timeout
    async def _timeout_response(self) -> AsyncGenerator[Dict[str, Any], None]:
        text_id = str(uuid.uuid4())
        yield TextStartEvent(block_id=text_id).to_dict()
        best = self.state.refined or self.state.analysis
        if best:
            response = f"Based on my analysis so far:\n\n{best[:2000]}"
        else:
            response = "I wasn't able to complete my analysis in time. Could you rephrase or simplify?"
        yield TextDeltaEvent(block_id=text_id, delta=response).to_dict()
        yield TextEndEvent(block_id=text_id).to_dict()

    # -------------------------------------------------------------- verify
    async def _verify(self, refine_text: str, trace_ctx: Any, observer: Any) -> float:
        """Produce a termination score in [0,1] and push it to observability.

        'none' -> the model's self-reported confidence; a callable ->
        ``(question, reasoning) -> float``; 'judge' -> external LLM-as-judge.
        """
        v = self.config.verify
        if callable(v):
            result = v(self.state.question, self.state.refined)
            score = await result if asyncio.iscoroutine(result) else result
        elif v == 'judge':
            score = await self._verify_judge()
        else:  # 'none'
            score = self._extract_confidence(refine_text)
        score = float(score)

        if observer is not None and trace_ctx is not None and hasattr(observer, 'set_score'):
            try:
                observer.set_score(trace_ctx, 'reflection_confidence', score)
            except Exception:
                pass
        return score

    async def _verify_judge(self) -> float:
        """External verification via vel.memory.judge.LLMJudge (CRITIC-style)."""
        from ..memory.judge import LLMJudge, JudgeConfig, JudgeOutcome

        cfg_kwargs = self.config.verify_model if isinstance(self.config.verify_model, dict) else {}
        judge = LLMJudge(JudgeConfig(**cfg_kwargs) if cfg_kwargs else JudgeConfig(),
                         llm_fn=self._judge_llm_fn())
        trajectory = {
            'run_id': 'reflection',
            'input_message': self.state.question,
            'final_answer': self.state.refined,
        }
        result = await judge.evaluate(trajectory)
        # Success keeps the judge's confidence; failure caps it low so the loop
        # keeps refining.
        if result.outcome == JudgeOutcome.SUCCESS:
            return result.confidence
        return min(result.confidence, 0.4)

    def _judge_llm_fn(self):
        """An llm_fn for LLMJudge that reuses the agent's provider."""
        agent = self.agent

        async def fn(messages, model, **kwargs):
            parts = []
            async for ev in agent._get_provider().stream(
                messages, model=model, tools=None, generation_config={}
            ):
                if getattr(ev, 'type', None) == 'text-delta':
                    parts.append(ev.delta)
            return ''.join(parts)

        return fn

    def _verify_event(self, score: float, converged: bool) -> Dict[str, Any]:
        method = 'judge' if self.config.verify == 'judge' else (
            'callable' if callable(self.config.verify) else 'self'
        )
        return DataEvent(
            type='data-thinking-verify',
            data={'confidence': score, 'converged': converged, 'method': method},
            transient=True,
        ).to_dict()

    # -------------------------------------------------------------- helpers
    def _extract_confidence(self, response: str) -> float:
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
                return value / 100 if value > 1 else value
        return 0.6

    def _extract_refinement(self, response: str) -> str:
        cleaned = re.sub(
            r'\n*[Cc]onfidence[:\s]+\d+\.?\d*\s*%?\s*$',
            '',
            response,
            flags=re.MULTILINE,
        )
        return cleaned.strip()

    def _should_show_phase(self, phase: ThinkingPhase) -> bool:
        if not self.config.stream_thinking:
            return False
        if phase == ThinkingPhase.ANALYZE:
            return self.config.show_analysis
        if phase == ThinkingPhase.CRITIQUE:
            return self.config.show_critiques
        if phase == ThinkingPhase.REFINE:
            return self.config.show_refinements
        if phase == ThinkingPhase.CONCLUDE:
            return True  # conclude streams as text (final answer)
        return True

    def _stage_event(self, phase: ThinkingPhase, step: int) -> Dict[str, Any]:
        data = {'stage': phase.value, 'step': step}
        if phase == ThinkingPhase.REFINE:
            data['iteration'] = self.state.iteration
            data['confidence'] = self.state.confidence
        return DataEvent(type='data-thinking-stage', data=data, transient=True).to_dict()
