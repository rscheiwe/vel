"""HarnessController — checkpointable loop wrapper around ``Agent._step_loop``.

Responsibilities:
  * M1: budget enforcement + ``data-harness-*`` lifecycle/progress events.
  * M3: durable HITL approval — a pre-pass over each step's tool calls that
    snapshots a checkpoint and raises ``SuspendRun`` when a tool awaits a human
    decision, plus :meth:`resume` to re-execute the suspended step and continue.

The controller deliberately does NOT re-implement ``run_stream``'s setup
(ctxmgr, guardrails, observability, scratchpad). ``run_stream`` keeps ownership
of that; the controller only interposes per-step logic via hooks. This keeps the
backwards-compatibility contract (§8) trivially satisfiable.
"""
from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional

from .approvals import (
    ApprovalContext,
    ApprovalDecision,
    ApprovalGate,
    InMemoryApprovalStore,
    SQLiteApprovalStore,
)
from .budget import HarnessBudget
from .checkpoint import (
    ApprovalMemoryStore,
    CheckpointStore,
    RunCheckpoint,
    SandboxSessionStore,
)
from .compaction import CompactionPolicy
from .config import HarnessConfig
from .events import (
    HarnessApprovalRequiredEvent,
    HarnessBudgetExhaustedEvent,
    HarnessRecoveredEvent,
    HarnessResumedEvent,
    HarnessRunFinishedEvent,
    HarnessRunStartedEvent,
    HarnessSandboxEvent,
    HarnessStepEvent,
    HarnessSuspendedEvent,
)
from .exceptions import BudgetExhausted, SandboxError, SuspendRun

if TYPE_CHECKING:
    from vel.agent import Agent


class HarnessController:
    """Interposes budget + durable-approval logic around the agent step loop."""

    def __init__(self, agent: 'Agent', config: HarnessConfig) -> None:
        self.agent = agent
        self.config = config
        self._budget: HarnessBudget = HarnessBudget.from_config(config.budget)
        self._base_steps = 0
        self._run_id: Optional[str] = None
        self._session_id: Optional[str] = None
        self._context: Optional[Dict[str, Any]] = None

        # Durable approval wiring. Durable mode persists to SQLite (survives a
        # process restart); inline/non-durable falls back to in-memory.
        self._durable_approvals = (
            config.approval.enabled and config.approval.mode == 'durable'
        )
        store = (
            SQLiteApprovalStore(config.db_path)
            if self._durable_approvals
            else InMemoryApprovalStore()
        )
        self._gate = ApprovalGate(config=config.approval, store=store)
        # Session-scoped approve-once memory (lazily opened; durable only).
        self._approval_memory: Optional[ApprovalMemoryStore] = None
        self._checkpoints = CheckpointStore(config.db_path)
        self._compaction = (
            CompactionPolicy(config.compaction, agent)
            if config.compaction.enabled
            else None
        )
        self._config_hash = self._compute_config_hash()

        # Synchronous decision cache read by approval_resolver (the resolver is
        # called inside the sync tool loop; async store lookups happen in the
        # pre-pass / resume which populate this map).
        self._decisions: Dict[str, bool] = {}
        # Set when a suspension occurs so on_suspend() can emit the right events.
        self._suspend_requests: List[Any] = []

        # Sandbox session state (created/connected at run start; §6.7).
        self._sandbox_session: Any = None
        self._sandbox_ref: Optional[str] = None
        self._sandbox_tool_names: List[str] = []
        # session_id -> sandbox_ref registry for per_session/persistent reuse.
        self._sandbox_sessions: Optional[SandboxSessionStore] = None

    # ------------------------------------------------------------------ setup
    def bind_run(
        self,
        *,
        run_id: str,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        budget_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._run_id = run_id
        self._session_id = session_id
        self._context = context
        self._budget = HarnessBudget.from_config(self.config.budget, restore=budget_state)
        # Steps already consumed before this (re)entry — keeps the budget's step
        # count cumulative across suspend/resume even though _step_loop's local
        # step counter restarts from 1 on each entry.
        self._base_steps = self._budget.steps
        # Resolve skills: inject their tools + prepend their instructions for this
        # run. (On resume the prepended message is harmlessly overwritten by the
        # rehydrated checkpoint, which already contains it; the tool injection —
        # not persisted — must be re-applied, so it lives here.)
        self._apply_skills(run_id)

    def _apply_skills(self, run_id: str) -> None:
        """Activate configured skills for the run (spec §6.8).

        v1: all configured skills are active for the whole run — union their
        tools into the agent's injected tools and prepend their instructions to
        the run's message window. Dynamic mid-run activation is future work.
        """
        if not self.config.skills:
            return
        from .skills import resolve_skills

        skills = resolve_skills(self.config.skills)
        instructions: List[str] = []
        for skill in skills:
            for tool in skill.tools:
                self.agent._injected_tools[tool.name] = tool
            if skill.instructions:
                instructions.append(skill.instructions)
        if instructions:
            self.agent.ctxmgr._by_run.setdefault(run_id, []).insert(
                0, {'role': 'system', 'content': '[Skills]\n' + '\n\n'.join(instructions)}
            )

    # ----------------------------------------------------------------- sandbox
    def _make_sandbox_provider(self) -> Any:
        name = self.config.sandbox.provider
        if name == 'local_subprocess':
            from .sandbox.providers import LocalSubprocessSandboxProvider
            return LocalSubprocessSandboxProvider()
        if name == 'e2b':
            from .sandbox.providers import E2BSandboxProvider
            return E2BSandboxProvider()
        raise SandboxError(
            f"sandbox provider {name!r} is not available "
            "(use 'e2b' or 'local_subprocess')"
        )

    async def ensure_sandbox(self, *, sandbox_ref: Optional[str] = None) -> List[Dict[str, Any]]:
        """Create (or connect to) a sandbox and inject its tools for the run.

        Returns ``data-harness-sandbox`` event dicts to emit (empty if sandbox
        is disabled). Idempotent within a run.
        """
        sb = self.config.sandbox
        if not sb.enabled or not sb.tools or self._sandbox_session is not None:
            return []
        provider = self._make_sandbox_provider()

        reusable = sb.lifecycle in ('per_session', 'persistent')
        # Resolve a ref to reconnect to: explicit (resume) wins, else a stored
        # per-session ref (so the SAME workspace — plan.md/findings — is reused
        # across runs in a session, even after a process restart with E2B).
        ref = sandbox_ref
        if ref is None and reusable and self._session_id:
            self._sandbox_sessions = self._sandbox_sessions or SandboxSessionStore(self.config.db_path)
            ref = self._sandbox_sessions.get(
                self._session_id,
                idle_ttl_seconds=sb.provider_options.get('idle_ttl_seconds'),
            )

        kind = 'created'
        if ref:
            try:
                self._sandbox_session = await provider.connect(ref)
                kind = 'connected'
            except Exception:
                # Stale/dead ref (e.g. expired remote sandbox) -> create fresh.
                self._sandbox_session = await provider.create(sb)
                if reusable and self._session_id and self._sandbox_sessions:
                    self._sandbox_sessions.delete(self._session_id)
        else:
            self._sandbox_session = await provider.create(sb)
        self._sandbox_ref = getattr(self._sandbox_session, 'id', None)

        # Remember the ref for reuse by later runs in this session.
        if reusable and self._session_id and self._sandbox_ref:
            self._sandbox_sessions = self._sandbox_sessions or SandboxSessionStore(self.config.db_path)
            self._sandbox_sessions.put(self._session_id, self._sandbox_ref, sb.provider)

        from .sandbox.tools import build_sandbox_tools

        specs = build_sandbox_tools(
            self._sandbox_session, sb.tools, timeout_seconds=sb.timeout_seconds
        )
        self._sandbox_tool_names = []
        for spec in specs:
            self.agent._injected_tools[spec.name] = spec
            self._sandbox_tool_names.append(spec.name)
        return [HarnessSandboxEvent(event=kind, sandbox_ref=self._sandbox_ref).to_dict()]

    async def close_sandbox(self, *, force: bool = False) -> List[Dict[str, Any]]:
        """Tear down the sandbox per lifecycle (per_run closes; per_session /
        persistent stay warm — reaping them is future work, spec §6.7)."""
        if self._sandbox_session is None:
            return []
        if not (force or self.config.sandbox.lifecycle == 'per_run'):
            return []
        ref = self._sandbox_ref
        try:
            await self._sandbox_session.close()
        except Exception:  # never let teardown failures break the run
            pass
        self._sandbox_session = None
        return [HarnessSandboxEvent(event='closed', sandbox_ref=ref).to_dict()]

    def _compute_config_hash(self) -> str:
        """Hash model + tool names + harness-relevant config for resume safety."""
        tool_names = sorted(getattr(self.agent, '_tool_names', []) or [])
        payload = {
            'model': self.agent.model_cfg.get('model'),
            'provider': self.agent.model_cfg.get('provider'),
            'tools': tool_names,
            'approval_mode': self.config.approval.mode,
            'durable': self.config.durable,
            'max_steps': self.config.budget.max_steps,
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    @property
    def config_hash(self) -> str:
        return self._config_hash

    @property
    def budget(self) -> HarnessBudget:
        return self._budget

    @property
    def effective_max_steps(self) -> Optional[int]:
        return self.config.budget.max_steps

    # ------------------------------------------------------------------ hooks
    async def budget_hook(self, step: int) -> None:
        """Pre-step budget gate. Raises ``BudgetExhausted`` when over limit."""
        self._budget.steps = self._base_steps + step
        self._budget.check()

    async def pre_step_hook(
        self, run_id: str, session_id: Optional[str], step: int
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run before each step: auto-compact if over the token threshold, persist
        a running checkpoint, then emit a progress event. Compaction never severs
        tool-call/result pairs (enforced in CompactionPolicy per spec §12 Q4)."""
        if self._compaction is not None:
            messages = self.agent.ctxmgr._by_run.get(run_id, [])
            if self._compaction.should_compact(messages, self.agent.model_cfg):
                event = await self._compaction.compact(
                    self.agent.ctxmgr, run_id, model_cfg=self.agent.model_cfg
                )
                if event is not None:
                    yield event
        # §6.3.3: persist a running checkpoint each step so in-flight state is
        # recoverable (not only at suspension). Cheap snapshot of the window.
        self._save_running_checkpoint(run_id, session_id, step)
        yield HarnessStepEvent(
            step=step, budget=self._budget.to_event_budget()
        ).to_dict()

    def _save_running_checkpoint(
        self, run_id: str, session_id: Optional[str], step: int
    ) -> None:
        if not self.config.durable:
            return
        ckpt = RunCheckpoint(
            run_id=run_id,
            agent_id=self.agent.id,
            session_id=session_id,
            status='running',
            step=step,
            messages=list(self.agent.ctxmgr._by_run.get(run_id, [])),
            pending_approvals=[],
            pending_tool_calls=[],
            budget_state=self._budget_state(),
            sandbox_ref=self._sandbox_ref,
            config_hash=self._config_hash,
        )
        self._checkpoints.save(ckpt)

    async def on_tool_completed(
        self,
        run_id: str,
        session_id: Optional[str],
        step: int,
        tool_calls: List[Dict[str, Any]],
        completed_ids: List[str],
    ) -> None:
        """Persist a mid-step running checkpoint after each committed tool result.

        Records the step's full tool_calls plus which have completed, so
        :meth:`recover` can re-run only the not-yet-completed ones. Only wired in
        when ``config.checkpoint_each_tool`` is set (crash recovery); otherwise
        the loop keeps its cheaper one-checkpoint-per-step behavior.
        """
        if not self.config.durable:
            return
        ckpt = RunCheckpoint(
            run_id=run_id,
            agent_id=self.agent.id,
            session_id=session_id,
            status='running',
            step=step,
            messages=list(self.agent.ctxmgr._by_run.get(run_id, [])),
            pending_approvals=[],
            pending_tool_calls=list(tool_calls),
            completed_tool_calls=list(completed_ids),
            budget_state=self._budget_state(),
            sandbox_ref=self._sandbox_ref,
            config_hash=self._config_hash,
        )
        self._checkpoints.save(ckpt)

    def mark_completed(self, run_id: str) -> None:
        """Mark a run's checkpoint terminal so it is no longer 'running'."""
        if not self.config.durable:
            return
        if self._checkpoints.load(run_id) is not None:
            self._checkpoints.set_status(run_id, 'completed')

    def _approval_memory_store(self) -> ApprovalMemoryStore:
        if self._approval_memory is None:
            self._approval_memory = ApprovalMemoryStore(self.config.db_path)
        return self._approval_memory

    def _remember_if_enabled(self, session_id: Optional[str], tool_name: str) -> None:
        """Record a session-level approval for approve-once-per-session."""
        if self.config.approval.remember_approvals and session_id:
            self._approval_memory_store().add(session_id, tool_name)

    def _load_approved_tools(self, session_id: Optional[str]) -> frozenset:
        if self.config.approval.remember_approvals and session_id:
            return frozenset(self._approval_memory_store().get(session_id))
        return frozenset()

    async def _resolve_decisions(
        self,
        tool_calls: List[Dict[str, Any]],
        run_id: str,
        session_id: Optional[str],
        step: int,
    ) -> List[Dict[str, Any]]:
        """Populate :attr:`_decisions` for a step's tool calls; return undecided.

        Runs the context-aware gate (:meth:`ApprovalGate.evaluate`) for each
        call: ``not-applicable`` needs no gate, ``approved``/``denied`` are
        cached (and approvals remembered), and ``user-approval`` consults the
        durable store — recorded → cached (+ remembered), else collected as
        undecided. Shared by the pre-pass (fresh step) and resume (re-execute a
        suspended step) so policy/memory outcomes are re-derived identically.
        """
        approved_tools = self._load_approved_tools(session_id)
        undecided: List[Dict[str, Any]] = []
        for tc in tool_calls:
            tool = self.agent._get_tool(tc['tool_name'])
            ctx = ApprovalContext(
                tool_name=tc['tool_name'],
                tool_input=tc.get('input', {}),
                tool_call_id=tc['tool_call_id'],
                run_id=run_id,
                session_id=session_id,
                step=step,
                approved_tools=approved_tools,
                requires_confirmation=bool(getattr(tool, 'requires_confirmation', False)),
            )
            status = self._gate.evaluate(ctx)
            if status == 'not-applicable':
                continue
            if status == 'approved':
                self._decisions[tc['tool_call_id']] = True
                self._remember_if_enabled(session_id, tc['tool_name'])
                continue
            if status == 'denied':
                self._decisions[tc['tool_call_id']] = False
                continue
            # status == 'user-approval' -> consult the durable store.
            decision = await self._gate.get_decision(tc['tool_call_id'])
            if decision is None:
                undecided.append(tc)
            else:
                approved = decision.decision == 'approve'
                self._decisions[tc['tool_call_id']] = approved
                if approved:
                    self._remember_if_enabled(session_id, tc['tool_name'])
        return undecided

    async def approval_prepass(
        self,
        tool_calls: List[Dict[str, Any]],
        run_id: str,
        session_id: Optional[str],
        step: int,
    ) -> None:
        """Resolve approvals for a step's tool calls before any execution.

        Each call is scored by the context-aware gate (policy predicate +
        session memory + static rules). Auto-approved/denied calls are cached
        for :meth:`approval_resolver`. If any require a human decision that has
        not been recorded, snapshot a checkpoint (assistant tool_calls message
        already appended by the caller, no tool results yet), persist the pending
        approvals, and raise :class:`SuspendRun`.
        """
        if not self._durable_approvals:
            return

        undecided = await self._resolve_decisions(tool_calls, run_id, session_id, step)

        if not undecided:
            return

        requests = [
            self._gate.build_request(
                run_id=run_id,
                tool_call_id=tc['tool_call_id'],
                tool_name=tc['tool_name'],
                args=tc['input'],
                reason=f"Tool '{tc['tool_name']}' requires approval",
            )
            for tc in undecided
        ]
        await self._gate.open(requests)

        ckpt = RunCheckpoint(
            run_id=run_id,
            agent_id=self.agent.id,
            session_id=session_id,
            status='suspended',
            step=step,
            messages=list(self.agent.ctxmgr._by_run.get(run_id, [])),
            pending_approvals=requests,
            pending_tool_calls=list(tool_calls),
            budget_state=self._budget_state(),
            sandbox_ref=self._sandbox_ref,
            config_hash=self._config_hash,
        )
        self._checkpoints.save(ckpt)
        self._suspend_requests = requests
        raise SuspendRun(ckpt, requests)

    def approval_resolver(self, tool_call_id: str) -> bool:
        """Sync approve/deny lookup for the tool loop (default approve)."""
        return self._decisions.get(tool_call_id, True)

    # ----------------------------------------------------------- observation
    def observe_event(self, event: Dict[str, Any]) -> None:
        if not isinstance(event, dict):
            return
        if event.get('type') == 'response-metadata':
            usage = event.get('usage')
            if usage:
                self._budget.bump({'usage': usage})

    def _budget_state(self) -> Dict[str, Any]:
        return {
            'started_at': self._budget.started_at,
            'steps': self._budget.steps,
            'prompt_tokens': self._budget.prompt_tokens,
            'completion_tokens': self._budget.completion_tokens,
            'cost_usd': self._budget.cost_usd,
        }

    # --------------------------------------------------------- lifecycle events
    def run_started_event(self) -> Dict[str, Any]:
        return HarnessRunStartedEvent(
            run_id=self._run_id or '',
            agent_id=self.agent.id,
            durable=self.config.durable,
        ).to_dict()

    def budget_exhausted_event(self, exc: Exception) -> Dict[str, Any]:
        reason = getattr(exc, 'reason', None) or str(exc) or 'budget exhausted'
        return HarnessBudgetExhaustedEvent(reason=reason).to_dict()

    def run_finished_event(self, status: str) -> Dict[str, Any]:
        return HarnessRunFinishedEvent(
            run_id=self._run_id or '',
            status=status,
            usage=self._budget.to_event_budget(),
        ).to_dict()

    def on_suspend(self, exc: SuspendRun) -> List[Dict[str, Any]]:
        """Events emitted when a run suspends (checkpoint already persisted)."""
        events: List[Dict[str, Any]] = []
        for req in exc.approval_requests:
            events.append(
                HarnessApprovalRequiredEvent(
                    approval_id=req.approval_id,
                    run_id=req.run_id,
                    tool_call_id=req.tool_call_id,
                    tool_name=req.tool_name,
                    reason=req.reason,
                ).to_dict()
            )
        events.append(
            HarnessSuspendedEvent(run_id=exc.checkpoint.run_id, reason='approval').to_dict()
        )
        return events

    # ------------------------------------------------------------------ resume
    def _assert_resumable(self, ckpt: RunCheckpoint, force: bool = False) -> None:
        if force:
            return
        if ckpt.config_hash and ckpt.config_hash != self._config_hash:
            raise ValueError(
                f"Refusing to resume run {ckpt.run_id}: config_hash changed "
                f"({ckpt.config_hash} != {self._config_hash}). Pass force=True to override."
            )

    async def resume(
        self,
        run_id: str,
        decisions: List[ApprovalDecision],
        *,
        context: Optional[Dict[str, Any]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        force: bool = False,
        observer: Optional[Any] = None,
        trace_ctx: Optional[Any] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Resume a suspended run: apply decisions, re-execute the suspended
        step's tools, then continue the loop. Re-suspends/exhausts as needed.

        ``observer``/``trace_ctx`` (set up by ``Agent.resume``) make the resumed
        leg traceable — without them, post-suspension work is invisible to
        observability."""
        ckpt = self._checkpoints.load(run_id)
        if ckpt is None or ckpt.status != 'suspended':
            raise ValueError(f"Run {run_id!r} is not suspended (cannot resume)")
        self._assert_resumable(ckpt, force=force)

        self.bind_run(
            run_id=run_id,
            session_id=ckpt.session_id,
            context=context,
            budget_state=ckpt.budget_state,
        )

        # Record the human decisions, then apply approval timeouts (on_timeout,
        # §6.4) so expired approvals are auto-decided in the store.
        for decision in decisions:
            await self._gate.record(decision)
        await self._gate.get_pending(run_id)  # auto-records expired per on_timeout

        # Re-derive every decision for the suspended step's tool calls through
        # the same gate the pre-pass used: policy / session-memory outcomes plus
        # the human/timeout decisions now recorded in the store. This also
        # remembers newly-approved tools for approve-once-per-session.
        await self._resolve_decisions(
            ckpt.pending_tool_calls, run_id, ckpt.session_id, ckpt.step
        )

        # Rehydrate the run's message window (assistant tool_calls msg included).
        self.agent.ctxmgr._by_run[run_id] = list(ckpt.messages)
        self._checkpoints.set_status(run_id, 'running')

        yield HarnessResumedEvent(run_id=run_id).to_dict()

        # Reconnect the sandbox (if any) before re-executing the suspended step.
        for ev in await self.ensure_sandbox(sandbox_ref=ckpt.sandbox_ref):
            yield ev

        loop_state: Dict[str, Any] = {'steps': ckpt.step, 'final_answer': '', 'control': 'continue'}

        _on_tool = self.on_tool_completed if self.config.checkpoint_each_tool else None

        # 1) Re-execute the suspended step's tools (no LLM call — decisions known).
        #    Skip any whose results are already in the rehydrated window (e.g. a
        #    crash between the approval resume and step completion).
        async for ev in self.agent._run_tool_calls(
            ckpt.pending_tool_calls,
            run_id,
            ckpt.session_id,
            steps=ckpt.step,
            context=context,
            trace_ctx=trace_ctx,
            observer=observer,
            loop_state=loop_state,
            approval_resolver=self.approval_resolver,
            on_tool_completed=_on_tool,
            skip_tool_call_ids=set(ckpt.completed_tool_calls),
        ):
            self.observe_event(ev)
            yield ev

        if loop_state.get('control') == 'terminate':
            for ev in await self.close_sandbox():
                yield ev
            self._checkpoints.set_status(run_id, 'completed')
            yield self.run_finished_event('completed')
            return

        # 2) Continue the loop for subsequent steps (fresh LLM calls).
        try:
            async for ev in self.agent._step_loop(
                run_id=run_id,
                session_id=ckpt.session_id,
                context=context,
                generation_config=generation_config,
                trace_ctx=trace_ctx,
                observer=observer,
                loop_state=loop_state,
                max_steps=self.effective_max_steps,
                pre_step_hook=self.pre_step_hook,
                approval_prepass=self.approval_prepass,
                approval_resolver=self.approval_resolver,
                budget_hook=self.budget_hook,
                on_tool_completed=_on_tool,
            ):
                self.observe_event(ev)
                yield ev
        except SuspendRun as s:
            for ev in self.on_suspend(s):
                yield ev
            return  # stays suspended; sandbox kept for the next resume
        except BudgetExhausted as b:
            yield self.budget_exhausted_event(b)
            async for ev in self.agent._synthesize_final(
                run_id, ckpt.session_id, context=context, reason=getattr(b, 'reason', str(b))
            ):
                yield ev

        for ev in await self.close_sandbox():
            yield ev
        self._checkpoints.set_status(run_id, 'completed')
        yield self.run_finished_event('completed')

    async def recover(
        self,
        run_id: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        force: bool = False,
        observer: Optional[Any] = None,
        trace_ctx: Optional[Any] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Recover a run that crashed while ``running`` (not suspended).

        Rehydrates the last running checkpoint and, if the crash was mid-step
        (``checkpoint_each_tool`` was on), re-executes only the not-yet-completed
        tools — the completed ones are skipped because their results are already
        in the rehydrated window. Then continues the loop. If only step-boundary
        checkpoints exist, recovery restarts from the last step boundary.
        """
        ckpt = self._checkpoints.load(run_id)
        if ckpt is None or ckpt.status != 'running':
            raise ValueError(
                f"Run {run_id!r} has no recoverable 'running' checkpoint "
                f"(status={getattr(ckpt, 'status', None)!r})"
            )
        self._assert_resumable(ckpt, force=force)

        self.bind_run(
            run_id=run_id,
            session_id=ckpt.session_id,
            context=context,
            budget_state=ckpt.budget_state,
        )

        completed = set(ckpt.completed_tool_calls)

        # Rehydrate the run's message window (already includes completed results).
        self.agent.ctxmgr._by_run[run_id] = list(ckpt.messages)
        self._checkpoints.set_status(run_id, 'running')

        yield HarnessRecoveredEvent(run_id=run_id, skipped_tools=len(completed)).to_dict()

        # Reconnect the sandbox (if any) before finishing the interrupted step.
        for ev in await self.ensure_sandbox(sandbox_ref=ckpt.sandbox_ref):
            yield ev

        loop_state: Dict[str, Any] = {'steps': ckpt.step, 'final_answer': '', 'control': 'continue'}
        _on_tool = self.on_tool_completed if self.config.checkpoint_each_tool else None

        # 1) Finish a mid-step crash: re-run the step's remaining tools only.
        pending = ckpt.pending_tool_calls
        if pending and len(completed) < len(pending):
            # Re-derive decisions for the step's gated tools (policy/memory/store).
            await self._resolve_decisions(pending, run_id, ckpt.session_id, ckpt.step)
            async for ev in self.agent._run_tool_calls(
                pending,
                run_id,
                ckpt.session_id,
                steps=ckpt.step,
                context=context,
                trace_ctx=trace_ctx,
                observer=observer,
                loop_state=loop_state,
                approval_resolver=self.approval_resolver,
                on_tool_completed=_on_tool,
                skip_tool_call_ids=completed,
            ):
                self.observe_event(ev)
                yield ev
            if loop_state.get('control') == 'terminate':
                for ev in await self.close_sandbox():
                    yield ev
                self._checkpoints.set_status(run_id, 'completed')
                yield self.run_finished_event('completed')
                return

        # 2) Continue the loop for subsequent steps (fresh LLM calls).
        try:
            async for ev in self.agent._step_loop(
                run_id=run_id,
                session_id=ckpt.session_id,
                context=context,
                generation_config=generation_config,
                trace_ctx=trace_ctx,
                observer=observer,
                loop_state=loop_state,
                max_steps=self.effective_max_steps,
                pre_step_hook=self.pre_step_hook,
                approval_prepass=self.approval_prepass,
                approval_resolver=self.approval_resolver,
                budget_hook=self.budget_hook,
                on_tool_completed=_on_tool,
            ):
                self.observe_event(ev)
                yield ev
        except SuspendRun as s:
            for ev in self.on_suspend(s):
                yield ev
            return  # a recovered run can itself suspend for approval
        except BudgetExhausted as b:
            yield self.budget_exhausted_event(b)
            async for ev in self.agent._synthesize_final(
                run_id, ckpt.session_id, context=context, reason=getattr(b, 'reason', str(b))
            ):
                yield ev

        for ev in await self.close_sandbox():
            yield ev
        self._checkpoints.set_status(run_id, 'completed')
        yield self.run_finished_event('completed')


__all__ = ['HarnessController']
