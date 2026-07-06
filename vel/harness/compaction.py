"""Automatic context compaction for Harness Mode (M2).

When the working message window approaches the model's context limit, the loop
shrinks it (summarize / reduce / memory-offload) so long-horizon runs survive.
Invoked from :meth:`HarnessController.pre_step_hook`; default-on only inside
Harness Mode.

Hard rule (spec §12 Q4): never compact across an unresolved tool-call/tool-
result boundary, and never drop the most recent user turn. Violating either
corrupts the provider message format (orphaned tool calls).
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional, Tuple

from .config import CompactionConfig
from .events import HarnessCompactionEvent
from .exceptions import CompactionError

if TYPE_CHECKING:
    from vel.agent import Agent

# Pluggable context-window table (tokens). Falls back to DEFAULT_CONTEXT_WINDOW.
CONTEXT_WINDOWS: Dict[str, int] = {
    'gpt-4o': 128_000,
    'gpt-4o-mini': 128_000,
    'gpt-4-turbo': 128_000,
    'o1': 200_000,
    'o3': 200_000,
    'claude-3-5-sonnet': 200_000,
    'claude-3-7-sonnet': 200_000,
    'claude-opus-4': 200_000,
    'claude-sonnet-4': 200_000,
    'gemini-1.5-pro': 1_000_000,
    'gemini-1.5-flash': 1_000_000,
    'gemini-2.0-flash': 1_000_000,
}
DEFAULT_CONTEXT_WINDOW = 128_000


def estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    """Heuristic token estimate (~4 chars/token over the serialized messages)."""
    total = 0
    for msg in messages:
        content = msg.get('content')
        if isinstance(content, str):
            total += len(content)
        elif content is not None:
            total += len(json.dumps(content, default=str))
        if msg.get('tool_calls'):
            total += len(json.dumps(msg['tool_calls'], default=str))
    return total // 4


def context_window_for(model_cfg: Dict[str, Any]) -> int:
    """Look up a model's context window, matching by prefix."""
    model = (model_cfg or {}).get('model', '') or ''
    if model in CONTEXT_WINDOWS:
        return CONTEXT_WINDOWS[model]
    for name, window in CONTEXT_WINDOWS.items():
        if model.startswith(name):
            return window
    return DEFAULT_CONTEXT_WINDOW


def _is_assistant_tool_call(msg: Dict[str, Any]) -> bool:
    return msg.get('role') == 'assistant' and bool(msg.get('tool_calls'))


def _is_tool_result(msg: Dict[str, Any]) -> bool:
    return msg.get('role') == 'tool'


class CompactionPolicy:
    """Decides when/how to compact a run's message window."""

    def __init__(self, config: CompactionConfig, agent: 'Agent') -> None:
        self.config = config
        self.agent = agent

    # ------------------------------------------------------------------ decide
    def should_compact(self, messages: List[Dict[str, Any]], model_cfg: Dict[str, Any]) -> bool:
        if not self.config.enabled:
            return False
        if len(messages) <= self.config.keep_last_messages + 1:
            return False
        window = context_window_for(model_cfg)
        return estimate_tokens(messages) > self.config.trigger_token_ratio * window

    # ------------------------------------------------------------------ split
    def _safe_split(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split into (head, middle, tail) without severing tool-call/result
        pairs or dropping the most recent user turn.

        - head: leading system messages (always preserved).
        - tail: at least ``keep_last_messages`` most recent, extended so it never
          starts mid tool-sequence.
        - middle: the older, fully-resolved turns eligible for compaction.

        The most recent user turn is preserved separately (pinned verbatim) in
        :meth:`compact`, so a long single-user-turn loop still compacts its
        tool/assistant churn without dropping the task prompt.
        """
        n = len(messages)
        # head = leading system messages
        head_end = 0
        while head_end < n and messages[head_end].get('role') == 'system':
            head_end += 1
        head = messages[:head_end]
        body = messages[head_end:]
        if not body:
            return head, [], []

        keep = max(1, self.config.keep_last_messages)
        tail_start = max(0, len(body) - keep)

        # Never start the tail on an orphaned tool result: pull its assistant
        # tool_calls message into the tail too.
        while tail_start > 0 and _is_tool_result(body[tail_start]):
            tail_start -= 1
        # If the last middle message is an assistant tool_calls, its results are
        # in the tail — move it into the tail so it is never summarized away.
        while tail_start > 0 and _is_assistant_tool_call(body[tail_start - 1]):
            tail_start -= 1

        middle = body[:tail_start]
        tail = body[tail_start:]
        return head, middle, tail

    # ------------------------------------------------------------------ compact
    async def compact(
        self,
        ctxmgr: Any,
        run_id: str,
        *,
        model_cfg: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Compact the run's window in place. Returns a HarnessCompactionEvent
        dict if compaction occurred, else None (nothing to do)."""
        messages = ctxmgr._by_run.get(run_id, [])
        if not messages:
            return None
        before_tokens = estimate_tokens(messages)
        head, middle, tail = self._safe_split(messages)
        if not middle:
            return None  # nothing safely compactable

        # Pin the most recent user turn that falls in the middle (the task
        # prompt in a long single-turn loop): keep it verbatim, don't summarize.
        pinned: List[Dict[str, Any]] = []
        last_user_idx = max(
            (i for i, m in enumerate(middle) if m.get('role') == 'user'), default=-1
        )
        if last_user_idx != -1:
            pinned = [middle[last_user_idx]]
            middle = middle[:last_user_idx] + middle[last_user_idx + 1:]
        if not middle:
            return None  # only the pinned user turn was compactable -> nothing to do

        strategy = self.config.strategy
        if strategy == 'summarize':
            summary_msgs = await self._summarize(middle, model_cfg)
        elif strategy == 'reduce':
            summary_msgs = self._reduce(middle)
        elif strategy == 'memory_offload':
            summary_msgs = self._memory_offload(ctxmgr, middle, run_id)
        else:  # pragma: no cover - guarded by Literal type
            raise CompactionError(f"unknown compaction strategy: {strategy}")

        new_messages = head + summary_msgs + pinned + tail
        ctxmgr._by_run[run_id] = new_messages
        after_tokens = estimate_tokens(new_messages)

        return HarnessCompactionEvent(
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            strategy=strategy,
            removed=len(messages) - len(new_messages),
        ).to_dict()

    # --------------------------------------------------------------- strategies
    def _reduce(self, middle: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collapse older turns into a compact textual digest (no LLM call)."""
        from vel.prompts import ContextCompactor

        # Drop tool-call/result scaffolding; summarize remaining text turns.
        digest_lines: List[str] = []
        for msg in middle:
            role = msg.get('role', '')
            content = msg.get('content')
            if isinstance(content, str) and content.strip():
                snippet = content.strip().replace('\n', ' ')
                if len(snippet) > 200:
                    snippet = snippet[:200] + '…'
                digest_lines.append(f"{role}: {snippet}")
            elif _is_assistant_tool_call(msg):
                names = ', '.join(
                    tc.get('function', {}).get('name', '?') for tc in msg.get('tool_calls', [])
                )
                digest_lines.append(f"assistant: [called tools: {names}]")
        # Reuse ContextCompactor's placeholder convention for the wrapper.
        _ = ContextCompactor  # documents the reuse intent; helpers are static
        summary = "[Earlier conversation compacted]\n" + "\n".join(digest_lines)
        return [{'role': 'system', 'content': summary}]

    async def _summarize(
        self, middle: List[Dict[str, Any]], model_cfg: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """LLM-summarize older turns into a single context-summary message."""
        transcript_lines: List[str] = []
        for msg in middle:
            role = msg.get('role', '')
            content = msg.get('content')
            if isinstance(content, str) and content.strip():
                transcript_lines.append(f"{role}: {content.strip()}")
            elif _is_assistant_tool_call(msg):
                names = ', '.join(
                    tc.get('function', {}).get('name', '?') for tc in msg.get('tool_calls', [])
                )
                transcript_lines.append(f"assistant: [called tools: {names}]")
        transcript = "\n".join(transcript_lines)
        if not transcript:
            return self._reduce(middle)

        summarizer_model = self.config.summarizer_model or self.agent.model_cfg
        prompt = [
            {
                'role': 'system',
                'content': (
                    'Summarize the following earlier conversation into a concise '
                    'context note that preserves decisions, facts, and open '
                    f'threads. Keep it under {self.config.summary_max_tokens} tokens.'
                ),
            },
            {'role': 'user', 'content': transcript},
        ]
        provider = self.agent._get_provider()
        text_parts: List[str] = []
        try:
            async for event in provider.stream(
                prompt, model=summarizer_model.get('model'), tools=[]
            ):
                if getattr(event, 'type', None) == 'text-delta':
                    text_parts.append(getattr(event, 'delta', '') or '')
        except Exception as exc:  # fall back to deterministic reduce on any error
            raise CompactionError(f"summarizer failed: {exc}") from exc

        summary = ''.join(text_parts).strip() or '[Earlier conversation summarized]'
        return [{'role': 'system', 'content': f"[Context summary]\n{summary}"}]

    def _memory_offload(
        self, ctxmgr: Any, middle: List[Dict[str, Any]], run_id: str
    ) -> List[Dict[str, Any]]:
        """Offload salient turns to memory, then drop them from the window.

        Routing reflects what each store is *for* (see investigation note below):

        * **FactStore** (raw store) gets the verbatim salient turns, namespaced
          ``compaction:{run_id}`` — this is the correct home for raw context.
        * **ReasoningBank** is a *strategy* store (signature + generalizable
          strategy_text + confidence + embeddings), NOT a raw-turn dump. If it is
          enabled we write a single low-confidence *distilled* note so RB stays
          coherent rather than polluted with conversation chunks. Curated
          strategy learning still belongs to the Phase-2 auto-learning pipeline.

        All writes go through the ``ctxmgr`` memory adapters and are no-ops when
        memory is not configured, so behavior is unchanged by default. If nothing
        could be offloaded, degrades to :meth:`_reduce`.
        """
        offloaded = 0
        for msg in middle:
            content = msg.get('content')
            if isinstance(content, str) and content.strip():
                try:
                    ctxmgr.fact_put(
                        f"compaction:{run_id}",
                        f"{msg.get('role', 'msg')}:{offloaded}",
                        content.strip(),
                    )
                    offloaded += 1
                except Exception:
                    pass  # memory disabled / backend error — never corrupt the run

        self._offload_to_reasoningbank(ctxmgr, middle, run_id)

        if offloaded == 0:
            # No fact store (or nothing offloadable) — still shrink the window.
            return self._reduce(middle)
        return [{
            'role': 'system',
            'content': (
                f'[{offloaded} earlier turns offloaded to memory '
                f'(namespace compaction:{run_id}); summary below]\n'
                + self._reduce(middle)[0]['content']
            ),
        }]

    def _offload_to_reasoningbank(
        self, ctxmgr: Any, middle: List[Dict[str, Any]], run_id: str
    ) -> None:
        """Optionally record ONE distilled strategy note in ReasoningBank.

        No-op unless a ReasoningBank store is configured on the context manager.
        RB is a strategy store, so we write a single low-confidence distilled
        item (not raw turns) referencing the FactStore offload as evidence.
        """
        adapters = getattr(ctxmgr, '_adapters', None)
        rb_store = adapters.get('rb_store') if adapters else None
        if rb_store is None:
            return
        digest = self._reduce(middle)[0]['content']
        try:
            rb_store.upsert_strategy(
                signature={'source': 'compaction', 'run': run_id},
                strategy_text=digest[:500],
                evidence_refs=[f'compaction:{run_id}'],
                confidence=0.4,
            )
        except Exception:
            pass  # RB optional / encoder missing — never break the run


__all__ = ['CompactionPolicy', 'estimate_tokens', 'context_window_for', 'CONTEXT_WINDOWS']
