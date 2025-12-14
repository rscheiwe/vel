# context.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Callable, TypedDict
from dataclasses import dataclass
import os
import warnings

# --- Optional memory backends (soft imports; safe if modules are absent) ---
try:
    # Fact store: namespaced key-value store for long-term structured data (SQLite)
    from vel.memory.fact_store import FactStore  # type: ignore
except Exception:  # pragma: no cover
    FactStore = None  # type: ignore

try:
    # ReasoningBank (strategy memory) over SQLite + embeddings
    from vel.memory.strategy_reasoningbank import (  # type: ignore
        ReasoningBank,
        ReasoningBankStore,
        Embeddings,
        StrategyItem,
    )
except Exception:  # pragma: no cover
    ReasoningBank = None  # type: ignore
    ReasoningBankStore = None  # type: ignore
    Embeddings = None  # type: ignore
    StrategyItem = None  # type: ignore


# =========================
# Optional memory config
# =========================

@dataclass
class MemoryConfig:
    """
    Optional runtime-owned memory (OFF by default).

    Vel has three distinct memory systems:
    1. Message History - Conversation turns (managed by ContextManager)
    2. Fact Store - Long-term structured facts (this config)
    3. Session Persistence - Where message history is saved

    Args:
        mode: Memory mode to enable
            - "none": No memory (default)
            - "facts": Fact store only (namespaced KV)
            - "reasoning": ReasoningBank only (strategy memory)
            - "all": Both fact store and ReasoningBank

            Deprecated (still work with warnings):
            - "episodic" → use "facts"
            - "reasoningbank" → use "reasoning"
            - "both" → use "all"

        db_path: SQLite file path (e.g., ".vel/vel.db")
        rb_top_k: Top-k ReasoningBank strategies to retrieve per run
        embeddings_fn: Embedding function for ReasoningBank
            - Must return np.ndarray with dtype=float32
            - Required if mode includes "reasoning"
    """
    mode: str = "none"
    db_path: str = ".vel/vel.db"
    rb_top_k: int = 5
    embeddings_fn: Optional[Callable[[List[str]], "object"]] = None  # return: np.ndarray

    def __post_init__(self):
        """Handle backwards compatibility for mode names."""
        mode_mapping = {
            "episodic": "facts",
            "reasoningbank": "reasoning",
            "both": "all"
        }

        if self.mode in mode_mapping:
            old_mode = self.mode
            self.mode = mode_mapping[old_mode]
            warnings.warn(
                f"MemoryConfig mode='{old_mode}' is deprecated and will be removed in v2.0. "
                f"Use mode='{self.mode}' instead.",
                DeprecationWarning,
                stacklevel=3
            )


def load_memory_config_from_env(default: Optional[MemoryConfig] = None) -> MemoryConfig:
    """
    Convenience helper to read env vars into a MemoryConfig.

    Note: embeddings_fn must be set in code (cannot be set via env var).

    Environment Variables:
      VEL_MEMORY_MODE: "none" | "facts" | "reasoning" | "all"
                       (old names still work: "episodic" | "reasoningbank" | "both")
      VEL_MEMORY_DB: Path to SQLite DB
      VEL_RB_TOP_K: Top-k strategies to retrieve (integer)
    """
    base = default or MemoryConfig()
    mode = os.environ.get("VEL_MEMORY_MODE", base.mode)
    db_path = os.environ.get("VEL_MEMORY_DB", base.db_path)
    try:
        topk = int(os.environ.get("VEL_RB_TOP_K", str(base.rb_top_k)))
    except ValueError:
        topk = base.rb_top_k
    return MemoryConfig(mode=mode, db_path=db_path, rb_top_k=topk, embeddings_fn=base.embeddings_fn)


class MemoryAdapters(TypedDict, total=False):
    """
    Memory adapter instances.

    Keys:
        facts: FactStore instance (namespaced KV store)
        rb: ReasoningBank instance (strategy memory)
    """
    facts: Any
    rb: Any


def _expand_path(path: str) -> str:
    return os.path.expanduser(path or ".vel/vel.db")


def build_memory_adapters(cfg: MemoryConfig) -> MemoryAdapters:
    """
    Build memory adapters based on config.

    Returns dictionary with 'facts' and/or 'rb' keys (values may be None if backend unavailable).
    This function never raises if a backend is missing; it just returns None for that adapter.

    Args:
        cfg: MemoryConfig with mode, db_path, and embeddings_fn

    Returns:
        MemoryAdapters dict with 'facts' and/or 'rb' keys
    """
    mode = (cfg.mode or "none").strip().lower()
    db_path = _expand_path(cfg.db_path)

    adapters: MemoryAdapters = {}

    # Note: mode has already been normalized by MemoryConfig.__post_init__
    if mode in ("facts", "all"):
        adapters["facts"] = FactStore(db_path) if FactStore else None

    if mode in ("reasoning", "all"):
        if ReasoningBank and ReasoningBankStore and Embeddings and cfg.embeddings_fn:
            emb = Embeddings(cfg.embeddings_fn)
            store = ReasoningBankStore(db_path=db_path, emb=emb)
            adapters["rb"] = ReasoningBank(store)
        else:
            adapters["rb"] = None

    return adapters


# =========================
# Existing Context Managers
# =========================

class ContextManager:
    """
    Manages message history for multi-turn conversations.

    Vel has three distinct memory systems:
    1. **Message History** (this class) - Conversation turns (automatic)
    2. **Fact Store** (optional) - Long-term structured facts (manual)
    3. **Session Persistence** - Where message history is saved (infrastructure)

    Message History (this class):
        - Stores conversation turns (user/assistant messages)
        - Automatically managed by Agent runtime
        - Configure window size with max_history parameter
        - Supports both run-based and session-based context

    Optional Runtime-Owned Memory (via set_memory_config):
        - Fact Store: fact_put(), fact_get(), fact_list()
          Store user preferences, project metadata, domain knowledge
        - ReasoningBank: prepare_for_run(), finalize_outcome()
          Strategy-level memory for learning reasoning patterns

    Example:
        ```python
        # Configure message history
        ctx = ContextManager(max_history=20)  # Keep last 20 messages

        # Enable optional memory
        mem = MemoryConfig(mode="all", db_path=".vel/vel.db", embeddings_fn=encode_fn)
        ctx.set_memory_config(mem)

        # Store facts
        ctx.fact_put("user:alice", "theme", "dark")

        # Get strategy advice before run
        advice = ctx.prepare_for_run({"intent": "planning"})
        ```
    """
    def __init__(self, max_history: Optional[int] = None, summarize: bool = False):
        """
        Args:
            max_history: Maximum number of messages to retain (None = unlimited)
            summarize: Whether to summarize old messages (not yet implemented)
        """
        self._by_run: Dict[str, List[Dict[str,Any]]] = {}
        self._by_session: Dict[str, List[Dict[str,Any]]] = {}
        self._inputs: Dict[str, Dict[str, Any]] = {}
        self.max_history = max_history
        self.summarize = summarize

        # Optional memory fields (remain None unless enabled)
        self._memory_cfg: Optional[MemoryConfig] = None
        self._adapters: MemoryAdapters = {}
        self._rb_last_ids: List[int] = []   # strategy_ids used for the current run (if any)

    # ---------- existing behavior stays the same ----------

    def set_input(self, run_id: str, input: Dict[str, Any], session_id: Optional[str] = None):
        """
        Store the initial input for a run.

        Supports two modes:
        1. Stateless with messages array: input={'messages': [...]}
           Client provides full conversation history, no session management
        2. Session-based: input={'message': '...'} with session_id
           Vel manages conversation history across calls
        """
        self._inputs[run_id] = input

        # Stateless mode: client provides full messages array
        if 'messages' in input and isinstance(input['messages'], list):
            # Use provided messages array directly (copy for safety during tool calls)
            self._by_run[run_id] = list(input['messages'])
            # Don't link to session - client manages history
            return

        # Legacy session-based mode: Vel manages history
        message = input.get('message', '') or str(input)

        if session_id:
            # Session-based: append to existing session or create new
            if session_id not in self._by_session:
                self._by_session[session_id] = []
            self._by_session[session_id].append({'role': 'user', 'content': message})
            # Link run to session
            self._by_run[run_id] = self._by_session[session_id]
        else:
            # Run-based: each run is independent
            self._by_run[run_id] = [{'role': 'user', 'content': message}]

    def messages_for_llm(self, run_id: str, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get messages for LLM, respecting max_history"""
        if session_id and session_id in self._by_session:
            messages = self._by_session[session_id]
        else:
            messages = self._by_run.get(run_id, [{'role': 'user', 'content': 'Hello'}])

        # Apply max_history limit
        if self.max_history and len(messages) > self.max_history:
            return messages[-self.max_history:]

        return messages

    def append(self, run_id: str, item: Dict[str,Any], session_id: Optional[str] = None):
        """Append a message to the conversation"""
        if session_id and session_id in self._by_session:
            self._by_session[session_id].append(item)
        else:
            self._by_run.setdefault(run_id, []).append(item)

    def append_assistant_message(self, run_id: str, content: str, session_id: Optional[str] = None):
        """Append an assistant message"""
        self.append(run_id, {'role': 'assistant', 'content': content}, session_id)

    def append_assistant_with_reasoning(
        self,
        run_id: str,
        reasoning: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ):
        """
        Append an assistant message with both reasoning and answer parts.

        Used by Extended Thinking to store the full reasoning trace alongside
        the final answer. The multi-part content format is compatible with
        Vercel AI SDK message structure.

        Args:
            run_id: Run identifier
            reasoning: The accumulated reasoning text (all phases)
            answer: The final answer text
            metadata: Optional thinking metadata (steps, confidence, etc.)
            session_id: Optional session identifier

        Example:
            ```python
            ctx.append_assistant_with_reasoning(
                run_id='run-123',
                reasoning='[Analysis]\\n...\\n[Critique]\\n...\\n[Refinement]\\n...',
                answer='Based on my analysis, the answer is...',
                metadata={'steps': 5, 'iterations': 2, 'final_confidence': 0.9}
            )
            ```
        """
        message = {
            'role': 'assistant',
            'content': [
                {'type': 'reasoning', 'text': reasoning},
                {'type': 'text', 'text': answer}
            ]
        }

        if metadata:
            message['thinking_metadata'] = metadata

        self.append(run_id, message, session_id)

    def append_tool_result(self, run_id: str, tool_name: str, result: Any, session_id: Optional[str] = None, tool_call_id: Optional[str] = None):
        """Append a tool result as a message.

        Uses OpenAI's expected format with role='tool' and tool_call_id when provided.
        Falls back to legacy format for backwards compatibility.
        """
        if tool_call_id:
            # OpenAI format: role='tool' with tool_call_id
            content = result if isinstance(result, str) else str(result)
            self.append(run_id, {
                'role': 'tool',
                'tool_call_id': tool_call_id,
                'content': content
            }, session_id)
        else:
            # Legacy format for backwards compatibility
            self.append(run_id, {
                'role': 'user',
                'content': f"Tool {tool_name} returned: {result}"
            }, session_id)

    def get_session_context(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a session"""
        return self._by_session.get(session_id, [])

    def set_session_context(self, session_id: str, messages: List[Dict[str, Any]]):
        """Set messages for a session (used when loading from storage)"""
        self._by_session[session_id] = messages

    def clear_session(self, session_id: str):
        """Clear a session from memory"""
        if session_id in self._by_session:
            del self._by_session[session_id]

    # ---------- optional memory: enable & helpers ----------

    def set_memory_config(self, cfg: MemoryConfig):
        """
        Enable optional memory. If you never call this, behavior is unchanged.
        """
        self._memory_cfg = cfg
        self._adapters = build_memory_adapters(cfg)

    # ---- Fact Store convenience wrappers (no LLM tool calls) ----

    def fact_put(self, namespace: str, key: str, value: Any):
        """
        Store a JSON-serializable value in the fact store (namespaced KV).

        The fact store is for long-term structured data that persists across conversations:
        - User preferences (theme, language, expertise)
        - Project metadata (current project, technologies)
        - Domain knowledge (company facts, endpoints)

        No-op if fact store is not enabled/available.

        Args:
            namespace: Logical grouping (e.g., "user:alice", "project:myapp")
            key: Fact key (e.g., "theme", "expertise_level")
            value: Any JSON-serializable value
        """
        store = self._adapters.get("facts") if self._adapters else None
        if store is not None:
            store.put(namespace, key, value)

    def fact_get(self, namespace: str, key: str) -> Optional[Any]:
        """
        Retrieve a value from the fact store.

        Returns None if fact store is not enabled/available or key not found.

        Args:
            namespace: Logical grouping
            key: Fact key

        Returns:
            Stored value or None
        """
        store = self._adapters.get("facts") if self._adapters else None
        if store is not None:
            return store.get(namespace, key)
        return None

    def fact_list(self, namespace: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List all facts in a namespace.

        Args:
            namespace: Logical grouping
            limit: Maximum number of items to return

        Returns:
            List of dicts with 'key', 'value', 'updated_at'
        """
        store = self._adapters.get("facts") if self._adapters else None
        if store is not None:
            return store.list(namespace, limit=limit)
        return []

    # ---- Backwards compatibility (deprecated) ----

    def kv_put(self, namespace: str, key: str, value: Any):
        """
        Deprecated: Use fact_put() instead.
        This method will be removed in v2.0.
        """
        warnings.warn(
            "kv_put() is deprecated and will be removed in v2.0. Use fact_put() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.fact_put(namespace, key, value)

    def kv_get(self, namespace: str, key: str) -> Optional[Any]:
        """
        Deprecated: Use fact_get() instead.
        This method will be removed in v2.0.
        """
        warnings.warn(
            "kv_get() is deprecated and will be removed in v2.0. Use fact_get() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.fact_get(namespace, key)

    def kv_list(self, namespace: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Deprecated: Use fact_list() instead.
        This method will be removed in v2.0.
        """
        warnings.warn(
            "kv_list() is deprecated and will be removed in v2.0. Use fact_list() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.fact_list(namespace, limit)

    # ---- ReasoningBank runtime hooks (no LLM tool calls) ----

    def prepare_for_run(self, signature: Dict[str, Any]) -> str:
        """
        Run-time pre-retrieval (bounded by your caller; this call itself is synchronous).
        Returns a short "Strategy Advice:\n..." text to append to your system prompt,
        or "" if ReasoningBank is disabled/unavailable or nothing is retrieved.

        Note: This method does not modify your message buffers; it just returns advice.
        """
        self._rb_last_ids = []
        rb = self._adapters.get("rb") if self._adapters else None
        if not rb or not self._memory_cfg:
            return ""

        try:
            top_k = max(1, int(self._memory_cfg.rb_top_k))
        except Exception:
            top_k = 5

        items: List[StrategyItem] = rb.get_advice(signature, k=top_k)  # type: ignore[attr-defined]
        if not items:
            return ""

        advice_lines: List[str] = []
        for i, s in enumerate(items, 1):
            # keep items tight (1 sentence preferred)
            advice_lines.append(f"{i}. {getattr(s, 'strategy_text', '')}")
            anti = getattr(s, "anti_patterns", None)
            if anti:
                advice_lines.append(f"   Avoid: {', '.join(anti[:3])}")
            sid = getattr(s, "id", None)
            if isinstance(sid, int):
                self._rb_last_ids.append(sid)

        return "Strategy Advice:\n" + "\n".join(advice_lines)

    def finalize_outcome(self, run_success: bool, fail_notes: Optional[List[str]] = None):
        """
        Post-run update for ReasoningBank (call AFTER you close the stream).
        This is a synchronous convenience wrapper; if you want fully async,
        call it from your background worker.

        No-op if RB is disabled/unavailable or if no strategies were used.
        """
        rb = self._adapters.get("rb") if self._adapters else None
        if not rb or not self._rb_last_ids:
            return

        notes = fail_notes or []
        rb.mark_outcome(self._rb_last_ids, success=bool(run_success), fail_notes=notes)  # type: ignore[attr-defined]
        # clear last ids after recording
        self._rb_last_ids = []


class StatelessContextManager(ContextManager):
    """
    Stateless context manager - never retains history.
    Each call is independent with no memory of previous turns.
    """
    def __init__(self):
        super().__init__(max_history=1)

    def set_input(self, run_id: str, input: Dict[str, Any], session_id: Optional[str] = None):
        """Store input but don't accumulate in sessions"""
        self._inputs[run_id] = input
        message = input.get('message', '') or str(input)
        # Always store in run, never accumulate in session
        self._by_run[run_id] = [{'role': 'user', 'content': message}]

    def messages_for_llm(self, run_id: str, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Always returns only the current input message, ignoring session history"""
        messages = self._by_run.get(run_id, [{'role': 'user', 'content': 'Hello'}])
        return messages[:1] if messages else [{'role': 'user', 'content': 'Hello'}]

    def append(self, run_id: str, item: Dict[str,Any], session_id: Optional[str] = None):
        """No-op: don't store subsequent messages"""
        pass

    def append_assistant_message(self, run_id: str, content: str, session_id: Optional[str] = None):
        """No-op: stateless mode doesn't track responses"""
        pass

    def append_tool_result(self, run_id: str, tool_name: str, result: Any, session_id: Optional[str] = None):
        """No-op: stateless mode doesn't track tool results"""
        pass
