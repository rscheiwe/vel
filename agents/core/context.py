from typing import Any, Dict, List, Optional

class ContextManager:
    """
    Context manager with configurable memory behavior.
    Stores conversation history for multi-turn interactions.
    Supports both run-based and session-based context.
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

    def set_input(self, run_id: str, input: Dict[str, Any], session_id: Optional[str] = None):
        """Store the initial input for a run"""
        self._inputs[run_id] = input
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

    def append_tool_result(self, run_id: str, tool_name: str, result: Any, session_id: Optional[str] = None):
        """Append a tool result as a message"""
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
