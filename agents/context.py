from typing import Any, Dict, List

class ContextManager:
    def __init__(self):
        self._by_run: Dict[str, List[Dict[str,Any]]] = {}
    def messages_for_llm(self, run_id: str):
        return self._by_run.get(run_id, [{'role':'user','content':'Hello'}])
    def append(self, run_id: str, item: Dict[str,Any]):
        self._by_run.setdefault(run_id, []).append(item)
