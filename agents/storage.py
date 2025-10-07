from __future__ import annotations
from typing import Any, Dict, List

class RunStore:
    def __init__(self):
        self._events: Dict[str, List[Dict[str,Any]]] = {}

    @classmethod
    def default(cls) -> "RunStore":
        return cls()

    async def append_event(self, run_id: str, event: Dict[str,Any]):
        self._events.setdefault(run_id, []).append(event)
