from __future__ import annotations
import json, os
from typing import Any, Dict, List, Optional

class RunStore:
    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn
        self._events: Dict[str, List[Dict[str,Any]]] = {}

    @classmethod
    def default(cls) -> 'RunStore':
        return cls(os.getenv('POSTGRES_DSN'))

    async def append_event(self, run_id: str, event: Dict[str,Any]):
        self._events.setdefault(run_id, []).append(event)
