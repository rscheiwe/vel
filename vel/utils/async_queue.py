# utils/async_queue.py
"""
Lightweight background work queue for non-blocking updates.

Used by memory system for async post-run updates.
"""
import queue
import threading
import atexit
from typing import Callable, Any, Tuple, Optional

class WorkQueue:
    """
    Thread-safe background work queue with graceful shutdown.

    Drops work on overflow to protect latency.
    """
    def __init__(self, maxsize: int = 1024):
        self.q: "queue.Queue[Tuple[str, dict]]" = queue.Queue(maxsize=maxsize)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._started = False

    def put_nowait(self, item: Tuple[str, dict]):
        """Add work item. Drops silently if queue is full."""
        try:
            self.q.put_nowait(item)
        except queue.Full:
            # Drop on floor to protect user latency
            pass

    def start(self, handler: Callable[[str, dict], None]):
        """Start background worker thread (idempotent)."""
        if self._started:
            return

        self._started = True
        self._running = True

        def _loop():
            while self._running:
                try:
                    kind, payload = self.q.get(timeout=0.25)
                    handler(kind, payload)
                except queue.Empty:
                    continue
                except Exception:
                    # Swallow to keep worker alive
                    pass

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

        # Register shutdown handler
        atexit.register(self.stop)

    def stop(self):
        """Stop background worker gracefully."""
        if not self._running:
            return

        self._running = False

        # Wait for thread to finish (with timeout)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
