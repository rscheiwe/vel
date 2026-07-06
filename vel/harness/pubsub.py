from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import AsyncContextManager, AsyncIterator, Dict, List, Optional, Protocol, Tuple


class PubSub(Protocol):
    async def publish(self, run_id: str, cursor: int, event: Optional[dict]) -> None: ...
    def subscribe(self, run_id: str) -> AsyncContextManager[AsyncIterator[Tuple[int, Optional[dict]]]]: ...
    async def close(self) -> None: ...


class InProcessPubSub:
    def __init__(self):
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}

    async def publish(self, run_id: str, cursor: int, event: Optional[dict]) -> None:
        for queue in list(self._subscribers.get(run_id, [])):
            queue.put_nowait((cursor, event))

    @asynccontextmanager
    async def subscribe(self, run_id: str) -> AsyncIterator[AsyncIterator[Tuple[int, Optional[dict]]]]:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(run_id, []).append(queue)
        try:
            yield _queue_iterator(queue)
        finally:
            subscribers = self._subscribers.get(run_id, [])
            if queue in subscribers:
                subscribers.remove(queue)

    async def close(self) -> None:
        self._subscribers.clear()


class RedisPubSub:
    def __init__(self, url: str, *, channel_prefix: str = 'vel:run:'):
        self.url = url
        self.channel_prefix = channel_prefix
        redis = _load_redis()
        self.redis = redis.from_url(url)

    async def publish(self, run_id: str, cursor: int, event: Optional[dict]) -> None:
        payload = json.dumps({'cursor': cursor, 'event': event})
        await self.redis.publish(self._channel(run_id), payload)

    @asynccontextmanager
    async def subscribe(self, run_id: str) -> AsyncIterator[AsyncIterator[Tuple[int, Optional[dict]]]]:
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(self._channel(run_id))
        try:
            yield _redis_iterator(pubsub)
        finally:
            await pubsub.unsubscribe(self._channel(run_id))
            await pubsub.close()

    async def close(self) -> None:
        await self.redis.aclose()

    def _channel(self, run_id: str) -> str:
        return f"{self.channel_prefix}{run_id}"


async def _queue_iterator(queue: asyncio.Queue) -> AsyncIterator[Tuple[int, Optional[dict]]]:
    while True:
        yield await queue.get()


async def _redis_iterator(pubsub) -> AsyncIterator[Tuple[int, Optional[dict]]]:
    async for message in pubsub.listen():
        if message.get('type') != 'message':
            continue
        data = message.get('data')
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        payload = json.loads(data)
        yield int(payload['cursor']), payload.get('event')


def _load_redis():
    try:
        import redis.asyncio as redis
    except ImportError as exc:
        raise ImportError(
            "Redis RunManager pub/sub requires the optional 'harness-redis' "
            "extra: pip install 'vel-ai[harness-redis]'"
        ) from exc
    return redis


__all__ = ['InProcessPubSub', 'PubSub', 'RedisPubSub']
