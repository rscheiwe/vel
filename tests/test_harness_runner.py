import asyncio
import os
import uuid

import pytest

from vel.harness.pubsub import RedisPubSub
from vel.harness.runner import PostgresEventLogStore, RunManager, SQLiteEventLogStore


class FakeAgent:
    id = 'fake'

    def __init__(self):
        self.run_calls = []
        self.resume_calls = []

    async def run_stream(self, input, **kwargs):
        self.run_calls.append((input, kwargs))
        yield {'type': 'start'}
        await asyncio.sleep(0)
        yield {
            'type': 'data-harness-suspended',
            'data': {'run_id': kwargs['external_run_id'], 'reason': 'approval'},
            'transient': False,
        }

    async def resume(self, run_id, decisions, **kwargs):
        self.resume_calls.append((run_id, decisions, kwargs))
        yield {'type': 'data-harness-resumed', 'data': {'run_id': run_id}, 'transient': True}
        yield {
            'type': 'data-harness-run-finished',
            'data': {'run_id': run_id, 'status': 'completed', 'usage': {}},
            'transient': False,
        }


@pytest.mark.asyncio
async def test_run_manager_starts_logs_and_replays_suspended_run(tmp_path):
    manager = RunManager(db_path=str(tmp_path / 'vel.db'))
    agent = FakeAgent()

    run_id = await manager.start(
        agent,
        input={'message': 'go'},
        session_id='s1',
        harness={'enabled': True},
        context={'tenant': 't1'},
    )
    await manager.wait(run_id)

    assert await manager.get_status(run_id) == 'suspended'
    assert agent.run_calls[0][1]['external_run_id'] == run_id
    assert agent.run_calls[0][1]['session_id'] == 's1'
    assert agent.run_calls[0][1]['context'] == {'tenant': 't1'}

    replay = [event async for event in manager.stream(run_id)]
    assert [event['type'] for event in replay] == ['start', 'data-harness-suspended']
    assert [event['_cursor'] for event in replay] == [1, 2]

    rows = manager.store.events_after(run_id)
    replay_after_first = [event async for event in manager.stream(run_id, cursor=rows[0][0])]
    assert [event['type'] for event in replay_after_first] == ['data-harness-suspended']
    assert replay_after_first[0]['_cursor'] == rows[1][0]
    assert '_cursor' not in rows[0][1]
    manager.close()


@pytest.mark.asyncio
async def test_run_manager_live_stream_and_resume(tmp_path):
    manager = RunManager(db_path=str(tmp_path / 'vel.db'))
    agent = FakeAgent()

    run_id = await manager.start(agent, input={'message': 'go'}, harness={'enabled': True})
    live = asyncio.create_task(_collect(manager.stream(run_id)))

    await manager.wait(run_id)
    live_events = await live
    assert [event['type'] for event in live_events] == ['start', 'data-harness-suspended']
    assert [event['_cursor'] for event in live_events] == [1, 2]

    await manager.resume(run_id, decisions=[{'approval_id': 'a1', 'decision': 'approve'}])
    await manager.wait(run_id)

    assert await manager.get_status(run_id) == 'completed'
    assert agent.resume_calls[0][0] == run_id
    assert agent.resume_calls[0][2]['harness'] == {'enabled': True}

    replay = [event async for event in manager.stream(run_id)]
    assert [event['type'] for event in replay] == [
        'start',
        'data-harness-suspended',
        'data-harness-resumed',
        'data-harness-run-finished',
    ]
    assert [event['_cursor'] for event in replay] == [1, 2, 3, 4]
    manager.close()


@pytest.mark.asyncio
async def test_run_manager_concurrent_subscribers(tmp_path):
    manager = RunManager(db_path=str(tmp_path / 'vel.db'))
    agent = FakeAgent()

    run_id = await manager.start(agent, input={'message': 'go'}, harness={'enabled': True})
    subscriber_a = asyncio.create_task(_collect(manager.stream(run_id)))
    subscriber_b = asyncio.create_task(_collect(manager.stream(run_id)))

    await manager.wait(run_id)

    events_a = await subscriber_a
    events_b = await subscriber_b
    assert [event['type'] for event in events_a] == ['start', 'data-harness-suspended']
    assert [event['type'] for event in events_b] == ['start', 'data-harness-suspended']
    assert [event['_cursor'] for event in events_a] == [1, 2]
    assert [event['_cursor'] for event in events_b] == [1, 2]
    manager.close()


def test_sqlite_event_log_store_interface(tmp_path):
    store = SQLiteEventLogStore(str(tmp_path / 'vel.db'))
    _assert_event_log_store_interface(store)
    store.close()


@pytest.mark.skipif(
    not os.environ.get('VEL_TEST_POSTGRES_DSN'),
    reason='VEL_TEST_POSTGRES_DSN not set',
)
def test_postgres_event_log_store_interface():
    store = PostgresEventLogStore(os.environ['VEL_TEST_POSTGRES_DSN'])
    try:
        _assert_event_log_store_interface(store, prefix=f"pg-{uuid.uuid4()}")
    finally:
        store.close()


@pytest.mark.skipif(
    not os.environ.get('VEL_TEST_POSTGRES_DSN'),
    reason='VEL_TEST_POSTGRES_DSN not set',
)
@pytest.mark.asyncio
async def test_run_manager_postgres_backend():
    manager = RunManager(
        store_backend='postgres',
        dsn=os.environ['VEL_TEST_POSTGRES_DSN'],
    )
    agent = FakeAgent()
    run_id = await manager.start(
        agent,
        input={'message': 'go'},
        harness={'enabled': True},
        run_id=f"pg-{uuid.uuid4()}",
    )
    await manager.wait(run_id)

    replay = [event async for event in manager.stream(run_id)]
    assert [event['type'] for event in replay] == ['start', 'data-harness-suspended']
    assert replay[0]['_cursor'] < replay[1]['_cursor']
    manager.close()


def test_run_manager_postgres_backend_requires_dsn():
    with pytest.raises(ValueError, match='requires dsn'):
        RunManager(store_backend='postgres')


@pytest.mark.skipif(
    not os.environ.get('VEL_TEST_REDIS_URL'),
    reason='VEL_TEST_REDIS_URL not set',
)
@pytest.mark.asyncio
async def test_redis_pubsub_publish_subscribe():
    pytest.importorskip('redis.asyncio')
    run_id = f"redis-{uuid.uuid4()}"
    pubsub = RedisPubSub(os.environ['VEL_TEST_REDIS_URL'])

    async with pubsub.subscribe(run_id) as subscription:
        await pubsub.publish(run_id, 7, {'type': 'start'})
        cursor, event = await asyncio.wait_for(subscription.__anext__(), timeout=5)

    assert cursor == 7
    assert event == {'type': 'start'}
    await pubsub.close()


@pytest.mark.skipif(
    not os.environ.get('VEL_TEST_REDIS_URL'),
    reason='VEL_TEST_REDIS_URL not set',
)
@pytest.mark.asyncio
async def test_run_manager_redis_pubsub_cross_worker_live_tail(tmp_path):
    pytest.importorskip('redis.asyncio')
    run_id = f"redis-run-{uuid.uuid4()}"
    db_path = str(tmp_path / 'vel.db')
    pubsub_a = RedisPubSub(os.environ['VEL_TEST_REDIS_URL'])
    pubsub_b = RedisPubSub(os.environ['VEL_TEST_REDIS_URL'])
    worker_a = RunManager(db_path=db_path, pubsub=pubsub_a)
    worker_b = RunManager(db_path=db_path, pubsub=pubsub_b)
    agent = FakeAgent()

    live = asyncio.create_task(_collect(worker_b.stream(run_id)))
    await asyncio.sleep(0.1)
    await worker_a.start(agent, input={'message': 'go'}, harness={'enabled': True}, run_id=run_id)
    await worker_a.wait(run_id)

    events = await asyncio.wait_for(live, timeout=5)
    assert [event['type'] for event in events] == ['start', 'data-harness-suspended']
    assert events[0]['_cursor'] < events[1]['_cursor']
    worker_a.close()
    worker_b.close()
    await pubsub_a.close()
    await pubsub_b.close()


def _assert_event_log_store_interface(store, prefix='run'):
    run_id = f"{prefix}-{uuid.uuid4()}"
    store.ensure_run(run_id, 'agent', 'running')
    assert store.get_run_status(run_id) == 'running'

    first = store.append_event(run_id, {'type': 'start'})
    second = store.append_event(run_id, {'type': 'data-harness-suspended', 'data': {'run_id': run_id}})
    assert second > first
    assert store.events_after(run_id, first) == [
        (second, {'type': 'data-harness-suspended', 'data': {'run_id': run_id}})
    ]

    store.set_run_status(run_id, 'suspended')
    assert store.get_run_status(run_id) == 'suspended'


async def _collect(agen):
    return [event async for event in agen]
