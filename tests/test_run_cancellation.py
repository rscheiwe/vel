"""`RunManager.cancel()` — stopping a detached run and settling it.

`RunManager` held its tasks in `self._tasks` and never cancelled any of them,
and `'cancelled'` was declared in `TERMINAL_STATUSES` and `checkpoint.Status`
without a single writer anywhere in the codebase. A client that walked away
left a task making paid provider calls to completion.

Five things have to move together or the run is not really cancelled: the task
stops, the status says so, a terminal event reaches the durable log, waiting
subscribers are woken, and the checkpoint is settled so `recover()` does not
restart it later. The last two are the ones that fail quietly — a subscriber
that is never woken blocks forever on the pub/sub queue.
"""
from __future__ import annotations

import asyncio
from typing import Any, List

import pytest

from vel import Agent, ToolSpec
from vel.events import (
    FinishMessageEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
)
from vel.harness import RunManager
from vel.providers import BaseProvider


class SlowProvider(BaseProvider):
    name = 'scripted'

    async def stream(self, messages, model, tools, generation_config=None):
        yield TextStartEvent(block_id='b')
        for i in range(200):
            await asyncio.sleep(0.02)
            yield TextDeltaEvent(block_id='b', delta=f'{i} ')
        yield TextEndEvent(block_id='b')
        yield FinishMessageEvent(finish_reason='stop')

    async def generate(self, messages, model, tools, generation_config=None):
        return {}


async def echo(x: str = '', ctx: dict = None) -> dict:
    return {'echo': x}


def _agent() -> Agent:
    agent = Agent(
        id='cancellable',
        model={'provider': 'scripted', 'model': 'm'},
        tools=[ToolSpec.from_function(echo, name='echo')],
    )
    agent._custom_provider = SlowProvider()
    return agent


@pytest.mark.asyncio
async def test_cancel_stops_the_run_and_records_the_status(tmp_path):
    runs = RunManager(db_path=str(tmp_path / 'vel.db'))
    run_id = await runs.start(_agent(), input={'message': 'hi'}, session_id='s1')

    await asyncio.sleep(0.15)  # let it get going
    assert await runs.cancel(run_id) is True

    assert await runs.get_status(run_id) == 'cancelled'
    task = runs._tasks[run_id]
    assert task.done(), 'the driving task must not still be running'


@pytest.mark.asyncio
async def test_cancel_emits_a_well_formed_ending_to_the_log(tmp_path):
    """Subscribers must see the run end, not a stream that simply stops."""
    runs = RunManager(db_path=str(tmp_path / 'vel.db'))
    run_id = await runs.start(_agent(), input={'message': 'hi'}, session_id='s1')

    await asyncio.sleep(0.15)
    await runs.cancel(run_id)

    logged = [event for _id, event in runs.store.events_after(run_id, 0)]
    types = [e.get('type') for e in logged]

    assert 'abort' in types, 'a cancelled run must say so on the wire'
    assert types.index('abort') < types.index('finish')
    assert types.count('text-start') == types.count('text-end') == 1
    assert 'error' not in types, 'cancel is an abort, not a failure'


@pytest.mark.asyncio
async def test_a_live_subscriber_is_woken_rather_than_left_hanging(tmp_path):
    """Without the sentinel, `stream()` blocks forever on the queue."""
    runs = RunManager(db_path=str(tmp_path / 'vel.db'))
    run_id = await runs.start(_agent(), input={'message': 'hi'}, session_id='s1')

    async def drain() -> List[Any]:
        return [e async for e in runs.stream(run_id, cursor=0)]

    consumer = asyncio.create_task(drain())
    await asyncio.sleep(0.15)
    await runs.cancel(run_id)

    events = await asyncio.wait_for(consumer, timeout=5.0)
    assert any(e.get('type') == 'abort' for e in events)


@pytest.mark.asyncio
async def test_cancelling_an_unknown_run_reports_it(tmp_path):
    runs = RunManager(db_path=str(tmp_path / 'vel.db'))
    assert await runs.cancel('no-such-run') is False
