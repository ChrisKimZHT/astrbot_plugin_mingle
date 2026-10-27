import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mingle_tests_plugin.scheduler import ChatScheduler
from mingle_tests_plugin.settings import Settings


class Event:
    def __init__(self, session="group"):
        self.unified_msg_origin = session
        self.extras = {}
        self.stopped = False

    def get_extra(self, key, default=None):
        return self.extras.get(key, default)

    def set_extra(self, key, value):
        self.extras[key] = value

    def stop_event(self):
        self.stopped = True

    def is_stopped(self):
        return self.stopped


@pytest.mark.asyncio
async def test_coalesces_messages_preserves_forced_intent_and_waits_for_completion():
    judge = SimpleNamespace(should_reply=AsyncMock(return_value=False))
    history = SimpleNamespace(get=AsyncMock(return_value=[]))
    scheduler = ChatScheduler(
        Settings(debounce_seconds=0.01), history, judge, logging.getLogger()
    )
    try:
        first = await scheduler.submit(Event(), 1, True)
        second = await scheduler.submit(Event(), 2, False)
        assert await asyncio.wait_for(first.ready, 1) is None
        assert await asyncio.wait_for(second.ready, 1) == [1, 2]
        judge.should_reply.assert_not_awaited()
        third = await scheduler.submit(Event(), 3, True)
        await asyncio.sleep(0.03)
        assert not third.ready.done()
        second.finished.set()
        assert await asyncio.wait_for(third.ready, 1) == [3]
        third.finished.set()
    finally:
        await scheduler.close()


@pytest.mark.asyncio
async def test_separate_sessions_are_not_blocked_by_active_turn():
    scheduler = ChatScheduler(
        Settings(debounce_seconds=0),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        SimpleNamespace(should_reply=AsyncMock(return_value=True)),
        logging.getLogger(),
    )
    try:
        a = await scheduler.submit(Event("a"), 1, True)
        assert await asyncio.wait_for(a.ready, 1) == [1]
        b = await scheduler.submit(Event("b"), 2, True)
        assert await asyncio.wait_for(b.ready, 1) == [2]
        a.finished.set()
        b.finished.set()
    finally:
        await scheduler.close()


@pytest.mark.asyncio
async def test_cooldown_only_blocks_unforced_group_replies():
    judge = SimpleNamespace(should_reply=AsyncMock(return_value=True))
    scheduler = ChatScheduler(
        Settings(debounce_seconds=0),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        judge,
        logging.getLogger(),
    )
    try:
        first = await scheduler.submit(Event(), 1, False)
        await asyncio.wait_for(first.ready, 1)
        first.event.set_extra("mingle_agent_started", True)
        first.finished.set()
        second = await scheduler.submit(Event(), 2, False)
        assert await asyncio.wait_for(second.ready, 1) is None
        third = await scheduler.submit(Event(), 3, True)
        assert await asyncio.wait_for(third.ready, 1) == [3]
        third.finished.set()
        judge.should_reply.assert_awaited_once()
    finally:
        await scheduler.close()


@pytest.mark.asyncio
async def test_shutdown_releases_judging_and_pending_handlers():
    started = asyncio.Event()

    async def judge_reply(*args):
        started.set()
        await asyncio.Event().wait()

    scheduler = ChatScheduler(
        Settings(debounce_seconds=0),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        SimpleNamespace(should_reply=judge_reply),
        logging.getLogger(),
    )
    first = await scheduler.submit(Event(), 1, False)
    await asyncio.wait_for(started.wait(), 1)
    second = await scheduler.submit(Event(), 2, True)
    await scheduler.close()
    assert await first.ready is None
    assert await second.ready is None


@pytest.mark.asyncio
async def test_stopped_pending_event_is_released():
    scheduler = ChatScheduler(
        Settings(debounce_seconds=0.01),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        SimpleNamespace(should_reply=AsyncMock()),
        logging.getLogger(),
    )
    try:
        first = await scheduler.submit(Event(), 1, True)
        first.event.stop_event()
        assert await asyncio.wait_for(first.ready, 1) is None
    finally:
        await scheduler.close()


@pytest.mark.asyncio
async def test_pending_limit_preserves_direct_address_when_events_are_merged():
    judge = SimpleNamespace(should_reply=AsyncMock(return_value=False))
    scheduler = ChatScheduler(
        Settings(debounce_seconds=0, max_pending_messages=2),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        judge,
        logging.getLogger(),
    )
    try:
        tickets = [
            await scheduler.submit(Event(), number, number == 1)
            for number in range(1, 5)
        ]
        assert await tickets[0].ready is None
        assert await tickets[1].ready is None
        assert await asyncio.wait_for(tickets[2].ready, 1) is None
        assert await asyncio.wait_for(tickets[3].ready, 1) == [3, 4]
        judge.should_reply.assert_not_awaited()
        tickets[3].finished.set()
    finally:
        await scheduler.close()
