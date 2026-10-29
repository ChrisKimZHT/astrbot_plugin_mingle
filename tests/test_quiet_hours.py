import asyncio
import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mingle_tests_plugin.src.scheduler import ChatScheduler, SessionState
from mingle_tests_plugin.src.settings import Settings
from test_scheduler import Event


@pytest.fixture
def clocks(monkeypatch):
    clock = SimpleNamespace(
        now=100.0, wall=datetime(2026, 10, 30, 9, 50, tzinfo=timezone.utc)
    )

    class WallClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock.wall.astimezone(tz)

    monkeypatch.setattr("mingle_tests_plugin.src.scheduler.datetime", WallClock)
    monkeypatch.setattr(
        "mingle_tests_plugin.src.scheduler.monotonic", lambda: clock.now
    )
    return clock


@pytest.mark.parametrize(
    ("wall", "zone", "start", "end", "remaining"),
    [
        ("2026-10-30T21:59:59", "UTC", "22:00", "10:00", 0),
        ("2026-10-30T22:00:00", "UTC", "22:00", "10:00", 43200),
        ("2026-10-30T23:30:00", "UTC", "22:00", "10:00", 37800),
        ("2026-10-30T00:00:00", "UTC", "22:00", "10:00", 36000),
        ("2026-10-30T09:59:59", "UTC", "22:00", "10:00", 1),
        ("2026-10-30T10:00:00", "UTC", "22:00", "10:00", 0),
        ("2026-10-30T14:00:00", "Asia/Shanghai", "22:00", "10:00", 43200),
        ("2026-10-30T09:00:00", None, "22:00", "10:00", 3600),
        ("2026-10-30T09:00:00", "Bad/Zone", "22:00", "10:00", 3600),
        ("2026-10-30T09:00:00", 123, "22:00", "10:00", 3600),
        ("2026-10-30T09:00:00", "UTC", "09:00", "17:30", 30600),
        ("2026-10-30T17:30:00", "UTC", "09:00", "17:30", 0),
        ("2026-10-30T09:00:00", "UTC", "10:00", "10:00", 0),
        ("2026-10-31T22:00:00", "America/New_York", "22:00", "10:00", 0),
        ("2026-11-01T02:00:00", "America/New_York", "22:00", "10:00", 46800),
    ],
)
def test_quiet_hours_boundaries_and_timezone(clocks, wall, zone, start, end, remaining):
    clocks.wall = datetime.fromisoformat(wall).replace(tzinfo=timezone.utc)
    config = Mock(return_value={"timezone": zone})
    scheduler = ChatScheduler(
        Settings(quiet_hours_start=start, quiet_hours_end=end),
        None,
        None,
        logging.getLogger(),
        get_config=config,
    )
    assert scheduler._quiet_seconds_remaining("group") == remaining
    if start == end:
        config.assert_not_called()
        return
    config.assert_called_once_with(umo="group")
    config.return_value = {"timezone": "UTC"}
    scheduler._quiet_seconds_remaining("group")
    assert config.call_count == 2


@pytest.mark.asyncio
async def test_due_bubble_resamples_after_quiet_hours_without_a_send(
    clocks, monkeypatch
):
    draw = Mock(side_effect=[10, 20])
    monkeypatch.setattr("mingle_tests_plugin.src.scheduler.expovariate", draw)
    queue = AsyncMock()
    scheduler = ChatScheduler(
        Settings(random_bubble_enabled=True, random_bubble_min_minutes=0),
        None,
        None,
        logging.getLogger(),
        queue_bubble=queue,
    )
    try:
        scheduler.watch_random_bubbles("group")
        state = scheduler.sessions["group"]
        clocks.now = 111
        state.changed.set()
        await asyncio.sleep(0.01)
        queue.assert_not_awaited()
        assert state.next_bubble_at == 111 + 600 + 20
        assert state.bubble_revision == 2
        assert state.last_agent_reply_at == 100
        clocks.wall = clocks.wall.replace(hour=10, minute=0)
        clocks.now = 712
        state.changed.set()
        await asyncio.sleep(0.01)
        queue.assert_not_awaited()
        clocks.now = 732
        state.changed.set()
        await asyncio.sleep(0.01)
        queue.assert_awaited_once_with("group", 2)
        assert state.next_bubble_at == float("inf")
        state.changed.set()
        await asyncio.sleep(0.01)
        queue.assert_awaited_once()
    finally:
        await scheduler.close()


@pytest.mark.asyncio
async def test_queued_bubble_entering_quiet_hours_is_invalidated(clocks, monkeypatch):
    monkeypatch.setattr("mingle_tests_plugin.src.scheduler.expovariate", lambda _: 10)
    scheduler = ChatScheduler(
        Settings(random_bubble_enabled=True, random_bubble_min_minutes=0),
        None,
        None,
        logging.getLogger(),
        queue_bubble=AsyncMock(),
    )
    try:
        scheduler.watch_random_bubbles("group")
        state = scheduler.sessions["group"]
        revision = state.bubble_revision
        assert await scheduler.admit_random_bubble(Event(), revision) is None
        assert state.active is None
        assert state.bubble_revision == revision + 1
        assert state.next_bubble_at == 710
        assert await scheduler.admit_random_bubble(Event(), revision) is None
    finally:
        await scheduler.close()


@pytest.mark.asyncio
async def test_backlog_skips_quiet_review_and_keeps_next_regular_slot(clocks):
    judge = SimpleNamespace(should_reply=AsyncMock(return_value=False))
    scheduler = ChatScheduler(
        Settings(debounce_seconds=0),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        judge,
        logging.getLogger(),
    )
    try:
        scheduler.sleep("group")
        ticket = await scheduler.submit(Event(), 1, False)
        state = scheduler.sessions["group"]
        clocks.now = 1900
        state.changed.set()
        await asyncio.sleep(0.01)
        assert state.next_sleep_backlog_at == 3700
        assert state.pending == [ticket]
        assert not ticket.ready.done()
        judge.should_reply.assert_not_awaited()
        scheduler.history.get.assert_not_awaited()
        clocks.wall = clocks.wall.replace(hour=10, minute=0)
        clocks.now = 2500
        state.changed.set()
        await asyncio.sleep(0.01)
        judge.should_reply.assert_not_awaited()
        clocks.now = 3700
        state.changed.set()
        assert await asyncio.wait_for(ticket.ready, 1) is None
        judge.should_reply.assert_awaited_once()
    finally:
        await scheduler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("crossing", ["history", "supplement", "judgment", "execution"])
async def test_backlog_crossing_quiet_start_retains_tickets(clocks, crossing):
    clocks.wall = clocks.wall.replace(hour=21, minute=59)
    entered_quiet = asyncio.Event()

    async def get_history(*args, **kwargs):
        if crossing == "supplement" and "after_id" not in kwargs:
            return []
        if crossing in {"history", "supplement"}:
            clocks.wall = clocks.wall.replace(hour=22, minute=0)
            entered_quiet.set()
        return [{"id": 1}]

    async def should_reply(*args, **kwargs):
        if crossing == "judgment":
            clocks.wall = clocks.wall.replace(hour=22, minute=0)
            entered_quiet.set()
        return True

    scheduler = ChatScheduler(
        Settings(debounce_seconds=0),
        SimpleNamespace(get=AsyncMock(side_effect=get_history)),
        SimpleNamespace(should_reply=AsyncMock(side_effect=should_reply)),
        logging.getLogger(),
    )
    try:
        scheduler.sleep("group")
        ticket = await scheduler.submit(Event(), 1, False)
        state = scheduler.sessions["group"]
        if crossing == "execution":
            state.pending = []
            state.approved = [ticket]
            ticket.backlog_sleep_revision = state.sleep_revision
            clocks.wall = clocks.wall.replace(hour=22, minute=0)
            await scheduler._execute_approved("group", state)
        clocks.now = 1900
        state.changed.set()
        if crossing != "execution":
            await asyncio.wait_for(entered_quiet.wait(), 1)
        for _ in range(20):
            if not state.judging:
                break
            await asyncio.sleep(0)
        assert state.pending == [ticket]
        assert state.approved == []
        assert state.active is None
        assert not ticket.ready.done()
        if crossing != "judgment":
            scheduler.judge.should_reply.assert_not_awaited()
    finally:
        await scheduler.close()


@pytest.mark.asyncio
async def test_deferred_approved_backlog_preserves_pending_queue_bound(clocks):
    scheduler = ChatScheduler(
        Settings(debounce_seconds=0, max_pending_messages=2),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        SimpleNamespace(should_reply=AsyncMock(return_value=True)),
        logging.getLogger(),
    )
    try:
        scheduler.sleep("group")
        first = await scheduler.submit(Event(), 1, True)
        state = scheduler.sessions["group"]
        state.pending = []
        state.approved = [first]
        first.backlog_sleep_revision = state.sleep_revision
        second = await scheduler.submit(Event(), 2, False)
        third = await scheduler.submit(Event(), 3, False)
        await scheduler._execute_approved("group", state)
        assert state.pending == [second, third]
        assert second.forced
        assert await first.ready is None
        assert state.finished[1] == "merged"
        assert not second.ready.done()
        assert not third.ready.done()
        scheduler.judge.should_reply.assert_not_awaited()
    finally:
        await scheduler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("wake", [False, True])
async def test_quiet_hours_preserve_normal_replies_and_explicit_wakes(clocks, wake):
    scheduler = ChatScheduler(
        Settings(debounce_seconds=0, cooldown_seconds=0),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        SimpleNamespace(should_reply=AsyncMock(return_value=True)),
        logging.getLogger(),
    )
    scheduler.sessions["group"] = SessionState()
    try:
        if wake:
            scheduler.sleep("group")
        ticket = await scheduler.submit(Event(), 1, True, wake=wake)
        assert await asyncio.wait_for(ticket.ready, 1) == [1]
        ticket.finished.set()
    finally:
        await scheduler.close()
