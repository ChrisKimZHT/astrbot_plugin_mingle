import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mingle_tests_plugin.src.decision import ReplyDecision
from mingle_tests_plugin.src.scheduler import ChatScheduler
from mingle_tests_plugin.src.settings import Settings


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

    def get_message_str(self):
        return "Current task"


@pytest.mark.asyncio
async def test_coalesces_messages_preserves_forced_intent_and_waits_for_completion():
    judge = SimpleNamespace(
        should_reply=AsyncMock(return_value=False),
        evaluate=AsyncMock(return_value=ReplyDecision()),
    )
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
async def test_default_cooldown_allows_the_next_group_batch_after_a_reply():
    judge = SimpleNamespace(should_reply=AsyncMock(return_value=True))
    scheduler = ChatScheduler(
        Settings(debounce_seconds=0),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        judge,
        logging.getLogger(),
    )
    try:
        first = await scheduler.submit(Event(), 1, False)
        assert await asyncio.wait_for(first.ready, 1) == [1]
        first.event.set_extra("mingle_agent_started", True)
        first.finished.set()
        second = await scheduler.submit(Event(), 2, False)
        assert await asyncio.wait_for(second.ready, 1) == [2]
        assert judge.should_reply.await_count == 2
        second.finished.set()
    finally:
        await scheduler.close()


@pytest.mark.asyncio
async def test_cooldown_only_blocks_unforced_group_replies():
    judge = SimpleNamespace(should_reply=AsyncMock(return_value=True))
    scheduler = ChatScheduler(
        Settings(debounce_seconds=0, cooldown_seconds=60),
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

    async def judge_reply(*args, **kwargs):
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


@pytest.mark.asyncio
async def test_busy_judgment_runs_concurrently_and_exposes_queue_status():
    judge = SimpleNamespace(
        should_reply=AsyncMock(return_value=True),
        evaluate=AsyncMock(return_value=ReplyDecision(True)),
    )
    scheduler = ChatScheduler(
        Settings(debounce_seconds=0),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        judge,
        logging.getLogger(),
    )
    try:
        first = await scheduler.submit(Event(), 1, True)
        assert await asyncio.wait_for(first.ready, 1) == [1]
        second = await scheduler.submit(Event(), 2, False)
        state = scheduler.sessions["group"]
        await asyncio.wait_for(state.task, 1)
        assert not second.ready.done()
        assert not first.finished.is_set()
        status = judge.evaluate.call_args.kwargs["message_status"]
        assert status["active"] == [1]
        assert "active_context" not in judge.evaluate.call_args.kwargs
        assert status["judging"] == [2]
        third = await scheduler.submit(Event(), 3, False)
        await asyncio.wait_for(state.task, 1)
        assert judge.evaluate.call_args.kwargs["message_status"]["queued"] == [2]
        first.finished.set()
        assert await asyncio.wait_for(second.ready, 1) is None
        assert await asyncio.wait_for(third.ready, 1) == [2, 3]
        assert state.finished[1] == "completed"
        third.finished.set()
    finally:
        await scheduler.close()


@pytest.mark.asyncio
async def test_rerun_requests_stop_but_waits_for_old_history_completion():
    judge = SimpleNamespace(
        evaluate=AsyncMock(return_value=ReplyDecision(True, "rerun"))
    )
    scheduler = ChatScheduler(
        Settings(debounce_seconds=0),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        judge,
        logging.getLogger(),
    )
    try:
        first = await scheduler.submit(Event(), 1, True)
        await asyncio.wait_for(first.ready, 1)
        second = await scheduler.submit(Event(), 2, False)
        state = scheduler.sessions["group"]
        await asyncio.wait_for(state.task, 1)
        assert first.event.get_extra("agent_stop_requested") is True
        assert not first.event.is_stopped()
        assert not second.ready.done()
        first.finished.set()
        assert await asyncio.wait_for(second.ready, 1) == [2]
        assert second.event.get_extra("mingle_resume_after_rerun") is True
        assert state.finished[1] == "interrupted"
        second.finished.set()
    finally:
        await scheduler.close()


@pytest.mark.asyncio
async def test_stop_discards_approved_work_without_starting_replacement_turn():
    judge = SimpleNamespace(
        evaluate=AsyncMock(
            side_effect=[ReplyDecision(True), ReplyDecision(False, "stop")]
        )
    )
    scheduler = ChatScheduler(
        Settings(debounce_seconds=0),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        judge,
        logging.getLogger(),
    )
    try:
        first = await scheduler.submit(Event(), 1, True)
        await asyncio.wait_for(first.ready, 1)
        queued = await scheduler.submit(Event(), 2, False)
        state = scheduler.sessions["group"]
        await asyncio.wait_for(state.task, 1)
        stop = await scheduler.submit(Event(), 3, True)
        await asyncio.wait_for(state.task, 1)
        assert first.event.get_extra("agent_stop_requested") is True
        assert await asyncio.wait_for(queued.ready, 1) is None
        assert await asyncio.wait_for(stop.ready, 1) is None
        first.finished.set()
        await asyncio.wait_for(state.execution_task, 1)
        assert state.active is None
        assert not state.approved
        assert not state.resume_after_rerun
    finally:
        await scheduler.close()


@pytest.mark.asyncio
async def test_late_rerun_result_cannot_stop_a_newer_active_turn():
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def evaluate(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ReplyDecision(True)
        started.set()
        await release.wait()
        return ReplyDecision(True, "rerun")

    scheduler = ChatScheduler(
        Settings(debounce_seconds=0),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        SimpleNamespace(evaluate=evaluate),
        logging.getLogger(),
    )
    try:
        first = await scheduler.submit(Event(), 1, True)
        await asyncio.wait_for(first.ready, 1)
        second = await scheduler.submit(Event(), 2, True)
        state = scheduler.sessions["group"]
        await asyncio.wait_for(state.task, 1)
        third = await scheduler.submit(Event(), 3, False)
        await asyncio.wait_for(started.wait(), 1)
        first.finished.set()
        assert await asyncio.wait_for(second.ready, 1) == [2]
        release.set()
        await asyncio.wait_for(state.task, 1)
        assert not second.event.get_extra("agent_stop_requested", False)
        assert not third.ready.done()
        second.finished.set()
        assert await asyncio.wait_for(third.ready, 1) == [3]
        assert not third.event.get_extra("mingle_resume_after_rerun", False)
        third.finished.set()
    finally:
        release.set()
        await scheduler.close()


@pytest.mark.asyncio
async def test_shutdown_cleans_both_execution_and_judgment_tasks():
    judging = asyncio.Event()

    async def evaluate(*args, **kwargs):
        judging.set()
        await asyncio.Event().wait()

    scheduler = ChatScheduler(
        Settings(debounce_seconds=0),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        SimpleNamespace(evaluate=evaluate),
        logging.getLogger(),
    )
    first = await scheduler.submit(Event(), 1, True)
    await asyncio.wait_for(first.ready, 1)
    second = await scheduler.submit(Event(), 2, False)
    await asyncio.wait_for(judging.wait(), 1)
    third = await scheduler.submit(Event(), 3, False)
    state = scheduler.sessions["group"]
    await scheduler.close()
    assert await second.ready is None
    assert await third.ready is None
    assert first.event.is_stopped()
    assert state.task.done() and state.execution_task.done()


@pytest.mark.asyncio
async def test_busy_judgment_includes_active_messages_outside_recent_window():
    records = [{"id": 1, "text": "Original task"}, {"id": 2, "text": "Correction"}]

    async def get(session, *, limit, before_id=None, after_id=None):
        return [
            record
            for record in records
            if (before_id is None or record["id"] < before_id)
            and (after_id is None or record["id"] > after_id)
        ][-limit:]

    judge = SimpleNamespace(
        evaluate=AsyncMock(return_value=ReplyDecision(True, "rerun"))
    )
    scheduler = ChatScheduler(
        Settings(debounce_seconds=0, recent_messages=1),
        SimpleNamespace(get=get),
        judge,
        logging.getLogger(),
    )
    try:
        first = await scheduler.submit(Event(), 1, True)
        assert await asyncio.wait_for(first.ready, 1) == [1]
        second = await scheduler.submit(Event(), 2, False)
        await asyncio.wait_for(scheduler.sessions["group"].task, 1)
        assert judge.evaluate.call_args.args[1] == records
        assert judge.evaluate.call_args.kwargs["message_status"]["active"] == [1]
        assert "active_context" not in judge.evaluate.call_args.kwargs
        assert first.event.get_extra("agent_stop_requested") is True
        assert not second.ready.done()
        first.finished.set()
        assert await asyncio.wait_for(second.ready, 1) == [2]
        second.finished.set()
    finally:
        await scheduler.close()
