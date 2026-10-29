import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mingle_tests_plugin.src.decision import ReplyDecision
from mingle_tests_plugin.src.scheduler import ChatScheduler, SessionState
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
@pytest.mark.parametrize(
    "late_decision",
    [ReplyDecision(True), ReplyDecision(True, "rerun"), ReplyDecision(False, "stop")],
)
async def test_recovery_claims_all_waiting_queues_and_ignores_late_judgment(
    late_decision,
):
    started, release = asyncio.Event(), asyncio.Event()

    async def evaluate(event, records, **kwargs):
        if kwargs["new_message_ids"] == [3]:
            started.set()
            await release.wait()
            return late_decision
        return ReplyDecision(True)

    scheduler = ChatScheduler(
        Settings(debounce_seconds=0, cooldown_seconds=0),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        SimpleNamespace(evaluate=evaluate),
        logging.getLogger(),
    )
    try:
        active = await scheduler.submit(Event(), 1, True)
        assert await asyncio.wait_for(active.ready, 1) == [1]
        approved = await scheduler.submit(Event(), 2, False)
        state = scheduler.sessions["group"]
        await asyncio.wait_for(state.judgment_task, 1)
        judging = await scheduler.submit(Event(), 3, False)
        await asyncio.wait_for(started.wait(), 1)
        pending = await scheduler.submit(Event(), 4, False)
        assert scheduler.claim_recovery_messages(active) == [2, 3, 4]
        assert state.active_ids == [1, 2, 3, 4]
        assert not active.finished.is_set()
        for item in (approved, judging, pending):
            assert await item.ready is None
        later = await scheduler.submit(Event(), 5, False)
        release.set()
        await asyncio.wait_for(state.judgment_task, 1)
        assert not active.event.get_extra("agent_stop_requested", False)
        assert state.active_ids == [1, 2, 3, 4]
        assert not state.finished
        assert [item.record_id for item in state.approved] == [5]
        assert not later.ready.done()
        active.finished.set()
        assert await asyncio.wait_for(later.ready, 1) == [5]
        assert all(
            state.finished[record_id] == "completed" for record_id in range(1, 5)
        )
        later.finished.set()
    finally:
        release.set()
        await scheduler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", ["agent_stop_requested", "agent_user_aborted"])
async def test_recovery_does_not_claim_messages_after_interruption(flag):
    scheduler = ChatScheduler(
        Settings(debounce_seconds=0),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        SimpleNamespace(evaluate=AsyncMock(return_value=ReplyDecision(True))),
        logging.getLogger(),
    )
    try:
        active = await scheduler.submit(Event(), 1, True)
        await asyncio.wait_for(active.ready, 1)
        pending = await scheduler.submit(Event(), 2, False)
        active.event.set_extra(flag, True)
        assert scheduler.claim_recovery_messages(active) is None
        assert not pending.ready.done()
    finally:
        await scheduler.close()


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
async def test_zero_cooldown_allows_the_next_group_batch_after_a_reply():
    judge = SimpleNamespace(should_reply=AsyncMock(return_value=True))
    scheduler = ChatScheduler(
        Settings(debounce_seconds=0, cooldown_seconds=0),
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


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(
        "mingle_tests_plugin.src.scheduler.monotonic", lambda: clock.now
    )
    return clock


@pytest.mark.asyncio
async def test_cooldown_expiry_automatically_judges_once_without_new_arrivals():
    from time import monotonic

    judged_at = []

    async def should_reply(*args, **kwargs):
        judged_at.append(monotonic())
        return False

    started = monotonic()
    scheduler = ChatScheduler(
        Settings(debounce_seconds=0, cooldown_seconds=0.03),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        SimpleNamespace(should_reply=should_reply),
        logging.getLogger(),
    )
    scheduler.sessions["group"] = SessionState(last_agent_finished_at=started)
    try:
        ticket = await scheduler.submit(Event(), 1, False)
        assert not ticket.ready.done()
        assert await asyncio.wait_for(ticket.ready, 1) is None
        assert len(judged_at) == 1
        assert judged_at[0] >= started + 0.03
        await asyncio.sleep(0.04)
        assert len(judged_at) == 1
        assert not scheduler.sessions["group"].pending
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
        await asyncio.wait_for(state.judgment_task, 1)
        assert not second.ready.done()
        assert not first.finished.is_set()
        status = judge.evaluate.call_args.kwargs["message_status"]
        assert status["active"] == [1]
        assert "active_context" not in judge.evaluate.call_args.kwargs
        assert status["judging"] == [2]
        third = await scheduler.submit(Event(), 3, False)
        await asyncio.wait_for(state.judgment_task, 1)
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
        await asyncio.wait_for(state.judgment_task, 1)
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
        await asyncio.wait_for(state.judgment_task, 1)
        stop = await scheduler.submit(Event(), 3, True)
        await asyncio.wait_for(state.judgment_task, 1)
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
        await asyncio.wait_for(state.judgment_task, 1)
        third = await scheduler.submit(Event(), 3, False)
        await asyncio.wait_for(started.wait(), 1)
        first.finished.set()
        assert await asyncio.wait_for(second.ready, 1) == [2]
        release.set()
        await asyncio.wait_for(state.judgment_task, 1)
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
    assert state.judgment_task.done() and state.execution_task.done()


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
        Settings(debounce_seconds=0, initial_context_messages=1),
        SimpleNamespace(get=get),
        judge,
        logging.getLogger(),
    )
    try:
        first = await scheduler.submit(Event(), 1, True)
        assert await asyncio.wait_for(first.ready, 1) == [1]
        second = await scheduler.submit(Event(), 2, False)
        await asyncio.wait_for(scheduler.sessions["group"].judgment_task, 1)
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


@pytest.mark.asyncio
async def test_cooldown_preserves_new_arrival_debounce_and_merges_the_batch(clock):
    judge = SimpleNamespace(should_reply=AsyncMock(return_value=True))
    scheduler = ChatScheduler(
        Settings(),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        judge,
        logging.getLogger(),
    )
    state = SessionState(last_agent_finished_at=clock.now)
    scheduler.sessions["group"] = state
    try:
        first = await scheduler.submit(Event(), 1, False)
        clock.now = 102.5
        second = await scheduler.submit(Event(), 2, False)
        clock.now = 103
        state.changed.set()
        await asyncio.sleep(0.01)
        assert not second.ready.done()
        judge.should_reply.assert_not_awaited()
        clock.now = 104.5
        state.changed.set()
        assert await asyncio.wait_for(second.ready, 1) == [1, 2]
        assert await first.ready is None
        assert judge.should_reply.call_args.kwargs["new_message_ids"] == [1, 2]
        second.finished.set()
    finally:
        await scheduler.close()


@pytest.mark.asyncio
async def test_continuous_arrivals_obey_maximum_merge_wait_during_cooldown(clock):
    judge = SimpleNamespace(should_reply=AsyncMock(return_value=True))
    scheduler = ChatScheduler(
        Settings(),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        judge,
        logging.getLogger(),
    )
    state = SessionState(last_agent_finished_at=clock.now)
    scheduler.sessions["group"] = state
    try:
        tickets = []
        for number, arrival in enumerate((100, 101.9, 103.8, 105.7), 1):
            clock.now = arrival
            tickets.append(await scheduler.submit(Event(), number, False))
        clock.now = 105.9
        state.changed.set()
        await asyncio.sleep(0.01)
        judge.should_reply.assert_not_awaited()
        clock.now = 106
        state.changed.set()
        assert await asyncio.wait_for(tickets[-1].ready, 1) == [1, 2, 3, 4]
        tickets[-1].finished.set()
    finally:
        await scheduler.close()


@pytest.mark.asyncio
async def test_maximum_merge_wait_does_not_bypass_cooldown_or_restart_after_expiry(
    clock,
):
    judge = SimpleNamespace(should_reply=AsyncMock(return_value=True))
    scheduler = ChatScheduler(
        Settings(cooldown_seconds=10),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        judge,
        logging.getLogger(),
    )
    state = SessionState(last_agent_finished_at=clock.now)
    scheduler.sessions["group"] = state
    try:
        ticket = await scheduler.submit(Event(), 1, False)
        clock.now = 106
        state.changed.set()
        await asyncio.sleep(0.01)
        assert not ticket.ready.done()
        judge.should_reply.assert_not_awaited()
        clock.now = 110
        state.changed.set()
        assert await asyncio.wait_for(ticket.ready, 1) == [1]
        ticket.finished.set()
    finally:
        await scheduler.close()


@pytest.mark.asyncio
async def test_direct_address_bypasses_cooldown_but_preserves_merge_wait(clock):
    judge = SimpleNamespace(should_reply=AsyncMock(return_value=False))
    scheduler = ChatScheduler(
        Settings(cooldown_seconds=60),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        judge,
        logging.getLogger(),
    )
    state = SessionState(last_agent_finished_at=clock.now)
    scheduler.sessions["group"] = state
    try:
        first = await scheduler.submit(Event(), 1, False)
        clock.now = 101
        direct = await scheduler.submit(Event(), 2, True)
        clock.now = 102
        state.changed.set()
        await asyncio.sleep(0.01)
        assert not direct.ready.done()
        clock.now = 103
        state.changed.set()
        assert await asyncio.wait_for(direct.ready, 1) == [1, 2]
        assert await first.ready is None
        judge.should_reply.assert_not_awaited()
        direct.finished.set()
    finally:
        await scheduler.close()


@pytest.mark.asyncio
async def test_busy_judgment_bypasses_cooldown(clock):
    judge = SimpleNamespace(evaluate=AsyncMock(return_value=ReplyDecision(True)))
    scheduler = ChatScheduler(
        Settings(debounce_seconds=0, cooldown_seconds=60),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        judge,
        logging.getLogger(),
    )
    state = SessionState(last_agent_finished_at=clock.now)
    scheduler.sessions["group"] = state
    try:
        first = await scheduler.submit(Event(), 1, True)
        assert await asyncio.wait_for(first.ready, 1) == [1]
        second = await scheduler.submit(Event(), 2, False)
        await asyncio.wait_for(state.judgment_task, 1)
        judge.evaluate.assert_awaited_once()
        assert not second.ready.done()
        first.finished.set()
        assert await asyncio.wait_for(second.ready, 1) == [2]
        second.finished.set()
    finally:
        await scheduler.close()


@pytest.mark.asyncio
async def test_agent_finishing_during_history_read_defers_batch_until_cooldown(clock):
    reading, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def get(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            reading.set()
            await release.wait()
        return []

    judge = SimpleNamespace(
        should_reply=AsyncMock(return_value=True), evaluate=AsyncMock()
    )
    scheduler = ChatScheduler(
        Settings(debounce_seconds=0),
        SimpleNamespace(get=get),
        judge,
        logging.getLogger(),
    )
    try:
        first = await scheduler.submit(Event(), 1, True)
        assert await asyncio.wait_for(first.ready, 1) == [1]
        first.event.set_extra("mingle_agent_started", True)
        second = await scheduler.submit(Event(), 2, False)
        await asyncio.wait_for(reading.wait(), 1)
        state = scheduler.sessions["group"]
        first.finished.set()
        await asyncio.wait_for(state.execution_task, 1)
        release.set()
        await asyncio.sleep(0.01)
        assert not second.ready.done()
        judge.should_reply.assert_not_awaited()
        judge.evaluate.assert_not_awaited()
        clock.now += 3
        state.changed.set()
        assert await asyncio.wait_for(second.ready, 1) == [2]
        judge.should_reply.assert_awaited_once()
        second.finished.set()
    finally:
        release.set()
        await scheduler.close()


@pytest.mark.asyncio
async def test_shutdown_releases_events_waiting_for_cooldown():
    from time import monotonic

    judge = SimpleNamespace(should_reply=AsyncMock())
    scheduler = ChatScheduler(
        Settings(debounce_seconds=0, cooldown_seconds=60),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        judge,
        logging.getLogger(),
    )
    scheduler.sessions["group"] = SessionState(last_agent_finished_at=monotonic())
    ticket = await scheduler.submit(Event(), 1, False)
    await asyncio.sleep(0)
    assert not ticket.ready.done()
    await scheduler.close()
    assert await ticket.ready is None
    judge.should_reply.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_source", ["history", "judge"])
async def test_worker_failure_releases_waiters_and_next_submission_can_restart(
    failure_source,
):
    started, release = asyncio.Event(), asyncio.Event()

    async def fail(*args, **kwargs):
        started.set()
        await release.wait()
        raise RuntimeError("Injected scheduling failure")

    history = SimpleNamespace(get=AsyncMock(return_value=[]))
    judge = SimpleNamespace(should_reply=AsyncMock(return_value=True))
    failing_call = history.get if failure_source == "history" else judge.should_reply
    failing_call.side_effect = fail
    scheduler = ChatScheduler(
        Settings(debounce_seconds=0), history, judge, logging.getLogger()
    )
    try:
        first = await scheduler.submit(Event(), 1, False)
        second = await scheduler.submit(Event(), 2, False)
        await asyncio.wait_for(started.wait(), 1)
        pending = await scheduler.submit(Event(), 3, False)
        state = scheduler.sessions["group"]
        release.set()
        await asyncio.wait_for(state.judgment_task, 1)
        for ticket in (first, second, pending):
            assert await asyncio.wait_for(ticket.ready, 1) is None
        assert not state.judging
        assert not state.pending
        assert not state.approved
        assert state.active is None

        failing_call.side_effect = None
        retry = await scheduler.submit(Event(), 4, False)
        assert await asyncio.wait_for(retry.ready, 1) == [4]
        retry.finished.set()
    finally:
        release.set()
        await scheduler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stopped_by", ["event", "agent", "cancelled_ticket"])
async def test_approved_limit_and_stopped_tickets_do_not_block_remaining_work(
    stopped_by,
):
    scheduler = ChatScheduler(
        Settings(debounce_seconds=0, max_pending_messages=2),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        SimpleNamespace(evaluate=AsyncMock(return_value=ReplyDecision(True))),
        logging.getLogger(),
    )
    try:
        active = await scheduler.submit(Event(), 1, True)
        assert await asyncio.wait_for(active.ready, 1) == [1]
        state = scheduler.sessions["group"]
        approved = []
        for record_id in (2, 3, 4):
            approved.append(await scheduler.submit(Event(), record_id, False))
            await asyncio.wait_for(state.judgment_task, 1)
        oldest, stopped, newest = approved
        assert await asyncio.wait_for(oldest.ready, 1) is None
        assert state.finished[2] == "merged"
        assert [item.record_id for item in state.approved] == [3, 4]
        assert not newest.ready.done()

        if stopped_by == "event":
            stopped.event.stop_event()
        elif stopped_by == "agent":
            stopped.event.set_extra("agent_stop_requested", True)
        else:
            stopped.ready.cancel()
        active.finished.set()
        assert await asyncio.wait_for(newest.ready, 1) == [4]
        if stopped_by == "cancelled_ticket":
            assert stopped.ready.cancelled()
        else:
            assert await asyncio.wait_for(stopped.ready, 1) is None
        assert state.active_ids == [4]
        newest.finished.set()
    finally:
        await scheduler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True])
@pytest.mark.parametrize("reply", [False, True])
async def test_wait_result_folds_only_current_pending_tail(busy, reply):
    started, release, judged_again = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []

    async def evaluate(*args, **kwargs):
        calls.append(kwargs["new_message_ids"])
        if len(calls) == 1:
            started.set()
            await release.wait()
            return ReplyDecision(reply, "wait")
        judged_again.set()
        return ReplyDecision(False)

    async def should_reply(*args, **kwargs):
        return (await evaluate(*args, **kwargs)).should_reply

    scheduler = ChatScheduler(
        Settings(debounce_seconds=0, cooldown_seconds=0),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        SimpleNamespace(evaluate=evaluate, should_reply=should_reply),
        logging.getLogger(),
    )
    try:
        active = None
        if busy:
            active = await scheduler.submit(Event(), 1, True)
            assert await asyncio.wait_for(active.ready, 1) == [1]
        first = await scheduler.submit(Event(), 2, False)
        await asyncio.wait_for(started.wait(), 1)
        tail = await scheduler.submit(Event(), 3, False)
        release.set()
        state = scheduler.sessions["group"]
        await asyncio.wait_for(state.judgment_task, 1)
        if reply:
            assert calls == [[2]]
            if busy:
                assert [item.record_id for item in state.approved] == [2, 3]
                assert not tail.ready.done()
                assert not active.event.get_extra("agent_stop_requested", False)
                active.finished.set()
            assert await asyncio.wait_for(tail.ready, 1) == [2, 3]
            assert await asyncio.wait_for(first.ready, 1) is None
            # The cutoff is fixed; a later arrival must get its own judgment.
            later = await scheduler.submit(Event(), 4, False)
            await asyncio.wait_for(judged_again.wait(), 1)
            assert calls == [[2], [4]]
            assert await asyncio.wait_for(later.ready, 1) is None
            tail.finished.set()
        else:
            assert calls == [[2], [3]]
            assert await asyncio.wait_for(first.ready, 1) is None
            assert await asyncio.wait_for(tail.ready, 1) is None
    finally:
        await scheduler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "decision",
    [ReplyDecision(True, "rerun"), ReplyDecision(False, "stop"), ReplyDecision(False)],
)
async def test_control_and_forced_negative_results_do_not_fold_pending(decision):
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def evaluate(*args, **kwargs):
        calls.append(kwargs["new_message_ids"])
        if len(calls) == 1:
            started.set()
            await release.wait()
            return decision
        return ReplyDecision(False)

    scheduler = ChatScheduler(
        Settings(debounce_seconds=0, cooldown_seconds=0),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        SimpleNamespace(evaluate=evaluate),
        logging.getLogger(),
    )
    try:
        active = await scheduler.submit(Event(), 1, True)
        assert await asyncio.wait_for(active.ready, 1) == [1]
        await scheduler.submit(Event(), 2, True)
        await asyncio.wait_for(started.wait(), 1)
        tail = await scheduler.submit(Event(), 3, False)
        release.set()
        await asyncio.wait_for(scheduler.sessions["group"].judgment_task, 1)
        assert calls == [[2], [3]]
        assert await asyncio.wait_for(tail.ready, 1) is None
    finally:
        await scheduler.close()
