"""Regression coverage for the unified coordinator and background IO boundaries."""

import asyncio
import logging
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from mingle_tests_plugin.src.decision import ReplyDecision
from mingle_tests_plugin.src.scheduler import ChatScheduler, SessionState
from mingle_tests_plugin.src.settings import Settings
from scheduler_helpers import wait_for_execution, wait_for_judgment
from test_scheduler import Event


@pytest.mark.asyncio
async def test_idle_coordinator_waits_without_polling_and_reuses_its_loop():
    scheduler = ChatScheduler(
        Settings(debounce_seconds=0, cooldown_seconds=0),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        SimpleNamespace(should_reply=AsyncMock(return_value=True)),
        logging.getLogger(),
    )
    scheduler.sessions["group"] = SessionState()
    try:
        first = await scheduler.submit(Event(), 1, False)
        assert await asyncio.wait_for(first.ready, 1) == [1]
        state = scheduler.sessions["group"]
        owner = scheduler._coordinator("group")
        loop_task = state.coordinator_task
        first.finished.set()
        await wait_for_execution(state)
        await asyncio.sleep(0.01)
        with patch.object(
            owner, "next_deadline", wraps=owner.next_deadline
        ) as calculate:
            await asyncio.sleep(0.03)
            calculate.assert_not_called()
            second = await scheduler.submit(Event(), 2, False)
            assert await asyncio.wait_for(second.ready, 1) == [2]
            assert state.coordinator_task is loop_task
            assert calculate.called
            second.finished.set()
    finally:
        await scheduler.close()


@pytest.mark.asyncio
async def test_blocked_dispatch_does_not_block_normal_admission_or_group_removal(
    monkeypatch,
):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(
        "mingle_tests_plugin.src.scheduling.timing.monotonic", lambda: clock.now
    )
    monkeypatch.setattr(
        "mingle_tests_plugin.src.scheduling.timing.expovariate", lambda _: 10
    )
    dispatching = asyncio.Event()

    async def dispatch(*args):
        dispatching.set()
        await asyncio.Event().wait()

    scheduler = ChatScheduler(
        Settings(
            random_bubble_enabled=True,
            random_bubble_min_minutes=0,
            debounce_seconds=0,
            cooldown_seconds=0,
        ),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        None,
        logging.getLogger(),
        queue_bubble=dispatch,
    )
    scheduler.sessions["group"] = SessionState()
    try:
        scheduler.watch_random_bubbles("group")
        state = scheduler.sessions["group"]
        clock.now = 111
        state.changed.set()
        await asyncio.wait_for(dispatching.wait(), 1)
        dispatch_task = state.dispatch_task
        ticket = await scheduler.submit(Event(), 1, True)
        assert await asyncio.wait_for(ticket.ready, 1) == [1]
        await scheduler.stop_random_bubbles("group")
        assert dispatch_task.cancelled()
        assert not state.coordinator_task.done()
        assert state.active is ticket
        assert not ticket.event.is_stopped()
        ticket.finished.set()
    finally:
        await scheduler.close()


@pytest.mark.asyncio
async def test_failed_proactive_dispatch_consumes_opportunity_until_next_send(
    monkeypatch,
):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(
        "mingle_tests_plugin.src.scheduling.timing.monotonic", lambda: clock.now
    )
    monkeypatch.setattr(
        "mingle_tests_plugin.src.scheduling.timing.expovariate", lambda _: 10
    )
    dispatch = AsyncMock(side_effect=RuntimeError("Injected dispatch failure"))
    scheduler = ChatScheduler(
        Settings(random_bubble_enabled=True, random_bubble_min_minutes=0),
        None,
        None,
        logging.getLogger(),
        queue_bubble=dispatch,
    )
    try:
        scheduler.watch_random_bubbles("group")
        state = scheduler.sessions["group"]
        clock.now = 111
        state.changed.set()
        await asyncio.sleep(0.02)
        dispatch.assert_awaited_once()
        assert state.next_bubble_at == float("inf")
        state.changed.set()
        await asyncio.sleep(0.02)
        dispatch.assert_awaited_once()
        scheduler.record_agent_reply("group")
        clock.now = 122
        state.changed.set()
        await asyncio.sleep(0.02)
        assert dispatch.await_count == 2
    finally:
        await scheduler.close()


@pytest.mark.asyncio
async def test_busy_bubble_clock_still_resamples_when_due_during_quiet_hours(
    monkeypatch,
):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(
        "mingle_tests_plugin.src.scheduling.timing.monotonic", lambda: clock.now
    )
    monkeypatch.setattr(
        "mingle_tests_plugin.src.scheduling.timing.expovariate", lambda _: 10
    )
    scheduler = ChatScheduler(
        Settings(
            random_bubble_enabled=True,
            random_bubble_min_minutes=0,
            debounce_seconds=0,
        ),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        None,
        logging.getLogger(),
        queue_bubble=AsyncMock(),
    )
    scheduler.sessions["group"] = SessionState()
    try:
        scheduler.watch_random_bubbles("group")
        ticket = await scheduler.submit(Event(), 1, True)
        assert await asyncio.wait_for(ticket.ready, 1) == [1]
        state = scheduler.sessions["group"]
        owner = scheduler._coordinator("group")
        assert owner.next_deadline() == 110
        with patch.object(scheduler, "_quiet_seconds_remaining", return_value=600):
            clock.now = 111
            state.changed.set()
            await asyncio.sleep(0.01)
            assert state.next_bubble_at == 721
            assert state.last_agent_reply_at == 100
            assert owner.next_deadline() == 721
            scheduler.queue_bubble.assert_not_awaited()
        ticket.finished.set()
    finally:
        await scheduler.close()


@pytest.mark.asyncio
async def test_replaced_settings_are_used_by_existing_coordinator():
    scheduler = ChatScheduler(
        Settings(debounce_seconds=60, cooldown_seconds=0),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        None,
        logging.getLogger(),
    )
    scheduler.sessions["group"] = SessionState()
    try:
        ticket = await scheduler.submit(Event(), 1, True)
        await asyncio.sleep(0.01)
        assert not ticket.ready.done()
        scheduler.settings = replace(scheduler.settings, debounce_seconds=0)
        scheduler.sessions["group"].changed.set()
        assert await asyncio.wait_for(ticket.ready, 1) == [1]
        ticket.finished.set()
    finally:
        await scheduler.close()


@pytest.mark.asyncio
async def test_submission_after_shutdown_does_not_recreate_session():
    scheduler = ChatScheduler(Settings(), None, None, logging.getLogger())
    await scheduler.close()
    ticket = await scheduler.submit(Event(), 1, True)
    assert await ticket.ready is None
    assert not scheduler.sessions


@pytest.mark.asyncio
async def test_shutdown_releases_pending_proactive_admission():
    scheduler = ChatScheduler(
        Settings(random_bubble_enabled=True, debounce_seconds=0),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        None,
        logging.getLogger(),
    )
    scheduler.sessions["group"] = SessionState()
    active = await scheduler.submit(Event(), 1, True)
    await asyncio.wait_for(active.ready, 1)
    waiting = asyncio.create_task(scheduler.admit_random_bubble(Event(), 0))
    await asyncio.sleep(0)
    assert not waiting.done()
    await scheduler.close()
    assert await asyncio.wait_for(waiting, 1) is None
    assert active.event.is_stopped()


@pytest.mark.asyncio
async def test_recovery_preserves_sleep_check_after_waiting_for_proactive_turn(
    monkeypatch,
):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(
        "mingle_tests_plugin.src.scheduling.timing.monotonic", lambda: clock.now
    )
    scheduler = ChatScheduler(
        Settings(random_bubble_enabled=True, debounce_seconds=0, sleep_minutes=0.1),
        SimpleNamespace(get=AsyncMock(return_value=[])),
        SimpleNamespace(evaluate=AsyncMock(return_value=ReplyDecision(True))),
        logging.getLogger(),
    )
    scheduler.sessions["group"] = SessionState()
    try:
        state = scheduler.sessions["group"]
        bubble = await scheduler.admit_random_bubble(Event(), state.bubble_revision)
        following = await scheduler.submit(Event(), 1, False)
        await wait_for_judgment(state)
        assert state.approved == [following]
        assert scheduler.claim_recovery_messages(bubble) == [1]
        assert not state.approved
        clock.now = 107
        bubble.finished.set()
        scheduler.finish_random_bubble(bubble)
        await asyncio.sleep(0.01)
        assert state.sleeping
        assert state.next_sleep_backlog_at == 1907
    finally:
        await scheduler.close()
