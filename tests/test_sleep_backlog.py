import asyncio
import json
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
from astrbot.core.message.components import At, Plain
from astrbot.core.provider.entities import ProviderRequest
from mingle_tests_plugin.src.scheduler import ChatScheduler
from scheduler_helpers import wait_for_execution, wait_for_judgment
from test_pipeline import make_event
from test_pipeline import plugin as plugin


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("trigger", "isolated"),
    [
        ("wake", False),
        ("wake", True),
        ("review", False),
        ("review", True),
        ("bubble", False),
    ],
)
async def test_sleep_backlog_survives_reload_and_uses_normal_pipeline(
    plugin, trigger, isolated
):
    plugin.settings = replace(
        plugin.settings,
        sleep_backlog_minutes=0,
        initial_context_messages=1,
        random_bubble_enabled=trigger == "bubble",
    )
    plugin.scheduler.settings = plugin.settings
    plugin.context.get_config = lambda **kwargs: {
        "platform_settings": {"unique_session": isolated}
    }
    source = make_event()
    plugin.context.get_platform_inst = lambda _: SimpleNamespace(
        meta=lambda: source.platform_meta
    )
    queue = asyncio.Queue()
    plugin.context.get_event_queue = lambda: queue
    record_ids = []
    for number in range(3):
        event = make_event(str(number))
        if isolated:
            event.session.session_id = "alice_group"
        record_id = await plugin.history.record(event, event.get_messages())
        record_ids.append(record_id)
        await plugin.scheduler.submit(event, record_id, False)
    key = event.unified_msg_origin
    await asyncio.sleep(0)
    await plugin.scheduler.flush_sleep_backlog()
    assert [
        row["record_id"] for row in await plugin.history.get_sleep_backlog()
    ] == record_ids
    # Close both owners, then reopen the same SQLite files as a plugin reload does.
    await plugin.scheduler.close()
    await plugin.history.close()
    await plugin.history.initialize()
    plugin.settings = replace(
        plugin.settings,
        sleep_backlog_minutes=0.0005 if trigger == "review" else 0,
    )
    plugin.scheduler = ChatScheduler(
        plugin.settings,
        plugin.history,
        plugin.scheduler.judge,
        plugin.logger,
        save_sleep_backlog=plugin.history.save_sleep_backlog,
        queue_bubble=plugin._queue_random_bubble,
    )
    plugin.context.llm_generate.return_value.completion_text = '{"should_reply": true}'
    await plugin._restore_sleep_backlog()
    restored = await asyncio.wait_for(queue.get(), 1)
    assert restored.unified_msg_origin == key
    assert restored.get_group_id() == "group"
    assert restored.get_self_id() == "bot"
    assert restored.role == "member"
    assert restored.get_sender_id() == "mingle_scheduler"
    assert not restored.is_at_or_wake_command
    restored_handler = plugin.on_message(restored)
    restored_request = asyncio.create_task(anext(restored_handler, None))
    handler = None
    try:
        if trigger == "review":
            request = await asyncio.wait_for(restored_request, 1)
        else:
            if trigger == "bubble":
                state = plugin.scheduler.sessions[key]
                await plugin._queue_random_bubble(key, state.bubble_revision)
                following = await asyncio.wait_for(queue.get(), 1)
            else:
                following = make_event("wake")
                if isolated:
                    following.session.session_id = "alice_group"
                following.message_obj.message.append(At(qq="bot"))
                following.is_at_or_wake_command = True
            handler = plugin.on_message(following)
            request = await asyncio.wait_for(anext(handler), 1)
            assert await asyncio.wait_for(restored_request, 1) is None
        assert isinstance(request, ProviderRequest)
        state = plugin.scheduler.sessions[key]
        assert state.active_ids[:3] == record_ids
        assert all(f"message {number}" in request.prompt for number in range(3))
        assert await plugin.history.get_sleep_backlog() == []
        assert len(await plugin.history.get(key)) == (4 if trigger == "wake" else 3)
        assert restored.get_extra("enable_streaming") is (
            False if trigger == "review" else None
        )
        if trigger != "bubble":
            judgment = json.loads(
                plugin.context.llm_generate.await_args.kwargs["prompt"]
            )
            statuses = [row["status"] for row in judgment["messages"]]
            assert statuses[-1] == "new"
            assert statuses.count("new") == 1
            assert plugin.context.llm_generate.await_count == 1
        else:
            plugin.context.llm_generate.assert_not_awaited()
    finally:
        restored_request.cancel()
        await asyncio.gather(restored_request, return_exceptions=True)
        if handler is not None:
            await handler.aclose()
        await restored_handler.aclose()
    await wait_for_execution(state)
    await plugin.scheduler.close()
    await plugin.history.close()
    await plugin.history.initialize()
    assert await plugin.history.get_sleep_backlog() == []


@pytest.mark.asyncio
async def test_declined_restored_backlog_finishes_once(plugin):
    event = make_event()
    record_id = await plugin.history.record(event, event.get_messages())
    await plugin.history.save_sleep_backlog(
        event.unified_msg_origin, [(record_id, "group", "bot", time.time())]
    )
    plugin.context.get_platform_inst = lambda _: SimpleNamespace(
        meta=lambda: event.platform_meta
    )
    queue = asyncio.Queue()
    plugin.context.get_event_queue = lambda: queue
    plugin.scheduler.settings = replace(plugin.settings, sleep_backlog_minutes=0.0005)
    await plugin._restore_sleep_backlog()
    restored = await queue.get()
    handler = plugin.on_message(restored)
    try:
        assert await asyncio.wait_for(anext(handler, None), 1) is None
        state = plugin.scheduler.sessions[event.unified_msg_origin]
        await wait_for_judgment(state)
        await plugin.scheduler.flush_sleep_backlog()
        assert await plugin.history.get_sleep_backlog() == []
        assert state.finished[record_id] == "no_reply"
        assert state.sleeping
        state.changed.set()
        await asyncio.sleep(0)
        plugin.context.llm_generate.assert_awaited_once()
    finally:
        await handler.aclose()


@pytest.mark.asyncio
async def test_backlog_write_finishes_before_agent_request(plugin):
    started = asyncio.Event()
    release = asyncio.Event()
    save = plugin.history.save_sleep_backlog

    async def delayed_save(key, records):
        if records:
            started.set()
            await release.wait()
        await save(key, records)

    plugin.scheduler._save_sleep_backlog = delayed_save
    first_handler = plugin.on_message(make_event("sleeping"))
    first = asyncio.create_task(anext(first_handler, None))
    wake = make_event("wake")
    wake.message_obj.message.append(At(qq="bot"))
    wake.is_at_or_wake_command = True
    plugin.context.llm_generate.return_value.completion_text = '{"should_reply": true}'
    wake_handler = plugin.on_message(wake)
    request_task = None
    try:
        await asyncio.wait_for(started.wait(), 1)
        request_task = asyncio.create_task(anext(wake_handler))
        state = plugin.scheduler.sessions[wake.unified_msg_origin]
        async with asyncio.timeout(1):
            while state.active is None:  # noqa: ASYNC110 - Observe admission.
                await asyncio.sleep(0)
        assert not request_task.done()
        release.set()
        assert isinstance(await asyncio.wait_for(request_task, 1), ProviderRequest)
        assert await plugin.history.get_sleep_backlog() == []
    finally:
        release.set()
        for task in (first, request_task):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            first, *([request_task] if request_task else []), return_exceptions=True
        )
        await wake_handler.aclose()
        await first_handler.aclose()


@pytest.mark.asyncio
async def test_backlog_follows_message_deletion_retention_and_session_bounds(plugin):
    plugin.history.settings = replace(plugin.settings, max_messages_per_session=2)
    event = make_event("first")
    key = event.unified_msg_origin
    first = await plugin.history.record(event, event.get_messages())
    await plugin.history.save_sleep_backlog(key, [(first, "group", "bot", time.time())])
    second = await plugin.history.record(make_event("second"), [Plain("second")])
    third = await plugin.history.record(make_event("third"), [Plain("third")])
    assert await plugin.history.get_sleep_backlog() == []
    await plugin.history.save_sleep_backlog(
        key,
        [(second, "group", "bot", time.time()), (third, "group", "bot", time.time())],
    )
    await plugin.history.delete_record(key, second)
    assert [row["record_id"] for row in await plugin.history.get_sleep_backlog()] == [
        third
    ]
    await plugin.history.db.execute(
        "UPDATE messages SET timestamp=0 WHERE id=?", (third,)
    )
    await plugin.history.db.commit()
    await plugin.history.cleanup()
    assert await plugin.history.get_sleep_backlog() == []


@pytest.mark.asyncio
async def test_restored_backlog_claim_before_coordinator_runs_is_persisted(plugin):
    plugin.scheduler.settings = replace(
        plugin.settings, random_bubble_enabled=True, sleep_backlog_minutes=0
    )
    event = make_event()
    key = event.unified_msg_origin
    record_id = await plugin.history.record(event, event.get_messages())
    await plugin.history.save_sleep_backlog(
        key, [(record_id, "group", "bot", time.time())]
    )
    rows = await plugin.history.get_sleep_backlog()
    plugin.scheduler.restore_sleep_backlog(event, rows)
    bubble = await plugin.scheduler.admit_random_bubble(make_event("bubble"), 0)
    assert bubble.ready.result() == [record_id]
    await plugin.scheduler.flush_sleep_backlog(key)
    assert await plugin.history.get_sleep_backlog() == []


@pytest.mark.asyncio
async def test_restored_backlog_obeys_new_queue_limit(plugin):
    event = make_event()
    key = event.unified_msg_origin
    record_ids = []
    for number in range(4):
        row = await plugin.history.record(make_event(str(number)), [Plain(str(number))])
        record_ids.append(row)
    await plugin.history.save_sleep_backlog(
        key, [(row, "group", "bot", time.time() - 30) for row in record_ids]
    )
    plugin.scheduler.settings = replace(
        plugin.settings, max_pending_messages=2, sleep_backlog_minutes=0
    )
    plugin.scheduler.restore_sleep_backlog(
        event, await plugin.history.get_sleep_backlog()
    )
    state = plugin.scheduler.sessions[key]
    assert [ticket.record_id for ticket in state.pending] == record_ids[-2:]
    await plugin.scheduler.close()
    restored_rows = await plugin.history.get_sleep_backlog()
    assert [row["record_id"] for row in restored_rows] == record_ids[-2:]
    assert all(row["received_at"] < time.time() - 20 for row in restored_rows)


@pytest.mark.asyncio
async def test_failed_backlog_removal_blocks_admission_and_can_retry(plugin):
    plugin.scheduler.settings = replace(
        plugin.settings, random_bubble_enabled=True, sleep_backlog_minutes=0
    )
    event = make_event()
    key = event.unified_msg_origin
    record_id = await plugin.history.record(event, event.get_messages())
    await plugin.history.save_sleep_backlog(
        key, [(record_id, "group", "bot", time.time())]
    )
    plugin.scheduler.restore_sleep_backlog(
        event, await plugin.history.get_sleep_backlog()
    )
    await plugin.scheduler.admit_random_bubble(make_event("bubble"), 0)
    save = plugin.scheduler._save_sleep_backlog

    async def fail_save(key, records):
        raise OSError("Storage unavailable")

    plugin.scheduler._save_sleep_backlog = fail_save
    with pytest.raises(RuntimeError, match="persistence failed"):
        await plugin.scheduler.flush_sleep_backlog(key)
    assert [row["record_id"] for row in await plugin.history.get_sleep_backlog()] == [
        record_id
    ]
    plugin.scheduler._save_sleep_backlog = save
    await plugin.scheduler.flush_sleep_backlog(key)
    assert await plugin.history.get_sleep_backlog() == []


@pytest.mark.asyncio
async def test_waking_backlog_is_retained_until_judgment_finishes(plugin):
    judge_started = asyncio.Event()
    release = asyncio.Event()

    async def judge(**kwargs):
        judge_started.set()
        await release.wait()
        return SimpleNamespace(completion_text='{"should_reply": false}')

    plugin.context.llm_generate.side_effect = judge
    event = make_event("sleeping")
    key = event.unified_msg_origin
    first_handler = plugin.on_message(event)
    first = asyncio.create_task(anext(first_handler, None))
    wake = make_event("wake")
    wake.message_obj.message.append(At(qq="bot"))
    wake.is_at_or_wake_command = True
    wake_handler = plugin.on_message(wake)
    waking = asyncio.create_task(anext(wake_handler, None))
    try:
        await asyncio.wait_for(judge_started.wait(), 1)
        await plugin.scheduler.flush_sleep_backlog(key)
        rows = await plugin.history.get_sleep_backlog()
        assert len(rows) == 1
        stored = await plugin.history.get(key)
        assert rows[0]["record_id"] == stored[0]["id"]
        assert not plugin.scheduler.sessions[key].sleeping
        release.set()
        assert await asyncio.wait_for(waking, 1) is None
        await plugin.scheduler.flush_sleep_backlog(key)
        assert await plugin.history.get_sleep_backlog() == []
    finally:
        release.set()
        first.cancel()
        waking.cancel()
        await asyncio.gather(first, waking, return_exceptions=True)
        await first_handler.aclose()
        await wake_handler.aclose()
