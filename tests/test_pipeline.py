import asyncio
import json
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio
from astrbot.api import AstrBotConfig
from astrbot.core.message.components import Plain
from astrbot.core.message.message_event_result import ResultContentType
from astrbot.core.pipeline.context_utils import call_handler
from astrbot.core.pipeline.process_stage.stage import ProcessStage
from astrbot.core.pipeline.scheduler import PipelineScheduler
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from astrbot.core.star.filter.command import CommandFilter
from mingle_tests_plugin.main import MinglePlugin
from mingle_tests_plugin.src.debug import DebugDumper
from mingle_tests_plugin.src.decision import ReplyJudge
from mingle_tests_plugin.src.history import ChatHistory
from mingle_tests_plugin.src.scheduler import ChatScheduler
from mingle_tests_plugin.src.settings import Settings
from mingle_tests_plugin.src.utils.history_formatting import format_prompt_history


def make_event(number="1", private=False):
    message = AstrBotMessage()
    message.type = MessageType.FRIEND_MESSAGE if private else MessageType.GROUP_MESSAGE
    message.self_id = "bot"
    message.sender = MessageMember("alice", "Alice")
    message.group_id = "" if private else "group"
    message.message_id = number
    message.message = [Plain(f"message {number}")]
    message.message_str = f"message {number}"
    event = AstrMessageEvent(
        message.message_str,
        message,
        PlatformMetadata("test", "Test", id="p"),
        "alice" if private else "group",
    )
    event.set_extra("activated_handlers", [])
    return event


@pytest_asyncio.fixture
async def plugin(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mingle_tests_plugin.main.SessionServiceManager.should_process_llm_request",
        AsyncMock(return_value=True),
    )
    instance = object.__new__(MinglePlugin)
    instance.logger = logging.getLogger("mingle.test")
    instance.settings = Settings(
        group_sessions=("group",), debounce_seconds=0, cooldown_seconds=0
    )
    instance.config = AstrBotConfig(
        str(tmp_path / "config.json"),
        default_config={
            "general": {
                "enabled": True,
                "private_enabled": True,
                "group_sessions": ["group"],
            },
            "debug": {"debug_sessions": []},
        },
    )
    instance.config_lock = asyncio.Lock()
    instance.debug_dumper = DebugDumper(
        tmp_path / "debug", instance.settings, instance.logger
    )
    instance.context = SimpleNamespace(
        get_config=lambda **kwargs: {"provider_settings": {"enable": True}},
        conversation_manager=SimpleNamespace(
            get_curr_conversation_id=AsyncMock(return_value="conversation"),
            get_conversation=AsyncMock(
                return_value=SimpleNamespace(
                    cid="conversation", history="[]", persona_id=None
                )
            ),
        ),
        persona_manager=SimpleNamespace(
            resolve_selected_persona=AsyncMock(return_value=(None, None, None, False))
        ),
        get_current_chat_provider_id=AsyncMock(return_value="default"),
        llm_generate=AsyncMock(
            return_value=SimpleNamespace(completion_text='{"should_reply": false}')
        ),
    )
    instance.history = ChatHistory(tmp_path, instance.settings, instance.logger)
    await instance.history.initialize()
    instance.scheduler = ChatScheduler(
        instance.settings,
        instance.history,
        ReplyJudge(
            instance.context, instance.settings, instance.logger, instance.debug_dumper
        ),
        instance.logger,
    )
    instance.available = True
    instance.cleanup_task = None
    instance.handlers = set()
    yield instance
    await instance.terminate()


@pytest.mark.asyncio
async def test_private_and_mentions_bypass_judgment_and_bind_conversation(plugin):
    for event in (make_event(private=True), make_event("2")):
        event.is_at_or_wake_command = True
        handler = plugin.on_message(event)
        request = await asyncio.wait_for(anext(handler), 1)
        assert isinstance(request, ProviderRequest)
        assert request.conversation.cid == "conversation"
        assert "message" in request.prompt
        assert request.contexts == []
        assert event.get_extra("mingle_managed") is True
        assert event.get_extra("enable_streaming") is False
        await handler.aclose()
    plugin.context.llm_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_declined_message_stops_fallback_and_is_still_queryable(plugin):
    event = make_event()
    assert [result async for result in plugin.on_message(event)] == []
    assert event.is_stopped()
    assert len(await plugin.history.get(event.unified_msg_origin)) == 1
    assert (
        json.loads(await plugin.chat_history(event))["messages"][0]["sender_id"]
        == "alice"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base,append", [(" Base {literal}\n", "\nAppend  "), ("", "Only append"), ("", "")]
)
@pytest.mark.parametrize("rerun", [False, True])
async def test_generator_prompt_uses_configured_text_verbatim(
    plugin, base, append, rerun
):
    plugin.settings = replace(
        plugin.settings, generator_prompt=base, generator_prompt_append=append
    )
    event = make_event(private=True)
    event.set_extra("mingle_resume_after_rerun", rerun)
    handler = plugin.on_message(event)
    try:
        request = await asyncio.wait_for(anext(handler), 1)
        records = await plugin.history.get(event.unified_msg_origin)
        record_id = records[0]["id"]
        expected = (
            f"New batch record IDs: [{record_id}]. Snapshot ends at record {record_id}.\n"
            + json.dumps(format_prompt_history(records, None), ensure_ascii=False)
        )
        if rerun:
            expected = (
                "The previous Agent turn was stopped because new messages corrected "
                "or redirected the task. Continue from its saved history, apply the "
                "new messages below, and reuse completed tool results when relevant.\n\n"
                + expected
            )
        assert request.prompt == expected
        request.system_prompt = "Persona instructions"
        await plugin.append_generator_instructions(event, request)
        assert request.system_prompt == (
            "Persona instructions" + ("\n\n" + base + append if base + append else "")
        )
        assert request.prompt == expected
        assert request.conversation.cid == "conversation"
        assert request.contexts == []
    finally:
        await handler.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "zone,expected_time",
    [
        ("Asia/Shanghai", "1970-01-01 08:00:00+08:00"),
        ("Invalid/Zone", "1970-01-01 00:00:00+00:00"),
        (None, "1970-01-01 00:00:00+00:00"),
    ],
)
async def test_generator_and_judge_share_chat_record_format(
    plugin, zone, expected_time
):
    plugin.context.get_config = lambda **kwargs: {
        "provider_settings": {"enable": True},
        "timezone": zone,
    }
    event = make_event(private=True)
    event.message_obj.timestamp = 0
    event.message_obj.sender.user_id = "full-sender-12345678"
    handler = plugin.on_message(event)
    try:
        request = await asyncio.wait_for(anext(handler), 1)
        original = await plugin.history.get(event.unified_msg_origin)
        await ReplyJudge(plugin.context, plugin.settings, plugin.logger).evaluate(
            event, original
        )
        judge_records = json.loads(
            plugin.context.llm_generate.call_args.kwargs["prompt"]
        )
        generator_records = json.loads(request.prompt.split("\n", 1)[1])
        assert generator_records == judge_records
        assert generator_records[0]["timestamp"] == expected_time
        assert generator_records[0]["sender_id"] == "12345678"
        assert "platform_message_id" not in generator_records[0]
        assert original[0]["timestamp"] == 0
        assert original[0]["sender_id"] == "full-sender-12345678"
        assert "platform_message_id" in original[0]
        assert await plugin.history.get(event.unified_msg_origin) == original
    finally:
        await handler.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("system_prompt", ["", None])
async def test_generator_instructions_support_empty_system_prompt(
    plugin, system_prompt
):
    event = make_event(private=True)
    event.set_extra("mingle_managed", True)
    request = ProviderRequest(prompt="Chat records")
    request.system_prompt = system_prompt
    await plugin.append_generator_instructions(event, request)
    assert request.system_prompt == plugin.settings.generator_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("managed,available", [(False, True), (True, False)])
async def test_generator_instructions_skip_unmanaged_or_disabled(
    plugin, managed, available
):
    event = make_event()
    event.set_extra("mingle_managed", managed)
    plugin.available = available
    request = ProviderRequest(prompt="Native input", system_prompt="Native persona")
    await plugin.append_generator_instructions(event, request)
    assert request.system_prompt == "Native persona"
    assert request.prompt == "Native input"


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [True, False])
async def test_commands_remain_native_even_outside_group_allowlist(plugin, configured):
    if not configured:
        plugin.settings = replace(plugin.settings, group_sessions=())
    event = make_event()
    event.is_at_or_wake_command = True
    event.set_extra(
        "activated_handlers", [SimpleNamespace(event_filters=[CommandFilter("stop")])]
    )
    assert [result async for result in plugin.on_message(event)] == []
    assert not event.is_stopped()
    assert not event.get_extra("mingle_managed")
    assert not event.call_llm
    assert not plugin.scheduler.sessions
    assert await plugin.history.get(event.unified_msg_origin) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("wake", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
async def test_nonconfigured_groups_block_native_fallback_and_ordinary_handlers(
    plugin, wake, enabled
):
    plugin.settings = replace(plugin.settings, enabled=enabled, group_sessions=())
    ordinary_calls = []
    agent_calls = []

    class PluginDispatch:
        async def process(self, event):
            async for value in call_handler(event, plugin.on_message):
                yield value
            if not event.is_stopped():
                ordinary_calls.append(event)

    class Agent:
        async def process(self, event):
            agent_calls.append(event)
            yield

    process = ProcessStage()
    process.ctx = SimpleNamespace(
        astrbot_config={"provider_settings": {"enable": True}}
    )
    process.star_request_sub_stage = PluginDispatch()
    process.agent_sub_stage = Agent()
    event = make_event()
    event.is_at_or_wake_command = wake
    event.set_extra("activated_handlers", [SimpleNamespace(event_filters=[])])
    async for _ in process.process(event):
        pass
    assert event.is_stopped() is enabled
    assert event.call_llm is enabled
    assert len(ordinary_calls) == (0 if enabled else 1)
    assert len(agent_calls) == (1 if not enabled and wake else 0)
    assert not event.get_extra("mingle_managed")
    assert not plugin.scheduler.sessions
    assert await plugin.history.get(event.unified_msg_origin) == []
    plugin.context.llm_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_default_pipeline_finishes_history_before_next_turn(plugin):
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    saved = []
    requests = []

    class PluginDispatch:
        async def process(self, event):
            async for value in call_handler(event, plugin.on_message):
                yield value

    class Agent:
        async def process(self, event):
            requests.append(event.get_extra("provider_request"))
            if event.message_obj.message_id == "1":
                first_started.set()
                await release_first.wait()
            else:
                assert saved == ["1"]
            event.set_result(event.plain_result("reply"))
            yield
            saved.append(event.message_obj.message_id)

    class Respond:
        async def process(self, event):
            await asyncio.sleep(0)

    process = ProcessStage()
    process.ctx = SimpleNamespace(
        astrbot_config={"provider_settings": {"enable": True}}
    )
    process.star_request_sub_stage = PluginDispatch()
    process.agent_sub_stage = Agent()
    scheduler = object.__new__(PipelineScheduler)
    scheduler.stages = [process, Respond()]
    a, b = make_event("1", private=True), make_event("2", private=True)
    a.is_at_or_wake_command = b.is_at_or_wake_command = True
    for event in (a, b):
        event.set_extra("activated_handlers", [SimpleNamespace(event_filters=[])])
    a_task = asyncio.create_task(scheduler._process_stages(a))
    await asyncio.wait_for(first_started.wait(), 1)
    b_task = asyncio.create_task(scheduler._process_stages(b))
    try:
        await asyncio.sleep(0.03)
        assert len(requests) == 1
        release_first.set()
        await asyncio.wait_for(asyncio.gather(a_task, b_task), 2)
        assert saved == ["1", "2"]
        assert len(requests) == 2
    finally:
        a_task.cancel()
        b_task.cancel()
        await asyncio.gather(a_task, b_task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("with_arrival", [False, True])
@pytest.mark.parametrize("recovery_final", ["<finished>", "Still not finished"])
async def test_protocol_recovery_saves_history_claims_arrivals_and_runs_only_once(
    plugin, with_arrival, recovery_final
):
    from astrbot.core.agent.message import Message, bind_checkpoint_messages
    from astrbot.core.agent.tool import ToolSet
    from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
        InternalAgentSubStage,
    )
    from astrbot.core.tools.message_tools import SendMessageToUserTool

    first_started, release_first = asyncio.Event(), asyncio.Event()
    arrival_judged = asyncio.Event()
    requests, saved = [], []
    turn_ids = []
    conversation = plugin.context.conversation_manager.get_conversation.return_value
    sender = SimpleNamespace(send_message=AsyncMock(return_value=True))
    tool = SendMessageToUserTool()
    saver = object.__new__(InternalAgentSubStage)
    saver.conv_manager = SimpleNamespace(update_conversation=AsyncMock())
    first = make_event("1", private=True)
    plugin.debug_dumper.settings = replace(
        plugin.settings, debug_sessions=(first.unified_msg_origin,)
    )

    async def judge(**kwargs):
        arrival_judged.set()
        return LLMResponse(role="assistant", completion_text='{"should_reply": true}')

    plugin.context.llm_generate.side_effect = judge

    class PluginDispatch:
        async def process(self, event):
            async for value in call_handler(event, plugin.on_message):
                yield value

    class Agent:
        async def process(self, event):
            request = event.get_extra("provider_request")
            requests.append(request)
            turn_ids.append(event.get_extra("mingle_turn_id"))
            request.func_tool = ToolSet([tool])
            await plugin.append_generator_instructions(event, request)
            await plugin.dump_agent_request(event, request)
            if len(requests) == 1:
                first_started.set()
                await release_first.wait()
                final_text = "Unsent final answer"
                sent_text = "First part"
            else:
                assert saved == ["Unsent final answer"]
                assert "Unsent final answer" in request.conversation.history
                assert request.contexts == []
                assert request.prompt.startswith(
                    "The previous turn did not follow the completion protocol: "
                    "end with exactly <finished>. Content in the final response "
                    "is not sent to the user. Review messages already sent "
                    "successfully and use the send_message_to_user tool to send "
                    "any remaining replies. Avoid duplicate messages and do not mention this "
                    "recovery run."
                )
                state = plugin.scheduler.sessions[event.unified_msg_origin]
                _, records_json = request.prompt.rsplit("\n", 1)
                records = json.loads(records_json)
                if with_arrival:
                    assert (
                        "Meanwhile, the following new messages have arrived."
                        in request.prompt
                    )
                    assert [row["text"] for row in records] == ["message 2"]
                    assert len(state.active_ids) == 2
                else:
                    assert (
                        "Meanwhile, the following new messages have arrived."
                        not in request.prompt
                    )
                    assert records == []
                    assert len(state.active_ids) == 1
                assert not state.active.finished.is_set()
                final_text = recovery_final
                sent_text = "Remaining answer"
            context = SimpleNamespace(
                context=SimpleNamespace(event=event, context=sender)
            )
            await tool.call(context, messages=[{"type": "plain", "text": sent_text}])
            response = LLMResponse(role="assistant", completion_text=final_text)
            runtime = bind_checkpoint_messages(json.loads(conversation.history))
            runtime.extend(
                [
                    Message(role="user", content=request.prompt),
                    Message(role="assistant", content=final_text),
                ]
            )
            await plugin.record_response(event, response)
            await plugin.dump_agent_response(event, response)
            await plugin.suppress_final_reply(event, response)
            assert response.completion_text == final_text
            event.set_result(
                event.plain_result(response.completion_text).set_result_content_type(
                    ResultContentType.LLM_RESULT
                )
            )
            yield
            await saver._save_to_history(event, request, response, runtime, None)
            conversation.history = json.dumps(
                saver.conv_manager.update_conversation.call_args.kwargs["history"]
            )
            saved.append(final_text)

    class Respond:
        async def process(self, event):
            await plugin.suppress_final_result(event)
            assert not event.get_result().chain

    process = ProcessStage()
    process.ctx = SimpleNamespace(
        astrbot_config={"provider_settings": {"enable": True}}
    )
    process.star_request_sub_stage = PluginDispatch()
    process.agent_sub_stage = Agent()
    pipeline = object.__new__(PipelineScheduler)
    pipeline.stages = [process, Respond()]
    first.set_extra("activated_handlers", [SimpleNamespace(event_filters=[])])
    tasks = [asyncio.create_task(pipeline._process_stages(first))]
    try:
        await asyncio.wait_for(first_started.wait(), 1)
        if with_arrival:
            second = make_event("2", private=True)
            second.set_extra("activated_handlers", [SimpleNamespace(event_filters=[])])
            tasks.append(asyncio.create_task(pipeline._process_stages(second)))
            state = plugin.scheduler.sessions[first.unified_msg_origin]
            await asyncio.wait_for(arrival_judged.wait(), 1)
            await asyncio.wait_for(state.judgment_task, 1)
        release_first.set()
        await asyncio.wait_for(asyncio.gather(*tasks), 2)
        assert len(requests) == 2
        assert len(set(turn_ids)) == 2
        assert saved == ["Unsent final answer", recovery_final]
        assert sender.send_message.await_count == 2
        rows = await plugin.history.get(first.unified_msg_origin)
        assert [row["text"] for row in rows if row["role"] == "bot"] == [
            "First part",
            "Remaining answer",
        ]
        outputs = list(plugin.debug_dumper.directory.rglob("*agent_output*.json"))
        assert len(outputs) == 2
        assert {
            json.loads(path.read_text(encoding="utf-8"))["payload"]["completion_text"]
            for path in outputs
        } == {"Unsent final answer", recovery_final}
    finally:
        release_first.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    [
        "finished",
        "error",
        "missing_tool",
        "inactive_tool",
        "stop",
        "aborted",
        "disabled",
        "group_off",
        "ai_off",
    ],
)
async def test_protocol_recovery_is_skipped_when_unnecessary_or_unavailable(
    plugin, reason
):
    from astrbot.core.agent.tool import ToolSet
    from astrbot.core.tools.message_tools import SendMessageToUserTool

    event = make_event()
    event.is_at_or_wake_command = True
    handler = plugin.on_message(event)
    try:
        request = await asyncio.wait_for(anext(handler), 1)
        if reason != "missing_tool":
            tool = SendMessageToUserTool()
            tool.active = reason != "inactive_tool"
            request.func_tool = ToolSet([tool])
        await plugin.append_generator_instructions(event, request)
        response = LLMResponse(role="assistant", completion_text="Unsent answer")
        if reason == "error":
            response.role = "err"
        elif reason == "finished":
            response.completion_text = " \n<finished>\t"
        await plugin.suppress_final_reply(event, response)
        if reason == "stop":
            event.set_extra("agent_stop_requested", True)
        elif reason == "aborted":
            event.set_extra("agent_user_aborted", True)
        elif reason == "disabled":
            plugin.settings = replace(plugin.settings, enabled=False)
        elif reason == "group_off":
            plugin.settings = replace(plugin.settings, group_sessions=())
        elif reason == "ai_off":
            plugin.context.get_config = lambda **kwargs: {
                "provider_settings": {"enable": False}
            }
        assert [request async for request in handler] == []
        assert event.is_stopped()
    finally:
        await handler.aclose()


@pytest.mark.asyncio
async def test_rerun_uses_framework_stop_signal_and_saves_before_replacement(plugin):
    from astrbot.core.astr_agent_run_util import _watch_agent_stop_signal

    plugin.context.llm_generate.return_value = LLMResponse(
        role="assistant", completion_text='{"should_reply": true, "action": "rerun"}'
    )
    first_started = asyncio.Event()
    stop_received = asyncio.Event()
    history_saved = []
    requests = []

    class PluginDispatch:
        async def process(self, event):
            async for value in call_handler(event, plugin.on_message):
                yield value

    class RunningAgent:
        def done(self):
            return stop_received.is_set()

        def request_stop(self):
            stop_received.set()

    class Agent:
        async def process(self, event):
            request = event.get_extra("provider_request")
            requests.append(request)
            if event.message_obj.message_id == "1":
                first_started.set()
                watcher = asyncio.create_task(
                    _watch_agent_stop_signal(RunningAgent(), event)
                )
                try:
                    await asyncio.wait_for(stop_received.wait(), 2)
                    assert not event.is_stopped()
                    event.set_extra("agent_user_aborted", True)
                    history_saved.append("1")
                finally:
                    watcher.cancel()
                    await asyncio.gather(watcher, return_exceptions=True)
                return
            assert history_saved == ["1"]
            assert "previous Agent turn was stopped" in request.prompt
            event.set_result(event.plain_result("Corrected answer"))
            yield
            history_saved.append("2")

    class Respond:
        async def process(self, event):
            await asyncio.sleep(0)

    process = ProcessStage()
    process.ctx = SimpleNamespace(
        astrbot_config={"provider_settings": {"enable": True}}
    )
    process.star_request_sub_stage = PluginDispatch()
    process.agent_sub_stage = Agent()
    pipeline = object.__new__(PipelineScheduler)
    pipeline.stages = [process, Respond()]
    first, second = make_event("1", private=True), make_event("2", private=True)
    for event in (first, second):
        event.set_extra("activated_handlers", [SimpleNamespace(event_filters=[])])
        event.is_at_or_wake_command = True
    first_task = asyncio.create_task(pipeline._process_stages(first))
    await asyncio.wait_for(first_started.wait(), 1)
    second_task = asyncio.create_task(pipeline._process_stages(second))
    try:
        await asyncio.wait_for(asyncio.gather(first_task, second_task), 3)
        assert history_saved == ["1", "2"]
        assert len(requests) == 2
        assert first.get_extra("agent_stop_requested") is True
        assert second.get_extra("mingle_resume_after_rerun") is True
        judgment = json.loads(plugin.context.llm_generate.call_args.kwargs["prompt"])
        assert any(item["status"] == "processing" for item in judgment["messages"])
        assert any(item["status"] == "new" for item in judgment["messages"])
        assert "running_agent" not in judgment
        assert any(
            item["status"] == "processing" and item["text"] == "message 1"
            for item in judgment["messages"]
        )
    finally:
        first_task.cancel()
        second_task.cancel()
        await asyncio.gather(first_task, second_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_sent_responses_are_recorded_once_without_final_output(plugin):
    event = make_event()
    event.set_extra("mingle_turn_id", "turn")
    event.set_extra("_send_message_to_user_current_session_plain_texts", ["hello"])
    response = LLMResponse(role="assistant", completion_text="Unsent final output")
    await plugin.record_response(event, response)
    await plugin.record_response(event, response)
    rows = await plugin.history.get(event.unified_msg_origin)
    assert len(rows) == 1
    assert rows[0]["role"] == "bot"
    assert rows[0]["text"] == "hello"


@pytest.mark.asyncio
async def test_session_ai_switch_also_disables_judgment(plugin, monkeypatch):
    monkeypatch.setattr(
        "mingle_tests_plugin.main.SessionServiceManager.should_process_llm_request",
        AsyncMock(return_value=False),
    )
    event = make_event()
    assert [result async for result in plugin.on_message(event)] == []
    plugin.context.llm_generate.assert_not_awaited()
    assert event.is_stopped()


@pytest.mark.asyncio
async def test_unload_cancels_waiting_handlers_before_database_close(plugin):
    plugin.scheduler.settings = replace(
        plugin.settings, debounce_seconds=30, max_wait_seconds=30
    )
    event = make_event(private=True)
    task = asyncio.create_task(anext(plugin.on_message(event)))
    for _ in range(100):
        if plugin.scheduler.sessions:
            break
        await asyncio.sleep(0.005)
    assert plugin.scheduler.sessions
    await plugin.terminate()
    assert task.done()
    await asyncio.gather(task, return_exceptions=True)
    assert plugin.history.db is None
    assert not plugin.handlers


@pytest.mark.asyncio
async def test_group_boundary_filters_agent_history_and_survives_reopen(plugin):
    old = make_event("old")
    old_id = await plugin.history.record(old, old.get_messages())
    assert "已设置" in (await anext(plugin.manage_barrier(old, "set"))).get_plain_text()
    new = make_event("new")
    new_id = await plugin.history.record(new, new.get_messages())
    new.set_extra("mingle_managed", True)
    result = json.loads(await plugin.chat_history(new))
    assert [item["id"] for item in result["messages"]] == [new_id]
    assert (
        json.loads(await plugin.chat_history(new, before_id=new_id))["messages"] == []
    )
    assert (
        await plugin.history.read_image(
            new.unified_msg_origin, old_id, 0, "p:GroupMessage:group"
        )
        is None
    )
    await plugin.history.close()
    await plugin.history.initialize()
    result = (await anext(plugin.manage_barrier(new, "find"))).get_plain_text()
    before, after = result.split("\n\n--- barrier ---\n\n")
    assert before.endswith("] message old")
    assert after.endswith("] message new")
    assert [
        item["id"] for item in json.loads(await plugin.chat_history(new))["messages"]
    ] == [new_id]
    assert (
        "已清除" in (await anext(plugin.manage_barrier(new, "clear"))).get_plain_text()
    )
    assert [
        item["id"] for item in json.loads(await plugin.chat_history(new))["messages"]
    ] == [old_id, new_id]


@pytest.mark.asyncio
async def test_barrier_preview_is_bounded_and_session_scoped(plugin):
    event = make_event("before")
    assert (
        "未设置" in (await anext(plugin.manage_barrier(event, "find"))).get_plain_text()
    )
    plugin.context.get_config = lambda **kwargs: {"timezone": "Asia/Shanghai"}
    event.message_obj.timestamp = 0
    await plugin.history.record(event, [Plain("a" * 150)])
    await anext(plugin.manage_barrier(event, "set"))
    other = make_event("other", private=True)
    await plugin.history.record(other, [Plain("private content")])
    result = (await anext(plugin.manage_barrier(event, "find"))).get_plain_text()
    assert (
        result
        == "[1970-01-01 08:00:00+08:00] "
        + "a" * 24
        + "…\n\n--- barrier ---\n\n（暂无保留的消息）"
    )
    await plugin.history.record(make_event("after"), [Plain("first\nsecond")])
    await plugin.history.record(make_event("later"), [Plain("later message")])
    result = await anext(plugin.manage_barrier(event, "find"))
    assert (
        result.get_plain_text()
        .split("\n\n--- barrier ---\n\n")[1]
        .endswith("] first second")
    )
    assert result.use_markdown_ is False
    await plugin.history.db.execute(
        "DELETE FROM messages WHERE session=?", (event.unified_msg_origin,)
    )
    await plugin.history.db.commit()
    assert await plugin.history.boundary_preview(
        "p:GroupMessage:group", event.unified_msg_origin
    ) == (None, None)


@pytest.mark.asyncio
async def test_boundary_filters_agent_prompt_without_filtering_judge_history(plugin):
    old = make_event("old")
    old_id = await plugin.history.record(old, old.get_messages())
    await anext(plugin.manage_barrier(old, "set"))
    event = make_event("new")
    event.is_at_or_wake_command = True
    handler = plugin.on_message(event)
    try:
        request = await asyncio.wait_for(anext(handler), 1)
        assert "message old" not in request.prompt
        assert "message new" in request.prompt
        assert old_id in [
            item["id"] for item in await plugin.history.get(event.unified_msg_origin)
        ]
    finally:
        await handler.aclose()


@pytest.mark.asyncio
async def test_group_boundary_overwrite_and_private_rejection(plugin):
    event = make_event("one")
    await plugin.history.record(event, event.get_messages())
    await anext(plugin.manage_barrier(event, "set"))
    new = make_event("two")
    new_id = await plugin.history.record(new, new.get_messages())
    await anext(plugin.manage_barrier(new, "set"))
    assert (await plugin.history.get_boundary("p:GroupMessage:group"))[0] == new_id
    private = make_event(private=True)
    assert (
        "群聊"
        in (await anext(plugin.manage_barrier(private, "clear"))).get_plain_text()
    )
    assert await plugin.history.get_boundary("p:GroupMessage:group") is not None


@pytest.mark.asyncio
async def test_group_management_persists_and_updates_all_runtime_settings(plugin):
    event = make_event()
    enabled = [value async for value in plugin.manage_group(event, "on")]
    assert "已启用" in enabled[0].get_plain_text()
    assert "p:GroupMessage:group" in plugin.settings.group_sessions
    assert plugin.scheduler.settings is plugin.settings
    assert plugin.scheduler.judge.settings is plugin.settings
    assert plugin.history.settings is plugin.settings
    persisted = json.loads(
        Path(plugin.config.config_path).read_text(encoding="utf-8-sig")
    )
    assert persisted["general"]["group_sessions"] == ["group", "p:GroupMessage:group"]
    assert persisted["general"]["enabled"] is True
    assert persisted["general"]["private_enabled"] is True
    [value async for value in plugin.manage_group(event, "on")]
    assert plugin.config["general"]["group_sessions"].count("p:GroupMessage:group") == 1
    plugin.config["general"]["group_sessions"].append("other:GroupMessage:other")
    disabled = [value async for value in plugin.manage_group(event, "off")]
    assert "已关闭" in disabled[0].get_plain_text()
    assert plugin.settings.group_sessions == ("other:GroupMessage:other",)
    persisted = json.loads(
        Path(plugin.config.config_path).read_text(encoding="utf-8-sig")
    )
    assert persisted["general"]["group_sessions"] == ["other:GroupMessage:other"]


@pytest.mark.asyncio
async def test_group_management_rejects_private_chat_and_bad_arguments(plugin):
    original = plugin.settings
    results = [
        value async for value in plugin.manage_group(make_event(private=True), "on")
    ]
    assert "群聊" in results[0].get_plain_text()
    results = [value async for value in plugin.manage_group(make_event(), "unknown")]
    assert "用法" in results[0].get_plain_text()
    assert plugin.settings is original


@pytest.mark.asyncio
async def test_group_management_save_failure_leaves_runtime_unchanged(
    plugin, monkeypatch
):
    original = plugin.settings
    monkeypatch.setattr(
        plugin.config, "save_config_async", AsyncMock(side_effect=OSError("disk full"))
    )
    results = [value async for value in plugin.manage_group(make_event(), "off")]
    assert "保存配置失败" in results[0].get_plain_text()
    assert plugin.settings is original
    assert plugin.config["general"]["group_sessions"] == ["group"]


def test_management_commands_use_native_group_and_admin_permissions():
    from astrbot.core.star.command_management import _build_descriptor
    from astrbot.core.star.filter.command_group import CommandGroupFilter
    from astrbot.core.star.filter.permission import PermissionType, PermissionTypeFilter
    from astrbot.core.star.star_handler import star_handlers_registry

    handlers = star_handlers_registry.get_handlers_by_module_name(
        MinglePlugin.__module__
    )
    root = next(item for item in handlers if item.handler_name == "mingle")
    group = next(
        item for item in root.event_filters if isinstance(item, CommandGroupFilter)
    )
    assert group.group_name == "mingle"
    assert {item.command_name for item in group.sub_command_filters} == {
        "group",
        "debug",
        "barrier",
        "whisper",
        "version",
    }
    assert not any(
        isinstance(item, PermissionTypeFilter) for item in root.event_filters
    )
    event = make_event()
    event.role = "admin"
    event.is_at_or_wake_command = True
    event.message_str = "mingle"
    with pytest.raises(ValueError) as error:
        group.filter(event, {})
    for name in ("group", "debug", "barrier"):
        assert name in str(error.value)
    assert "Args:" not in str(error.value)
    assert "Yields:" not in str(error.value)
    assert "开启或关闭当前群聊：on|off" in str(error.value)
    assert "开启或关闭当前会话调试：on|off" in str(error.value)
    assert "管理群聊记录边界：set 设置，find 查看，clear 清除" in str(error.value)
    from mingle_tests_plugin.src.utils.version import VERSION

    assert f"查看版本：Mingle {VERSION}" in str(error.value)
    version_command = next(
        item for item in group.sub_command_filters if item.command_name == "version"
    )
    assert not any(
        isinstance(item, PermissionTypeFilter)
        for item in version_command.get_handler_md().event_filters
    )
    assert root.desc == "管理 Mingle 群聊、调试和记录边界"
    for command in group.sub_command_filters:
        if command.command_name in {"whisper", "version"}:
            continue
        handler = command.get_handler_md()
        assert handler.extras_configs.get("sub_command") is True
        descriptor = _build_descriptor(handler)
        assert descriptor.is_sub_command
        assert descriptor.parent_group_handler == root.handler_full_name
        permission = next(
            item
            for item in handler.event_filters
            if isinstance(item, PermissionTypeFilter)
        )
        assert permission.permission_type == PermissionType.ADMIN
        event = make_event()
        assert not permission.filter(event, {})
        event.role = "admin"
        assert permission.filter(event, {})
        assert command.get_complete_command_names() == [
            f"mingle {command.command_name}"
        ]


@pytest.mark.asyncio
async def test_version_command_matches_installed_metadata(plugin):
    import yaml

    metadata = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "metadata.yaml").read_text(
            encoding="utf-8"
        )
    )
    results = [value async for value in plugin.show_version(make_event())]
    assert results[0].get_plain_text() == f"Mingle {metadata['version']}"


@pytest.mark.asyncio
async def test_group_enable_applies_to_isolated_members_and_platform_scope(plugin):
    [value async for value in plugin.manage_group(make_event(), "on")]
    isolated = make_event("2")
    isolated.session_id = "bob_group"
    assert [value async for value in plugin.on_message(isolated)] == []
    assert isolated.get_extra("mingle_managed")
    # Remove the legacy bare ID so only the newly added platform-scoped rule applies.
    plugin.settings = replace(plugin.settings, group_sessions=("p:GroupMessage:group",))
    other = make_event("3")
    other.session.platform_id = "other"
    other.platform_meta.id = "other"
    assert [value async for value in plugin.on_message(other)] == []
    assert not other.get_extra("mingle_managed")


@pytest.mark.asyncio
@pytest.mark.parametrize("private", [False, True])
async def test_debug_command_persists_current_chat_and_updates_runtime(plugin, private):
    event = make_event(private=private)
    original_groups = plugin.settings.group_sessions
    results = [value async for value in plugin.manage_debug(event, "on")]
    assert "已开启" in results[0].get_plain_text()
    assert plugin.settings.debug_sessions == (event.unified_msg_origin,)
    assert plugin.debug_dumper.is_enabled(event)
    assert plugin.debug_dumper.settings is plugin.settings
    assert plugin.scheduler.judge.settings is plugin.settings
    persisted = json.loads(
        Path(plugin.config.config_path).read_text(encoding="utf-8-sig")
    )
    assert persisted["debug"]["debug_sessions"] == [event.unified_msg_origin]
    assert plugin.settings.group_sessions == original_groups
    [value async for value in plugin.manage_debug(event, "on")]
    assert plugin.config["debug"]["debug_sessions"] == [event.unified_msg_origin]
    results = [value async for value in plugin.manage_debug(event, "off")]
    assert "已关闭" in results[0].get_plain_text()
    assert not plugin.debug_dumper.is_enabled(event)
    persisted = json.loads(
        Path(plugin.config.config_path).read_text(encoding="utf-8-sig")
    )
    assert persisted["debug"]["debug_sessions"] == []


@pytest.mark.asyncio
async def test_debug_command_rejects_bad_arguments_and_recovers_from_save_failure(
    plugin, monkeypatch
):
    original = plugin.settings
    results = [value async for value in plugin.manage_debug(make_event(), "maybe")]
    assert "用法" in results[0].get_plain_text()
    assert plugin.settings is original
    monkeypatch.setattr(
        plugin.config, "save_config_async", AsyncMock(side_effect=OSError("disk full"))
    )
    results = [value async for value in plugin.manage_debug(make_event(), "on")]
    assert "保存配置失败" in results[0].get_plain_text()
    assert plugin.settings is original
    assert plugin.debug_dumper.settings is original
    assert plugin.config["debug"]["debug_sessions"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["assistant", "err"])
async def test_agent_debug_hooks_capture_request_and_final_response_outside_managed_flow(
    plugin, role
):
    event = make_event()
    [value async for value in plugin.manage_debug(event, "on")]
    assert not event.get_extra("mingle_managed")
    request = ProviderRequest(
        prompt="Original question",
        system_prompt="Configured persona",
        contexts=[{"role": "user", "content": "History"}],
    )
    await plugin.dump_agent_request(event, request)
    request.prompt = "Changed after the capture"
    response = LLMResponse(role=role, completion_text="Final model result")
    await plugin.dump_agent_response(event, response)
    documents = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in plugin.debug_dumper.directory.rglob("*.json")
    ]
    assert len(documents) == 2
    assert len({item["request_id"] for item in documents}) == 1
    assert {item["kind"] for item in documents} == {"agent"}
    initial = next(item for item in documents if item["phase"] == "input")["payload"][
        "request"
    ]
    assert initial["prompt"] == "Original question"
    assert initial["system_prompt"] == "Configured persona"
    assert initial["contexts"] == [{"role": "user", "content": "History"}]
    final = next(item for item in documents if item["phase"] == "output")["payload"]
    assert final["completion_text"] == "Final model result"
    assert final["response"]["role"] == role


@pytest.mark.asyncio
async def test_debug_off_stops_new_dumps_without_removing_existing_files(plugin):
    event = make_event(private=True)
    [value async for value in plugin.manage_debug(event, "on")]
    await plugin.dump_agent_request(event, ProviderRequest(prompt="question"))
    [value async for value in plugin.manage_debug(event, "off")]
    await plugin.dump_agent_response(
        event, LLMResponse(role="assistant", completion_text="answer")
    )
    documents = list(plugin.debug_dumper.directory.rglob("*.json"))
    assert len(documents) == 1
    assert json.loads(documents[0].read_text(encoding="utf-8"))["phase"] == "input"


@pytest.mark.asyncio
async def test_agent_debug_hooks_create_nothing_when_disabled(plugin):
    event = make_event()
    await plugin.dump_agent_request(event, ProviderRequest(prompt="question"))
    await plugin.dump_agent_response(
        event, LLMResponse(role="assistant", completion_text="answer")
    )
    assert not plugin.debug_dumper.directory.exists()


@pytest.mark.asyncio
async def test_group_off_during_busy_judgment_does_not_stop_active_agent(plugin):
    from mingle_tests_plugin.src.decision import ReplyDecision

    started, release = asyncio.Event(), asyncio.Event()

    async def evaluate(*args, **kwargs):
        started.set()
        await release.wait()
        return ReplyDecision(True, "rerun")

    plugin.scheduler.judge.evaluate = evaluate
    first = make_event("1")
    first.is_at_or_wake_command = True
    first_handler = plugin.on_message(first)
    await asyncio.wait_for(anext(first_handler), 1)
    second_task = asyncio.create_task(anext(plugin.on_message(make_event("2"))))
    try:
        await asyncio.wait_for(started.wait(), 1)
        [value async for value in plugin.manage_group(make_event("command"), "off")]
        release.set()
        await asyncio.wait_for(
            plugin.scheduler.sessions[first.unified_msg_origin].judgment_task, 1
        )
        result = await asyncio.gather(second_task, return_exceptions=True)
        assert isinstance(result[0], StopAsyncIteration)
        assert not first.get_extra("agent_stop_requested", False)
    finally:
        release.set()
        second_task.cancel()
        await asyncio.gather(second_task, return_exceptions=True)
        await first_handler.aclose()


@pytest.mark.asyncio
async def test_history_tool_returns_stored_records(plugin):
    event = make_event()
    event.set_extra("mingle_managed", True)
    first_id = await plugin.history.record(event, event.get_messages())
    result = json.loads(await plugin.chat_history(event))
    assert result["messages"][0]["id"] == first_id


@pytest.mark.asyncio
@pytest.mark.parametrize("subcommand", ["group", "debug", "barrier"])
async def test_management_invalid_action_does_not_change_settings(plugin, subcommand):
    original = plugin.settings
    results = [
        value
        async for value in getattr(plugin, f"manage_{subcommand}")(
            make_event(), "unknown"
        )
    ]
    assert "用法" in results[0].get_plain_text()
    assert plugin.settings is original


@pytest.mark.asyncio
@pytest.mark.parametrize("subcommand", ["group", "debug"])
async def test_management_subcommands_parse_and_bypass_mingle(plugin, subcommand):
    from astrbot.core.star.star_handler import star_handlers_registry

    handler = next(
        item
        for item in star_handlers_registry.get_handlers_by_module_name(
            MinglePlugin.__module__
        )
        if item.handler_name == f"manage_{subcommand}"
    )
    command = next(
        item for item in handler.event_filters if isinstance(item, CommandFilter)
    )
    event = make_event()
    event.role = "admin"
    event.is_at_or_wake_command = True
    for state in ("on", "off"):
        event.message_str = f"mingle {subcommand} {state}"
        assert command.filter(event, {})
        results = [
            value
            async for value in getattr(plugin, f"manage_{subcommand}")(
                event, **event.get_extra("parsed_params")
            )
        ]
        assert len(results) == 1
        sessions = (
            plugin.settings.group_sessions
            if subcommand == "group"
            else plugin.settings.debug_sessions
        )
        assert (event.unified_msg_origin in sessions) is (state == "on")

        event.set_extra("activated_handlers", [handler])
        assert [value async for value in plugin.on_message(event)] == []
        assert not event.get_extra("mingle_managed")


@pytest.mark.asyncio
@pytest.mark.parametrize("format", ["JPEG", "GIF", "PNG"])
async def test_view_image_prepares_model_copy_and_preserves_original(
    plugin, tmp_path, monkeypatch, format
):
    import base64
    import io

    from astrbot.core.message.components import Image as ImageComponent
    from PIL import Image

    temporary = tmp_path / "compressed"
    temporary.mkdir()
    monkeypatch.setattr(
        "astrbot.core.utils.media_utils.get_astrbot_temp_path", lambda: str(temporary)
    )
    source = tmp_path / f"source.{format.lower()}"
    with Image.new("RGB", (1800, 900), "red") as first:
        if format == "GIF":
            with Image.new("RGB", first.size, "blue") as second:
                first.save(
                    source,
                    format=format,
                    save_all=True,
                    append_images=[second],
                    duration=100,
                    loop=0,
                )
        else:
            first.save(source, format=format)
    original = source.read_bytes()
    event = make_event()
    event.set_extra("mingle_managed", True)
    record_id = await plugin.history.record(
        event, [ImageComponent.fromFileSystem(source)]
    )
    result = await plugin.view_image(event, record_id)
    assert not result.isError
    content = result.content[0]
    output = base64.b64decode(content.data)
    with Image.open(io.BytesIO(output)) as image:
        assert max(image.size) == 1280
        assert image.size == (1280, 640)
        assert getattr(image, "n_frames", 1) == 1
        red, green, blue = image.convert("RGB").getpixel((0, 0))
        assert red > 240 and green < 10 and blue < 10
        assert content.mimeType == Image.MIME[image.format]
    assert (await plugin.history.read_image(event.unified_msg_origin, record_id, 0))[
        0
    ] == original
    assert source.read_bytes() == original
    assert list(temporary.iterdir()) == []


@pytest.mark.asyncio
async def test_view_image_reports_preparation_failure_without_returning_original(
    plugin, monkeypatch
):
    event = make_event()
    event.set_extra("mingle_managed", True)
    monkeypatch.setattr(
        plugin.history, "read_image", AsyncMock(return_value=(b"invalid", "image/jpeg"))
    )
    monkeypatch.setattr(
        "mingle_tests_plugin.src.tools.images.prepare_image",
        AsyncMock(side_effect=ValueError("invalid image")),
    )
    result = await plugin.view_image(event, 1)
    assert result.isError is True
    assert all(item.type == "text" for item in result.content)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(10, 10), (1800, 900)])
async def test_unchanged_compressor_output_requires_valid_dimensions(
    plugin, monkeypatch, size
):
    import io

    from PIL import Image

    with Image.new("RGB", size, "red") as image, io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
    event = make_event()
    event.set_extra("mingle_managed", True)
    monkeypatch.setattr(
        plugin.history, "read_image", AsyncMock(return_value=(data, "image/png"))
    )
    monkeypatch.setattr(
        "mingle_tests_plugin.src.utils.image_preprocessing.compress_image",
        AsyncMock(side_effect=lambda source, **kwargs: source),
    )
    result = await plugin.view_image(event, 1)
    assert bool(result.isError) is (max(size) > 1280)
    if result.isError:
        assert all(item.type == "text" for item in result.content)
    else:
        assert result.content[0].type == "image"


@pytest.mark.asyncio
async def test_unchanged_compressor_output_rejects_corrupt_image(plugin, monkeypatch):
    event = make_event()
    event.set_extra("mingle_managed", True)
    monkeypatch.setattr(
        plugin.history, "read_image", AsyncMock(return_value=(b"invalid", "image/png"))
    )
    monkeypatch.setattr(
        "mingle_tests_plugin.src.utils.image_preprocessing.compress_image",
        AsyncMock(side_effect=lambda source, **kwargs: source),
    )
    result = await plugin.view_image(event, 1)
    assert result.isError is True
    assert all(item.type == "text" for item in result.content)


@pytest.mark.asyncio
async def test_history_images_are_removed_before_debug_and_saved_history_keeps_current_turn_images(
    plugin,
):
    import copy

    from astrbot.core.agent.message import (
        ImageURLPart,
        Message,
        bind_checkpoint_messages,
    )
    from astrbot.core.db.po import Conversation
    from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
        InternalAgentSubStage,
    )

    event = make_event()
    event.set_extra("mingle_managed", True)
    plugin.debug_dumper.settings = replace(
        plugin.settings, debug_sessions=(event.unified_msg_origin,)
    )
    old_image = "data:image/png;base64,OLD_IMAGE"
    history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Original caption"},
                {"type": "image_url", "image_url": {"url": old_image}},
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call1",
                    "type": "function",
                    "function": {
                        "name": "mingle_view_image",
                        "arguments": '{"record_id":60}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call1", "content": "Viewed record 60"},
    ]
    original = copy.deepcopy(history)
    conversation = Conversation(
        platform_id="p",
        user_id="group",
        cid="conversation",
        history=json.dumps(history),
    )
    current_image = ImageURLPart(
        image_url=ImageURLPart.ImageURL(url="data:image/png;base64,CURRENT_IMAGE")
    )
    request = ProviderRequest(
        prompt="Follow-up",
        contexts=history,
        conversation=conversation,
        image_urls=["current.png"],
        extra_user_content_parts=[current_image],
    )
    await plugin.clear_history_images(event, request)
    assert history == original
    assert conversation.history == json.dumps(original)
    assert old_image not in json.dumps(request.contexts)
    assert request.contexts[0]["content"][0]["text"] == "Original caption"
    assert request.contexts[1:] == original[1:]
    assert request.image_urls == ["current.png"]
    assert request.extra_user_content_parts == [current_image]
    assert json.loads(request.conversation.history) == request.contexts
    await plugin.dump_agent_request(event, request)
    dumps = list(plugin.debug_dumper.directory.rglob("*agent_input*.json"))
    assert len(dumps) == 1
    assert old_image not in dumps[0].read_text(encoding="utf-8")

    # Exercise the framework saver: a new tool image remains for this turn,
    # while images cleaned from the previous turn are no longer persisted.
    runtime = bind_checkpoint_messages(request.contexts)
    runtime.append(Message(role="user", content=[current_image]))
    runtime.append(Message(role="assistant", content="Answer"))
    stage = object.__new__(InternalAgentSubStage)
    stage.conv_manager = SimpleNamespace(update_conversation=AsyncMock())
    await stage._save_to_history(
        event,
        request,
        LLMResponse(role="assistant", completion_text="Answer"),
        runtime,
        None,
    )
    saved = stage.conv_manager.update_conversation.call_args.kwargs["history"]
    assert old_image not in json.dumps(saved)
    assert "CURRENT_IMAGE" in json.dumps(saved)
    next_request = ProviderRequest(prompt="Next turn", contexts=saved)
    await plugin.clear_history_images(event, next_request)
    assert "CURRENT_IMAGE" not in json.dumps(next_request.contexts)
    assert next_request.contexts[-1]["content"] == "Answer"


@pytest.mark.asyncio
async def test_history_image_cleanup_does_not_affect_unmanaged_requests(plugin):
    event = make_event()
    contexts = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,KEEP"},
                }
            ],
        }
    ]
    request = ProviderRequest(prompt="Native request", contexts=contexts)
    await plugin.clear_history_images(event, request)
    assert request.contexts is contexts
    assert "KEEP" in json.dumps(request.contexts)


def test_history_image_cleanup_runs_before_request_debug_capture():
    from astrbot.core.star.star_handler import star_handlers_registry

    handlers = {
        item.handler_name: item
        for item in star_handlers_registry.get_handlers_by_module_name(
            MinglePlugin.__module__
        )
    }
    assert handlers["clear_history_images"].extras_configs.get(
        "priority", 0
    ) > handlers["dump_agent_request"].extras_configs.get("priority", 0)
    assert handlers["append_generator_instructions"].extras_configs.get(
        "priority", 0
    ) > handlers["dump_agent_request"].extras_configs.get("priority", 0)


@pytest.mark.asyncio
async def test_time_tool_follows_session_config_without_managed_turn(plugin):
    from datetime import datetime
    from unittest.mock import Mock

    event = make_event()
    config = {"timezone": "Asia/Shanghai"}
    plugin.context.get_config = Mock(return_value=config)
    result = json.loads(await plugin.get_time(event))
    plugin.context.get_config.assert_called_once_with(umo=event.unified_msg_origin)
    assert not event.get_extra("mingle_managed")
    assert result["timezone"] == "Asia/Shanghai"
    assert (
        datetime.fromisoformat(result["datetime"]).utcoffset().total_seconds()
        == 8 * 3600
    )
    config.clear()
    result = json.loads(await plugin.get_time(event))
    assert result["timezone"] == "UTC"
    assert datetime.fromisoformat(result["datetime"]).utcoffset().total_seconds() == 0


def test_time_tool_registration_has_no_model_supplied_parameters():
    from astrbot.core.provider.register import llm_tools

    tool = llm_tools.get_func("mingle_get_time")
    assert tool is not None
    assert not tool.parameters.get("properties")
    assert not tool.parameters.get("required")
    assert tool.handler.__module__ == MinglePlugin.__module__


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "master,individual", [(True, True), (True, False), (False, True), (False, False)]
)
async def test_toolbox_switches_filter_requests_and_guard_execution(
    plugin, master, individual
):
    from astrbot.core.agent.tool import FunctionTool, ToolSet

    plugin.settings = replace(
        plugin.settings, toolbox_enabled=master, toolbox_time_enabled=individual
    )
    names = [
        "mingle_get_time",
        "mingle_chat_history",
        "mingle_view_image",
        "web_search_deepseek",
    ]
    shared = ToolSet(
        [FunctionTool(name=name, parameters={}, description="") for name in names]
    )
    request = ProviderRequest(prompt="question", func_tool=shared)
    event = make_event()
    assert not event.get_extra("mingle_managed")
    await plugin.apply_toolbox_switches(event, request)
    actual = [tool.name for tool in request.func_tool.tools]
    assert ("mingle_get_time" in actual) is (master and individual)
    assert set(actual) - {"mingle_get_time"} == set(names) - {"mingle_get_time"}
    assert [tool.name for tool in shared.tools] == names
    result = json.loads(await plugin.get_time(event))
    assert ("error" not in result) is (master and individual)
    assert plugin.settings.toolbox_time_enabled is individual


@pytest.mark.asyncio
async def test_toolbox_switches_do_not_add_tools_excluded_by_persona(plugin):
    from astrbot.core.agent.tool import ToolSet

    event = make_event()
    request = ProviderRequest(prompt="question", func_tool=ToolSet())
    await plugin.apply_toolbox_switches(event, request)
    assert request.func_tool.empty()
    request.func_tool = None
    await plugin.apply_toolbox_switches(event, request)
    assert request.func_tool is None


@pytest.mark.asyncio
@pytest.mark.parametrize("master", [True, False])
@pytest.mark.parametrize("individual", [True, False])
@pytest.mark.parametrize(
    "name,setting,method,arguments",
    [
        (
            "mingle_calculate",
            "toolbox_calculate_enabled",
            "calculate",
            {"expression": "0.1+0.2"},
        ),
        ("mingle_random", "toolbox_random_enabled", "random", {"choices": ["a"]}),
    ],
)
async def test_new_tool_switches_are_independent(
    plugin, master, individual, name, setting, method, arguments
):
    from astrbot.core.agent.tool import FunctionTool, ToolSet

    plugin.settings = replace(
        plugin.settings, toolbox_enabled=master, **{setting: individual}
    )
    names = [
        "mingle_get_time",
        "mingle_calculate",
        "mingle_random",
        "mingle_chat_history",
        "native_tool",
    ]
    shared = ToolSet(
        [FunctionTool(name=item, parameters={}, description="") for item in names]
    )
    request = ProviderRequest(prompt="test", func_tool=shared)
    event = make_event()
    await plugin.apply_toolbox_switches(event, request)
    actual = {tool.name for tool in request.func_tool.tools}
    for item in names:
        expected = (
            True
            if item in {"mingle_chat_history", "native_tool"}
            else master and (individual if item == name else True)
        )
        assert (item in actual) is expected
    assert [tool.name for tool in shared.tools] == names
    assert (
        "error" not in json.loads(await getattr(plugin, method)(event, **arguments))
    ) is (master and individual)
    assert getattr(plugin.settings, setting) is individual
    plugin.available = False
    assert "error" in json.loads(await getattr(plugin, method)(event, **arguments))
    await plugin.apply_toolbox_switches(event, request)
    assert {tool.name for tool in request.func_tool.tools} == {
        "mingle_chat_history",
        "native_tool",
    }
    plugin.available = True


def test_new_tool_schemas_only_expose_model_arguments():
    from astrbot.core.provider.register import llm_tools

    calculate_tool = llm_tools.get_func("mingle_calculate")
    random_tool = llm_tools.get_func("mingle_random")
    assert set(calculate_tool.parameters["properties"]) == {"expression"}
    assert set(random_tool.parameters["properties"]) == {
        "minimum",
        "maximum",
        "count",
        "choices",
    }
    assert random_tool.parameters["properties"]["choices"]["items"]["type"] == "string"
    edit_tool = llm_tools.get_func("mingle_edit_image")
    assert set(edit_tool.parameters["properties"]) == {
        "prompt",
        "record_ids",
        "image_indices",
        "size",
        "quality",
    }
    assert edit_tool.parameters["properties"]["record_ids"]["items"]["type"] == "number"
    image_tool = llm_tools.get_func("mingle_generate_image")
    assert set(image_tool.parameters["properties"]) == {"prompt", "size", "quality"}
    assert all(
        item["type"] == "string"
        for item in image_tool.parameters["properties"].values()
    )


@pytest.mark.parametrize("master", [True, False])
@pytest.mark.parametrize("individual", [True, False])
async def test_image_tool_switches_and_explicit_provider(
    plugin, monkeypatch, master, individual
):
    from astrbot.core.agent.tool import FunctionTool, ToolSet

    provider = object()
    lookup = Mock(return_value=provider)
    plugin.context.get_provider_by_id = lookup
    generate = AsyncMock(return_value='{"sent": true}')
    monkeypatch.setattr("mingle_tests_plugin.main.generate_image", generate)
    plugin.settings = replace(
        plugin.settings,
        toolbox_enabled=master,
        toolbox_image_generate_enabled=individual,
        toolbox_image_generate_provider_id="images-provider",
    )
    names = ["mingle_generate_image", "mingle_get_time", "mingle_view_image", "native"]
    shared = ToolSet(
        [FunctionTool(name=name, parameters={}, description="") for name in names]
    )
    event = make_event(private=True)
    request = ProviderRequest(prompt="draw", func_tool=shared)
    await plugin.apply_toolbox_switches(event, request)
    actual = {tool.name for tool in request.func_tool.tools}
    assert ("mingle_generate_image" in actual) is (master and individual)
    assert ("mingle_get_time" in actual) is master
    assert {"mingle_view_image", "native"} <= actual
    assert [tool.name for tool in shared.tools] == names
    result = json.loads(await plugin.generate_image(event, "a bird"))
    if master and individual:
        assert result["sent"] is True
        lookup.assert_called_once_with("images-provider")
        generate.assert_awaited_once_with(
            event,
            provider,
            plugin.logger,
            prompt="a bird",
            size="auto",
            quality="auto",
        )
    else:
        assert "error" in result
        generate.assert_not_awaited()
        lookup.assert_not_called()
    plugin.available = False
    generate.reset_mock()
    assert "error" in json.loads(await plugin.generate_image(event, "bird"))
    generate.assert_not_awaited()
    plugin.available = True


async def test_image_tool_missing_selection_does_not_choose_another_provider(plugin):
    plugin.context.get_provider_by_id = Mock()
    result = json.loads(await plugin.generate_image(make_event(), "a bird"))
    assert "Select an image generation provider" in result["error"]
    plugin.context.get_provider_by_id.assert_not_called()


def test_toolbox_switches_run_before_request_debug_capture():
    from astrbot.core.star.star_handler import star_handlers_registry

    handlers = {
        item.handler_name: item
        for item in star_handlers_registry.get_handlers_by_module_name(
            MinglePlugin.__module__
        )
    }
    assert (
        handlers["apply_toolbox_switches"].extras_configs["priority"]
        > handlers["dump_agent_request"].extras_configs["priority"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text,managed,llm,hidden",
    [
        ("<finished>", True, True, True),
        (" \n<finished>\t", True, True, True),
        ("Answer <finished>", True, True, True),
        ("<finished> is a marker", True, True, True),
        ("<finished>", False, True, False),
        ("<finished>", True, False, True),
        ("Normal reply", True, True, True),
    ],
)
async def test_final_output_is_suppressed_only_for_managed_turns(
    plugin, text, managed, llm, hidden, caplog
):
    event = make_event()
    event.set_extra("mingle_managed", managed)
    response = LLMResponse(role="assistant" if llm else "err", completion_text=text)
    await plugin.suppress_final_reply(event, response)
    event.set_result(
        event.plain_result(text).set_result_content_type(ResultContentType.LLM_RESULT)
    )
    await plugin.suppress_final_result(event)
    assert (event.get_result().chain == []) is hidden
    assert response.completion_text == text
    assert ("final output was not <finished>" in caplog.text) is (
        managed and text.strip() != "<finished>"
    )
    assert not event.is_stopped()


@pytest.mark.asyncio
async def test_final_output_discards_media_and_reasoning(plugin):
    from astrbot.core.message.components import Image

    event = make_event()
    event.set_extra("mingle_managed", True)
    result = event.plain_result("<finished>")
    result.chain.append(Image.fromURL("https://example.com/image.png"))
    response = LLMResponse(role="assistant", result_chain=result)
    await plugin.suppress_final_reply(event, response)
    assert response.result_chain is None
    assert response.completion_text == "<finished>"
    assert response.reasoning_content == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("final_text", ["<finished>", "Unsent final output"])
async def test_native_split_replies_are_recorded_and_final_output_stays_in_debug_only(
    plugin,
    final_text,
):
    from astrbot.core.tools.message_tools import SendMessageToUserTool

    event = make_event()
    event.set_extra("mingle_managed", True)
    event.set_extra("mingle_turn_id", "split-turn")
    plugin.debug_dumper.settings = replace(
        plugin.settings, debug_sessions=(event.unified_msg_origin,)
    )
    await plugin.dump_agent_request(
        event, ProviderRequest(prompt="Two members asked different questions")
    )
    sender = SimpleNamespace(send_message=AsyncMock(return_value=True))
    context = SimpleNamespace(context=SimpleNamespace(event=event, context=sender))
    tool = SendMessageToUserTool()
    for text in ("Alice: first answer", "Bob: second answer"):
        result = await tool.call(context, messages=[{"type": "plain", "text": text}])
        assert result.startswith("Message sent")
    response = LLMResponse(role="assistant", completion_text=final_text)
    await plugin.record_response(event, response)
    await plugin.record_response(event, response)
    await plugin.dump_agent_response(event, response)
    await plugin.suppress_final_reply(event, response)
    assert response.completion_text == final_text
    assert response.result_chain is None
    assert not event.is_stopped()
    assert sender.send_message.await_count == 2
    rows = await plugin.history.get(event.unified_msg_origin)
    assert [row["text"] for row in rows] == [
        "Alice: first answer",
        "Bob: second answer",
    ]
    assert all(row["role"] == "bot" for row in rows)
    outputs = list(plugin.debug_dumper.directory.rglob("*agent_output*.json"))
    assert (
        json.loads(outputs[0].read_text(encoding="utf-8"))["payload"]["completion_text"]
        == final_text
    )


@pytest.mark.asyncio
async def test_agent_context_is_incremental_and_resets_with_conversation(plugin):
    event = make_event("first", private=True)
    first_id = await plugin.history.record(event, event.get_messages())
    conversation = SimpleNamespace(history="[]")
    initial = await plugin._agent_prompt_history(event, conversation, [first_id])
    assert [row["id"] for row in initial] == [first_id]
    conversation.history = json.dumps(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"New batch record IDs: [{first_id}]. Snapshot ends at record {first_id}.\n[]"
                        ),
                    }
                ],
            },
            {"role": "assistant", "content": "Previous response"},
        ]
    )
    # More than one query page, including messages not admitted individually.
    ids = []
    for index in range(105):
        arrival = make_event(f"next-{index}", private=True)
        ids.append(await plugin.history.record(arrival, arrival.get_messages()))
    incremental = await plugin._agent_prompt_history(event, conversation, [ids[-1]])
    assert [row["id"] for row in incremental] == ids
    conversation.history = "[]"
    plugin.settings = replace(plugin.settings, initial_context_messages=3)
    reset = await plugin._agent_prompt_history(event, conversation, [ids[-1]])
    assert [row["id"] for row in reset] == ids[-3:]
    conversation.history = json.dumps([{"role": "user", "content": "Native chat"}])
    native = await plugin._agent_prompt_history(event, conversation, [ids[-1]])
    assert [row["id"] for row in native] == ids[-1:]


@pytest.mark.parametrize(
    "master,individual", [(True, True), (True, False), (False, True)]
)
async def test_edit_tool_switches_and_provider(plugin, monkeypatch, master, individual):
    from astrbot.core.agent.tool import FunctionTool, ToolSet

    provider = object()
    plugin.context.get_provider_by_id = Mock(return_value=provider)
    edit = AsyncMock(return_value='{"sent": true}')
    monkeypatch.setattr("mingle_tests_plugin.main.edit_image", edit)
    plugin.settings = replace(
        plugin.settings,
        toolbox_enabled=master,
        toolbox_image_edit_enabled=individual,
        toolbox_image_edit_provider_id="edit-provider",
    )
    shared = ToolSet(
        [FunctionTool(name="mingle_edit_image", parameters={}, description="")]
    )
    event = make_event(private=True)
    request = ProviderRequest(prompt="edit", func_tool=shared)
    await plugin.apply_toolbox_switches(event, request)
    assert bool(request.func_tool.tools) is (master and individual)
    assert len(shared.tools) == 1
    result = json.loads(await plugin.edit_image(event, "Watercolor", [1], [0]))
    if master and individual:
        assert result["sent"]
        plugin.context.get_provider_by_id.assert_called_once_with("edit-provider")
        assert edit.await_args.args == (event, plugin.history, provider, plugin.logger)
    else:
        assert "error" in result
        edit.assert_not_awaited()
    plugin.available = False
    edit.reset_mock()
    assert "error" in json.loads(await plugin.edit_image(event, "Edit", [1], [0]))
    edit.assert_not_awaited()
    plugin.available = True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,text",
    [
        ("whisper", "mingle whisper hello\nworld"),
        ("whisper_alias", "w hello"),
        ("whisper", "mingle whisper"),
        ("whisper_alias", "w"),
    ],
)
@pytest.mark.parametrize("private", [False, True])
async def test_whisper_discards_entire_message_before_history(
    plugin, name, text, private
):
    from astrbot.core.star.filter.permission import PermissionTypeFilter
    from astrbot.core.star.star_handler import star_handlers_registry

    metadata = next(
        item
        for item in star_handlers_registry.get_handlers_by_module_name(
            MinglePlugin.__module__
        )
        if item.handler_name == name
    )
    assert not any(
        isinstance(item, PermissionTypeFilter) for item in metadata.event_filters
    )
    command = next(
        item for item in metadata.event_filters if isinstance(item, CommandFilter)
    )
    event = make_event(private=private)
    event.message_str = text
    event.is_at_or_wake_command = True
    assert command.filter(event, {})
    event.set_extra("activated_handlers", [metadata])
    plugin.settings = replace(plugin.settings, group_sessions=(), private_enabled=False)
    assert [result async for result in plugin.on_message(event)] == []
    await getattr(plugin, name)(event)
    assert event.is_stopped()
    assert event.call_llm
    assert not plugin.scheduler.sessions
    assert await plugin.history.get(event.unified_msg_origin) == []
    plugin.context.llm_generate.assert_not_awaited()
