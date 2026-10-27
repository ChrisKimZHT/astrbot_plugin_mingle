import asyncio
import json
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from astrbot.api import AstrBotConfig
from astrbot.core.message.components import Plain
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
    instance.settings = Settings(group_sessions=("group",), debounce_seconds=0)
    instance.config = AstrBotConfig(
        str(tmp_path / "config.json"), default_config={"group_sessions": ["group"]}
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
async def test_commands_and_nonconfigured_groups_remain_unmanaged(plugin):
    event = make_event()
    event.set_extra(
        "activated_handlers", [SimpleNamespace(event_filters=[CommandFilter("stop")])]
    )
    assert [result async for result in plugin.on_message(event)] == []
    assert not event.is_stopped()
    assert not event.get_extra("mingle_managed")
    event = make_event("2")
    plugin.settings = replace(plugin.settings, group_sessions=())
    assert [result async for result in plugin.on_message(event)] == []
    assert not event.is_stopped()


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
async def test_completed_responses_are_recorded_once(plugin):
    event = make_event()
    event.set_extra("mingle_turn_id", "turn")
    response = LLMResponse(role="assistant", completion_text="hello")
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
    assert persisted["group_sessions"] == ["group", "p:GroupMessage:group"]
    [value async for value in plugin.manage_group(event, "on")]
    assert plugin.config["group_sessions"].count("p:GroupMessage:group") == 1
    plugin.config["group_sessions"].append("other:GroupMessage:other")
    disabled = [value async for value in plugin.manage_group(event, "off")]
    assert "已关闭" in disabled[0].get_plain_text()
    assert plugin.settings.group_sessions == ("other:GroupMessage:other",)
    persisted = json.loads(
        Path(plugin.config.config_path).read_text(encoding="utf-8-sig")
    )
    assert persisted["group_sessions"] == ["other:GroupMessage:other"]


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
    assert plugin.config["group_sessions"] == ["group"]


def test_group_management_permission_is_registered_on_group_and_subcommand():
    from astrbot.core.star.filter.permission import PermissionType, PermissionTypeFilter
    from astrbot.core.star.star_handler import star_handlers_registry

    handlers = star_handlers_registry.get_handlers_by_module_name(
        MinglePlugin.__module__
    )
    for name in ("mingle", "manage_group", "manage_debug"):
        handler = next(item for item in handlers if item.handler_name == name)
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
    assert persisted["debug_sessions"] == [event.unified_msg_origin]
    assert plugin.settings.group_sessions == original_groups
    [value async for value in plugin.manage_debug(event, "on")]
    assert plugin.config["debug_sessions"] == [event.unified_msg_origin]
    results = [value async for value in plugin.manage_debug(event, "off")]
    assert "已关闭" in results[0].get_plain_text()
    assert not plugin.debug_dumper.is_enabled(event)
    persisted = json.loads(
        Path(plugin.config.config_path).read_text(encoding="utf-8-sig")
    )
    assert persisted["debug_sessions"] == []


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
    assert plugin.config["debug_sessions"] == []


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
            plugin.scheduler.sessions[first.unified_msg_origin].task, 1
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
