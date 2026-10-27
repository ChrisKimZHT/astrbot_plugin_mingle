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
from mingle_tests_plugin.decision import ReplyJudge
from mingle_tests_plugin.history import ChatHistory
from mingle_tests_plugin.main import MinglePlugin
from mingle_tests_plugin.scheduler import ChatScheduler
from mingle_tests_plugin.settings import Settings


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
        ReplyJudge(instance.context, instance.settings, instance.logger),
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
    for name in ("mingle", "manage_group"):
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
