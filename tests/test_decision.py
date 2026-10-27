import copy
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mingle_tests_plugin.src.decision import ReplyJudge
from mingle_tests_plugin.src.settings import Settings


@pytest.fixture
def context():
    return SimpleNamespace(
        get_config=lambda **kwargs: {"timezone": "Asia/Shanghai"},
        get_current_chat_provider_id=AsyncMock(return_value="default"),
        llm_generate=AsyncMock(
            return_value=SimpleNamespace(completion_text='{"should_reply": true}')
        ),
        conversation_manager=SimpleNamespace(
            get_curr_conversation_id=AsyncMock(return_value="conversation"),
            get_conversation=AsyncMock(
                return_value=SimpleNamespace(persona_id="persona")
            ),
        ),
        persona_manager=SimpleNamespace(
            resolve_selected_persona=AsyncMock(
                return_value=(
                    "persona",
                    {"prompt": "You enjoy discussing astronomy."},
                    None,
                    False,
                )
            )
        ),
    )


@pytest.fixture
def event():
    return SimpleNamespace(unified_msg_origin="group", get_platform_name=lambda: "test")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"should_reply": true}', True),
        ('```json\n{"should_reply": true}\n```', True),
        ('{"should_reply": "true"}', False),
        ('{"should_reply": false}', False),
        ("not JSON", False),
        ("[]", False),
    ],
)
async def test_strict_judgment_and_default_provider(text, expected, context, event):
    context.llm_generate.return_value = SimpleNamespace(completion_text=text)
    judge = ReplyJudge(context, Settings(), logging.getLogger())
    assert await judge.should_reply(event, []) is expected
    assert context.llm_generate.call_args.kwargs["chat_provider_id"] == "default"


@pytest.mark.asyncio
async def test_selected_provider_and_failure_are_isolated(context, event):
    context.llm_generate.side_effect = RuntimeError("unavailable")
    judge = ReplyJudge(
        context, Settings(judge_provider_id="small"), logging.getLogger()
    )
    assert not await judge.should_reply(event, [])
    assert context.llm_generate.call_args.kwargs["chat_provider_id"] == "small"


@pytest.mark.asyncio
async def test_persona_uses_framework_selection_and_stable_prefix(context, event):
    judge = ReplyJudge(context, Settings(), logging.getLogger())
    first_messages = [{"id": 1, "text": "A first question"}]
    second_messages = [{"id": 2, "text": "A follow-up"}]
    assert await judge.should_reply(event, first_messages)
    assert await judge.should_reply(event, second_messages)
    calls = context.llm_generate.call_args_list
    assert calls[0].kwargs["system_prompt"] == calls[1].kwargs["system_prompt"]
    assert (
        "```quote\nYou enjoy discussing astronomy.\n```"
        in calls[0].kwargs["system_prompt"]
    )
    assert "A first question" not in calls[0].kwargs["system_prompt"]
    assert calls[0].kwargs["prompt"] != calls[1].kwargs["prompt"]
    context.persona_manager.resolve_selected_persona.assert_awaited_with(
        umo="group", conversation_persona_id="persona", platform_name="test"
    )


@pytest.mark.asyncio
async def test_new_session_resolves_default_persona_without_creating_conversation(
    context, event
):
    context.conversation_manager.get_curr_conversation_id.return_value = None
    judge = ReplyJudge(context, Settings(), logging.getLogger())
    assert await judge.should_reply(event, [])
    context.conversation_manager.get_conversation.assert_not_awaited()
    context.persona_manager.resolve_selected_persona.assert_awaited_once_with(
        umo="group", conversation_persona_id=None, platform_name="test"
    )


@pytest.mark.asyncio
async def test_persona_edits_and_disabled_persona_are_not_cached(context, event):
    judge = ReplyJudge(
        context, Settings(judge_provider_id="small"), logging.getLogger()
    )
    assert await judge.should_reply(event, [])
    context.persona_manager.resolve_selected_persona.return_value = (
        "changed",
        {"prompt": "A different persona"},
        "changed",
        False,
    )
    assert await judge.should_reply(event, [])
    updated = context.llm_generate.call_args.kwargs["system_prompt"]
    assert "A different persona" in updated
    assert "astronomy" not in updated
    context.persona_manager.resolve_selected_persona.return_value = (
        "[%None]",
        None,
        None,
        False,
    )
    assert await judge.should_reply(event, [])
    assert "```quote" not in context.llm_generate.call_args.kwargs["system_prompt"]
    assert context.llm_generate.call_args.kwargs["chat_provider_id"] == "small"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base, appended",
    [("Custom rules.\n", "Additional rules."), ("", "Append only"), ("", "")],
)
async def test_configured_instructions_are_concatenated_verbatim(
    context, event, base, appended
):
    context.persona_manager.resolve_selected_persona.return_value = (
        "[%None]",
        None,
        None,
        False,
    )
    judge = ReplyJudge(
        context,
        Settings(judge_prompt=base, judge_prompt_append=appended),
        logging.getLogger(),
    )
    assert await judge.should_reply(event, [{"text": "A chat message"}])
    request = context.llm_generate.call_args.kwargs
    assert request["system_prompt"] == base + appended
    assert "A chat message" in request["prompt"]


@pytest.mark.asyncio
async def test_persona_is_appended_after_both_configured_instruction_parts(
    context, event
):
    settings = Settings(
        judge_prompt="Base rules.\n", judge_prompt_append="Extra rules."
    )
    judge = ReplyJudge(context, settings, logging.getLogger())
    assert await judge.should_reply(event, [])
    system = context.llm_generate.call_args.kwargs["system_prompt"]
    prefix = settings.judge_prompt + settings.judge_prompt_append
    assert system.startswith(prefix)
    assert system.index("```quote") > len(prefix)
    assert "You enjoy discussing astronomy." in system


@pytest.mark.asyncio
async def test_judgment_history_is_compact_without_mutating_original_records(
    context, event
):
    messages = [
        {
            "id": 1,
            "timestamp": 0,
            "sender_id": "123456789012",
            "sender_name": "Alice",
            "text": "hello",
            "platform_message_id": "platform-message",
            "reply_to": "previous",
            "images": [{"index": 0, "available": True}],
        }
    ]
    original = copy.deepcopy(messages)
    judge = ReplyJudge(context, Settings(), logging.getLogger())
    assert await judge.should_reply(event, messages)
    record = json.loads(context.llm_generate.call_args.kwargs["prompt"])[0]
    assert record["timestamp"] == "1970-01-01 08:00:00+08:00"
    assert record["sender_id"] == "56789012"
    assert "platform_message_id" not in record
    assert record["text"] == "hello"
    assert record["reply_to"] == "previous"
    assert record["images"] == messages[0]["images"]
    assert messages == original


@pytest.mark.asyncio
@pytest.mark.parametrize("sender", ["short", "", None, 1234567890])
async def test_short_or_missing_sender_ids_are_preserved(context, event, sender):
    judge = ReplyJudge(context, Settings(), logging.getLogger())
    assert await judge.should_reply(event, [{"sender_id": sender, "timestamp": None}])
    record = json.loads(context.llm_generate.call_args.kwargs["prompt"])[0]
    assert record["sender_id"] == (str(sender)[-8:] if sender is not None else None)
    assert record["timestamp"] is None


@pytest.mark.asyncio
async def test_invalid_timezone_falls_back_to_readable_utc(context, event):
    context.get_config = lambda **kwargs: {"timezone": "Invalid/Zone"}
    judge = ReplyJudge(context, Settings(), logging.getLogger())
    assert await judge.should_reply(event, [{"timestamp": 0}])
    assert (
        json.loads(context.llm_generate.call_args.kwargs["prompt"])[0]["timestamp"]
        == "1970-01-01 00:00:00+00:00"
    )
