import asyncio
import copy
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mingle_tests_plugin.src.decision import ReplyDecision, ReplyJudge
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
    record = json.loads(context.llm_generate.call_args.kwargs["prompt"])["messages"][0]
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
    record = json.loads(context.llm_generate.call_args.kwargs["prompt"])["messages"][0]
    assert record["sender_id"] == (str(sender)[-8:] if sender is not None else None)
    assert record["timestamp"] is None


@pytest.mark.asyncio
async def test_invalid_timezone_falls_back_to_readable_utc(context, event):
    context.get_config = lambda **kwargs: {"timezone": "Invalid/Zone"}
    judge = ReplyJudge(context, Settings(), logging.getLogger())
    assert await judge.should_reply(event, [{"timestamp": 0}])
    assert (
        json.loads(context.llm_generate.call_args.kwargs["prompt"])["messages"][0][
            "timestamp"
        ]
        == "1970-01-01 00:00:00+00:00"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result, expected",
    [
        ({"should_reply": True, "action": "rerun"}, ReplyDecision(True, "rerun")),
        ({"should_reply": False, "action": "stop"}, ReplyDecision(False, "stop")),
        ({"should_reply": True, "action": "wait"}, ReplyDecision(True)),
        ({"should_reply": True}, ReplyDecision(True)),
        ({"should_reply": True, "action": "stop"}, ReplyDecision()),
        ({"should_reply": False, "action": "rerun"}, ReplyDecision()),
        ({"should_reply": "true", "action": "rerun"}, ReplyDecision()),
        ({"should_reply": True, "action": "interrupt"}, ReplyDecision()),
    ],
)
async def test_busy_decisions_are_validated_before_control_actions(
    context, event, result, expected
):
    context.llm_generate.return_value.completion_text = json.dumps(result)
    judge = ReplyJudge(context, Settings(), logging.getLogger())
    actual = await judge.evaluate(
        event, [], message_status={"active": [1]}, new_message_ids=[2]
    )
    assert actual == expected


@pytest.mark.asyncio
async def test_busy_context_is_explicit_without_changing_the_system_prefix(
    context, event
):
    judge = ReplyJudge(context, Settings(), logging.getLogger())
    await judge.should_reply(event, [])
    original = context.llm_generate.call_args.kwargs["system_prompt"]
    status = {
        "finished": [{"id": 1, "outcome": "completed"}],
        "active": [2],
        "queued": [3],
        "awaiting_judgment": [5],
        "judging": [4],
    }
    await judge.evaluate(
        event,
        [{"id": number, "text": f"Message {number}"} for number in range(6)],
        new_message_ids=[4],
        message_status=status,
    )
    request = context.llm_generate.call_args.kwargs
    assert request["system_prompt"] == original
    payload = json.loads(request["prompt"])
    assert "message_status" not in payload
    assert "new_message_ids" not in payload
    assert "running_agent" not in payload
    records = {item["id"]: item for item in payload["messages"]}
    assert [records[number]["status"] for number in range(6)] == [
        "history",
        "history",
        "processing",
        "queued",
        "new",
        "history",
    ]
    assert all(
        "is_new" not in item and "sent_to_agent" not in item
        for item in payload["messages"]
    )
    assert [item["id"] for item in payload["messages"] if item["status"] == "new"] == [
        4
    ]
    assert records[4]["text"] == "Message 4"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "age, expected",
    [
        (0, "0 seconds"),
        (59, "59 seconds"),
        (60, "1 minute"),
        (3599, "59 minutes"),
        (3600, "1 hour"),
        (86400, "1 day"),
        (172800, "2 days"),
        (-10, "0 seconds"),
        (None, "unknown"),
        (float("nan"), "unknown"),
    ],
)
async def test_judgment_time_context(context, event, monkeypatch, age, expected):
    from datetime import datetime, timezone

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 29, 7, 30, 45, tzinfo=timezone.utc)

    monkeypatch.setattr("mingle_tests_plugin.src.decision.datetime", FixedDatetime)
    now = FixedDatetime.now().timestamp()
    messages = [
        {"id": 1, "timestamp": now - 999999},
        {"id": 2, "timestamp": None if age is None else now - age},
        {"id": 3, "timestamp": now - 1},
        {"id": 4, "timestamp": now},
    ]
    judge = ReplyJudge(context, Settings(), logging.getLogger())
    assert await judge.should_reply(
        event,
        messages,
        new_message_ids=[3, 4],
        message_status={"active": [2]},
    )
    request = context.llm_generate.call_args.kwargs
    payload = json.loads(request["prompt"])
    assert list(payload) == ["messages", "time_context"]
    assert payload["time_context"].startswith(
        "Current time: 2026-10-29 15:30:45+08:00\n"
        "Time since the last message before the new messages: "
    )
    assert payload["time_context"].split(": ")[-1].startswith(expected)
    assert "Current time:" not in request["system_prompt"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "messages,new_ids",
    [
        ([], []),
        ([{"id": 1, "timestamp": 0}], [1]),
        ([{"id": 1, "timestamp": 0}], []),
        ([{"id": 1, "timestamp": 0}], [2]),
    ],
)
async def test_time_context_without_previous_message(context, event, messages, new_ids):
    judge = ReplyJudge(context, Settings(), logging.getLogger())
    assert await judge.should_reply(event, messages, new_message_ids=new_ids)
    payload = json.loads(context.llm_generate.call_args.kwargs["prompt"])
    assert payload["time_context"].endswith(
        "unknown (no valid message timestamp before new messages)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "votes,expected,choices",
    [
        (
            [(True, "rerun")] * 2 + [(False, "wait")] * 2 + [(False, "stop")],
            ReplyDecision(False, "wait"),
            [[False], ["wait"]],
        ),
        (
            [(False, "stop")] * 2 + [(True, "wait")] * 2 + [(True, "rerun")],
            ReplyDecision(True, "wait"),
            [[True], ["wait"]],
        ),
        (
            [(True, "rerun"), (False, "wait")],
            ReplyDecision(True, "rerun"),
            [[True, False], ["rerun", "wait"]],
        ),
        (
            [(True, "wait"), (False, "stop"), (False, "stop")],
            ReplyDecision(False, "stop"),
            [[False], ["stop"]],
        ),
    ],
)
async def test_parallel_two_level_vote(
    context, event, monkeypatch, votes, expected, choices
):
    selected = []

    def choose(values):
        selected.append(values)
        return values[0]

    monkeypatch.setattr("mingle_tests_plugin.src.decision.random.choice", choose)
    context.llm_generate.side_effect = [
        SimpleNamespace(
            completion_text=json.dumps({"should_reply": reply, "action": action})
        )
        for reply, action in votes
    ]
    judge = ReplyJudge(
        context, Settings(judge_parallel_requests=len(votes)), logging.getLogger()
    )
    assert await judge.evaluate(event, []) == expected
    assert selected == choices
    assert context.llm_generate.await_count == len(votes)
    assert all(
        call == context.llm_generate.call_args_list[0]
        for call in context.llm_generate.call_args_list
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["should_reply", "evaluate"])
@pytest.mark.parametrize("waking,expected_count", [(False, 2), (True, 3)])
async def test_parallel_count_follows_batch_wake_state(
    context, event, method, waking, expected_count
):
    judge = ReplyJudge(
        context,
        Settings(judge_parallel_requests=2, judge_wake_parallel_requests=3),
        logging.getLogger(),
    )
    evaluate = getattr(judge, method)
    await evaluate(event, [], waking=waking)
    assert context.llm_generate.await_count == expected_count
    assert judge.settings.judge_parallel_requests == 2
    assert judge.settings.judge_wake_parallel_requests == 3


@pytest.mark.asyncio
async def test_parallel_requests_overlap_and_have_distinct_debug_ids(context, event):
    started = 0
    ready = asyncio.Event()

    async def generate(**kwargs):
        nonlocal started
        started += 1
        if started == 3:
            ready.set()
        await ready.wait()
        return SimpleNamespace(completion_text='{"should_reply": true}')

    context.llm_generate.side_effect = generate
    debug = SimpleNamespace(dump=AsyncMock())
    judge = ReplyJudge(
        context, Settings(judge_parallel_requests=3), logging.getLogger(), debug
    )
    assert (await asyncio.wait_for(judge.evaluate(event, []), 2)).should_reply
    calls = debug.dump.call_args_list
    inputs = [call.args for call in calls if call.args[2] == "input"]
    outputs = [call.args for call in calls if call.args[2] == "output"]
    assert len(inputs) == len(outputs) == 3
    assert len({args[4] for args in inputs}) == 3
    assert {args[4] for args in inputs} == {args[4] for args in outputs}
    assert all(args[3] == inputs[0][3] for args in inputs)


@pytest.mark.asyncio
async def test_parallel_failures_and_timeouts_vote_silent(context, event):
    async def generate(**kwargs):
        index = context.llm_generate.await_count
        if index == 1:
            return SimpleNamespace(
                completion_text='{"should_reply": true, "action": "rerun"}'
            )
        if index == 2:
            raise RuntimeError("unavailable")
        if index == 3:
            return SimpleNamespace(completion_text="invalid")
        await asyncio.Event().wait()

    context.llm_generate.side_effect = generate
    judge = ReplyJudge(
        context,
        Settings(judge_parallel_requests=4, judge_timeout=0.02),
        logging.getLogger(),
    )
    assert await judge.evaluate(event, []) == ReplyDecision()
    assert context.llm_generate.await_count == 4


@pytest.mark.asyncio
async def test_parallel_cancellation_cancels_all_requests(context, event):
    started = 0
    cancelled = 0
    ready = asyncio.Event()

    async def generate(**kwargs):
        nonlocal started, cancelled
        started += 1
        if started == 3:
            ready.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled += 1

    context.llm_generate.side_effect = generate
    judge = ReplyJudge(
        context, Settings(judge_parallel_requests=3), logging.getLogger()
    )
    task = asyncio.create_task(judge.evaluate(event, []))
    await asyncio.wait_for(ready.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled == 3
