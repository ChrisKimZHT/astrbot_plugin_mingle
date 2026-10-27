import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mingle_tests_plugin.decision import ReplyJudge
from mingle_tests_plugin.settings import Settings


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
async def test_strict_judgment_and_default_provider(text, expected):
    context = SimpleNamespace(
        get_current_chat_provider_id=AsyncMock(return_value="default"),
        llm_generate=AsyncMock(return_value=SimpleNamespace(completion_text=text)),
    )
    judge = ReplyJudge(context, Settings(), logging.getLogger())
    assert (
        await judge.should_reply(SimpleNamespace(unified_msg_origin="group"), [])
        is expected
    )
    assert context.llm_generate.call_args.kwargs["chat_provider_id"] == "default"


@pytest.mark.asyncio
async def test_selected_provider_and_failure_are_isolated():
    context = SimpleNamespace(
        llm_generate=AsyncMock(side_effect=RuntimeError("unavailable"))
    )
    judge = ReplyJudge(
        context, Settings(judge_provider_id="small"), logging.getLogger()
    )
    assert not await judge.should_reply(SimpleNamespace(unified_msg_origin="group"), [])
    assert context.llm_generate.call_args.kwargs["chat_provider_id"] == "small"
