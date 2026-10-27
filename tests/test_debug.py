import asyncio
import json
import logging
from dataclasses import replace
from html.parser import HTMLParser
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from mingle_tests_plugin.src.debug import DebugDumper, render_debug_html
from mingle_tests_plugin.src.decision import ReplyJudge
from mingle_tests_plugin.src.settings import Settings


@pytest.fixture
def event():
    return SimpleNamespace(
        unified_msg_origin="p:GroupMessage:group",
        is_private_chat=lambda: False,
        get_group_id=lambda: "group",
        get_platform_id=lambda: "p",
        get_platform_name=lambda: "test",
    )


@pytest.fixture
def dumper(tmp_path):
    return DebugDumper(
        tmp_path / "debug",
        Settings(debug_sessions=("p:GroupMessage:group",)),
        logging.getLogger("mingle.debug.test"),
    )


@pytest.mark.asyncio
async def test_dump_files_are_correlated_unique_and_paths_logged(dumper, event, caplog):
    with caplog.at_level(logging.INFO, logger="mingle.debug.test"):
        paths = await asyncio.gather(
            *[
                dumper.dump(event, "judge", phase, {"message": "你好"}, "request-1")
                for phase in ("input", "output", "output")
            ]
        )
    assert len(set(paths)) == 3
    for path in paths:
        assert path.is_absolute()
        html_path = path.with_suffix(".html")
        assert html_path.is_file()
        assert str(html_path) in caplog.text
        assert str(path) not in caplog.text
        content = json.loads(path.read_text(encoding="utf-8"))
        assert content["request_id"] == "request-1"
        assert content["capture_layer"] == "plugin"
        assert content["session"] == event.unified_msg_origin
        assert content["payload"]["message"] == "你好"
    assert not list(dumper.directory.rglob("*.tmp"))


@pytest.mark.asyncio
async def test_disabled_and_other_session_requests_do_not_create_files(dumper, event):
    dumper.settings = Settings()
    assert await dumper.dump(event, "agent", "input", {}, "request") is None
    assert not dumper.directory.exists()
    dumper.settings = Settings(debug_sessions=("other:GroupMessage:group",))
    assert await dumper.dump(event, "agent", "input", {}, "request") is None
    assert not dumper.directory.exists()


def test_debug_scope_supports_private_and_isolated_group_members(dumper, event):
    event.unified_msg_origin = "p:GroupMessage:user_group"
    assert dumper.is_enabled(event)
    event.unified_msg_origin = "p:FriendMessage:user"
    event.is_private_chat = lambda: True
    assert not dumper.is_enabled(event)
    dumper.settings = Settings(debug_sessions=(event.unified_msg_origin,))
    assert dumper.is_enabled(event)


@pytest.mark.asyncio
async def test_request_dump_serializes_tools_without_executable_handlers(dumper, event):
    tool = FunctionTool(
        name="lookup",
        description="Look up a value",
        parameters={"type": "object"},
        handler=lambda: None,
    )
    request = ProviderRequest(
        prompt="question",
        system_prompt="persona",
        contexts=[{"role": "assistant", "content": "previous reply"}],
        func_tool=ToolSet([tool]),
    )
    path = await dumper.dump(event, "agent", "input", {"request": request}, "request")
    payload = json.loads(path.read_text(encoding="utf-8"))["payload"]["request"]
    assert payload["prompt"] == "question"
    assert payload["system_prompt"] == "persona"
    assert payload["contexts"] == request.contexts
    assert payload["func_tool"]["tools"][0]["name"] == "lookup"
    assert "handler" not in payload["func_tool"]["tools"][0]


@pytest.mark.asyncio
async def test_write_failure_does_not_propagate(dumper, event, tmp_path, caplog):
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    dumper.directory = blocked
    assert await dumper.dump(event, "agent", "input", {}, "request") is None
    assert "Failed to write Mingle debug dump" in caplog.text
    assert "debug dump written" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, TimeoutError("provider timed out")])
async def test_judge_dumps_input_and_output_or_error(dumper, event, failure):
    context = SimpleNamespace(
        get_config=lambda **kwargs: {"timezone": "UTC"},
        conversation_manager=SimpleNamespace(
            get_curr_conversation_id=AsyncMock(return_value=None),
        ),
        persona_manager=SimpleNamespace(
            resolve_selected_persona=AsyncMock(
                return_value=("persona", {"prompt": "Be friendly"}, None, False)
            )
        ),
        llm_generate=AsyncMock(
            return_value=LLMResponse(
                role="assistant", completion_text='{"should_reply": true}'
            ),
            side_effect=failure,
        ),
    )
    settings = replace(dumper.settings, judge_provider_id="judge")
    judge = ReplyJudge(context, settings, logging.getLogger(), dumper)
    messages = [{"text": '聊天 "内容"\n第二行'}]
    assert await judge.should_reply(event, messages) is (failure is None)
    documents = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in dumper.directory.rglob("*.json")
    ]
    assert {item["phase"] for item in documents} == {
        "input",
        "output" if failure is None else "error",
    }
    assert len({item["request_id"] for item in documents}) == 1
    request = next(item for item in documents if item["phase"] == "input")["payload"]
    assert "Be friendly" in request["system_prompt"]
    assert request["prompt"] == context.llm_generate.call_args.kwargs["prompt"]
    assert isinstance(request["prompt"], str)
    assert json.loads(request["prompt"]) == messages
    assert json.loads(context.llm_generate.call_args.kwargs["prompt"]) == messages


class DebugHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.text = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        if tag == "a":
            self.links.append(dict(attrs))

    def handle_data(self, data):
        self.text.append(data)


def test_html_contains_all_fields_and_escapes_untrusted_content():
    attack = '<script>alert("test")</script>\n<img src=x onerror=alert(1)>'
    prompt = json.dumps([{"sender": "群友", "text": attack}], ensure_ascii=False)
    document = {
        "request_id": "id-123",
        "payload": {
            "system_prompt": "System prompt\nSecond line",
            "prompt": prompt,
            "response": {"raw_completion": {"content": attack}},
            "records": [{"id": 1, "reply_to": None}, {"id": 2, "enabled": False}],
            "mixed": [True, 3.5, None, "plain", [], {}],
            "empty_records": [{}, {}],
        },
        attack: "Untrusted key",
    }
    html = render_debug_html(document, 'original" onmouseover="bad.json')
    parser = DebugHTMLParser()
    parser.feed(html)
    for text in (
        "request_id",
        "id-123",
        "payload",
        "system_prompt",
        "System prompt\nSecond line",
        "prompt",
        prompt,
        "sender",
        "群友",
        "text",
        attack,
        "response",
        "raw_completion",
        "content",
        "records",
        "id",
        "1",
        "2",
        "reply_to",
        "null",
        "enabled",
        "false",
        "mixed",
        "true",
        "3.5",
        "plain",
        "[]",
        "{}",
        "empty_records",
        "Untrusted key",
    ):
        assert text in parser.text
    assert parser.tags.count("table") > 3
    assert "details" in parser.tags
    assert not {"script", "img", "iframe", "link"}.intersection(parser.tags)
    assert parser.tags.count("style") == 1
    assert parser.links == [{"href": 'original" onmouseover="bad.json'}]
    assert "(missing)" in parser.text


@pytest.mark.asyncio
async def test_html_render_failure_preserves_json_without_logging_success(
    dumper, event, monkeypatch, caplog
):
    def fail_render(*args):
        raise ValueError("render failed")

    monkeypatch.setattr("mingle_tests_plugin.src.debug.render_debug_html", fail_render)
    assert (
        await dumper.dump(event, "agent", "input", {"prompt": "question"}, "request")
        is None
    )
    files = list(dumper.directory.rglob("*.json"))
    assert len(files) == 1
    assert (
        json.loads(files[0].read_text(encoding="utf-8"))["payload"]["prompt"]
        == "question"
    )
    assert not list(dumper.directory.rglob("*.html"))
    assert not list(dumper.directory.rglob("*.tmp"))
    assert "Failed to write Mingle debug dump" in caplog.text
    assert "Mingle debug dump written" not in caplog.text
