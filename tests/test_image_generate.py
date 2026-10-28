import asyncio
import base64
import io
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from mingle_tests_plugin.toolbox import image_generate
from PIL import Image


@pytest.fixture
def image_request(monkeypatch):
    with io.BytesIO() as buffer:
        Image.new("RGB", (8, 8), "blue").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    provider = SimpleNamespace(
        provider_config={
            "type": "openai_responses",
            "api_base": "https://images.example/v1",
            "timeout": "90",
            "proxy": "http://proxy.example:8080",
        },
        get_current_key=Mock(return_value="test-current-key"),
        get_model=Mock(return_value="custom-image-model"),
        request_headers={"X-Test-Header": "selected-provider"},
    )
    event = SimpleNamespace(send=AsyncMock(), chain_result=lambda chain: chain)
    calls = []
    clients = []
    result = {"created": 1, "data": [{"b64_json": encoded}]}

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=result)

    def client_factory(label, proxy):
        assert proxy == provider.provider_config["proxy"]
        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        clients.append(client)
        return client

    monkeypatch.setattr(image_generate, "create_proxy_client", client_factory)
    return SimpleNamespace(
        provider=provider,
        event=event,
        calls=calls,
        clients=clients,
        result=result,
        encoded=encoded,
        logger=logging.getLogger("mingle.image.test"),
    )


async def test_standard_openai_request_sends_once_without_returning_image(
    image_request,
):
    case = image_request
    config = case.provider.provider_config.copy()
    result = json.loads(
        await image_generate.generate_image(
            case.event,
            case.provider,
            case.logger,
            prompt="Draw a blue bird",
            size="1536x1024",
            quality="high",
        )
    )
    assert result["sent"] is True
    assert result["count"] == 1
    assert case.encoded not in json.dumps(result)
    assert len(case.calls) == 1
    request = case.calls[0]
    assert request.method == "POST"
    assert str(request.url) == "https://images.example/v1/images/generations"
    assert request.headers["authorization"] == "Bearer test-current-key"
    assert request.headers["x-test-header"] == "selected-provider"
    assert request.extensions["timeout"]["read"] == 90
    assert json.loads(request.content) == {
        "model": "custom-image-model",
        "prompt": "Draw a blue bird",
        "n": 1,
        "size": "1536x1024",
        "quality": "high",
        "output_format": "png",
    }
    case.event.send.assert_awaited_once()
    assert case.event.send.await_args.args[0][0].file == f"base64://{case.encoded}"
    assert all(client.is_closed for client in case.clients)
    assert case.provider.provider_config == config


async def test_image_model_is_resolved_from_provider_on_each_call(image_request):
    case = image_request
    case.provider.get_model.side_effect = ["first-image-model", "second-image-model"]
    for _ in range(2):
        result = json.loads(
            await image_generate.generate_image(
                case.event, case.provider, case.logger, prompt="bird"
            )
        )
        assert result["sent"] is True
    assert [json.loads(request.content)["model"] for request in case.calls] == [
        "first-image-model",
        "second-image-model",
    ]
    assert case.event.send.await_count == 2


@pytest.mark.parametrize(
    "arguments",
    [
        {"prompt": ""},
        {"prompt": " "},
        {"prompt": "x" * 32001},
        {"prompt": None},
        {"prompt": "bird", "size": "4000x4000"},
        {"prompt": "bird", "quality": "hd"},
    ],
)
async def test_invalid_arguments_never_call_api(image_request, arguments):
    case = image_request
    result = await image_generate.generate_image(
        case.event, case.provider, case.logger, **arguments
    )
    assert "error" in json.loads(result)
    assert not case.calls
    case.event.send.assert_not_awaited()


@pytest.mark.parametrize(
    "problem",
    ["missing", "type", "azure", "key", "model", "timeout", "nan", "infinity"],
)
async def test_unusable_provider_fails_before_network(image_request, problem):
    case = image_request
    provider = case.provider
    if problem == "missing":
        provider = None
    elif problem == "type":
        provider.provider_config["type"] = "anthropic"
    elif problem == "azure":
        provider.provider_config["api_version"] = "2025-01-01"
    elif problem == "key":
        provider.get_current_key.return_value = ""
    elif problem == "model":
        provider.get_model.return_value = " "
    elif problem == "timeout":
        provider.provider_config["timeout"] = 0
    elif problem == "nan":
        provider.provider_config["timeout"] = float("nan")
    else:
        provider.provider_config["timeout"] = float("inf")
    result = await image_generate.generate_image(
        case.event, provider, case.logger, prompt="bird"
    )
    assert "error" in json.loads(result)
    assert not case.calls
    case.event.send.assert_not_awaited()


@pytest.mark.parametrize(
    "data",
    [
        [],
        [{"url": "https://untrusted.example/image.png"}],
        [{"b64_json": "broken base64"}],
        [{"b64_json": base64.b64encode(b"not an image").decode()}],
        [{"b64_json": "a"}, {"b64_json": "b"}],
    ],
)
async def test_bad_responses_are_never_sent_or_fetched(image_request, data):
    case = image_request
    case.result["data"] = data
    result = await image_generate.generate_image(
        case.event, case.provider, case.logger, prompt="bird"
    )
    assert "error" in json.loads(result)
    assert len(case.calls) == 1
    assert all(client.is_closed for client in case.clients)
    case.event.send.assert_not_awaited()


def test_generated_image_validation_bounds_bytes_and_format(image_request, monkeypatch):
    with io.BytesIO() as buffer:
        Image.new("RGB", (2, 2)).save(buffer, format="JPEG")
        jpeg = base64.b64encode(buffer.getvalue()).decode()
    with pytest.raises(ValueError):
        image_generate._validate_image(jpeg)
    monkeypatch.setattr(image_generate, "MAX_IMAGE_BYTES", 4)
    with pytest.raises(ValueError):
        image_generate._validate_image(image_request.encoded)
    with pytest.raises(ValueError):
        image_generate._validate_image(base64.b64encode(b"12345").decode())


async def test_api_failure_is_safe_and_not_retried(image_request, monkeypatch, caplog):
    case = image_request
    requests = []

    def fail(request):
        requests.append(request)
        return httpx.Response(
            429,
            json={"error": {"message": "SECRET-UPSTREAM-BODY", "type": "rate_limit"}},
        )

    monkeypatch.setattr(
        image_generate,
        "create_proxy_client",
        lambda *args: httpx.AsyncClient(transport=httpx.MockTransport(fail)),
    )
    result = await image_generate.generate_image(
        case.event, case.provider, case.logger, prompt="bird"
    )
    assert json.loads(result)["status_code"] == 429
    assert len(requests) == 1
    assert "SECRET-UPSTREAM-BODY" not in result + caplog.text
    assert "test-current-key" not in result + caplog.text
    case.event.send.assert_not_awaited()


async def test_platform_failure_does_not_claim_success_or_regenerate(image_request):
    case = image_request
    case.event.send.side_effect = RuntimeError("SECRET-PLATFORM-BODY")
    result = json.loads(
        await image_generate.generate_image(
            case.event, case.provider, case.logger, prompt="bird"
        )
    )
    assert "sent" not in result
    assert "platform send failed" in result["error"]
    assert "SECRET-PLATFORM-BODY" not in json.dumps(result)
    assert len(case.calls) == 1
    case.event.send.assert_awaited_once()


async def test_api_timeout_is_not_retried_or_sent(image_request, monkeypatch):
    case = image_request
    requests = []

    def timeout(request):
        requests.append(request)
        raise httpx.ReadTimeout("SECRET-UPSTREAM-BODY", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(timeout))
    monkeypatch.setattr(image_generate, "create_proxy_client", lambda *args: client)
    result = json.loads(
        await image_generate.generate_image(
            case.event, case.provider, case.logger, prompt="bird"
        )
    )
    assert "timed out" in result["error"]
    assert "SECRET-UPSTREAM-BODY" not in json.dumps(result)
    assert len(requests) == 1
    assert client.is_closed
    case.event.send.assert_not_awaited()


async def test_cancellation_propagates_without_sending(image_request, monkeypatch):
    case = image_request
    started = asyncio.Event()

    async def blocked(request):
        started.set()
        await asyncio.Event().wait()

    client = httpx.AsyncClient(transport=httpx.MockTransport(blocked))
    monkeypatch.setattr(image_generate, "create_proxy_client", lambda *args: client)
    task = asyncio.create_task(
        image_generate.generate_image(
            case.event, case.provider, case.logger, prompt="bird"
        )
    )
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.is_closed
    case.event.send.assert_not_awaited()
