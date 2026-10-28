"""Text-to-image generation through the standard OpenAI Images API."""

import asyncio
import base64
import io
import json
import logging
import math

from astrbot.api.event import AstrMessageEvent
from astrbot.core.message.components import Image as ImageComponent
from astrbot.core.provider.provider import Provider
from astrbot.core.utils.network_utils import create_proxy_client
from openai import APIStatusError, APITimeoutError, AsyncOpenAI
from PIL import Image

MAX_IMAGE_BYTES = 20 * 1024 * 1024


def _validate_image(encoded: str) -> None:
    """Validate bounded PNG data off the event loop before platform submission.

    Args:
        encoded: Base64 image data returned by the generation endpoint.

    Raises:
        ValueError: If the image is oversized, invalid or not a PNG.
    """
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("The API did not return base64 image data")
    if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ValueError("Generated image exceeds the byte limit")
    data = base64.b64decode(encoded, validate=True)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Generated image exceeds the byte limit")
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PNG" or image.width * image.height > 16_000_000:
            raise ValueError("Generated image has an unsupported format or size")
        image.verify()


async def generate_image(
    event: AstrMessageEvent,
    provider: Provider | None,
    logger: logging.Logger,
    *,
    prompt: str,
    size: str = "auto",
    quality: str = "auto",
) -> str:
    """Generate and send one image without modifying the selected provider.

    Args:
        event: Event defining the destination chat.
        provider: Selected OpenAI-compatible AstrBot provider supplying the model
            name, credentials and connection settings.
        logger: Plugin logger; upstream error bodies and credentials are never logged.
        prompt: Text description of the requested image.
        size: Standard GPT Image resolution or auto.
        quality: Standard GPT Image quality or auto.

    Returns:
        JSON acknowledging a completed send call, or a safe explicit error.
    """
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 32000:
        return json.dumps({"error": "prompt must contain 1 to 32000 characters."})
    if size not in ("auto", "1024x1024", "1536x1024", "1024x1536"):
        return json.dumps({"error": "Unsupported image size."})
    if quality not in ("auto", "low", "medium", "high"):
        return json.dumps({"error": "Unsupported image quality."})
    if provider is None:
        return json.dumps(
            {
                "error": "Select an image generation provider in Mingle's toolbox settings."
            }
        )
    config = provider.provider_config
    if config.get("type") not in (
        "openai_chat_completion",
        "openai_responses",
    ) or config.get("api_version"):
        return json.dumps(
            {
                "error": "Select a standard OpenAI-compatible provider; Azure endpoints are unsupported."
            }
        )
    try:
        model = provider.get_model()
        if not isinstance(model, str) or not model.strip():
            return json.dumps({"error": "The selected provider has no model name."})
        key = provider.get_current_key()
        if not key:
            return json.dumps({"error": "The selected provider has no API key."})
        timeout = float(config.get("timeout", 120))
        if not math.isfinite(timeout) or timeout <= 0:
            return json.dumps(
                {
                    "error": "The selected provider timeout must be a positive finite number of seconds."
                }
            )
        async with create_proxy_client(
            "Mingle image generation", config.get("proxy")
        ) as http_client:
            async with AsyncOpenAI(
                api_key=key,
                base_url=config.get("api_base") or None,
                default_headers=provider.request_headers,
                http_client=http_client,
                timeout=timeout,
                max_retries=0,
            ) as client:
                response = await client.images.generate(
                    model=model,
                    prompt=prompt,
                    n=1,
                    size=size,
                    quality=quality,
                    output_format="png",
                )
        if not response.data or len(response.data) != 1:
            raise ValueError("Expected exactly one generated image")
        encoded = response.data[0].b64_json
        await asyncio.to_thread(_validate_image, encoded)
    except (APITimeoutError, TimeoutError):
        logger.warning("Mingle image generation request timed out")
        return json.dumps(
            {
                "error": "Image generation timed out; no image was sent. Check the provider timeout before trying again."
            }
        )
    except APIStatusError as error:
        logger.warning("Mingle image generation failed with HTTP %s", error.status_code)
        return json.dumps(
            {
                "error": "Image generation API request failed. Check the provider and image model configuration.",
                "status_code": error.status_code,
            }
        )
    except Exception as error:
        logger.warning("Mingle image generation failed (%s)", type(error).__name__)
        return json.dumps(
            {
                "error": "Image generation failed or returned invalid image data. No image was sent."
            }
        )
    try:
        await event.send(event.chain_result([ImageComponent.fromBase64(encoded)]))
    except Exception as error:
        logger.warning("Mingle generated image send failed (%s)", type(error).__name__)
        return json.dumps(
            {
                "error": "Image generated, but the platform send failed; delivery is unconfirmed. Do not automatically regenerate."
            }
        )
    return json.dumps(
        {
            "sent": True,
            "count": 1,
            "message": "Image sent to the current chat. Do not send it again.",
        }
    )
