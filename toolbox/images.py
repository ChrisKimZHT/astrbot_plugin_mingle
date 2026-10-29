"""Image generation and reference editing through the OpenAI Images API."""

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

from ..src.history import ChatHistory, boundary_group
from ..src.utils.image_preprocessing import prepare_image

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
    max_encoded_length = 4 * ((MAX_IMAGE_BYTES + 2) // 3)
    if len(encoded) > max_encoded_length:
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
    """Generate a new image from text and send it to the current chat.

    Args:
        event: Event defining the destination chat.
        provider: Explicitly selected OpenAI-compatible image provider.
        logger: Logger for safe failure metadata.
        prompt: Description of the requested image.
        size: Output resolution or auto.
        quality: Output quality or auto.

    Returns:
        JSON send result or an explicit error, without image content.
    """
    return await _request_image(
        event,
        provider,
        logger,
        prompt=prompt,
        size=size,
        quality=quality,
    )


async def _request_image(
    event: AstrMessageEvent,
    provider: Provider | None,
    logger: logging.Logger,
    *,
    prompt: str,
    size: str = "auto",
    quality: str = "auto",
    reference_images: list[tuple[str, bytes, str]] | None = None,
) -> str:
    """Request and send one image without modifying the selected provider.

    Args:
        event: Event defining the destination chat.
        provider: Selected OpenAI-compatible AstrBot provider supplying the model
            name, credentials and connection settings.
        logger: Plugin logger; upstream error bodies and credentials are never logged.
        prompt: Text description of the requested image.
        size: Standard GPT Image resolution or auto.
        quality: Standard GPT Image quality or auto.
        reference_images: Validated uploads for editing, or None for text generation.

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
    supported_provider = config.get("type") in (
        "openai_chat_completion",
        "openai_responses",
    )
    if not supported_provider or config.get("api_version"):
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
                if reference_images:
                    method = client.images.edit
                    image_args = {"image": reference_images}
                else:
                    method = client.images.generate
                    image_args = {}
                response = await method(
                    **image_args,
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


async def edit_image(
    event: AstrMessageEvent,
    history: ChatHistory,
    provider: Provider | None,
    logger: logging.Logger,
    *,
    prompt: str,
    record_ids: list[int],
    image_indices: list[int],
    size: str = "auto",
    quality: str = "auto",
) -> str:
    """Load authorized reference images and submit a single paid edit request.

    Args:
        event: Current managed event defining session and destination.
        history: Mingle image storage enforcing session and boundary visibility.
        provider: Explicitly selected OpenAI-compatible image provider.
        logger: Logger for safe failure metadata.
        prompt: Instructions defining changes and reference roles.
        record_ids: One to four Mingle records, in reference order.
        image_indices: Matching zero-based image indices.
        size: Output resolution or auto.
        quality: Output quality or auto.

    Returns:
        JSON send result or an explicit error. No image content is returned.
    """
    if not event.get_extra("mingle_managed", False):
        return json.dumps({"error": "Image editing requires a managed Mingle turn."})
    if (
        not isinstance(record_ids, list)
        or not isinstance(image_indices, list)
        or not 1 <= len(record_ids) <= 4
        or len(record_ids) != len(image_indices)
        or any(type(value) is not int or value <= 0 for value in record_ids)
        or any(type(value) is not int or value < 0 for value in image_indices)
    ):
        return json.dumps(
            {
                "error": "Supply one to four record IDs and matching nonnegative image indices."
            }
        )
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 32000:
        return json.dumps({"error": "prompt must contain 1 to 32000 characters."})
    references = []
    total_bytes = 0
    for record_id, image_index in zip(record_ids, image_indices):
        image = await history.read_image(
            event.unified_msg_origin,
            record_id,
            image_index,
            boundary=boundary_group(event),
        )
        if image is None:
            return json.dumps(
                {
                    "error": "Reference image unavailable, expired, or outside the current session or boundary."
                }
            )
        data, mime = image
        if len(data) > MAX_IMAGE_BYTES:
            return json.dumps({"error": "Reference image exceeds the byte limit."})
        try:
            data, mime = await prepare_image(data, mime)
        except Exception as error:
            logger.warning(
                "Mingle edit image preparation failed (%s)", type(error).__name__
            )
            return json.dumps({"error": "Reference image preparation failed."})
        total_bytes += len(data)
        if total_bytes > MAX_IMAGE_BYTES:
            return json.dumps(
                {"error": "Combined reference images exceed the byte limit."}
            )
        extensions = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}
        extension = extensions.get(mime)
        if extension is None:
            return json.dumps({"error": "Unsupported prepared reference image format."})
        filename = f"reference_{len(references) + 1}.{extension}"
        references.append((filename, data, mime))
    return await _request_image(
        event,
        provider,
        logger,
        prompt=prompt,
        size=size,
        quality=quality,
        reference_images=references,
    )
