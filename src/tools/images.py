import asyncio
import base64
import logging

from astrbot.api.event import AstrMessageEvent
from mcp.types import CallToolResult, ImageContent, TextContent

from ..history import ChatHistory
from ..images import prepare_image


async def view_image(
    event: AstrMessageEvent,
    history: ChatHistory,
    logger: logging.Logger,
    *,
    available: bool,
    record_id: int,
    image_index: int = 0,
) -> CallToolResult:
    """Read and prepare a cached image from the managed execution's session.

    Args:
        event: Event defining the session and managed execution status.
        history: Plugin-owned image and message store.
        logger: Plugin logger for image preparation failures.
        available: Whether the plugin's storage is ready for use.
        record_id: Message record referencing the image.
        image_index: Zero-based image index within the record.

    Returns:
        Prepared image content or an explicit unavailable or error result.
    """
    image = None
    if available and event.get_extra("mingle_managed", False):
        image = await history.read_image(
            event.unified_msg_origin, record_id, image_index
        )
    if image is None:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text="Image unavailable, expired, or outside the current session.",
                )
            ]
        )
    data, mime = image
    try:
        data, mime = await prepare_image(data, mime)
    except Exception:
        logger.exception("Failed to prepare Mingle image for record %s", record_id)
        return CallToolResult(
            content=[TextContent(type="text", text="Image preparation failed.")],
            isError=True,
        )
    encoded = await asyncio.to_thread(base64.b64encode, data)
    return CallToolResult(
        content=[
            ImageContent(type="image", data=encoded.decode("ascii"), mimeType=mime)
        ]
    )
