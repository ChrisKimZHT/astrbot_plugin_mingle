import asyncio
import base64
import io
from pathlib import Path

from astrbot.core.utils.media_utils import (
    compress_image,
    detect_image_mime_type_async,
)
from PIL import Image


async def prepare_image(data: bytes, mime: str) -> tuple[bytes, str]:
    """Prepare a static, compressed image without modifying the original cache.

    Args:
        data: Original encoded image bytes.
        mime: Original image MIME type.

    Returns:
        Encoded image bytes and their actual MIME type for model input.
    """
    if mime == "image/gif":
        # The framework compressor skips animations, so give it a static frame.
        with await asyncio.to_thread(Image.open, io.BytesIO(data)) as image:
            with await asyncio.to_thread(image.convert, "RGBA") as frame:
                with io.BytesIO() as buffer:
                    await asyncio.to_thread(frame.save, buffer, format="PNG")
                    data = buffer.getvalue()
        mime = "image/png"

    encoded = await asyncio.to_thread(base64.b64encode, data)
    source = f"data:{mime};base64,{encoded.decode('ascii')}"
    compressed = await compress_image(source, max_size=1280, quality=85)
    if compressed == source:
        return data, mime

    path = Path(compressed)
    try:
        output = await asyncio.to_thread(path.read_bytes)
        output_mime = await detect_image_mime_type_async(
            str(path), default_mime_type=None
        )
        if not output_mime or not output_mime.startswith("image/"):
            raise ValueError("Compressed output is not a recognized image")
        return output, output_mime
    finally:
        await asyncio.to_thread(path.unlink, missing_ok=True)
