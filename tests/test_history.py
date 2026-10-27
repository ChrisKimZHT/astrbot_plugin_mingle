import base64
import logging
from dataclasses import replace
from types import SimpleNamespace

import pytest
import pytest_asyncio
from astrbot.core.message.components import Image, Plain, Reply
from mingle_tests_plugin.src.history import ChatHistory
from mingle_tests_plugin.src.settings import Settings


class Event:
    def __init__(self, message_id="1", session="p:GroupMessage:g"):
        self.unified_msg_origin = session
        self.message_obj = SimpleNamespace(message_id=message_id)

    def get_sender_id(self):
        return "alice"

    def get_sender_name(self):
        return "Alice"

    def get_self_id(self):
        return "bot"


@pytest_asyncio.fixture
async def history(tmp_path):
    store = ChatHistory(tmp_path, Settings(), logging.getLogger("mingle.test"))
    await store.initialize()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_deduplication_pagination_and_scope(history):
    ids = []
    for number in range(5):
        ids.append(await history.record(Event(str(number)), [Plain(f"line {number}")]))
    assert await history.record(Event("0"), [Plain("duplicate")]) is None
    other = await history.record(Event("0", "other"), [Plain("private")])
    assert other is not None
    assert [r["id"] for r in await history.get("p:GroupMessage:g", limit=2)] == ids[-2:]
    assert [
        r["id"] for r in await history.get("p:GroupMessage:g", before_id=ids[2])
    ] == ids[:2]
    assert [
        r["id"] for r in await history.get("p:GroupMessage:g", after_id=ids[1], limit=2)
    ] == ids[2:4]
    assert len(await history.get("p:GroupMessage:g", keyword="LINE 3")) == 1
    assert await history.get("missing") == []


@pytest.mark.asyncio
async def test_record_count_retention_and_reply_identity(history):
    history.settings = replace(history.settings, max_messages_per_session=2)
    for number in range(4):
        await history.record(
            Event(str(number)),
            [Plain(str(number)), Reply(id="original", message_str="quoted")],
        )
    records = await history.get("p:GroupMessage:g")
    assert len(records) == 2
    assert records[-1]["reply_to"] == "original"
    await history.db.execute("UPDATE messages SET timestamp=0")
    await history.db.commit()
    await history.cleanup()
    assert await history.get("p:GroupMessage:g") == []


@pytest.mark.asyncio
async def test_image_cache_dedup_expiration_and_cross_session_access(history, tmp_path):
    image = tmp_path / "source.png"
    data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aXioAAAAASUVORK5CYII="
    )
    image.write_bytes(data)
    first = await history.record(Event("1"), [Image.fromFileSystem(image)])
    second = await history.record(Event("2"), [Image.fromFileSystem(image)])
    assert len(list(history.images.iterdir())) == 1
    image.unlink()
    assert (await history.read_image("p:GroupMessage:g", first, 0))[0] == data
    assert await history.read_image("other", first, 0) is None
    assert await history.read_image("p:GroupMessage:g", first, -1) is None
    await history.db.execute("UPDATE images SET last_seen=0")
    await history.db.commit()
    await history.cleanup()
    assert await history.read_image("p:GroupMessage:g", second, 0) is None
    assert (await history.get("p:GroupMessage:g"))[0]["images"][0]["available"] is False


@pytest.mark.asyncio
async def test_image_size_limit_and_invalid_content(history, tmp_path):
    invalid = tmp_path / "invalid.png"
    invalid.write_text("not an image")
    record_id = await history.record(Event(), [Image.fromFileSystem(invalid)])
    assert await history.read_image("p:GroupMessage:g", record_id, 0) is None
    assert not list(history.images.iterdir())


@pytest.mark.asyncio
async def test_deduplication_survives_restart(history):
    await history.record(Event(), [Plain("hello")])
    await history.close()
    await history.initialize()
    assert await history.record(Event(), [Plain("hello")]) is None
    assert len(await history.get("p:GroupMessage:g")) == 1


@pytest.mark.asyncio
async def test_image_quota_evicts_older_cache_without_deleting_messages(
    history, tmp_path
):
    history.settings = replace(history.settings, image_cache_mb=1)
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aXioAAAAASUVORK5CYII="
    )
    ids = []
    for index in range(2):
        image = tmp_path / f"{index}.png"
        image.write_bytes(png + bytes([index]) * 700000)
        ids.append(
            await history.record(Event(str(index)), [Image.fromFileSystem(image)])
        )
    assert await history.read_image("p:GroupMessage:g", ids[0], 0) is None
    assert await history.read_image("p:GroupMessage:g", ids[1], 0) is not None
    assert len(await history.get("p:GroupMessage:g")) == 2
    assert sum(path.stat().st_size for path in history.images.iterdir()) <= 1024 * 1024
