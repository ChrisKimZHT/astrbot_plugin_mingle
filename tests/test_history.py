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
async def test_redaction_is_stored_before_truncation_and_preserves_identity(history):
    event = Event()
    chain = [
        Plain("联系13800138000"),
        Reply(id="13800138000", message_str="a@example.com"),
        Plain("-----BEGIN PRIVATE KEY-----" + "x" * 9000 + "-----END PRIVATE KEY-----"),
    ]
    record_id = await history.record(event, chain)
    assert await history.record(event, chain) is None
    async with history.db.execute(
        "SELECT * FROM messages WHERE id=?", (record_id,)
    ) as cursor:
        row = await cursor.fetchone()
    assert "a@example.com" not in row["text"]
    assert "[邮箱]" in row["text"] and "[私钥]" in row["text"]
    assert "x" * 20 not in row["text"]
    assert row["reply_to"] == "13800138000"
    assert row["sender_id"] == "alice"
    assert chain[0].text == "联系13800138000"
    assert await history.get(event.unified_msg_origin, keyword="a@example.com") == []
    assert (await history.get(event.unified_msg_origin))[0]["text"] == row["text"]
    history.settings = replace(history.settings, redaction_enabled=False)
    await history.record(Event("2"), [Plain("a@example.com")])
    history.settings = replace(history.settings, redaction_enabled=True)
    await history.record(Event("3"), [Plain("a@example.com")], role="bot")
    records = await history.get(event.unified_msg_origin)
    assert records[-2]["text"] == "a@example.com"
    assert records[-1]["text"] == "[邮箱]"


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
async def test_keyword_boundary_searches_hidden_records_and_preserves_first_match(
    history,
):
    group = "p:GroupMessage:g"
    first_id = await history.record(Event("first"), [Plain("Topic 100%_literal")])
    await history.record(Event("gap", "other"), [Plain("unrelated")])
    await history.record(Event("later"), [Plain("Topic 100XXliteral")])
    await history.record(
        Event("member", "p:GroupMessage:alice_g"), [Plain("Topic 100%_literal")]
    )
    await history.set_boundary(group)
    assert await history.get(group, boundary=group) == []

    boundary = await history.set_boundary(
        group, session=group, keyword="topic 100%_literal"
    )
    assert boundary[0] == first_id - 1 == 0
    assert [row["id"] for row in await history.get(group, boundary=group)][
        0
    ] == first_id
    before, after = await history.boundary_preview(group, group)
    assert before is None
    assert after["text"] == "Topic 100%_literal"
    await history.close()
    await history.initialize()
    assert await history.get_boundary(group) == boundary

    assert await history.set_boundary(group, session=group, keyword="absent") is None
    assert await history.get_boundary(group) == boundary
    assert await history.set_boundary("unset", session="empty", keyword="Topic") is None
    assert await history.get_boundary("unset") is None


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
async def test_find_and_delete_are_literal_session_scoped_and_keep_shared_images(
    history, tmp_path
):
    image = tmp_path / "delete-source.png"
    image.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aXioAAAAASUVORK5CYII="
        )
    )
    session = "p:GroupMessage:g"
    first_id = await history.record(
        Event("first"), [Plain("Topic 100%_"), Image.fromFileSystem(image)]
    )
    latest_id = await history.record(
        Event("latest"), [Plain("Topic 100%_"), Image.fromFileSystem(image)]
    )
    await history.record(Event("other", "other"), [Plain("Topic 100%_")])
    await history.record(Event("wildcard"), [Plain("Topic 100XX")])
    record = await history.find_message(session, "topic 100%_")
    assert record["id"] == latest_id
    assert record["source_key"] == "user:alice:latest"
    assert not await history.delete_record("other", latest_id)
    assert await history.delete_record(session, latest_id)
    assert not await history.delete_record(session, latest_id)
    assert (await history.find_message(session, "topic 100%_"))["id"] == first_id
    assert await history.read_image(session, first_id, 0) is not None
    assert await history.read_image(session, latest_id, 0) is None
    await history.close()
    await history.initialize()
    assert (await history.find_message(session, "Topic 100%_"))["id"] == first_id


@pytest.mark.asyncio
async def test_delete_session_is_exact_cascades_backlog_and_preserves_shared_cache(
    history, tmp_path
):
    group = "p:GroupMessage:g"
    member = "p:GroupMessage:alice_g"
    image = tmp_path / "session-source.png"
    image.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aXioAAAAASUVORK5CYII="
        )
    )
    group_id = await history.record(Event("group"), [Image.fromFileSystem(image)])
    member_id = await history.record(
        Event("member", member), [Image.fromFileSystem(image)]
    )
    await history.save_sleep_backlog(group, [(group_id, "g", "bot", 100.0)])
    await history.save_sleep_backlog(member, [(member_id, "g", "bot", 101.0)])
    boundary = await history.set_boundary(group)
    assert await history.list_sessions() == sorted([group, member])

    assert await history.delete_session(member) == member_id
    assert await history.get(member) == []
    assert await history.get_boundary(group) == boundary
    assert [row["record_id"] for row in await history.get_sleep_backlog()] == [group_id]
    assert await history.read_image(group, group_id, 0) is not None

    assert await history.delete_session(group) == member_id
    assert await history.list_sessions() == []
    assert await history.get_sleep_backlog() == []
    assert await history.get_boundary(group) is None
    assert len(list(history.images.iterdir())) == 1
    new_id = await history.record(Event("new"), [Plain("new session")])
    assert new_id > member_id
    await history.close()
    await history.initialize()
    assert [row["id"] for row in await history.get(group)] == [new_id]


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
async def test_valid_image_header_cannot_bypass_cache_byte_limit(history, tmp_path):
    history.settings = replace(history.settings, max_image_mb=1)
    path = tmp_path / "oversized.png"
    data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aXioAAAAASUVORK5CYII="
    )
    path.write_bytes(data + b"\0" * (1024 * 1024))
    record_id = await history.record(Event(), [Image.fromFileSystem(path)])
    assert await history.read_image("p:GroupMessage:g", record_id, 0) is None
    records = await history.get("p:GroupMessage:g")
    assert records[0]["images"] == [{"index": 0, "available": False}]
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
