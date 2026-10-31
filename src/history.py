import asyncio
import hashlib
import json
import logging
import time
import uuid
from pathlib import Path

import aiosqlite
from astrbot.api.event import AstrMessageEvent
from astrbot.core.message.components import (
    At,
    BaseMessageComponent,
    Image,
    Plain,
    Reply,
)
from astrbot.core.platform.message_type import MessageType
from astrbot.core.utils.media_utils import detect_image_mime_type_async

from .settings import Settings
from .utils.redaction import redact_text


def boundary_group(event: AstrMessageEvent) -> str | None:
    """Return the platform-scoped group identity for a group event."""
    if event.is_private_chat() or not event.get_group_id():
        return None
    return (
        f"{event.get_platform_id()}:{MessageType.GROUP_MESSAGE.value}:"
        f"{event.get_group_id()}"
    )


def _copy_image(path: Path, directory: Path, max_bytes: int) -> tuple[str, int]:
    """Copy bounded image bytes to an atomic content-addressed cache file.

    Args:
        path: Incoming image file available during event handling.
        directory: Plugin-owned image cache directory.
        max_bytes: Maximum number of bytes permitted per image.

    Returns:
        Content digest and encoded byte count.

    Raises:
        ValueError: If the input exceeds the byte limit.
        OSError: If reading or writing fails.
    """
    # Limit reads too: a file can grow after a size check.
    with path.open("rb") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("Image exceeds the per-file cache limit")
    digest = hashlib.sha256(data).hexdigest()
    target = directory / digest
    if not target.exists():
        temporary = directory / f"{digest}.{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_bytes(data)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return digest, len(data)


class ChatHistory:
    """Own chat records and bounded, content-addressed image storage."""

    def __init__(
        self, directory: Path, settings: Settings, logger: logging.Logger
    ) -> None:
        self.directory = directory
        self.images = directory / "images"
        self.settings = settings
        self.logger = logger
        self.db: aiosqlite.Connection | None = None
        self.lock = asyncio.Lock()
        self.image_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Create plugin-owned storage and indexes without changing AstrBot tables."""
        self.images.mkdir(parents=True, exist_ok=True)
        self.db = await aiosqlite.connect(self.directory / "history.sqlite3")
        self.db.row_factory = aiosqlite.Row
        await self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA busy_timeout=5000;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session TEXT NOT NULL,
                source_key TEXT NOT NULL,
                timestamp REAL NOT NULL,
                sender_id TEXT NOT NULL,
                sender_name TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                reply_to TEXT,
                images TEXT NOT NULL,
                UNIQUE(session, source_key)
            );
            CREATE INDEX IF NOT EXISTS messages_session_id ON messages(session, id);
            CREATE INDEX IF NOT EXISTS messages_timestamp ON messages(timestamp);
            CREATE TABLE IF NOT EXISTS images (
                digest TEXT PRIMARY KEY,
                mime TEXT NOT NULL,
                size INTEGER NOT NULL,
                last_seen REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS boundaries (
                group_session TEXT PRIMARY KEY,
                record_id INTEGER NOT NULL,
                set_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sleep_backlog (
                record_id INTEGER PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
                session TEXT NOT NULL,
                group_id TEXT NOT NULL,
                self_id TEXT NOT NULL,
                received_at REAL NOT NULL
            );
            PRAGMA user_version=1;
        """)
        await self.db.commit()

    async def save_sleep_backlog(self, session: str, records: list[tuple]) -> None:
        """Replace one session's bounded sleeping queue without copying message bodies.

        Args:
            session: Full UMO owning the queue.
            records: Tuples of record ID, actual group ID, bot ID and Unix arrival time.
        """
        async with self.lock:
            try:
                await self.db.execute(
                    "DELETE FROM sleep_backlog WHERE session=?", (session,)
                )
                for record_id, group_id, self_id, received_at in records:
                    await self.db.execute(
                        "INSERT INTO sleep_backlog "
                        "SELECT id, session, ?, ?, ? FROM messages "
                        "WHERE id=? AND session=? AND role='user'",
                        (group_id, self_id, received_at, record_id, session),
                    )
                await self.db.commit()
            except BaseException:
                # Cancellation must also roll back the partial queue replacement.
                await self.db.rollback()
                raise

    async def get_sleep_backlog(self) -> list[dict]:
        """Read retained sleeping tickets in session and record order."""
        async with self.lock:
            async with self.db.execute(
                "SELECT * FROM sleep_backlog ORDER BY session, record_id"
            ) as cursor:
                rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def set_boundary(
        self, group: str, *, session: str = "", keyword: str = ""
    ) -> tuple[int, float] | None:
        """Set a boundary at the end of history or before the latest text match.

        Args:
            group: Platform-scoped group owning the boundary.
            session: Current full UMO to search when a keyword is supplied.
            keyword: Optional literal substring, using the history search's
                case-insensitive matching across all retained session records.

        Returns:
            Exclusive lower record bound and set time, or None when the keyword
            does not match. A missing match leaves the existing boundary intact.
        """
        async with self.lock:
            if keyword:
                async with self.db.execute(
                    "SELECT id FROM messages WHERE session=? "
                    "AND instr(lower(text), lower(?)) > 0 "
                    "ORDER BY id DESC LIMIT 1",
                    (session, keyword),
                ) as cursor:
                    row = await cursor.fetchone()
                if row is None:
                    return None
                record_id = row[0] - 1
            else:
                async with self.db.execute(
                    "SELECT seq FROM sqlite_sequence WHERE name='messages'"
                ) as cursor:
                    row = await cursor.fetchone()
                record_id = row[0] if row else 0
            set_at = time.time()
            await self.db.execute(
                "INSERT OR REPLACE INTO boundaries VALUES (?, ?, ?)",
                (group, record_id, set_at),
            )
            await self.db.commit()
            return record_id, set_at

    async def get_boundary(self, group: str) -> tuple[int, float] | None:
        """Read the persistent boundary for a platform-scoped group."""
        async with self.lock:
            async with self.db.execute(
                "SELECT record_id, set_at FROM boundaries WHERE group_session=?",
                (group,),
            ) as cursor:
                row = await cursor.fetchone()
            return (row[0], row[1]) if row else None

    async def boundary_preview(
        self, group: str, session: str
    ) -> tuple[dict | None, dict | None] | None:
        """Read bounded text immediately around a group's boundary in one session.

        Args:
            group: Platform-scoped group owning the boundary.
            session: Current full UMO; other member sessions remain isolated.

        Returns:
            Before and after text/timestamp records (None for missing sides), or
            None when no boundary is set.
        """
        async with self.lock:
            async with self.db.execute(
                "SELECT record_id FROM boundaries WHERE group_session=?", (group,)
            ) as cursor:
                boundary = await cursor.fetchone()
            if boundary is None:
                return None
            snippets = []
            for comparison, ordering in (("<=", "DESC"), (">", "ASC")):
                async with self.db.execute(
                    f"SELECT substr(text, 1, 26), timestamp FROM messages WHERE session=? "
                    f"AND id{comparison}? ORDER BY id {ordering} LIMIT 1",
                    (session, boundary[0]),
                ) as cursor:
                    row = await cursor.fetchone()
                if row is None:
                    snippets.append(None)
                    continue
                text = row[0].replace("\r", " ").replace("\n", " ")
                if len(text) > 25:
                    preview_text = text[:24] + "…"
                else:
                    preview_text = text or "（非文本消息）"
                snippets.append({"text": preview_text, "timestamp": row[1]})
            return snippets[0], snippets[1]

    async def clear_boundary(self, group: str) -> bool:
        """Remove a group boundary without deleting chat records."""
        async with self.lock:
            cursor = await self.db.execute(
                "DELETE FROM boundaries WHERE group_session=?", (group,)
            )
            removed = cursor.rowcount > 0
            await cursor.close()
            await self.db.commit()
            return removed

    async def find_message(self, session: str, keyword: str) -> dict | None:
        """Find the latest retained literal text match in exactly one session.

        Args:
            session: Full UMO supplied by the administrator event.
            keyword: Nonempty substring using ordinary history search semantics.

        Returns:
            Raw stored record including its source key, or None for no match.
        """
        async with self.lock:
            async with self.db.execute(
                "SELECT * FROM messages WHERE session=? "
                "AND instr(lower(text), lower(?)) > 0 ORDER BY id DESC LIMIT 1",
                (session, keyword),
            ) as cursor:
                row = await cursor.fetchone()
            return dict(row) if row else None

    async def delete_record(self, session: str, record_id: int) -> bool:
        """Delete one authorized message without resetting IDs or shared images.

        Args:
            session: Full UMO containing the record.
            record_id: Mingle database record ID selected in this session.

        Returns:
            Whether a record was deleted.
        """
        async with self.lock:
            try:
                cursor = await self.db.execute(
                    "DELETE FROM messages WHERE session=? AND id=?",
                    (session, record_id),
                )
                removed = cursor.rowcount > 0
                await cursor.close()
                await self.db.commit()
                return removed
            except BaseException:
                # Keep the transaction atomic even when the caller is cancelled.
                await self.db.rollback()
                raise

    async def list_sessions(self) -> list[str]:
        """List full UMOs with retained records or a persistent group boundary."""
        async with self.lock:
            async with self.db.execute(
                "SELECT session FROM messages UNION "
                "SELECT group_session FROM boundaries ORDER BY 1"
            ) as cursor:
                rows = await cursor.fetchall()
            return [row[0] for row in rows]

    async def delete_session(self, session: str) -> int:
        """Delete one full UMO's records, linked backlog and own group boundary.

        Shared cached images remain subject to ordinary retention. A boundary
        keyed by a different group UMO is preserved for isolated member sessions.

        Args:
            session: Exact administrator-selected full UMO.

        Returns:
            Global record high-water mark at deletion, covering late submissions
            of already deleted records while retaining subsequent arrivals.
        """
        async with self.lock:
            try:
                async with self.db.execute(
                    "SELECT seq FROM sqlite_sequence WHERE name='messages'"
                ) as cursor:
                    row = await cursor.fetchone()
                deleted_through_id = row[0] if row else 0
                await self.db.execute(
                    "DELETE FROM messages WHERE session=?", (session,)
                )
                await self.db.execute(
                    "DELETE FROM boundaries WHERE group_session=?", (session,)
                )
                await self.db.commit()
                return deleted_through_id
            except BaseException:
                await self.db.rollback()
                raise

    async def record(
        self,
        event: AstrMessageEvent,
        chain: list[BaseMessageComponent],
        role: str = "user",
        source_key: str = "",
    ) -> int | None:
        """Persist a normalized message and cache incoming images before event cleanup.

        Args:
            event: Original event defining the session and sender identity.
            chain: Message components in platform order.
            role: Either user or bot.
            source_key: Optional stable identity for deduplicating output hooks.

        Returns:
            Inserted record ID, or None for an empty or duplicate message.
        """
        parts = []
        images = []
        reply_to = None
        for component in chain:
            if isinstance(component, Plain):
                parts.append(component.text)
            elif isinstance(component, At):
                parts.append(f"@{component.name or component.qq}")
            elif isinstance(component, Reply):
                reply_to = str(component.id)
                parts.append(f"[reply to {reply_to}: {component.message_str or ''}]")
            elif isinstance(component, Image):
                digest = None
                try:
                    path = Path(await component.convert_to_file_path())
                    async with self.image_lock:
                        mime = await detect_image_mime_type_async(
                            str(path), default_mime_type=None
                        )
                        if not mime or not mime.startswith("image/"):
                            raise ValueError("Unsupported image content")

                        digest, size = await asyncio.to_thread(
                            _copy_image,
                            path,
                            self.images,
                            self.settings.max_image_mb * 1024 * 1024,
                        )
                        async with self.lock:
                            await self.db.execute(
                                "INSERT INTO images VALUES (?, ?, ?, ?) "
                                "ON CONFLICT(digest) DO UPDATE SET last_seen=excluded.last_seen",
                                (digest, mime, size, time.time()),
                            )
                            await self.db.commit()
                except Exception:
                    self.logger.warning(
                        "Image caching failed for session %s",
                        event.unified_msg_origin,
                        exc_info=True,
                    )
                images.append(digest)
                availability = "available" if digest else "unavailable"
                parts.append(f"[image {len(images) - 1}: {availability}]")
            else:
                parts.append(f"[{component.type.value}]")
        text = " ".join(parts).strip()
        if self.settings.redaction_enabled:
            text = redact_text(text)
        text = text[:8000]
        if not text:
            return None
        sender_id = event.get_self_id() if role == "bot" else event.get_sender_id()
        sender_name = "bot" if role == "bot" else event.get_sender_name()
        if not source_key:
            message_id = str(getattr(event.message_obj, "message_id", "") or "")
            source_key = (
                f"{role}:{sender_id}:{message_id}" if message_id else uuid.uuid4().hex
            )
        async with self.lock:
            session = event.unified_msg_origin
            if role == "bot":
                timestamp = time.time()
            else:
                timestamp = getattr(event.message_obj, "timestamp", time.time())
            cursor = await self.db.execute(
                "INSERT OR IGNORE INTO messages "
                "(session, source_key, timestamp, sender_id, sender_name, role, text, reply_to, images) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session,
                    source_key,
                    timestamp,
                    str(sender_id),
                    str(sender_name or ""),
                    role,
                    text,
                    reply_to,
                    json.dumps(images),
                ),
            )
            record_id = cursor.lastrowid if cursor.rowcount == 1 else None
            await cursor.close()
            await self.db.execute(
                "DELETE FROM messages WHERE session=? AND id NOT IN "
                "(SELECT id FROM messages WHERE session=? ORDER BY id DESC LIMIT ?)",
                (
                    event.unified_msg_origin,
                    event.unified_msg_origin,
                    self.settings.max_messages_per_session,
                ),
            )
            await self.db.commit()
        if any(images):
            await self.cleanup()
        return record_id

    async def get(
        self,
        session: str,
        limit: int = 30,
        before_id: int = 0,
        after_id: int = 0,
        keyword: str = "",
        boundary: str | None = None,
    ) -> list[dict]:
        """Query chronological records using stable, session-scoped cursors.

        Args:
            session: Authorized session, supplied by the tool environment.
            limit: Number of records, capped at 100.
            before_id: Exclusive upper record bound, or zero.
            after_id: Exclusive lower record bound, or zero.
            keyword: Optional literal text search.
            boundary: Group identity whose visibility boundary applies, if any.

        Returns:
            Chronological normalized messages with image availability.
        """
        limit = max(1, min(100, int(limit)))
        query = "SELECT * FROM messages WHERE session=?"
        values = [session]
        if boundary:
            query += " AND id>COALESCE((SELECT record_id FROM boundaries WHERE group_session=?), 0)"
            values.append(boundary)
        if before_id:
            query += " AND id<?"
            values.append(int(before_id))
        if after_id:
            query += " AND id>?"
            values.append(int(after_id))
        if keyword:
            query += " AND instr(lower(text), lower(?)) > 0"
            values.append(keyword)
        ordering = "ASC" if after_id else "DESC"
        query += f" ORDER BY id {ordering} LIMIT ?"
        values.append(limit)
        async with self.lock:
            async with self.db.execute(query, values) as cursor:
                rows = await cursor.fetchall()
        records = []
        for row in sorted(rows, key=lambda item: item["id"]):
            record = dict(row)
            source_key = record.pop("source_key")
            if record["role"] == "user" and source_key.startswith("user:"):
                record["platform_message_id"] = source_key.split(":", 2)[2]
            else:
                record["platform_message_id"] = None
            record.pop("session")
            cached_images = json.loads(record["images"])
            image_details = []
            for index, digest in enumerate(cached_images):
                available = bool(digest and (self.images / digest).is_file())
                image_details.append({"index": index, "available": available})
            record["images"] = image_details
            records.append(record)
        # Keep tool output and prompt snapshots bounded even for very long posts.
        budget = 24000
        for record in reversed(records):
            record["text"] = record["text"][: max(0, budget)]
            budget -= len(record["text"])
        return records

    async def read_image(
        self,
        session: str,
        record_id: int,
        image_index: int,
        boundary: str | None = None,
    ) -> tuple[bytes, str] | None:
        """Read a cached image only through a message in the authorized session.

        Args:
            session: Current session; models cannot override it.
            record_id: Message record that references the image.
            image_index: Zero-based image index within that record.
            boundary: Group identity whose visibility boundary applies, if any.

        Returns:
            Image bytes and MIME type, or None if unavailable or unauthorized.
        """
        if image_index < 0:
            return None
        async with self.image_lock, self.lock:
            async with self.db.execute(
                "SELECT images FROM messages WHERE session=? AND id=? AND "
                "id>COALESCE((SELECT record_id FROM boundaries WHERE group_session=?), 0)",
                (session, record_id, boundary),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return None
            images = json.loads(row["images"])
            if image_index >= len(images) or not images[image_index]:
                return None
            digest = images[image_index]
            async with self.db.execute(
                "SELECT mime FROM images WHERE digest=?", (digest,)
            ) as cursor:
                image = await cursor.fetchone()
            path = self.images / digest
            if image is None or not path.is_file():
                return None
            return await asyncio.to_thread(path.read_bytes), image["mime"]

    async def cleanup(self) -> None:
        """Expire chat rows and images by age and total image storage size."""
        now = time.time()
        async with self.image_lock, self.lock:
            await self.db.execute(
                "DELETE FROM messages WHERE timestamp<?",
                (now - self.settings.retention_days * 86400,),
            )
            async with self.db.execute(
                "SELECT * FROM images ORDER BY last_seen DESC"
            ) as cursor:
                rows = await cursor.fetchall()
            retained = 0
            for row in rows:
                retention_cutoff = now - self.settings.image_retention_days * 86400
                keep = (
                    row["last_seen"] >= retention_cutoff
                    and retained + row["size"]
                    <= self.settings.image_cache_mb * 1024 * 1024
                )
                if keep:
                    retained += row["size"]
                    continue
                await asyncio.to_thread(
                    (self.images / row["digest"]).unlink, missing_ok=True
                )
                await self.db.execute(
                    "DELETE FROM images WHERE digest=?", (row["digest"],)
                )
            # A crash between file creation and DB commit can leave unreferenced files.
            known = set()
            for row in rows:
                path = self.images / row["digest"]
                if path.exists():
                    known.add(row["digest"])
            for path in self.images.iterdir():
                if path.is_file() and path.name not in known:
                    await asyncio.to_thread(path.unlink, missing_ok=True)
            await self.db.commit()

    async def close(self) -> None:
        """Close the connection after plugin background tasks have stopped."""
        async with self.lock:
            if self.db is not None:
                await self.db.close()
                self.db = None
