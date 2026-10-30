"""Public scheduler entry point; internal coordination lives in scheduling/."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

from astrbot.api.event import AstrMessageEvent
from astrbot.core.platform.message_type import MessageType

from .decision import ReplyJudge
from .history import ChatHistory
from .scheduling import timing
from .scheduling.session import SessionCoordinator
from .scheduling.state import PendingMessage, SessionState
from .settings import Settings

__all__ = ["ChatScheduler", "PendingMessage", "SessionState"]


class ChatScheduler:
    """Coordinate independent sessions with one Agent slot per full UMO."""

    def __init__(
        self,
        settings: Settings,
        history: ChatHistory,
        judge: ReplyJudge,
        logger: logging.Logger,
        queue_bubble: Callable[[str, int], Awaitable[None]] | None = None,
        get_config: Callable[..., dict] | None = None,
        save_sleep_backlog: Callable[[str, list[tuple]], Awaitable[None]] | None = None,
    ) -> None:
        self.settings = settings
        self.history = history
        self.judge = judge
        self.logger = logger
        self.sessions: dict[str, SessionState] = {}
        self.closed = False
        self.queue_bubble = queue_bubble
        self.get_config = get_config
        self._coordinators: dict[str, SessionCoordinator] = {}
        self._save_sleep_backlog = save_sleep_backlog
        self._backlog_snapshots: dict[str, list[tuple]] = {}
        self._backlog_updates: dict[str, list[tuple]] = {}
        self._backlog_write_task: asyncio.Task | None = None
        self._backlog_write_errors: dict[str, Exception] = {}

    def persist_sleep_backlog(self, key: str) -> None:
        """Capture sleeping tickets and serialize coalesced writes off the coordinator.

        Args:
            key: Full UMO whose synchronous scheduling transition has completed.
        """
        if self._save_sleep_backlog is None:
            return
        state = self.sessions[key]
        previous = self._backlog_snapshots.get(key, [])
        retained_ids = {row[0] for row in previous}
        records = []
        offset = time.time() - timing.monotonic()
        for item in state.pending + state.judging + state.approved:
            if not state.sleeping and item.record_id not in retained_ids:
                continue
            if (
                item.ready.done()
                or item.event.is_stopped()
                or item.event.get_extra("agent_stop_requested", False)
            ):
                continue
            records.append(
                (
                    item.record_id,
                    item.event.get_group_id(),
                    item.event.get_self_id(),
                    item.received_at + offset,
                )
            )
        records.sort()
        # Compare IDs and routing only: converting clock domains can introduce jitter.
        unchanged = [row[:3] for row in records] == [row[:3] for row in previous]
        if unchanged and key not in self._backlog_write_errors:
            return
        self._backlog_snapshots[key] = records
        self._backlog_updates[key] = records
        if self._backlog_write_task is None or self._backlog_write_task.done():
            self._backlog_write_task = asyncio.create_task(self._write_sleep_backlogs())

    async def _write_sleep_backlogs(self) -> None:
        """Write captured snapshots in order without changing scheduling state."""
        while self._backlog_updates:
            key = next(iter(self._backlog_updates))
            records = self._backlog_updates.pop(key)
            try:
                await self._save_sleep_backlog(key, records)
                self._backlog_write_errors.pop(key, None)
            except Exception as error:
                self._backlog_write_errors[key] = error
                self.logger.exception("Failed to persist sleeping backlog for %s", key)

    async def flush_sleep_backlog(self, key: str | None = None) -> None:
        """Finish writes, including admission outside the coordinator's loop.

        Args:
            key: Admitted session to capture before waiting for writes, if any.
        """
        if key is not None:
            self.persist_sleep_backlog(key)
        if self._backlog_write_task is not None:
            await asyncio.shield(self._backlog_write_task)
        if key is not None and key in self._backlog_write_errors:
            raise RuntimeError(
                "Sleeping backlog persistence failed"
            ) from self._backlog_write_errors[key]
        if key is None and self._backlog_write_errors:
            error = next(iter(self._backlog_write_errors.values()))
            raise RuntimeError("Sleeping backlog persistence failed") from error

    def restore_sleep_backlog(
        self, event: AstrMessageEvent, records: list[dict]
    ) -> PendingMessage:
        """Rebuild sleeping tickets using one fresh internal pipeline event.

        Args:
            event: Neutral internal event retaining the original full UMO and group.
            records: Nonempty retained sleeping rows in ascending record order.

        Returns:
            Latest ticket whose handler can execute an admitted restored batch.
        """
        coordinator = self._coordinator(event.unified_msg_origin)
        self._backlog_snapshots[event.unified_msg_origin] = [
            (
                row["record_id"],
                row["group_id"],
                row["self_id"],
                row["received_at"],
            )
            for row in records
        ]
        now = timing.monotonic()
        wall_now = time.time()
        for record in records:
            received_at = now - max(0.0, wall_now - record["received_at"])
            ticket = coordinator.submit(
                event, record["record_id"], False, False, received_at=received_at
            )
        return ticket

    def _coordinator(self, key: str) -> SessionCoordinator:
        """Resolve state, starting groups asleep and keeping private chats awake.

        Args:
            key: Full unified session identifier, including member isolation.
        """
        coordinator = self._coordinators.get(key)
        if coordinator is not None:
            return coordinator
        state = self.sessions.get(key)
        if state is None:
            parts = key.split(":", 2)
            is_private = (
                len(parts) == 3 and parts[1] == MessageType.FRIEND_MESSAGE.value
            )
            next_review_at = float("inf")
            interval = self.settings.sleep_backlog_minutes * 60
            if not is_private and interval > 0:
                next_review_at = timing.monotonic() + interval
            state = SessionState(
                sleep_enabled=not is_private,
                sleeping=not is_private,
                next_sleep_backlog_at=next_review_at,
            )
            self.sessions[key] = state
        coordinator = SessionCoordinator(self, key, state)
        self._coordinators[key] = coordinator
        return coordinator

    def initialize_session(self, key: str) -> None:
        """Start sleep and backlog clocks for a configured session at startup.

        Args:
            key: Configured full session UMO.
        """
        self._coordinator(key)

    def _quiet_seconds_remaining(self, key: str) -> float:
        return timing.quiet_seconds_remaining(key, self.settings, self.get_config)

    async def submit(
        self,
        event: AstrMessageEvent,
        record_id: int,
        forced: bool,
        *,
        wake: bool = False,
    ) -> PendingMessage:
        """Register input without blocking other arrivals or an active Agent.

        Args:
            event: Original event kept alive by its plugin handler.
            record_id: Persisted Mingle record identifier.
            forced: Private or directly addressed input bypassing idle judgment.
            wake: Direct mention or literal configured keyword.

        Returns:
            Ticket whose ready future grants a batch to its original handler.
        """
        if self.closed:
            ticket = PendingMessage(
                event,
                record_id,
                forced,
                asyncio.get_running_loop().create_future(),
                timing.monotonic(),
            )
            ticket.release()
            return ticket
        return self._coordinator(event.unified_msg_origin).submit(
            event, record_id, forced, wake
        )

    def sleep(self, key: str) -> None:
        if not self.closed:
            self._coordinator(key).sleep()

    def record_agent_reply(self, key: str) -> None:
        if not self.closed:
            self._coordinator(key).record_reply()

    def claim_recovery_messages(self, ticket: PendingMessage) -> list[int] | None:
        coordinator = self._coordinators.get(ticket.event.unified_msg_origin)
        if coordinator is None:
            return None
        return coordinator.claim_recovery(ticket)

    def watch_random_bubbles(self, key: str) -> None:
        """Enable a random deadline on the existing session coordinator.

        Args:
            key: Shared group UMO whose eligibility was checked by the plugin.
        """
        if self.closed or not self.settings.enabled:
            return
        if not self.settings.random_bubble_enabled or self.queue_bubble is None:
            return
        coordinator = self._coordinator(key)
        coordinator.proactive.watch()

    async def stop_random_bubbles(self, key: str) -> None:
        coordinator = self._coordinators.get(key)
        if coordinator is None:
            return
        task = coordinator.proactive.stop()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    async def admit_random_bubble(
        self, event: AstrMessageEvent, revision: int
    ) -> PendingMessage | None:
        if event.unified_msg_origin not in self.sessions:
            return None
        coordinator = self._coordinator(event.unified_msg_origin)
        return await coordinator.proactive.request(event, revision)

    def finish_random_bubble(self, ticket: PendingMessage) -> None:
        coordinator = self._coordinators.get(ticket.event.unified_msg_origin)
        if coordinator is not None:
            coordinator.finish_active(ticket)

    def release_group_waiters(self, group_session: str) -> None:
        """Release waiting work without interrupting active group turns.

        Args:
            group_session: Removed platform-scoped whole-group UMO.
        """
        for coordinator in self._coordinators.values():
            coordinator.release_group_waiters(group_session)

    def begin_history_edit(self, key: str) -> bool:
        """Reserve an idle session while its persisted histories are edited.

        Args:
            key: Current full UMO supplied by an administrator command.

        Returns:
            Whether this caller acquired the session's edit reservation.
        """
        if self.closed:
            return False
        coordinator = self._coordinator(key)
        state = coordinator.state
        if state.active is not None or state.history_editing:
            return False
        state.history_editing = True
        coordinator.notify()
        return True

    def end_history_edit(self, key: str, deleted_id: int | None) -> None:
        """Resume coordination and invalidate work that saw a deleted record.

        Args:
            key: Session whose edit reservation is held by the caller.
            deleted_id: Successfully deleted record, or None if nothing changed.
        """
        coordinator = self._coordinators.get(key)
        if coordinator is not None:
            coordinator.end_history_edit(deleted_id)

    async def close(self) -> None:
        """Release waiting handlers, cancel owned work, and stop active events."""
        self.closed = True
        tasks = []
        for key, coordinator in self._coordinators.items():
            self.persist_sleep_backlog(key)
            tasks.extend(coordinator.close())
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._backlog_write_task is not None:
            await asyncio.shield(self._backlog_write_task)
        self._coordinators.clear()
        self.sessions.clear()
