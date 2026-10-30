"""Public scheduler entry point; internal coordination lives in scheduling/."""

import asyncio
import logging
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

    async def close(self) -> None:
        """Release waiting handlers, cancel owned work, and stop active events."""
        self.closed = True
        tasks = []
        for coordinator in self._coordinators.values():
            tasks.extend(coordinator.close())
        await asyncio.gather(*tasks, return_exceptions=True)
        self._coordinators.clear()
        self.sessions.clear()
