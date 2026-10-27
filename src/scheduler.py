import asyncio
import logging
from dataclasses import dataclass, field
from time import monotonic

from astrbot.api.event import AstrMessageEvent

from .decision import ReplyJudge
from .history import ChatHistory
from .settings import Settings


@dataclass
class PendingMessage:
    event: AstrMessageEvent
    record_id: int
    forced: bool
    ready: asyncio.Future[list[int] | None]
    finished: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class SessionState:
    pending: list[PendingMessage] = field(default_factory=list)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    active: PendingMessage | None = None
    last_reply: float = float("-inf")


class ChatScheduler:
    """Serialize complete pipeline turns while collecting concurrent messages."""

    def __init__(
        self,
        settings: Settings,
        history: ChatHistory,
        judge: ReplyJudge,
        logger: logging.Logger,
    ) -> None:
        self.settings = settings
        self.history = history
        self.judge = judge
        self.logger = logger
        self.sessions: dict[str, SessionState] = {}
        self.closed = False

    async def submit(
        self, event: AstrMessageEvent, record_id: int, forced: bool
    ) -> PendingMessage:
        """Queue an event without starting a second Agent for the same session.

        Args:
            event: Original event kept alive by its plugin handler.
            record_id: Persisted chat record identifier.
            forced: Whether a private message or explicit mention bypasses judgment.

        Returns:
            Admission ticket whose ready future contains a batch or None.
        """
        ticket = PendingMessage(
            event, record_id, forced, asyncio.get_running_loop().create_future()
        )
        if self.closed:
            ticket.ready.set_result(None)
            return ticket
        key = event.unified_msg_origin
        state = self.sessions.setdefault(key, SessionState())
        state.pending.append(ticket)
        if len(state.pending) > self.settings.max_pending_messages:
            oldest = state.pending.pop(0)
            # Keep direct-address intent even when its original event is merged away.
            state.pending[0].forced |= oldest.forced
            if not oldest.ready.done():
                oldest.ready.set_result(None)
        state.changed.set()
        if state.task is None or state.task.done():
            state.task = asyncio.create_task(self._run(key, state))
        return ticket

    async def _run(self, key: str, state: SessionState) -> None:
        """Admit one handler and wait until its entire Agent pipeline unwinds.

        Args:
            key: Unified session identifier.
            state: Pending and active work for this session.
        """
        batch: list[PendingMessage] = []
        try:
            while state.pending and not self.closed:
                deadline = monotonic() + self.settings.max_wait_seconds
                while not self.closed:
                    state.changed.clear()
                    delay = min(
                        self.settings.debounce_seconds, max(0, deadline - monotonic())
                    )
                    try:
                        await asyncio.wait_for(state.changed.wait(), timeout=delay)
                    except asyncio.TimeoutError:
                        break
                    if monotonic() >= deadline:
                        break
                batch, state.pending = state.pending, []
                for item in batch:
                    if (
                        item.event.is_stopped()
                        or item.event.get_extra("agent_stop_requested", False)
                    ) and not item.ready.done():
                        item.ready.set_result(None)
                batch = [item for item in batch if not item.ready.done()]
                if not batch:
                    continue
                leader = batch[-1]
                forced = any(item.forced for item in batch)
                records = await self.history.get(
                    key, limit=self.settings.recent_messages
                )
                allowed = forced
                if (
                    not allowed
                    and monotonic() - state.last_reply >= self.settings.cooldown_seconds
                ):
                    allowed = await self.judge.should_reply(leader.event, records)
                # Messages arriving during judgment belong to the next batch.
                for item in batch[:-1]:
                    if not item.ready.done():
                        item.ready.set_result(None)
                if leader.ready.done():
                    continue
                if not allowed or self.closed:
                    leader.ready.set_result(None)
                    continue
                state.active = leader
                leader.ready.set_result([item.record_id for item in batch])
                await leader.finished.wait()
                if leader.event.get_extra("mingle_agent_started", False):
                    state.last_reply = monotonic()
                state.active = None
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.exception("Chat scheduling failed for session %s", key)
        finally:
            for item in batch + state.pending:
                if not item.ready.done():
                    item.ready.set_result(None)
            state.pending.clear()
            state.active = None

    async def close(self) -> None:
        """Stop admission, release waiting handlers, and stop owned active events."""
        self.closed = True
        tasks = []
        for state in self.sessions.values():
            if state.active is not None:
                state.active.event.set_extra("agent_stop_requested", True)
                state.active.event.stop_event()
            for item in state.pending:
                if not item.ready.done():
                    item.ready.set_result(None)
            if state.task is not None:
                state.task.cancel()
                tasks.append(state.task)
        await asyncio.gather(*tasks, return_exceptions=True)
        self.sessions.clear()
