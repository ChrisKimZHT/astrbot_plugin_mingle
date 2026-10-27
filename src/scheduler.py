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
    judging: list[PendingMessage] = field(default_factory=list)
    approved: list[PendingMessage] = field(default_factory=list)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    execution_task: asyncio.Task | None = None
    active: PendingMessage | None = None
    last_reply: float = float("-inf")
    resume_after_rerun: bool = False
    active_ids: list[int] = field(default_factory=list)
    finished: dict[int, str] = field(default_factory=dict)


class ChatScheduler:
    """Judge incoming batches concurrently with one serialized Agent per session."""

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
            self._finish_records(state, [oldest.record_id], "merged")
            # Keep direct-address intent even when its original event is merged away.
            state.pending[0].forced |= oldest.forced
            if not oldest.ready.done():
                oldest.ready.set_result(None)
        state.changed.set()
        if state.task is None or state.task.done():
            state.task = asyncio.create_task(self._run(key, state))
        return ticket

    async def _run(self, key: str, state: SessionState) -> None:
        """Judge new batches without waiting for the running Agent to finish.

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
                state.judging = batch
                forced = any(item.forced for item in batch)
                records = await self.history.get(
                    key,
                    limit=self.settings.recent_messages,
                    before_id=max(item.record_id for item in batch) + 1,
                )
                active = state.active
                if active is not None and active.finished.is_set():
                    active = None
                active_ids = list(state.active_ids) if active is not None else []
                if active is not None:
                    missing_ids = set(active_ids) - {item["id"] for item in records}
                    if missing_ids:
                        # Keep the active task visible when new arrivals have
                        # pushed its input outside the recent-history window.
                        active_records = await asyncio.gather(
                            *(
                                self.history.get(
                                    key,
                                    limit=1,
                                    after_id=record_id - 1,
                                    before_id=record_id + 1,
                                )
                                for record_id in missing_ids
                            )
                        )
                        records = sorted(
                            records
                            + [
                                item
                                for rows in active_records
                                for item in rows
                                if item["id"] in missing_ids
                            ],
                            key=lambda item: item["id"],
                        )
                message_status = {
                    "finished": [
                        {"id": record_id, "outcome": outcome}
                        for record_id, outcome in state.finished.items()
                    ],
                    "active": active_ids,
                    "queued": [
                        item.record_id
                        for item in state.approved
                        if not item.ready.done()
                    ],
                    "awaiting_judgment": [
                        item.record_id
                        for item in state.pending
                        if not item.ready.done()
                    ],
                    "judging": [item.record_id for item in batch],
                }
                allowed = forced
                if active is not None:
                    decision = await self.judge.evaluate(
                        leader.event,
                        records,
                        message_status=message_status,
                        new_message_ids=[item.record_id for item in batch],
                    )
                    allowed = forced or decision.should_reply
                    if decision.action == "stop":
                        allowed = False
                    # A slow judgment must never stop a different, newer turn.
                    if (
                        decision.action in {"rerun", "stop"}
                        and state.active is active
                        and not active.finished.is_set()
                        and not leader.ready.done()
                        and not leader.event.is_stopped()
                        and not leader.event.get_extra("agent_stop_requested", False)
                    ):
                        active.event.set_extra("agent_stop_requested", True)
                        state.resume_after_rerun = decision.action == "rerun"
                        self.logger.info(
                            "Mingle requested Agent %s for session %s",
                            decision.action,
                            key,
                        )
                        if decision.action == "stop":
                            self._finish_records(
                                state,
                                [item.record_id for item in state.approved],
                                "discarded_by_stop",
                            )
                            for item in state.approved:
                                if not item.ready.done():
                                    item.ready.set_result(None)
                            state.approved.clear()
                elif (
                    not allowed
                    and monotonic() - state.last_reply >= self.settings.cooldown_seconds
                ):
                    allowed = await self.judge.should_reply(
                        leader.event,
                        records,
                        message_status=message_status,
                        new_message_ids=[item.record_id for item in batch],
                    )
                if not allowed or self.closed or leader.ready.done():
                    self._finish_records(
                        state, [item.record_id for item in batch], "no_reply"
                    )
                    for item in batch:
                        if not item.ready.done():
                            item.ready.set_result(None)
                    batch = []
                    state.judging = []
                    continue
                state.approved.extend(batch)
                state.judging = []
                batch = []
                while len(state.approved) > self.settings.max_pending_messages:
                    oldest = state.approved.pop(0)
                    self._finish_records(state, [oldest.record_id], "merged")
                    if not oldest.ready.done():
                        oldest.ready.set_result(None)
                if state.execution_task is None or state.execution_task.done():
                    state.execution_task = asyncio.create_task(
                        self._execute(key, state)
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.exception("Chat scheduling failed for session %s", key)
        finally:
            for item in batch + state.pending:
                if not item.ready.done():
                    item.ready.set_result(None)
            state.pending.clear()
            state.judging = []

    async def _execute(self, key: str, state: SessionState) -> None:
        """Admit approved batches only after the previous pipeline has unwound.

        Args:
            key: Unified session identifier.
            state: Approved work and active execution for this session.
        """
        batch: list[PendingMessage] = []
        try:
            while state.approved and not self.closed:
                batch, state.approved = state.approved, []
                for item in batch:
                    if (
                        item.event.is_stopped()
                        or item.event.get_extra("agent_stop_requested", False)
                    ) and not item.ready.done():
                        item.ready.set_result(None)
                batch = [item for item in batch if not item.ready.done()]
                if not batch:
                    state.resume_after_rerun = False
                    continue
                leader = batch[-1]
                for item in batch[:-1]:
                    item.ready.set_result(None)
                if state.resume_after_rerun:
                    leader.event.set_extra("mingle_resume_after_rerun", True)
                    state.resume_after_rerun = False
                state.active = leader
                state.active_ids = [item.record_id for item in batch]
                leader.ready.set_result([item.record_id for item in batch])
                # The handler sets finished after the framework's history save,
                # not merely after its final response hook or first send.
                await leader.finished.wait()
                outcome = (
                    "interrupted"
                    if (
                        leader.event.get_extra("agent_user_aborted", False)
                        or leader.event.get_extra("agent_stop_requested", False)
                    )
                    else "completed"
                )
                self._finish_records(state, state.active_ids, outcome)
                if leader.event.get_extra("mingle_agent_started", False):
                    state.last_reply = monotonic()
                state.active = None
                state.active_ids = []
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.exception("Mingle Agent admission failed for session %s", key)
        finally:
            for item in batch + state.approved:
                if not item.ready.done():
                    item.ready.set_result(None)
            state.approved.clear()
            state.active = None
            state.active_ids = []

    def _finish_records(
        self, state: SessionState, record_ids: list[int], outcome: str
    ) -> None:
        """Retain bounded live outcomes for later judgment context.

        Args:
            state: Session whose records reached a terminal scheduling state.
            record_ids: Mingle record identifiers, not platform message IDs.
            outcome: Completion, interruption, or no-reply outcome.
        """
        for record_id in record_ids:
            state.finished[record_id] = outcome
        while (
            len(state.finished)
            > self.settings.max_pending_messages + self.settings.recent_messages
        ):
            state.finished.pop(next(iter(state.finished)))

    async def close(self) -> None:
        """Stop admission, release waiting handlers, and stop owned active events."""
        self.closed = True
        tasks = []
        for state in self.sessions.values():
            if state.active is not None:
                state.active.event.set_extra("agent_stop_requested", True)
                state.active.event.stop_event()
            for item in state.pending + state.approved:
                if not item.ready.done():
                    item.ready.set_result(None)
            if state.task is not None:
                state.task.cancel()
                tasks.append(state.task)
            if state.execution_task is not None:
                state.execution_task.cancel()
                tasks.append(state.execution_task)
        await asyncio.gather(*tasks, return_exceptions=True)
        self.sessions.clear()
