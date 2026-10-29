import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from math import ceil
from random import expovariate
from time import monotonic
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from astrbot.api.event import AstrMessageEvent

from .decision import ReplyDecision, ReplyJudge
from .history import ChatHistory
from .settings import Settings


@dataclass
class PendingMessage:
    """Keep the original handler alive through admission and pipeline completion.

    Only the newest ticket in an admitted batch receives record IDs via ready;
    the others receive None. The admitted handler sets finished after its entire
    pipeline unwinds, including framework history saving.
    Internal proactive tickets use record_id=0 without creating a stored row;
    their ready future contains only the real backlog record IDs.
    """

    event: AstrMessageEvent
    record_id: int
    forced: bool
    ready: asyncio.Future[list[int] | None]
    received_at: float
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    backlog_sleep_revision: int | None = None

    def release(self) -> None:
        """Release a waiting handler without admitting another Agent turn."""
        if not self.ready.done():
            self.ready.set_result(None)


def _waiting_messages(messages: list[PendingMessage]) -> list[PendingMessage]:
    """Release stopped events and retain tickets still awaiting admission.

    Args:
        messages: Pending or approved tickets to inspect without changing the list.

    Returns:
        Unresolved tickets whose events have not been stopped.
    """
    waiting = []
    for item in messages:
        if item.event.is_stopped() or item.event.get_extra(
            "agent_stop_requested", False
        ):
            item.release()
        if not item.ready.done():
            waiting.append(item)
    return waiting


@dataclass
class SessionState:
    """Shared state for the session's judgment and execution workers.

    Tickets move through pending -> judging -> approved -> active. Judgment owns
    pending/judging and appends to approved; execution drains approved and owns
    active/active_ids. A stop decision may discard approved tickets. changed
    wakes judgment when arrivals or Agent transitions affect its deadline.
    Proactive handlers can reserve the same active slot directly while idle.

    State transitions contain no awaits. After IO, workers must recheck any
    live state needed to act on a previously captured Agent or batch.
    """

    pending: list[PendingMessage] = field(default_factory=list)
    judging: list[PendingMessage] = field(default_factory=list)
    approved: list[PendingMessage] = field(default_factory=list)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    judgment_task: asyncio.Task | None = None
    execution_task: asyncio.Task | None = None
    active: PendingMessage | None = None
    last_agent_finished_at: float = float("-inf")
    last_agent_reply_at: float = field(default_factory=lambda: monotonic())
    last_woken_at: float = float("-inf")
    sleeping: bool = False
    waking: bool = False
    sleep_revision: int = 0
    next_sleep_backlog_at: float = float("inf")
    bubble_task: asyncio.Task | None = None
    next_bubble_at: float = float("inf")
    bubble_revision: int = 0
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

    def _quiet_seconds_remaining(self, key: str) -> float:
        """Return time until quiet hours end in the current session timezone.

        The start is inclusive and the end exclusive. Equal endpoints disable
        quiet hours. Resolve the profile on every check; invalid zones use UTC.

        Args:
            key: Full session UMO used to resolve AstrBot's timezone.

        Returns:
            Remaining seconds, or zero outside quiet hours.
        """
        start = self.settings.quiet_hours_start
        end = self.settings.quiet_hours_end
        if start == end:
            return 0.0
        timezone_name = None
        if self.get_config is not None:
            timezone_name = self.get_config(umo=key).get("timezone")
        try:
            session_timezone = ZoneInfo(timezone_name or "UTC")
        except (ZoneInfoNotFoundError, ValueError, TypeError):
            session_timezone = timezone.utc
        now = datetime.now(session_timezone)
        current = now.strftime("%H:%M")
        if start < end:
            quiet = start <= current < end
        else:
            quiet = current >= start or current < end
        if not quiet:
            return 0.0
        hour, minute = map(int, end.split(":"))
        end_date = now.date()
        if start > end and current >= start:
            end_date += timedelta(days=1)
        quiet_end = datetime.combine(
            end_date, datetime.min.time(), tzinfo=session_timezone
        ).replace(hour=hour, minute=minute)
        return max(0.0, quiet_end.timestamp() - now.timestamp())

    def _defer_quiet_batch(
        self, state: SessionState, batch: list[PendingMessage]
    ) -> None:
        """Return a quiet-hour batch to the bounded pending queue.

        Args:
            state: Session retaining input until a later review or explicit wake.
            batch: Judging or approved tickets whose admission was postponed.
        """
        state.pending = sorted(
            _waiting_messages(batch + state.pending), key=lambda item: item.record_id
        )
        while len(state.pending) > self.settings.max_pending_messages:
            oldest = state.pending.pop(0)
            self._finish_records(state, [oldest.record_id], "merged")
            state.pending[0].forced |= oldest.forced
            oldest.release()
        state.changed.set()

    def watch_random_bubbles(self, key: str) -> None:
        """Start one independent random clock for an eligible shared group.

        Args:
            key: Full group UMO already checked by the plugin entry point.
        """
        if self.closed or not self.settings.enabled:
            return
        if not self.settings.random_bubble_enabled or self.queue_bubble is None:
            return
        state = self.sessions.setdefault(key, SessionState())
        if state.bubble_task is not None and not state.bubble_task.done():
            return
        self._reset_bubble_deadline(state)
        state.bubble_task = asyncio.create_task(self._random_bubbles(key, state))

    async def stop_random_bubbles(self, key: str) -> None:
        """Invalidate queued triggers and stop a removed group's owned timer.

        Args:
            key: Shared-group UMO removed from the allowlist.
        """
        state = self.sessions.get(key)
        if state is None:
            return
        state.bubble_revision += 1
        state.next_bubble_at = float("inf")
        state.changed.set()
        if state.bubble_task is not None:
            state.bubble_task.cancel()
            await asyncio.gather(state.bubble_task, return_exceptions=True)
            state.bubble_task = None

    def _reset_bubble_deadline(
        self, state: SessionState, *, anchor_at: float | None = None
    ) -> None:
        """Draw a shifted exponential deadline anchored to the last real send.

        Args:
            state: Session whose previous queued trigger must become stale.
            anchor_at: Quiet-hours end when postponing a skipped opportunity.
        """
        minimum = self.settings.random_bubble_min_minutes * 60
        random_mean = (
            self.settings.random_bubble_mean_minutes
            - self.settings.random_bubble_min_minutes
        ) * 60
        state.bubble_revision += 1
        if anchor_at is None:
            anchor_at = state.last_agent_reply_at
        state.next_bubble_at = anchor_at + minimum + expovariate(1 / random_mean)
        state.changed.set()

    async def _random_bubbles(self, key: str, state: SessionState) -> None:
        """Queue at most one internal event per sampled opportunity.

        Sending refreshes the deadline, including sends made by backlog turns.
        A consumed opportunity cannot spin if dispatch or the Agent fails.

        Args:
            key: Group receiving proactive opportunities.
            state: Shared scheduling state, unaffected by sleep transitions.
        """
        while not self.closed and self.settings.random_bubble_enabled:
            state.changed.clear()
            if not self.settings.enabled or state.next_bubble_at == float("inf"):
                await state.changed.wait()
                continue
            deadline = max(
                state.next_bubble_at,
                state.last_agent_finished_at + self.settings.cooldown_seconds,
            )
            delay = deadline - monotonic()
            if delay > 0:
                try:
                    await asyncio.wait_for(state.changed.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
                continue
            quiet_delay = self._quiet_seconds_remaining(key)
            if quiet_delay > 0:
                self._reset_bubble_deadline(state, anchor_at=monotonic() + quiet_delay)
                continue
            if state.active is not None or state.approved or state.judging:
                await state.changed.wait()
                continue
            # Consume before dispatch. A later successful send schedules the
            # next opportunity and invalidates this event's revision.
            revision = state.bubble_revision
            state.next_bubble_at = float("inf")
            try:
                await self.queue_bubble(key, revision)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.exception("Failed to queue Mingle random bubble: %s", key)

    async def admit_random_bubble(
        self, event: AstrMessageEvent, revision: int
    ) -> PendingMessage | None:
        """Claim bounded pending input without judgment once ordinary work is idle.

        Args:
            event: Fresh internal event, never inserted into chat storage.
            revision: Clock generation captured when the event was queued.

        Returns:
            Active ticket with its backlog IDs, or None for a stale trigger.
        """
        key = event.unified_msg_origin
        state = self.sessions.get(key)
        if state is None:
            return None
        while not self.closed and self.settings.enabled:
            if not self.settings.random_bubble_enabled:
                return None
            if revision != state.bubble_revision or event.is_stopped():
                return None
            quiet_delay = self._quiet_seconds_remaining(key)
            if quiet_delay > 0:
                self._reset_bubble_deadline(state, anchor_at=monotonic() + quiet_delay)
                return None
            state.changed.clear()
            if state.active is not None or state.approved or state.judging:
                await state.changed.wait()
                continue
            cooldown_delay = (
                state.last_agent_finished_at
                + self.settings.cooldown_seconds
                - monotonic()
            )
            if cooldown_delay > 0:
                try:
                    await asyncio.wait_for(state.changed.wait(), timeout=cooldown_delay)
                except asyncio.TimeoutError:
                    pass
                continue
            self._check_sleep(key, state)
            waiting = _waiting_messages(state.pending)
            record_ids = [item.record_id for item in waiting]
            state.pending = []
            state.waking = False
            state.resume_after_rerun = False
            for item in waiting:
                item.release()
            ticket = PendingMessage(
                event,
                0,
                False,
                asyncio.get_running_loop().create_future(),
                monotonic(),
                backlog_sleep_revision=state.sleep_revision,
            )
            state.active = ticket
            state.active_ids = record_ids
            event.set_extra("mingle_woke_from_sleep", True)
            ticket.ready.set_result(record_ids)
            state.changed.set()
            return ticket
        return None

    def finish_random_bubble(self, ticket: PendingMessage) -> None:
        """Release a proactive turn only after its complete pipeline unwinds.

        Args:
            ticket: Internal event's active ticket, including claimed backlog.
        """
        state = self.sessions.get(ticket.event.unified_msg_origin)
        if state is None or state.active is not ticket:
            return
        interrupted = ticket.event.get_extra(
            "agent_user_aborted", False
        ) or ticket.event.get_extra("agent_stop_requested", False)
        outcome = "interrupted" if interrupted else "completed"
        self._finish_records(state, state.active_ids, outcome)
        if ticket.event.get_extra("mingle_agent_started", False):
            state.last_agent_finished_at = monotonic()
        state.active = None
        state.active_ids = []
        state.changed.set()

    def sleep(self, key: str) -> None:
        """Retain waiting work for a wake or timed review, preserving the active turn.

        Args:
            key: Full unified session identifier, respecting AstrBot isolation.
        """
        if self.closed:
            return
        state = self.sessions.setdefault(key, SessionState())
        state.sleeping = True
        state.waking = False
        state.sleep_revision += 1
        backlog_interval_seconds = self.settings.sleep_backlog_minutes * 60
        state.next_sleep_backlog_at = monotonic() + backlog_interval_seconds
        waiting_tickets = state.pending + state.judging + state.approved
        waiting_tickets = _waiting_messages(waiting_tickets)
        state.pending = sorted(
            waiting_tickets,
            key=lambda item: item.record_id,
        )
        state.judging = []
        state.approved = []
        while len(state.pending) > self.settings.max_pending_messages:
            oldest = state.pending.pop(0)
            self._finish_records(state, [oldest.record_id], "merged")
            oldest.release()
        state.resume_after_rerun = False
        state.changed.set()
        if state.pending and (
            state.judgment_task is None or state.judgment_task.done()
        ):
            state.judgment_task = asyncio.create_task(self._judge_pending(key, state))
        self.logger.info("Mingle session entered sleep: %s", key)

    def record_agent_reply(self, key: str) -> None:
        """Refresh inactivity and wake an admitted backlog turn after a reply.

        Args:
            key: Session in which the managed Agent sent a reply.
        """
        if self.closed:
            return
        state = self.sessions.setdefault(key, SessionState())
        state.last_agent_reply_at = monotonic()
        if state.bubble_task is not None:
            self._reset_bubble_deadline(state)
        if not state.sleeping:
            state.changed.set()
            return

        active = state.active
        if active is None:
            state.changed.set()
            return
        if active.backlog_sleep_revision != state.sleep_revision:
            state.changed.set()
            return

        state.sleeping = False
        state.last_woken_at = state.last_agent_reply_at
        self.logger.info("Mingle session woke after a backlog reply: %s", key)
        state.changed.set()

    def _check_sleep(self, key: str, state: SessionState) -> None:
        """Apply an expired reply deadline without interrupting the active Agent.

        Args:
            key: Session whose timeout may have elapsed.
            state: Live session queues and inactivity timestamps.
        """
        if state.sleeping:
            return
        timeout_seconds = self.settings.sleep_minutes * 60
        last_activity = max(state.last_agent_reply_at, state.last_woken_at)
        if timeout_seconds > 0 and monotonic() >= last_activity + timeout_seconds:
            self.sleep(key)

    async def submit(
        self,
        event: AstrMessageEvent,
        record_id: int,
        forced: bool,
        *,
        wake: bool = False,
    ) -> PendingMessage:
        """Queue an event without starting a second Agent for the same session.

        Args:
            event: Original event kept alive by its plugin handler.
            record_id: Persisted chat record identifier.
            forced: Whether a private message or explicit mention bypasses judgment.
            wake: Whether this message directly mentions the bot or contains a keyword.

        Returns:
            Admission ticket whose ready future contains a batch or None.
        """
        ticket = PendingMessage(
            event,
            record_id,
            forced,
            asyncio.get_running_loop().create_future(),
            monotonic(),
        )
        if self.closed:
            ticket.release()
            return ticket
        key = event.unified_msg_origin
        state = self.sessions.setdefault(key, SessionState())
        self._check_sleep(key, state)
        if state.sleeping and wake:
            # Invalidate a timed review and return its snapshot to the wake batch.
            state.sleep_revision += 1
            waiting_tickets = state.pending + state.judging
            waiting_tickets = _waiting_messages(waiting_tickets)
            state.pending = sorted(
                waiting_tickets,
                key=lambda item: item.record_id,
            )
            state.judging = []
            state.sleeping = False
            state.waking = True
            state.last_woken_at = monotonic()
            self.logger.info("Mingle session woke from sleep: %s", key)
        state.pending.append(ticket)
        while len(state.pending) > self.settings.max_pending_messages:
            oldest = state.pending.pop(0)
            self._finish_records(state, [oldest.record_id], "merged")
            # Keep direct-address intent even when its original event is merged away.
            state.pending[0].forced |= oldest.forced
            oldest.release()
        self.logger.info(
            "Mingle queue for session %s: pending=%d, judging=%d, approved=%d, active=%d",
            key,
            len(state.pending),
            len(state.judging),
            len(state.approved),
            len(state.active_ids),
        )
        state.changed.set()
        if state.judgment_task is None or state.judgment_task.done():
            state.judgment_task = asyncio.create_task(self._judge_pending(key, state))
        return ticket

    async def _judge_pending(self, key: str, state: SessionState) -> None:
        """Merge and judge arrivals while execution independently runs the Agent.

        Args:
            key: Unified session identifier.
            state: Pending and active work for this session.
        """
        batch: list[PendingMessage] = []
        try:
            while state.pending and not self.closed:
                await self._wait_for_pending(key, state)
                batch, state.pending = _waiting_messages(state.pending), []
                if not batch:
                    continue
                state.judging = batch
                revision = state.sleep_revision
                sleep_backlog = state.sleeping
                waking = state.waking or sleep_backlog
                state.waking = False
                if sleep_backlog:
                    backlog_interval_seconds = self.settings.sleep_backlog_minutes * 60
                    state.next_sleep_backlog_at = monotonic() + backlog_interval_seconds
                leader = batch[-1]
                forced = not waking and any(item.forced for item in batch)
                records = await self.history.get(
                    key,
                    limit=self.settings.initial_context_messages,
                    before_id=max(item.record_id for item in batch) + 1,
                )
                self._check_sleep(key, state)
                if revision != state.sleep_revision:
                    batch = []
                    continue
                if leader.ready.done():
                    state.judging = []
                    batch = []
                    continue
                if sleep_backlog and self._quiet_seconds_remaining(key) > 0:
                    self._defer_quiet_batch(state, batch)
                    state.judging = []
                    batch = []
                    continue

                # Snapshot only after IO: the previous Agent may have finished
                # during the history read, starting a fresh cooldown.
                active = state.active
                if active is not None and active.finished.is_set():
                    active = None
                cooldown_until = (
                    state.last_agent_finished_at + self.settings.cooldown_seconds
                )
                if (
                    active is None
                    and not forced
                    and not waking
                    and monotonic() < cooldown_until
                ):
                    state.pending = batch + state.pending
                    state.judging = []
                    batch = []
                    continue

                decision = await self._evaluate_batch(
                    key,
                    state,
                    batch,
                    records,
                    active,
                    waking=waking,
                    sleep_backlog=sleep_backlog,
                )
                self._check_sleep(key, state)
                if revision != state.sleep_revision:
                    batch = []
                    continue
                # Recovery may have claimed this batch while judgment awaited IO.
                if leader.ready.done():
                    state.judging = []
                    batch = []
                    continue
                if sleep_backlog and self._quiet_seconds_remaining(key) > 0:
                    self._defer_quiet_batch(state, batch)
                    state.judging = []
                    batch = []
                    continue
                self._apply_control(key, state, leader, active, decision)
                reply_requested = forced or decision.should_reply
                allowed = reply_requested and decision.action != "stop"
                if not allowed or self.closed or leader.ready.done():
                    self._finish_records(
                        state, [item.record_id for item in batch], "no_reply"
                    )
                    for item in batch:
                        item.release()
                else:
                    should_fold_pending = (
                        decision.should_reply
                        and decision.action == "wait"
                        and (active is not None or not forced)
                    )
                    if should_fold_pending:
                        # Freeze the tail once judgment returns. Later arrivals
                        # still need judgment, including stop/rerun requests.
                        batch.extend(_waiting_messages(state.pending))
                        state.pending = []
                    if waking:
                        batch[-1].event.set_extra("mingle_woke_from_sleep", True)
                    if sleep_backlog:
                        for item in batch:
                            item.backlog_sleep_revision = revision
                            item.event.set_extra("mingle_woke_from_sleep", True)
                    state.approved.extend(batch)
                    while len(state.approved) > self.settings.max_pending_messages:
                        oldest = state.approved.pop(0)
                        self._finish_records(state, [oldest.record_id], "merged")
                        oldest.release()
                    if state.execution_task is None or state.execution_task.done():
                        state.execution_task = asyncio.create_task(
                            self._execute_approved(key, state)
                        )
                state.judging = []
                self.logger.info(
                    "Mingle batch for session %s: messages=%d, forced=%s, admitted=%s; pending=%d, approved=%d, active=%d",
                    key,
                    len(batch),
                    forced,
                    allowed and not self.closed and not leader.ready.done(),
                    len(state.pending),
                    len(state.approved),
                    len(state.active_ids),
                )
                batch = []
                state.changed.set()
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.exception("Chat scheduling failed for session %s", key)
        finally:
            for item in batch + state.pending:
                item.release()
            state.pending.clear()
            state.judging = []

    async def _wait_for_pending(self, key: str, state: SessionState) -> None:
        """Wait until the current arrivals satisfy merge timing and cooldown.

        Recompute from original arrival times on every wakeup. An active Agent
        or direct-address ticket bypasses cooldown, but never merge timing.
        Sleeping sessions wait for the backlog interval and an idle Agent instead.

        Args:
            key: Unified session identifier for sleep transitions.
            state: Session whose pending queue remains open to new arrivals.
        """
        while state.pending and not self.closed:
            state.pending = _waiting_messages(state.pending)
            if not state.pending:
                return
            self._check_sleep(key, state)
            merge_at = min(
                state.pending[-1].received_at + self.settings.debounce_seconds,
                state.pending[0].received_at + self.settings.max_wait_seconds,
            )
            active = state.active
            busy = active is not None and not active.finished.is_set()
            if state.sleeping:
                state.changed.clear()
                backlog_processing_disabled = self.settings.sleep_backlog_minutes == 0
                if backlog_processing_disabled or busy:
                    await state.changed.wait()
                    continue
                backlog_ready_at = max(state.next_sleep_backlog_at, merge_at)
                backlog_delay = backlog_ready_at - monotonic()
                if backlog_delay <= 0:
                    quiet_delay = self._quiet_seconds_remaining(key)
                    if quiet_delay <= 0:
                        return
                    # Skip this review and subsequent quiet-hour slots without
                    # releasing retained input or making model requests.
                    interval = self.settings.sleep_backlog_minutes * 60
                    skipped_intervals = max(1, ceil(quiet_delay / interval))
                    state.next_sleep_backlog_at = (
                        monotonic() + skipped_intervals * interval
                    )
                    backlog_delay = state.next_sleep_backlog_at - monotonic()
                try:
                    await asyncio.wait_for(
                        state.changed.wait(),
                        timeout=backlog_delay,
                    )
                except asyncio.TimeoutError:
                    pass
                continue
            forced = state.waking or any(item.forced for item in state.pending)
            ready_at = merge_at
            if not busy and not forced:
                ready_at = max(
                    merge_at,
                    state.last_agent_finished_at + self.settings.cooldown_seconds,
                )
            delay = ready_at - monotonic()
            if delay <= 0:
                return
            # No await between inspecting state and clearing the notification:
            # arrivals and Agent transitions cannot be lost in that interval.
            state.changed.clear()
            if self.settings.sleep_minutes > 0:
                timeout_seconds = self.settings.sleep_minutes * 60
                last_activity = max(state.last_agent_reply_at, state.last_woken_at)
                sleep_at = last_activity + timeout_seconds
                remaining_until_sleep = max(0, sleep_at - monotonic())
                delay = min(delay, remaining_until_sleep)
            try:
                await asyncio.wait_for(state.changed.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass

    async def _evaluate_batch(
        self,
        key: str,
        state: SessionState,
        batch: list[PendingMessage],
        records: list[dict],
        active: PendingMessage | None,
        *,
        waking: bool = False,
        sleep_backlog: bool = False,
    ) -> ReplyDecision:
        """Build judgment context and evaluate against a captured active turn.

        This method does not change queue ownership or stop an Agent. The caller
        must validate the captured active ticket before applying control actions.

        Args:
            key: Unified session identifier for retained history reads.
            state: Live queues and bounded completion outcomes.
            batch: Arrivals being judged, with the newest event as leader.
            records: Recent history already loaded for this batch.
            active: Active ticket captured after the initial history read.
            waking: Judge the latest trigger with every retained waiting record.
            sleep_backlog: Permit a timed review while retaining the sleeping state.

        Returns:
            Reply admission and any requested control action. Idle direct-address
            batches are admitted without calling the judge.
        """
        active_ids = list(state.active_ids) if active is not None else []
        revision = state.sleep_revision
        loaded_record_ids = {item["id"] for item in records}
        missing_ids = set(active_ids) - loaded_record_ids
        if waking:
            batch_record_ids = {item.record_id for item in batch}
            missing_ids.update(batch_record_ids - loaded_record_ids)
        if missing_ids:
            # Preserve active input even when arrivals push it outside the
            # recent-history window. Keep IDs tied to the captured Agent.
            history_requests = []
            for record_id in missing_ids:
                history_requests.append(
                    self.history.get(
                        key,
                        limit=1,
                        after_id=record_id - 1,
                        before_id=record_id + 1,
                    )
                )
            active_records = await asyncio.gather(*history_requests)
            missing_records = []
            for rows in active_records:
                for item in rows:
                    if item["id"] in missing_ids:
                        missing_records.append(item)
            records = sorted(records + missing_records, key=lambda item: item["id"])
        if revision != state.sleep_revision:
            return ReplyDecision(False)
        if state.sleeping and not sleep_backlog:
            return ReplyDecision(False)
        if sleep_backlog and self._quiet_seconds_remaining(key) > 0:
            return ReplyDecision(False)
        message_status = {
            "finished": [
                {"id": record_id, "outcome": outcome}
                for record_id, outcome in state.finished.items()
            ],
            "active": [] if waking else active_ids,
            "queued": [
                item.record_id for item in state.approved if not item.ready.done()
            ],
            "awaiting_judgment": [
                item.record_id for item in state.pending if not item.ready.done()
            ],
            "judging": [item.record_id for item in batch],
        }
        leader = batch[-1]
        new_message_ids = [item.record_id for item in batch]
        if waking:
            new_message_ids = [leader.record_id]
        if active is not None:
            return await self.judge.evaluate(
                leader.event,
                records,
                message_status=message_status,
                new_message_ids=new_message_ids,
                waking=waking,
                sleep_backlog=sleep_backlog,
            )
        if not waking and any(item.forced for item in batch):
            self.logger.info(
                "Mingle judgment bypassed for forced idle batch in session %s", key
            )
            return ReplyDecision(True)
        should_reply = await self.judge.should_reply(
            leader.event,
            records,
            message_status=message_status,
            new_message_ids=new_message_ids,
            waking=waking,
            sleep_backlog=sleep_backlog,
        )
        return ReplyDecision(should_reply)

    def _apply_control(
        self,
        key: str,
        state: SessionState,
        leader: PendingMessage,
        active: PendingMessage | None,
        decision: ReplyDecision,
    ) -> None:
        """Apply stop/rerun only while the captured Agent and request remain valid.

        Args:
            key: Unified session identifier for logging.
            state: Live active turn and approved queue.
            leader: Ticket whose event requested judgment.
            active: Agent ticket captured before awaiting judgment.
            decision: Validated control action returned by the judge.
        """
        if decision.action not in {"rerun", "stop"} or active is None:
            return
        # A slow judgment must never stop a different, newer turn. A released
        # leader (for example, after group off) no longer authorizes control.
        if state.active is not active or active.finished.is_set():
            return
        if (
            leader.ready.done()
            or leader.event.is_stopped()
            or leader.event.get_extra("agent_stop_requested", False)
        ):
            return
        active.event.set_extra("agent_stop_requested", True)
        state.resume_after_rerun = decision.action == "rerun"
        self.logger.info(
            "Mingle requested Agent %s for session %s", decision.action, key
        )
        if decision.action == "stop":
            self._finish_records(
                state,
                [item.record_id for item in state.approved],
                "discarded_by_stop",
            )
            for item in state.approved:
                item.release()
            state.approved.clear()

    def claim_recovery_messages(self, ticket: PendingMessage) -> list[int] | None:
        """Fold waiting arrivals into the active handler's single recovery run.

        Release claimed handlers atomically so in-flight judgments cannot queue
        their messages again or stop the recovery with a stale control result.

        Args:
            ticket: Active handler after its previous Agent pipeline has unwound.

        Returns:
            Claimed record IDs, or None when the active handler is no longer valid.
        """
        if self.closed:
            return None
        event = ticket.event
        state = self.sessions.get(event.unified_msg_origin)
        if state is None or state.active is not ticket:
            return None
        if state.sleeping:
            return None
        if ticket.finished.is_set() or event.is_stopped():
            return None
        if event.get_extra("agent_stop_requested", False):
            return None
        if event.get_extra("agent_user_aborted", False):
            return None
        waiting = _waiting_messages(state.pending + state.judging + state.approved)
        record_ids = sorted(item.record_id for item in waiting)
        state.pending = []
        state.judging = []
        state.approved = []
        state.active_ids.extend(record_ids)
        for item in waiting:
            item.release()
        state.changed.set()
        return record_ids

    async def _execute_approved(self, key: str, state: SessionState) -> None:
        """Admit approved batches only after the previous pipeline has unwound.

        Args:
            key: Unified session identifier.
            state: Approved work and active execution for this session.
        """
        batch: list[PendingMessage] = []
        try:
            while state.approved and not self.closed:
                # A proactive turn owns the same execution slot, but its event
                # handler releases it after the normal pipeline saves history.
                while state.active is not None and not self.closed:
                    state.changed.clear()
                    await state.changed.wait()
                if self.closed:
                    break
                self._check_sleep(key, state)
                if state.sleeping and not state.approved:
                    break
                if state.sleeping:
                    backlog_leader = state.approved[-1]
                    backlog_matches_sleep = (
                        backlog_leader.backlog_sleep_revision == state.sleep_revision
                    )
                    if not backlog_matches_sleep:
                        break
                    if self._quiet_seconds_remaining(key) > 0:
                        self._defer_quiet_batch(state, state.approved)
                        state.approved = []
                        if state.judgment_task is None or state.judgment_task.done():
                            state.judgment_task = asyncio.create_task(
                                self._judge_pending(key, state)
                            )
                        state.changed.set()
                        break
                batch, state.approved = _waiting_messages(state.approved), []
                if not batch:
                    state.resume_after_rerun = False
                    continue
                leader = batch[-1]
                for item in batch:
                    if item.event.get_extra("mingle_woke_from_sleep", False):
                        leader.event.set_extra("mingle_woke_from_sleep", True)
                        break
                for item in batch[:-1]:
                    item.release()
                if state.resume_after_rerun:
                    leader.event.set_extra("mingle_resume_after_rerun", True)
                    state.resume_after_rerun = False
                state.active = leader
                state.active_ids = [item.record_id for item in batch]
                state.changed.set()
                leader.ready.set_result([item.record_id for item in batch])
                # The handler sets finished after the framework's history save,
                # not merely after its final response hook or first send.
                await leader.finished.wait()
                interrupted = leader.event.get_extra(
                    "agent_user_aborted", False
                ) or leader.event.get_extra("agent_stop_requested", False)
                outcome = "interrupted" if interrupted else "completed"
                self._finish_records(state, state.active_ids, outcome)
                if leader.event.get_extra("mingle_agent_started", False):
                    state.last_agent_finished_at = monotonic()
                state.active = None
                state.active_ids = []
                self.logger.info(
                    "Mingle Agent finished for session %s: outcome=%s, pending=%d, approved=%d",
                    key,
                    outcome,
                    len(state.pending),
                    len(state.approved),
                )
                state.changed.set()
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.exception("Mingle Agent admission failed for session %s", key)
        finally:
            for item in batch + state.approved:
                item.release()
            state.approved.clear()
            if any(state.active is item for item in batch):
                state.active = None
                state.active_ids = []
            state.changed.set()

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
            > self.settings.max_pending_messages
            + self.settings.initial_context_messages
        ):
            state.finished.pop(next(iter(state.finished)))

    async def close(self) -> None:
        """Stop admission, release waiting handlers, and stop owned active events."""
        self.closed = True
        tasks = []
        for state in self.sessions.values():
            if state.bubble_task is not None:
                state.bubble_task.cancel()
                tasks.append(state.bubble_task)
            if state.active is not None:
                state.active.event.set_extra("agent_stop_requested", True)
                state.active.event.stop_event()
            for item in state.pending + state.approved:
                item.release()
            if state.judgment_task is not None:
                state.judgment_task.cancel()
                tasks.append(state.judgment_task)
            if state.execution_task is not None:
                state.execution_task.cancel()
                tasks.append(state.execution_task)
        await asyncio.gather(*tasks, return_exceptions=True)
        self.sessions.clear()
