"""One event-driven coordinator owns all scheduling transitions for a session."""

import asyncio
from math import ceil
from typing import TYPE_CHECKING

from astrbot.api.event import AstrMessageEvent
from astrbot.core.platform.message_type import MessageType

from ..decision import ReplyDecision
from . import timing
from .judgment import JudgmentAttempt
from .proactive import ProactiveSchedule
from .state import CompletionSignal, PendingMessage, SessionState, waiting_messages

if TYPE_CHECKING:
    from ..scheduler import ChatScheduler


class SessionCoordinator:
    """Apply atomic entry operations and coordinate background results centrally.

    All mutations happen on the framework event loop, without awaits inside a
    transition. Entry operations retain their synchronous semantics; they notify
    this owner's loop rather than starting competing scheduling workers. Only
    history/model IO and event dispatch run in separate, read-only tasks.
    """

    def __init__(
        self, scheduler: "ChatScheduler", key: str, state: SessionState
    ) -> None:
        self.scheduler = scheduler
        self.key = key
        self.state = state
        self.attempt: JudgmentAttempt | None = None
        self.proactive = ProactiveSchedule(self)
        self.waiting_for_proactive = False

    def notify(self) -> None:
        """Wake the owner, restarting it if a previous coordinator exited."""
        self.state.changed.set()
        if self.scheduler.closed:
            return
        task = self.state.coordinator_task
        if task is None or task.done():
            self.state.coordinator_task = asyncio.create_task(self.run())

    def submit(
        self, event: AstrMessageEvent, record_id: int, forced: bool, wake: bool
    ) -> PendingMessage:
        """Retain an arrival and preserve wake intent through bounded merging.

        Args:
            event: Original framework event retained by its handler.
            record_id: Persisted Mingle database record ID.
            forced: Private or directly addressed input bypassing idle judgment.
            wake: Direct mention or literal keyword that wakes a sleeping group.
        """
        state = self.state
        ticket = PendingMessage(
            event,
            record_id,
            forced,
            asyncio.get_running_loop().create_future(),
            timing.monotonic(),
            finished=CompletionSignal(state.changed),
        )
        if self.scheduler.closed:
            ticket.release()
            return ticket
        self.check_sleep()
        if state.sleeping and wake:
            state.sleep_revision += 1
            state.pending = sorted(
                waiting_messages(state.pending + state.judging),
                key=lambda item: item.record_id,
            )
            state.judging = []
            state.sleeping = False
            state.waking = True
            state.last_woken_at = timing.monotonic()
            self.scheduler.logger.info("Mingle session woke from sleep: %s", self.key)
        state.pending.append(ticket)
        self.bound_pending(preserve_forced=True)
        self.scheduler.logger.info(
            "Mingle queue for session %s: pending=%d, judging=%d, approved=%d, active=%d",
            self.key,
            len(state.pending),
            len(state.judging),
            len(state.approved),
            len(state.active_ids),
        )
        self.notify()
        return ticket

    def sleep(self) -> None:
        """Invalidate old judgments and retain waiting work without stopping AI."""
        state = self.state
        if self.scheduler.closed or not state.sleep_enabled:
            return
        state.sleeping = True
        state.waking = False
        state.sleep_revision += 1
        interval = self.scheduler.settings.sleep_backlog_minutes * 60
        state.next_sleep_backlog_at = timing.monotonic() + interval
        waiting = waiting_messages(state.pending + state.judging + state.approved)
        state.pending = sorted(waiting, key=lambda item: item.record_id)
        state.judging = []
        state.approved = []
        self.bound_pending(preserve_forced=False)
        state.resume_after_rerun = False
        self.notify()
        self.scheduler.logger.info("Mingle session entered sleep: %s", self.key)

    def record_reply(self) -> None:
        """Refresh send clocks and wake only an authorized sleeping backlog turn."""
        state = self.state
        state.last_agent_reply_at = timing.monotonic()
        if state.bubbles_enabled:
            self.proactive.reset_deadline()
        active = state.active
        if (
            state.sleeping
            and active is not None
            and active.backlog_sleep_revision == state.sleep_revision
        ):
            state.sleeping = False
            state.last_woken_at = state.last_agent_reply_at
            self.scheduler.logger.info(
                "Mingle session woke after a backlog reply: %s", self.key
            )
        self.notify()

    def check_sleep(self) -> None:
        """Apply inactivity at the same admission and judgment boundaries."""
        if timing.monotonic() >= timing.sleep_deadline(
            self.state, self.scheduler.settings
        ):
            self.sleep()

    def bound_pending(self, *, preserve_forced: bool) -> None:
        """Release overflow tickets with the caller's existing merge semantics.

        Args:
            preserve_forced: Carry direct-address intent to the next ticket.
                Manual sleep intentionally does not carry that intent.
        """
        state = self.state
        while len(state.pending) > self.scheduler.settings.max_pending_messages:
            oldest = state.pending.pop(0)
            self.finish_records([oldest.record_id], "merged")
            if preserve_forced:
                state.pending[0].forced |= oldest.forced
            oldest.release()

    def defer_quiet_batch(self, batch: list[PendingMessage]) -> None:
        """Return postponed backlog to bounded pending input in record order.

        Args:
            batch: Judging or approved tickets crossing into quiet hours.
        """
        state = self.state
        state.pending = sorted(
            waiting_messages(batch + state.pending), key=lambda item: item.record_id
        )
        self.bound_pending(preserve_forced=True)
        state.changed.set()

    def claim_recovery(self, ticket: PendingMessage) -> list[int] | None:
        """Atomically claim waiting input for the active handler's recovery run.

        Args:
            ticket: Handler whose previous pipeline has fully saved history.

        Returns:
            Claimed IDs, or None if recovery is no longer eligible.
        """
        state = self.state
        event = ticket.event
        if self.scheduler.closed or state.active is not ticket or state.sleeping:
            return None
        if ticket.finished.is_set() or event.is_stopped():
            return None
        if event.get_extra("agent_stop_requested", False):
            return None
        if event.get_extra("agent_user_aborted", False):
            return None
        waiting = waiting_messages(state.pending + state.judging + state.approved)
        record_ids = sorted(item.record_id for item in waiting)
        state.pending = []
        state.judging = []
        state.approved = []
        state.active_ids.extend(record_ids)
        for item in waiting:
            item.release()
        self.notify()
        return record_ids

    def release_group_waiters(self, group_session: str) -> None:
        """Release a removed group's waiting tickets, including isolated members.

        Args:
            group_session: Platform-scoped whole-group UMO removed by a command.
        """
        state = self.state
        for ticket in state.pending + state.approved + state.judging:
            event = ticket.event
            current_group = (
                f"{event.get_platform_id()}:"
                f"{MessageType.GROUP_MESSAGE.value}:{event.get_group_id()}"
            )
            if current_group == group_session:
                ticket.release()
        self.notify()

    def close(self) -> list[asyncio.Task]:
        """Stop admission and return canceled owned tasks for the entry to join."""
        state = self.state
        if state.active is not None:
            state.active.event.set_extra("agent_stop_requested", True)
            state.active.event.stop_event()
        for item in state.pending + state.judging + state.approved:
            item.release()
        state.pending = []
        state.judging = []
        state.approved = []
        if self.proactive.waiting is not None:
            result = self.proactive.waiting[2]
            if not result.done():
                result.set_result(None)
            self.proactive.waiting = None
        tasks = []
        for task in (
            state.coordinator_task,
            state.judgment_task,
            state.dispatch_task,
        ):
            if task is not None:
                task.cancel()
                tasks.append(task)
        return tasks

    def finish_active(self, ticket: PendingMessage) -> None:
        """Release the shared slot only after the handler's full pipeline unwinds.

        Args:
            ticket: Exact active handler, never a newer replacement turn.
        """
        state = self.state
        if state.active is not ticket:
            return
        interrupted = ticket.event.get_extra(
            "agent_user_aborted", False
        ) or ticket.event.get_extra("agent_stop_requested", False)
        outcome = "interrupted" if interrupted else "completed"
        self.finish_records(state.active_ids, outcome)
        if ticket.event.get_extra("mingle_agent_started", False):
            state.last_agent_finished_at = timing.monotonic()
        state.active = None
        state.active_ids = []
        self.scheduler.logger.info(
            "Mingle Agent finished for session %s: outcome=%s, pending=%d, approved=%d",
            self.key,
            outcome,
            len(state.pending),
            len(state.approved),
        )
        state.changed.set()

    def finish_records(self, record_ids: list[int], outcome: str) -> None:
        """Keep bounded runtime outcomes without claiming a durable audit trail.

        Args:
            record_ids: Mingle database IDs reaching a scheduling outcome.
            outcome: Completion, interruption, rejection, or merge result.
        """
        state = self.state
        for record_id in record_ids:
            state.finished[record_id] = outcome
        settings = self.scheduler.settings
        limit = settings.max_pending_messages + settings.initial_context_messages
        while len(state.finished) > limit:
            state.finished.pop(next(iter(state.finished)))

    async def run(self) -> None:
        """Wait for notifications or the nearest useful deadline without polling."""
        state = self.state
        try:
            while not self.scheduler.closed:
                state.changed.clear()
                active = state.active
                if active is not None and active.finished.is_set():
                    self.finish_active(active)
                self.advance_judgment()
                self.admit_approved()
                self.proactive.advance()
                self.start_due_judgment()
                self.proactive.admit()
                deadline = self.next_deadline()
                if deadline == float("inf"):
                    await state.changed.wait()
                    continue
                delay = max(0.0, deadline - timing.monotonic())
                try:
                    await asyncio.wait_for(state.changed.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception:
            self.scheduler.logger.exception(
                "Chat scheduling failed for session %s", self.key
            )
            for item in state.pending + state.judging + state.approved:
                item.release()
            state.pending = []
            state.judging = []
            state.approved = []

    def next_deadline(self) -> float:
        """Return the next actionable deadline; busy work waits for notification."""
        state = self.state
        settings = self.scheduler.settings
        deadline = float("inf")
        if state.pending and self.attempt is None:
            deadline = timing.pending_deadline(state, settings)
            deadline = min(deadline, timing.sleep_deadline(state, settings))
        return min(deadline, self.proactive.next_deadline())

    def start_due_judgment(self) -> None:
        """Freeze one ready batch and launch history IO without blocking arrivals."""
        state = self.state
        if self.attempt is not None or not state.pending:
            return
        state.pending = waiting_messages(state.pending)
        if not state.pending:
            return
        self.check_sleep()
        settings = self.scheduler.settings
        if timing.pending_deadline(state, settings) > timing.monotonic():
            return
        if state.sleeping:
            quiet_delay = self.scheduler._quiet_seconds_remaining(self.key)
            if quiet_delay > 0:
                interval = settings.sleep_backlog_minutes * 60
                skipped_intervals = max(1, ceil(quiet_delay / interval))
                state.next_sleep_backlog_at = (
                    timing.monotonic() + skipped_intervals * interval
                )
                return
        batch, state.pending = state.pending, []
        state.judging = batch
        sleep_backlog = state.sleeping
        waking = state.waking or sleep_backlog
        state.waking = False
        if sleep_backlog:
            state.next_sleep_backlog_at = (
                timing.monotonic() + settings.sleep_backlog_minutes * 60
            )
        self.attempt = JudgmentAttempt(
            batch,
            state.sleep_revision,
            waking,
            sleep_backlog,
            forced=not waking and any(item.forced for item in batch),
        )
        state.judgment_task = asyncio.create_task(
            self.attempt.load_history(
                self.scheduler.history, self.key, settings.initial_context_messages
            )
        )
        state.judgment_task.add_done_callback(self.background_finished)

    def background_finished(self, task: asyncio.Task) -> None:
        """Notify the owner; background callbacks never mutate scheduling state.

        Args:
            task: Completed history, judgment, or dispatch task.
        """
        self.state.changed.set()

    def advance_judgment(self) -> None:
        """Apply completed IO only while its captured batch still owns judgment."""
        state = self.state
        attempt = self.attempt
        task = state.judgment_task
        if attempt is None or task is None or not task.done():
            return
        try:
            result = task.result()
        except asyncio.CancelledError:
            raise
        except Exception:
            self.scheduler.logger.exception(
                "Chat scheduling failed for session %s", self.key
            )
            for item in attempt.batch + state.pending:
                item.release()
            state.pending = []
            state.judging = []
            self.attempt = None
            return
        if attempt.stage != "supplement":
            self.check_sleep()
        stale = (
            attempt.revision != state.sleep_revision or attempt.batch[-1].ready.done()
        )
        if stale:
            if state.judging is attempt.batch:
                state.judging = []
            self.attempt = None
            return
        if (
            attempt.sleep_backlog
            and self.scheduler._quiet_seconds_remaining(self.key) > 0
        ):
            self.defer_quiet_batch(attempt.batch)
            state.judging = []
            self.attempt = None
            return
        if attempt.stage == "decision":
            self.apply_decision(attempt, result)
            state.judging = []
            self.attempt = None
            return
        attempt.records = result
        if attempt.stage == "history":
            attempt.active = state.active
            if attempt.active is not None and attempt.active.finished.is_set():
                attempt.active = None
            cooldown_at = (
                state.last_agent_finished_at + self.scheduler.settings.cooldown_seconds
            )
            defer_for_cooldown = (
                attempt.active is None
                and not attempt.forced
                and not attempt.waking
                and timing.monotonic() < cooldown_at
            )
            if defer_for_cooldown:
                state.pending = attempt.batch + state.pending
                state.judging = []
                self.attempt = None
                return
            if attempt.active is not None:
                attempt.active_ids = list(state.active_ids)
            missing_ids = attempt.missing_record_ids()
            if missing_ids:
                attempt.stage = "supplement"
                state.judgment_task = asyncio.create_task(
                    attempt.supplement_history(
                        self.scheduler.history, self.key, missing_ids
                    )
                )
                state.judgment_task.add_done_callback(self.background_finished)
                return
        attempt.stage = "decision"
        state.judgment_task = asyncio.create_task(
            attempt.evaluate(
                self.scheduler.judge,
                self.scheduler.logger,
                self.key,
                attempt.message_status(state),
            )
        )
        state.judgment_task.add_done_callback(self.background_finished)

    def apply_decision(self, attempt: JudgmentAttempt, decision: ReplyDecision) -> None:
        """Apply control and admission together, freezing only the current tail.

        Args:
            attempt: Still-valid judgment, including the captured active turn.
            decision: Validated result; forced busy replies retain admission.
        """
        state = self.state
        batch = attempt.batch
        leader = batch[-1]
        active = attempt.active
        control_allowed = (
            decision.action in {"rerun", "stop"}
            and active is not None
            and state.active is active
            and not active.finished.is_set()
            and not leader.event.is_stopped()
            and not leader.event.get_extra("agent_stop_requested", False)
        )
        if control_allowed:
            active.event.set_extra("agent_stop_requested", True)
            state.resume_after_rerun = decision.action == "rerun"
            self.scheduler.logger.info(
                "Mingle requested Agent %s for session %s", decision.action, self.key
            )
            if decision.action == "stop":
                self.finish_records(
                    [item.record_id for item in state.approved], "discarded_by_stop"
                )
                for item in state.approved:
                    item.release()
                state.approved = []
        allowed = (
            attempt.forced or decision.should_reply
        ) and decision.action != "stop"
        if not allowed:
            self.finish_records([item.record_id for item in batch], "no_reply")
            for item in batch:
                item.release()
        else:
            fold_pending = (
                decision.should_reply
                and decision.action == "wait"
                and (active is not None or not attempt.forced)
            )
            if fold_pending:
                batch.extend(waiting_messages(state.pending))
                state.pending = []
            if attempt.waking:
                batch[-1].event.set_extra("mingle_woke_from_sleep", True)
            if attempt.sleep_backlog:
                for item in batch:
                    item.backlog_sleep_revision = attempt.revision
                    item.event.set_extra("mingle_woke_from_sleep", True)
            state.approved.extend(batch)
            while len(state.approved) > self.scheduler.settings.max_pending_messages:
                oldest = state.approved.pop(0)
                self.finish_records([oldest.record_id], "merged")
                oldest.release()
        self.scheduler.logger.info(
            "Mingle batch for session %s: messages=%d, forced=%s, admitted=%s; pending=%d, approved=%d, active=%d",
            self.key,
            len(batch),
            attempt.forced,
            allowed and not leader.ready.done(),
            len(state.pending),
            len(state.approved),
            len(state.active_ids),
        )
        state.changed.set()

    def admit_approved(self) -> None:
        """Grant the shared slot to approved input after the old pipeline ends."""
        state = self.state
        if state.active is not None:
            if state.approved and state.active.record_id == 0:
                self.waiting_for_proactive = True
            return
        if self.waiting_for_proactive:
            # Recovery or sleep may have reclaimed the approved queue while an
            # ordinary admission was waiting for a proactive pipeline to unwind.
            # Preserve that admission's inactivity check even with an empty queue.
            self.waiting_for_proactive = False
            self.check_sleep()
        if not state.approved:
            return
        self.check_sleep()
        if not state.approved:
            return
        if state.sleeping:
            leader = state.approved[-1]
            if leader.backlog_sleep_revision != state.sleep_revision:
                return
            if self.scheduler._quiet_seconds_remaining(self.key) > 0:
                self.defer_quiet_batch(state.approved)
                state.approved = []
                return
        batch, state.approved = waiting_messages(state.approved), []
        if not batch:
            state.resume_after_rerun = False
            return
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
        leader.ready.set_result(list(state.active_ids))
        state.changed.set()
