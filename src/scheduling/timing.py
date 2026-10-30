"""Deadline calculations shared by ordinary and proactive session work."""

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from random import expovariate
from time import monotonic as monotonic
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..settings import Settings

if TYPE_CHECKING:
    from .state import SessionState


def quiet_seconds_remaining(
    key: str, settings: Settings, get_config: Callable[..., dict] | None
) -> float:
    """Return seconds until quiet hours end in the current session timezone.

    Args:
        key: Full session UMO used to resolve AstrBot's current timezone.
        settings: Current immutable runtime configuration.
        get_config: Session profile resolver; missing or invalid zones use UTC.

    Returns:
        Remaining seconds, or zero outside the inclusive-start/exclusive-end
        interval. Equal endpoints disable quiet hours.
    """
    start = settings.quiet_hours_start
    end = settings.quiet_hours_end
    if start == end:
        return 0.0
    timezone_name = None
    if get_config is not None:
        timezone_name = get_config(umo=key).get("timezone")
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


def pending_deadline(state: "SessionState", settings: Settings) -> float:
    """Compute readiness from original arrivals without restarting merge timing.

    Args:
        state: Session with at least one valid pending ticket.
        settings: Current debounce, maximum wait, and cooldown configuration.

    Returns:
        Monotonic deadline, or infinity while sleeping work cannot be reviewed.
    """
    merge_at = min(
        state.pending[-1].received_at + settings.debounce_seconds,
        state.pending[0].received_at + settings.max_wait_seconds,
    )
    active = state.active
    busy = active is not None and not active.finished.is_set()
    if state.sleeping:
        if settings.sleep_backlog_minutes == 0 or busy:
            return float("inf")
        return max(state.next_sleep_backlog_at, merge_at)
    forced = state.waking or any(item.forced for item in state.pending)
    if busy or forced:
        return merge_at
    return max(merge_at, state.last_agent_finished_at + settings.cooldown_seconds)


def sleep_deadline(state: "SessionState", settings: Settings) -> float:
    """Return the inactivity deadline, or infinity when natural sleep is disabled.

    Args:
        state: Session's last successful send and explicit wake timestamps.
        settings: Current inactivity timeout.
    """
    if not state.sleep_enabled or state.sleeping or settings.sleep_minutes <= 0:
        return float("inf")
    last_activity = max(state.last_agent_reply_at, state.last_woken_at)
    return last_activity + settings.sleep_minutes * 60


def bubble_deadline(settings: Settings, anchor_at: float) -> float:
    """Sample a shifted exponential deadline without changing the send clock.

    Args:
        settings: Current minimum and total mean interval.
        anchor_at: Last real send or the end of quiet hours.
    """
    minimum = settings.random_bubble_min_minutes * 60
    random_mean = (
        settings.random_bubble_mean_minutes - settings.random_bubble_min_minutes
    ) * 60
    return anchor_at + minimum + expovariate(1 / random_mean)
