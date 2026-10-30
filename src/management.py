"""Management operations behind the plugin's native command registrations."""

import logging
from collections.abc import Callable
from dataclasses import replace

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent

from .history import ChatHistory
from .scheduler import ChatScheduler
from .settings import Settings
from .utils.history_formatting import format_prompt_history


async def manage_boundary(
    event: AstrMessageEvent,
    action: str,
    group: str,
    *,
    keyword: str = "",
    history: ChatHistory,
    get_config: Callable[..., dict],
    logger: logging.Logger,
) -> str:
    """Apply a validated group boundary operation and format its result.

    Args:
        event: Administrator command defining the session used for previews.
        action: Validated set, find, or clear operation.
        group: Platform-scoped group boundary identifier.
        keyword: Optional text to place the boundary before the latest match.
        history: Independent Mingle storage.
        get_config: Resolver for the current AstrBot session timezone.
        logger: Plugin logger for storage or formatting failures.

    Returns:
        User-facing result text, including the existing failure message.
    """
    try:
        if action == "set":
            boundary = await history.set_boundary(
                group, session=event.unified_msg_origin, keyword=keyword
            )
            if boundary is None:
                return "当前会话保留的聊天记录中未找到匹配消息，群聊记录边界未更改。"
            record_id, set_at = boundary
            if keyword:
                return (
                    "已设置群聊记录边界：位于最新匹配消息的上方，"
                    "该消息及之后的消息可见。"
                )
            return f"已设置群聊记录边界：记录 ID {record_id} 之后的消息可见。"
        if action == "clear":
            removed = await history.clear_boundary(group)
            if removed:
                return "已清除群聊记录边界。"
            return "当前群聊未设置记录边界。"
        preview = await history.boundary_preview(group, event.unified_msg_origin)
        if preview is None:
            return "当前群聊未设置记录边界。"
        timezone_name = get_config(umo=event.unified_msg_origin).get("timezone")
        lines = []
        for record in preview:
            if record is None:
                lines.append("（暂无保留的消息）")
                continue
            formatted_records = format_prompt_history([record], timezone_name)
            formatted = formatted_records[0]
            timestamp = formatted["timestamp"] or "未知时间"
            lines.append(f"[{timestamp}] {formatted['text']}")
        return f"{lines[0]}\n\n--- barrier ---\n\n{lines[1]}"
    except Exception:
        logger.exception("Failed to manage history boundary for %s", group)
        return "操作群聊记录边界失败，请查看日志。"


async def update_group(
    event: AstrMessageEvent,
    state: str,
    group_session: str,
    *,
    config: AstrBotConfig,
    settings: Settings,
    apply_settings: Callable[[Settings], None],
    scheduler: ChatScheduler,
    bubble_self_ids: dict[str, str],
    logger: logging.Logger,
) -> str:
    """Persist a validated group change under the caller's configuration lock.

    The caller captures settings after acquiring the lock. Publish the new
    snapshot only after saving succeeds, before updating group scheduling.

    Args:
        event: Group administrator command, already validated by its handler.
        state: Validated on or off value.
        group_session: Platform-scoped whole-group UMO.
        config: Plugin configuration whose unrelated grouped fields are retained.
        settings: Current runtime snapshot, captured while holding the lock.
        apply_settings: Synchronous publisher for the plugin and its components.
        scheduler: Owner of waiting tickets and proactive deadlines.
        bubble_self_ids: Last observed bot identity per shared group.
        logger: Plugin logger.

    Returns:
        User-facing success or save-failure text.
    """
    previous = dict(config["general"])
    sessions = list(previous.get("group_sessions", []))
    if state == "on":
        if group_session not in sessions:
            sessions.append(group_session)
    else:
        matching = {
            group_session,
            event.unified_msg_origin,
            event.get_group_id(),
        }
        sessions = [item for item in sessions if item.strip() not in matching]
    try:
        await config.save_config_async(
            {"general": {**previous, "group_sessions": sessions}}
        )
    except Exception:
        config["general"] = previous
        logger.exception(
            "Failed to update Mingle group allowlist for %s", group_session
        )
        return "保存配置失败，当前群聊的 Mingle 设置未更改。"

    updated_settings = replace(settings, group_sessions=tuple(sessions))
    apply_settings(updated_settings)
    if state == "on":
        bubble_self_ids[group_session] = event.get_self_id()
        scheduler.watch_random_bubbles(group_session)
    if state == "off":
        await scheduler.stop_random_bubbles(group_session)
        scheduler.release_group_waiters(group_session)
    logger.info("Mingle group allowlist updated for %s: %s", group_session, state)
    if state == "on":
        message = "已启用当前群聊的 Mingle。"
    else:
        message = "已关闭当前群聊的 Mingle。"
    if state == "on" and not updated_settings.enabled:
        message += "插件总开关当前关闭，需在配置中启用。"
    return message


async def update_debug(
    event: AstrMessageEvent,
    state: str,
    session: str,
    *,
    config: AstrBotConfig,
    settings: Settings,
    apply_settings: Callable[[Settings], None],
    logger: logging.Logger,
) -> str:
    """Persist a validated debug change under the caller's configuration lock.

    Args:
        event: Administrator command defining the current full UMO.
        state: Validated on or off value.
        session: Group-scoped UMO or current private session.
        config: Plugin configuration with all unrelated debug fields retained.
        settings: Current runtime snapshot, captured after acquiring the lock.
        apply_settings: Synchronous publisher for the plugin and its components.
        logger: Plugin logger.

    Returns:
        User-facing success or save-failure text.
    """
    previous = dict(config["debug"])
    sessions = []
    for item in previous.get("debug_sessions", []):
        stripped = item.strip()
        if stripped:
            sessions.append(stripped)
    if state == "on":
        if session not in sessions:
            sessions.append(session)
    else:
        sessions = [
            item for item in sessions if item not in {session, event.unified_msg_origin}
        ]
    try:
        await config.save_config_async(
            {"debug": {**previous, "debug_sessions": sessions}}
        )
    except Exception:
        config["debug"] = previous
        logger.exception("Failed to update Mingle debug sessions for %s", session)
        return "保存配置失败，当前会话的 debug 设置未更改。"

    updated_settings = replace(settings, debug_sessions=tuple(sessions))
    apply_settings(updated_settings)
    logger.info("Mingle debug setting updated for %s: %s", session, state)
    if state == "on":
        return "已开启当前会话的 debug，dump 文件路径会输出到日志。"
    return "已关闭当前会话的 debug，已有 dump 文件保留。"
