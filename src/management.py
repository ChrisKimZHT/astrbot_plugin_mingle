"""Management operations behind the plugin's native command registrations."""

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import replace

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent
from astrbot.core.conversation_mgr import ConversationManager
from astrbot.core.utils.session_lock import session_lock_manager

from .generation import agent_history_cursor, remove_message_turns
from .history import ChatHistory
from .scheduler import ChatScheduler
from .settings import Settings
from .utils.history_formatting import format_prompt_history


async def list_sessions(
    current_session: str,
    *,
    history: ChatHistory,
    scheduler: ChatScheduler,
    conversation_manager: ConversationManager,
    logger: logging.Logger,
) -> str:
    """List distinct full UMOs without loading native Agent message histories.

    Args:
        current_session: Command's full UMO, used only to mark the current entry.
        history: Independent Mingle storage.
        scheduler: Live session coordinator owner.
        conversation_manager: Native paginated conversation metadata API.
        logger: Plugin logger.

    Returns:
        Sorted session identifiers or a user-facing failure message.
    """
    try:
        sessions = set(await history.list_sessions())
        for key, state in scheduler.sessions.items():
            if (
                state.deleted_through_id is None
                or state.pending
                or state.judging
                or state.approved
                or state.active is not None
            ):
                sessions.add(key)
        page = 1
        page_size = 100
        while True:
            (
                conversations,
                total,
            ) = await conversation_manager.get_filtered_conversations(
                page=page, page_size=page_size, include_history=False
            )
            for conversation in conversations:
                sessions.add(conversation.user_id)
            if page * page_size >= total or not conversations:
                break
            page += 1
        if not sessions:
            return "暂无会话。"
        lines = [f"会话列表（共 {len(sessions)} 个，使用完整 UMO 删除）："]
        for session in sorted(sessions):
            marker = "🟢" if session == current_session else "⚪"
            lines.append(f"{marker} {session}")
        return "\n".join(lines)
    except Exception:
        logger.exception("Failed to list Mingle and Agent sessions")
        return "获取会话列表失败，请查看日志。"


async def delete_session(
    session: str,
    *,
    history: ChatHistory,
    scheduler: ChatScheduler,
    conversation_manager: ConversationManager,
    logger: logging.Logger,
) -> tuple[str, int | None]:
    """Delete native conversations and retained Mingle data under native locking.

    The caller reserves the idle coordinator until this operation returns. Native
    deletion uses AstrBot's public API, including selected-conversation cleanup.
    The two databases cannot share a transaction; report partial failure explicitly.

    Args:
        session: Validated full UMO, defaulted to the command's session by caller.
        history: Independent Mingle storage.
        scheduler: Owner of serialized sleeping-backlog writes.
        conversation_manager: Native API for deleting all conversations of a UMO.
        logger: Plugin logger.

    Returns:
        User-facing outcome and deletion high-water mark, or None on failure.
    """
    async with session_lock_manager.acquire_lock(session):
        agent_deleted = False
        try:
            await scheduler.flush_sleep_backlog(session)
            await conversation_manager.delete_conversations_by_user_id(session)
            agent_deleted = True
            deleted_through_id = await history.delete_session(session)
            return (
                f"已删除会话 {session} 的 Mingle 记录及全部 Agent 对话历史。",
                deleted_through_id,
            )
        except Exception:
            logger.exception("Failed to delete session %s", session)
            if agent_deleted:
                return "Agent 历史已删除，但 Mingle 会话清理失败，请重试。", None
            return "删除会话失败，部分数据可能已删除，请查看日志并重试。", None


async def delete_message(
    event: AstrMessageEvent,
    keyword: str,
    *,
    history: ChatHistory,
    conversation_manager: ConversationManager,
    logger: logging.Logger,
) -> tuple[str, int | None]:
    """Delete a selected record and complete linked requests under native locking.

    The caller reserves an idle Mingle coordinator until this operation returns.
    Native history is updated first; a storage failure restores its old value.

    Args:
        event: Administrator command restricting all access to its full UMO.
        keyword: Validated nonempty literal substring for latest-match selection.
        history: Independent Mingle storage.
        conversation_manager: Native conversation API, never raw framework SQL.
        logger: Plugin logger for failures without logging message content.

    Returns:
        User-facing outcome and the deleted record ID, or None on failure.
    """
    session = event.unified_msg_origin
    async with session_lock_manager.acquire_lock(session):
        try:
            record = await history.find_message(session, keyword)
            if record is None:
                return "当前会话保留的聊天记录中未找到匹配消息。", None
            conversation_id = await conversation_manager.get_curr_conversation_id(
                session
            )
            conversation = None
            original = []
            if conversation_id:
                conversation = await conversation_manager.get_conversation(
                    session, conversation_id
                )
                if conversation is None:
                    return "当前 Agent 对话无法读取，未删除消息。", None
                original = json.loads(conversation.history or "[]")
            try:
                edited, count = remove_message_turns(original, record)
            except ValueError:
                return "关联的 Agent 轮次已不完整，未删除消息。", None
            if count == 0 and original:
                cursor = agent_history_cursor(original)
                cannot_identify = (
                    not cursor or record["id"] <= cursor or record["role"] == "bot"
                )
                if cannot_identify:
                    return (
                        "未找到完整的新 Agent 轮次，记录可能已压缩或属于旧版本，未删除消息。",
                        None,
                    )
            history_attempted = False
            try:
                if count:
                    history_attempted = True
                    await conversation_manager.update_conversation(
                        session, conversation_id, history=edited
                    )
                removed = await history.delete_record(session, record["id"])
                if not removed:
                    raise RuntimeError("The selected message expired before deletion")
            except (Exception, asyncio.CancelledError) as error:
                if history_attempted:
                    try:
                        await conversation_manager.update_conversation(
                            session, conversation_id, history=original
                        )
                    except Exception:
                        logger.exception(
                            "Failed to restore Agent history for %s", session
                        )
                        if isinstance(error, asyncio.CancelledError):
                            raise error
                        return "删除失败，且 Agent 历史恢复失败，请查看日志。", None
                raise
            if count:
                return (
                    f"已删除消息 ID {record['id']}，并移除关联的 {count} 轮 Agent 历史"
                    "（含工具调用）；后续轮次保留。",
                    record["id"],
                )
            return (
                f"已删除消息 ID {record['id']}；当前 Agent 对话没有可关联的已保存轮次。",
                record["id"],
            )
        except Exception:
            logger.exception("Failed to delete Mingle message for %s", session)
            return "删除消息失败，请查看日志。", None


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
