from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Settings:
    enabled: bool = True
    group_sessions: tuple[str, ...] = ()
    private_enabled: bool = True
    judge_provider_id: str = ""
    judge_timeout: float = 20.0
    debounce_seconds: float = 2.0
    max_wait_seconds: float = 8.0
    cooldown_seconds: float = 60.0
    recent_messages: int = 30
    max_pending_messages: int = 100
    retention_days: int = 30
    max_messages_per_session: int = 10000
    image_retention_days: int = 7
    image_cache_mb: int = 512
    max_image_mb: int = 20

    @classmethod
    def from_config(cls, config: dict) -> "Settings":
        """Validate plugin configuration before starting background work.

        Args:
            config: Values supplied by the AstrBot plugin configuration loader.

        Returns:
            Immutable runtime settings.

        Raises:
            ValueError: If a setting has an invalid type or range.
        """
        values = {
            field.name: config[field.name]
            for field in fields(cls)
            if field.name in config
        }
        sessions = values.get("group_sessions", ())
        if not isinstance(sessions, (list, tuple)) or not all(
            isinstance(item, str) for item in sessions
        ):
            raise ValueError("group_sessions must be a list of session identifiers")
        values["group_sessions"] = tuple(
            item.strip() for item in sessions if item.strip()
        )
        settings = cls(**values)
        for name in ("enabled", "private_enabled"):
            if not isinstance(getattr(settings, name), bool):
                raise ValueError(f"{name} must be a boolean")
        for name in (
            "judge_timeout",
            "debounce_seconds",
            "max_wait_seconds",
            "cooldown_seconds",
        ):
            value = getattr(settings, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value <= 86400
            ):
                raise ValueError(f"{name} must be between 0 and 86400")
        for name in (
            "recent_messages",
            "max_pending_messages",
            "retention_days",
            "max_messages_per_session",
            "image_retention_days",
            "image_cache_mb",
            "max_image_mb",
        ):
            value = getattr(settings, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            settings.judge_timeout <= 0
            or settings.max_wait_seconds < settings.debounce_seconds
        ):
            raise ValueError(
                "judge_timeout must be positive and max_wait_seconds must cover debounce_seconds"
            )
        if not isinstance(settings.judge_provider_id, str):
            raise ValueError("judge_provider_id must be a string")
        if settings.recent_messages > 100:
            raise ValueError("recent_messages must not exceed 100")
        return settings
