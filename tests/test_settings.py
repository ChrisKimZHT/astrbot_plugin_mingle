import pytest
from mingle_tests_plugin.src.settings import Settings


@pytest.mark.parametrize(
    "config",
    [
        {"scheduler": {"debounce_seconds": -1}},
        {"scheduler": {"debounce_seconds": float("nan")}},
        {"scheduler": {"max_wait_seconds": 1, "debounce_seconds": 2}},
        {"judge": {"judge_timeout": 0}},
        {"history": {"recent_messages": True}},
        {"history": {"redaction_enabled": "true"}},
        {"history": {"recent_messages": 101}},
        {"general": {"enabled": "false"}},
        {"general": {"group_sessions": "group"}},
        {"general": {"group_sessions": [1]}},
        {"debug": {"debug_sessions": "group"}},
        {"debug": {"debug_sessions": [1]}},
        {"judge": {"judge_provider_id": 1}},
        {"judge": {"judge_prompt": None}},
        {"judge": {"judge_prompt_append": 123}},
        {"general": []},
        {"scheduler": None},
        {"toolbox": {"toolbox_enabled": "false"}},
        {"toolbox": {"toolbox_time_enabled": None}},
    ],
)
def test_invalid_settings_fail_at_startup(config):
    with pytest.raises(ValueError):
        Settings.from_config(config)


def test_schema_defaults_match_runtime_settings(tmp_path):
    import json
    from pathlib import Path

    from astrbot.api import AstrBotConfig

    schema = json.loads(
        (Path(__file__).parents[1] / "_conf_schema.json").read_text(encoding="utf-8")
    )
    config = AstrBotConfig(str(tmp_path / "config.json"), schema=schema)
    assert Settings.from_config(config) == Settings()
    assert Settings.from_config(
        {"general": {"group_sessions": ["  group  ", ""]}}
    ).group_sessions == ("group",)


def test_grouped_settings_load_custom_values():
    settings = Settings.from_config(
        {
            "general": {"enabled": False, "private_enabled": False},
            "scheduler": {"debounce_seconds": 3, "max_wait_seconds": 9},
            "judge": {"judge_provider_id": "custom", "judge_prompt_append": "extra"},
            "history": {"recent_messages": 15, "retention_days": 10},
            "images": {"image_cache_mb": 128, "max_image_mb": 5},
            "debug": {"debug_sessions": ["p:FriendMessage:alice"]},
            "toolbox": {"toolbox_enabled": False, "toolbox_time_enabled": False},
        }
    )
    assert settings == Settings(
        enabled=False,
        private_enabled=False,
        debounce_seconds=3,
        max_wait_seconds=9,
        judge_provider_id="custom",
        judge_prompt_append="extra",
        recent_messages=15,
        retention_days=10,
        image_cache_mb=128,
        max_image_mb=5,
        debug_sessions=("p:FriendMessage:alice",),
        toolbox_enabled=False,
        toolbox_time_enabled=False,
    )
