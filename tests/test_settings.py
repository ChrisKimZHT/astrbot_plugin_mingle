import pytest
from mingle_tests_plugin.src.settings import Settings


@pytest.mark.parametrize(
    "config",
    [
        {"scheduler": {"debounce_seconds": -1}},
        {"scheduler": {"sleep_seconds": -1}},
        {"scheduler": {"sleep_seconds": float("nan")}},
        {"scheduler": {"sleep_seconds": True}},
        {"scheduler": {"sleep_backlog_minutes": -1}},
        {"scheduler": {"sleep_backlog_minutes": float("nan")}},
        {"scheduler": {"sleep_backlog_minutes": float("inf")}},
        {"scheduler": {"sleep_backlog_minutes": True}},
        {"scheduler": {"sleep_backlog_minutes": "30"}},
        {"scheduler": {"sleep_wake_keywords": "name"}},
        {"scheduler": {"sleep_wake_keywords": [None]}},
        {"scheduler": {"debounce_seconds": float("nan")}},
        {"scheduler": {"max_wait_seconds": 1, "debounce_seconds": 2}},
        {"judge": {"judge_timeout": 0}},
        {"history": {"initial_context_messages": True}},
        {"history": {"redaction_enabled": "true"}},
        {"history": {"initial_context_messages": 101}},
        {"general": {"enabled": "false"}},
        {"general": {"group_sessions": "group"}},
        {"general": {"group_sessions": [1]}},
        {"debug": {"debug_sessions": "group"}},
        {"debug": {"debug_sessions": [1]}},
        {"judge": {"judge_provider_id": 1}},
        {"judge": {"judge_prompt": None}},
        {"judge": {"judge_prompt_append": 123}},
        {"generator": {"generator_prompt": None}},
        {"generator": {"generator_prompt_append": 123}},
        {"generator": []},
        {"general": []},
        {"scheduler": None},
        {"toolbox": {"toolbox_enabled": "false"}},
        {"toolbox": {"toolbox_time_enabled": None}},
        {"toolbox": {"toolbox_calculate_enabled": 1}},
        {"toolbox": {"toolbox_random_enabled": "false"}},
        {"toolbox": {"toolbox_image_generate_enabled": "true"}},
        {"mingle_generate_image": {"toolbox_image_generate_provider_id": 1}},
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
    assert list(schema) == [
        "general",
        "scheduler",
        "judge",
        "generator",
        "history",
        "images",
        "toolbox",
        "mingle_generate_image",
        "mingle_edit_image",
        "debug",
        "experimental",
    ]
    assert Settings.from_config(
        {"general": {"group_sessions": ["  group  ", ""]}}
    ).group_sessions == ("group",)


def test_grouped_settings_load_custom_values():
    settings = Settings.from_config(
        {
            "general": {"enabled": False, "private_enabled": False},
            "scheduler": {"debounce_seconds": 3, "max_wait_seconds": 9},
            "judge": {"judge_provider_id": "custom", "judge_prompt_append": "extra"},
            "generator": {
                "generator_prompt": "base",
                "generator_prompt_append": "\nextra",
            },
            "history": {"initial_context_messages": 15, "retention_days": 10},
            "images": {"image_cache_mb": 128, "max_image_mb": 5},
            "debug": {"debug_sessions": ["p:FriendMessage:alice"]},
            "toolbox": {"toolbox_enabled": False, "toolbox_time_enabled": False},
            "mingle_generate_image": {
                "toolbox_image_generate_provider_id": "image-provider"
            },
        }
    )
    assert settings == Settings(
        enabled=False,
        private_enabled=False,
        debounce_seconds=3,
        max_wait_seconds=9,
        judge_provider_id="custom",
        judge_prompt_append="extra",
        generator_prompt="base",
        generator_prompt_append="\nextra",
        initial_context_messages=15,
        retention_days=10,
        image_cache_mb=128,
        max_image_mb=5,
        debug_sessions=("p:FriendMessage:alice",),
        toolbox_enabled=False,
        toolbox_time_enabled=False,
        toolbox_image_generate_provider_id="image-provider",
    )


def test_sleep_settings():
    assert Settings().sleep_seconds == 180
    assert Settings().sleep_backlog_minutes == 30
    settings = Settings.from_config(
        {
            "scheduler": {
                "sleep_seconds": 0,
                "sleep_backlog_minutes": 0,
                "sleep_wake_keywords": ["", "Mingle", "名字"],
            }
        }
    )
    assert settings.sleep_seconds == 0
    assert settings.sleep_backlog_minutes == 0
    assert settings.sleep_wake_keywords == ("Mingle", "名字")


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "3", None])
@pytest.mark.parametrize(
    "name", ["judge_parallel_requests", "judge_wake_parallel_requests"]
)
def test_invalid_parallel_request_count(value, name):
    with pytest.raises(ValueError):
        Settings.from_config({"experimental": {name: value}})


def test_parallel_request_count_configuration():
    assert Settings().judge_parallel_requests == 1
    assert Settings().judge_wake_parallel_requests == 1
    settings = Settings.from_config(
        {
            "experimental": {
                "judge_parallel_requests": 5,
                "judge_wake_parallel_requests": 3,
            }
        }
    )
    assert settings.judge_parallel_requests == 5
    assert settings.judge_wake_parallel_requests == 3
