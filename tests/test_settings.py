import pytest
from mingle_tests_plugin.src.settings import Settings


@pytest.mark.parametrize(
    "config",
    [
        {"debounce_seconds": -1},
        {"debounce_seconds": float("nan")},
        {"max_wait_seconds": 1, "debounce_seconds": 2},
        {"judge_timeout": 0},
        {"recent_messages": True},
        {"recent_messages": 101},
        {"enabled": "false"},
        {"group_sessions": "group"},
        {"group_sessions": [1]},
        {"debug_sessions": "group"},
        {"debug_sessions": [1]},
        {"judge_provider_id": 1},
        {"judge_prompt": None},
        {"judge_prompt_append": 123},
    ],
)
def test_invalid_settings_fail_at_startup(config):
    with pytest.raises(ValueError):
        Settings.from_config(config)


def test_schema_defaults_match_runtime_settings():
    import json
    from pathlib import Path

    schema = json.loads(
        (Path(__file__).parents[1] / "_conf_schema.json").read_text(encoding="utf-8")
    )
    config = {name: value["default"] for name, value in schema.items()}
    assert Settings.from_config(config) == Settings()
    assert Settings.from_config(
        {"group_sessions": ["  group  ", ""]}
    ).group_sessions == ("group",)
