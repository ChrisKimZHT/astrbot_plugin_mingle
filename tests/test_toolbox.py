import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from mingle_tests_plugin.toolbox import time as time_tool


@pytest.mark.parametrize(
    ("zone", "expected", "weekday"),
    [
        ("Asia/Shanghai", "2026-10-28 01:30:00+08:00", "Wednesday"),
        ("UTC", "2026-10-27 17:30:00+00:00", "Tuesday"),
        ("America/New_York", "2026-10-27 13:30:00-04:00", "Tuesday"),
    ],
)
def test_time_tool_respects_timezone_date_boundary_and_offset(
    monkeypatch, zone, expected, weekday
):
    fixed = datetime(2026, 10, 27, 17, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(
        time_tool, "datetime", SimpleNamespace(now=lambda tz: fixed.astimezone(tz))
    )
    result = json.loads(time_tool.get_current_time(zone))
    assert result == {"datetime": expected, "timezone": zone, "weekday": weekday}


def test_time_tool_falls_back_to_utc_for_invalid_timezone(monkeypatch, caplog):
    fixed = datetime(2026, 10, 27, 17, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(
        time_tool, "datetime", SimpleNamespace(now=lambda tz: fixed.astimezone(tz))
    )
    result = json.loads(time_tool.get_current_time("Invalid/Zone"))
    assert result["timezone"] == "UTC"
    assert result["datetime"] == "2026-10-27 17:30:00+00:00"
    assert "using UTC" in caplog.text
