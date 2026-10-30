import importlib.util
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

# Framework imports initialize runtime files before fixtures run. Isolate only
# this test process from the checkout and any configured production data root.
_FRAMEWORK_ROOT = TemporaryDirectory(prefix="mingle-tests-", ignore_cleanup_errors=True)
_PREVIOUS_ROOT = os.environ.get("ASTRBOT_ROOT")
os.environ["ASTRBOT_ROOT"] = _FRAMEWORK_ROOT.name


def pytest_unconfigure(config):
    """Restore the test process environment and remove disposable framework data."""
    if _PREVIOUS_ROOT is None:
        os.environ.pop("ASTRBOT_ROOT", None)
    else:
        os.environ["ASTRBOT_ROOT"] = _PREVIOUS_ROOT
    _FRAMEWORK_ROOT.cleanup()


# Load the plugin under a stable package name, including in standalone checkouts.
PLUGIN_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "mingle_tests_plugin",
    PLUGIN_ROOT / "__init__.py",
    submodule_search_locations=[str(PLUGIN_ROOT)],
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


@pytest.fixture(autouse=True)
def scheduler_wall_clock(monkeypatch):
    """Keep scheduling tests independent of the actual time of day."""

    class Daytime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 30, 12, tzinfo=timezone.utc).astimezone(tz)

    monkeypatch.setattr("mingle_tests_plugin.src.scheduling.timing.datetime", Daytime)
