import importlib.util
import sys
from pathlib import Path

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
