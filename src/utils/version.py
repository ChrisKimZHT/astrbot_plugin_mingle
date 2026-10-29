"""Read the release version from the plugin's authoritative metadata."""

from pathlib import Path

import yaml
from astrbot.api import logger

try:
    with (Path(__file__).resolve().parents[2] / "metadata.yaml").open(
        encoding="utf-8"
    ) as metadata_file:
        metadata = yaml.safe_load(metadata_file)
    if not isinstance(metadata, dict):
        raise ValueError("Plugin metadata must be a mapping")
    VERSION = metadata.get("version")
    if not isinstance(VERSION, str) or not VERSION.strip():
        raise ValueError("Plugin metadata must contain a non-empty version string")
except (OSError, UnicodeError, yaml.YAMLError, ValueError):
    VERSION = "unknown"
    logger.warning("Failed to read Mingle version from metadata.yaml")
