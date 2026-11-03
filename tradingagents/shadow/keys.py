"""Read data-source keys from a file that also holds keys this package must never see."""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
KEY_FILE_ENV = "SHADOW_KEY_FILE"
_DATA_KEYS = ("FRED_API_KEY", "FINNHUB_API_KEY", "SEC_USER_AGENT")


def load_data_keys(path: Path | str) -> list[str]:
    """Set the named keys in this process and return their names. Other lines are not kept."""
    try:
        lines = Path(path).read_text().splitlines()
    except OSError as exc:
        logger.warning("Key file unreadable (%s); data sources needing keys will be skipped", type(exc).__name__)
        return []
    loaded = []
    for line in lines:
        name, _, value = line.partition("=")
        if name in _DATA_KEYS and value:
            os.environ[name] = value.strip().strip("\"'")
            loaded.append(name)
    return loaded
