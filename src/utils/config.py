"""Configuration management module.

Loads configs/config.yaml and provides typed access to all settings.
"""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to config YAML file. Defaults to 'configs/config.yaml'.

    Returns:
        Dict containing configuration values.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
