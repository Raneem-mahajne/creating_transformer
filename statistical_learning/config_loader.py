"""Load YAML configs from statistical_learning/configs/."""
from pathlib import Path

import yaml

from statistical_learning.word_sets import validate_words

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


def load_config(config_name: str) -> dict:
    config_path = CONFIGS_DIR / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for field in ("name", "complexity", "words", "data", "model", "training"):
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")

    validate_words(config["words"], config["complexity"])
    return config
