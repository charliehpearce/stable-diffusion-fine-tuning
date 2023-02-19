import json
import os
from constants import DATA_MOUNT_DIR
from .config_model import StableDiffusionConfig


def load_config(config_uuid: str) -> StableDiffusionConfig:
    """
    Load config, this could eventually from a database that can
    be exposed to users/engineers.
    """
    with open(os.path.join(DATA_MOUNT_DIR, "config", f"{config_uuid}.json"), "r") as f:
        config_json = json.load(f)

    return StableDiffusionConfig.parse_obj(config_json)
