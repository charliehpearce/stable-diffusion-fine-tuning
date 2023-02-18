from enum import Enum
from sconf import Config


class StableDiffusionConfig(Enum):
    TRAIN_1 = "fine_tune.yaml"


def load_config(config: StableDiffusionConfig) -> Config:
    config = Config(config.value)
    return config
