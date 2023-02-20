import asyncio
from .train import main
from .utils import load_config


class DiffusionPipeline:
    def __init__(self, dataset_uuid: str, config_uuid: str) -> None:
        self.dataset_uuid = dataset_uuid
        self.config_uuid = config_uuid

    def run(self):
        asyncio.run(self._pipeline())

    async def _pipeline(self) -> bool:
        # Load data from storage into tmp directory
        config = load_config(self.config_uuid)

        # Run training pipeline
        main(config=config, dataset_uuid=self.dataset_uuid)

        return True
