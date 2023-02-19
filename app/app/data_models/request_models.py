from pydantic import BaseModel


class FineTuneStableDiffusionRequest(BaseModel):
    dataset_uuid: str
    config_uuid: str
