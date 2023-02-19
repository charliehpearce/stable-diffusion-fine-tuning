from typing import Optional
from pydantic import BaseModel


class StableDiffusionConfig(BaseModel):
    mixed_precision: str = "no"
    seed: Optional[int]
    pretrained_model_name_or_path: str
    revision: str = "main"
    image_column: str = "image"
    caption_column: str = "caption"
    max_train_samples: Optional[int]
    cache_dir: str = "/tmp/model_cache/"
    center_crop: bool = False
    random_flip: bool = False
    train_batch_size: int = 16
    num_train_epochs: int = 1
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    learning_rate: float = 0.0001
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    use_8bit_adam: bool = False
    allow_tf32: bool = False
    use_ema: bool = False
    non_ema_revision: str = "main"
    dataloader_num_workers: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: int = 1
    max_train_steps: Optional[int]
    resume_from_checkpoint: int = 500
    enable_xformers_memory_efficient_attention: bool = False
    checkpoints_total_limit: Optional[int]
    report_to: str = "tensorboard"
    resolution: int = 512
