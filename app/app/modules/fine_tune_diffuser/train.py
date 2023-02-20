import logging
import math
import os
import random

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

from app.constants import DATA_MOUNT_DIR, HUGGINGFACE_TOKEN

from .config_model import StableDiffusionConfig

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.14.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def main(config: StableDiffusionConfig, dataset_uuid: str):
    output_dir = os.path.join(DATA_MOUNT_DIR, "models", dataset_uuid)

    logging_dir = os.path.join(output_dir, "logging")

    accelerator_project_config = ProjectConfiguration(
        total_limit=config.checkpoints_total_limit
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if config.seed is not None:
        set_seed(config.seed)

    # Handle the repository creation
    os.makedirs(output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="scheduler",
        use_auth_token=HUGGINGFACE_TOKEN,
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=config.revision,
        use_auth_token=HUGGINGFACE_TOKEN,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=config.revision,
        use_auth_token=HUGGINGFACE_TOKEN,
    )
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="vae",
        revision=config.revision,
        use_auth_token=HUGGINGFACE_TOKEN,
    )
    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="unet",
        revision=config.non_ema_revision,
        use_auth_token=HUGGINGFACE_TOKEN,
    )

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Create EMA for the unet.
    if config.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="unet",
            revision=config.revision,
            use_auth_token=HUGGINGFACE_TOKEN,
        )
        ema_unet = EMAModel(
            ema_unet.parameters(),
            model_cls=UNet2DConditionModel,
            model_config=ema_unet.config,
        )

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if config.use_ema:
            ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

        for i, model in enumerate(models):
            model.save_pretrained(os.path.join(output_dir, "unet"))

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

    def load_model_hook(models, input_dir):
        if config.use_ema:
            load_model = EMAModel.from_pretrained(
                os.path.join(input_dir, "unet_ema"), UNet2DConditionModel
            )
            ema_unet.load_state_dict(load_model.state_dict())
            ema_unet.to(accelerator.device)
            del load_model

        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            load_model = UNet2DConditionModel.from_pretrained(
                input_dir, subfolder="unet"
            )
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.scale_lr:
        config.learning_rate = (
            config.learning_rate
            * config.gradient_accumulation_steps
            * config.train_batch_size
            * accelerator.num_processes
        )

    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.

    data_files = {}
    logger.info("loading dataset")
    data_files["train"] = os.path.join(
        DATA_MOUNT_DIR, "datasets", f"{dataset_uuid}.tar.gz"
    )
    dataset = load_dataset(
        "imagefolder",
        data_files=data_files,
        cache_dir=config.cache_dir,
    )

    # See more about loading custom images at
    # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    image_column = config.image_column
    if image_column not in column_names:
        raise ValueError(
            f"--image_column' value '{config.image_column}' needs to be one of: {', '.join(column_names)}"
        )
    caption_column = config.caption_column
    if caption_column not in column_names:
        raise ValueError(
            f"--caption_column' value '{config.caption_column}' needs to be one of: {', '.join(column_names)}"
        )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                config.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(config.resolution)
            if config.center_crop
            else transforms.RandomCrop(config.resolution),
            transforms.RandomHorizontalFlip()
            if config.random_flip
            else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    with accelerator.main_process_first():
        if config.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"]
                .shuffle(seed=config.seed)
                .select(range(config.max_train_samples))
            )
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=config.train_batch_size,
        num_workers=config.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps
    )
    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * config.gradient_accumulation_steps,
        num_training_steps=config.max_train_steps * config.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if config.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    config.num_train_epochs = math.ceil(
        config.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.

    # Train!
    total_batch_size = (
        config.train_batch_size
        * accelerator.num_processes
        * config.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, config.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, config.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(
                    batch["pixel_values"].to(weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                # Predict the noise residual and compute loss
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(config.train_batch_size)
                ).mean()
                train_loss += avg_loss.item() / config.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if config.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                train_loss = 0.0

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= config.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if config.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionPipeline.from_pretrained(
            config.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            revision=config.revision,
            use_auth_token=HUGGINGFACE_TOKEN,
        )
        pipeline.save_pretrained(output_dir)

    return True
