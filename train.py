"""
Training script for the handwriting generation model.

Usage:
    python train.py --cfg configs/iam.yml \
        --image_path /path/to/IAM/images \
        --style_path /path/to/IAM/style_crops \
        --label_path /path/to/iam_labels.txt \
        --vae_path runwayml/stable-diffusion-v1-5

The script trains the UNet + Mix_TR conditioning encoder. The VAE is frozen
throughout. The training loss is:

    L = MSE(noise_pred, eps) + lambda_v * vertical_proxy_loss
                             + lambda_h * horizontal_proxy_loss

An EMA model is maintained alongside the main model and saved separately.
"""

import argparse
import copy
import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader.dataset import IAMDataset
from models.diffusion import Diffusion, EMA
from models.unet import UNetModel
from utils.config import cfg, cfg_from_file, assert_and_infer_cfg
from utils.helpers import fix_seed, get_device, save_checkpoint

logger = logging.getLogger(__name__)

WRITER_NUMS = 496
PROXY_LOSS_WEIGHT = 0.1


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def build_lr_scheduler(optimizer, warmup_iters: int, total_iters: int):
    """
    Linear warmup followed by cosine decay to zero.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_iters:
            return float(step) / max(1, warmup_iters)
        progress = (step - warmup_iters) / max(1, total_iters - warmup_iters)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Encode a batch of images to latents using the frozen VAE
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_to_latent(vae: nn.Module, images: torch.Tensor,
                     device: torch.device) -> torch.Tensor:
    """
    Encode RGB images [-1, 1] to 4-channel latents.

    Args:
        images: [B, 3, H, W] float32 in [-1, 1]
        device: target device

    Returns:
        latents: [B, 4, H/8, W/8]
    """
    images = images.to(device)
    latent_dist = vae.encode(images).latent_dist
    latents = latent_dist.sample() * 0.18215
    return latents


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    # Config
    cfg_from_file(args.cfg)
    assert_and_infer_cfg(make_immutable=False)

    fix_seed(cfg.TRAIN.SEED)
    device = get_device(args.device)
    logger.info("Using device: %s", device)

    # Dataset and dataloader
    dataset = IAMDataset(
        image_path=args.image_path or cfg.TRAIN.IMAGE_PATH,
        style_path=args.style_path or cfg.TRAIN.STYLE_PATH,
        text_path=args.label_path or cfg.TRAIN.LABEL_PATH,
        split="train",
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.TRAIN.IMS_PER_BATCH,
        shuffle=True,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        collate_fn=dataset.collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    logger.info("Dataset size: %d samples", len(dataset))

    # VAE (frozen)
    vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder="vae")
    vae = vae.to(device)
    vae.requires_grad_(False)
    vae.eval()
    logger.info("VAE loaded and frozen")

    # UNet
    unet = UNetModel(
        in_channels=cfg.MODEL.IN_CHANNELS,
        model_channels=cfg.MODEL.EMB_DIM,
        out_channels=cfg.MODEL.OUT_CHANNELS,
        num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS,
        attention_resolutions=(1, 1),
        channel_mult=(1, 1),
        num_heads=cfg.MODEL.NUM_HEADS,
        context_dim=cfg.MODEL.EMB_DIM,
        nb_classes=WRITER_NUMS,
    ).to(device)

    # EMA model (copy of unet, not updated by optimizer)
    ema = EMA(beta=0.9999)
    ema_unet = copy.deepcopy(unet)
    ema_unet.requires_grad_(False)

    # Resume from checkpoint if provided
    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        unet.load_state_dict(ckpt["unet"])
        ema_unet.load_state_dict(ckpt["ema_unet"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        logger.info("Resumed from %s (epoch %d)", args.resume, start_epoch)

    logger.info("UNet parameters: {:,}".format(
        sum(p.numel() for p in unet.parameters() if p.requires_grad)
    ))

    # Diffusion process
    diffusion = Diffusion(noise_steps=1000, noise_offset=0.0, device=device)

    # Optimizer
    optimizer = AdamW(
        unet.parameters(),
        lr=cfg.SOLVER.BASE_LR,
        weight_decay=1e-4,
    )

    iters_per_epoch = len(loader)
    total_iters = cfg.SOLVER.EPOCHS * iters_per_epoch
    scheduler = build_lr_scheduler(optimizer, cfg.SOLVER.WARMUP_ITERS, total_iters)

    # Optional W&B logging
    use_wandb = args.wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project="handwriting-gen", config=vars(args))
        except ImportError:
            logger.warning("wandb not installed; disabling W&B logging")
            use_wandb = False

    os.makedirs(args.save_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Epoch loop
    # ----------------------------------------------------------------
    for epoch in range(start_epoch, cfg.SOLVER.EPOCHS):
        unet.train()
        running_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.SOLVER.EPOCHS}", leave=False)

        for batch in pbar:
            imgs = batch["img"].to(device)          # [B, 3, H, W]
            style = batch["style"].to(device)        # [B, 1, H_s, W_s]
            content = batch["content"].to(device)    # [B, T, 16, 16]
            wid = batch["wid"].to(device)            # [B]

            # Encode images to latent space
            latents = encode_to_latent(vae, imgs, device)   # [B, 4, H/8, W/8]

            # Sample random timesteps and add noise
            t = diffusion.sample_timesteps(latents.shape[0]).to(device)
            x_noisy, eps = diffusion.noise_images(latents, t)

            # Forward pass through UNet
            noise_pred, v_loss, h_loss = unet(
                x_noisy, t, style, content, wid, tag="train"
            )

            # Total loss: diffusion MSE + proxy losses
            diff_loss = F.mse_loss(noise_pred, eps)
            loss = diff_loss + PROXY_LOSS_WEIGHT * (v_loss + h_loss)

            # Backward + optimizer step
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(unet.parameters(), cfg.SOLVER.GRAD_L2_CLIP)
            optimizer.step()
            scheduler.step()

            # EMA update
            ema.step_ema(ema_unet, unet)

            running_loss += loss.item()
            global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "diff": f"{diff_loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            if use_wandb:
                import wandb
                wandb.log({
                    "loss/total": loss.item(),
                    "loss/diffusion": diff_loss.item(),
                    "loss/vertical_proxy": v_loss.item(),
                    "loss/horizontal_proxy": h_loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "step": global_step,
                })

        avg_loss = running_loss / max(1, len(loader))
        logger.info("Epoch %d  avg_loss=%.4f  lr=%.2e",
                    epoch, avg_loss, scheduler.get_last_lr()[0])

        # Checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch == cfg.SOLVER.EPOCHS - 1:
            ckpt_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch:04d}.pt")
            save_checkpoint({
                "unet": unet.state_dict(),
                "ema_unet": ema_unet.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
            }, ckpt_path)
            logger.info("Saved checkpoint: %s", ckpt_path)

    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train handwriting generation model")
    parser.add_argument("--cfg", dest="cfg", default="configs/iam.yml",
                        help="Path to YAML config file")
    parser.add_argument("--image_path", type=str, default="",
                        help="Path to IAM image directory (overrides config)")
    parser.add_argument("--style_path", type=str, default="",
                        help="Path to style reference crops (overrides config)")
    parser.add_argument("--label_path", type=str, default="",
                        help="Path to IAM label text file (overrides config)")
    parser.add_argument("--vae_path", type=str,
                        default="runwayml/stable-diffusion-v1-5",
                        help="HuggingFace model ID or local path for VAE")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cuda', 'mps', or 'cpu'")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save a checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to a checkpoint to resume from")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
