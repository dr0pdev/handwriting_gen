"""
Inference / generation script for the handwriting generation model.

Usage:
    python generate.py \\
        --cfg configs/iam.yml \\
        --pretrained_model checkpoints/checkpoint_epoch_0099.pt \\
        --csv_path samples.csv \\
        --save_dir Generated/ \\
        [--style_image /path/to/style.png | --writer_id some_writer_folder] \\
        [--sampling_timesteps 50] [--eta 0.0]

CSV format (columns: Images, Text):
    Images,Text
    output_filename.png,Hello world

Style selection (in order of priority):
  1. --style_image: a specific grayscale image file
  2. --writer_id:  a folder name under cfg.TEST.STYLE_PATH
  3. (default):   a new random writer/style for each CSV row
"""

import argparse
import csv
import logging
import os
import random

import cv2
import torch
import torchvision
from diffusers import AutoencoderKL
from PIL import Image
from tqdm import tqdm

from data_loader.dataset import IAMGenerateDataset, LETTERS, FIXED_LEN
from models.diffusion import Diffusion
from models.unet import UNetModel
from utils.config import cfg, cfg_from_file, assert_and_infer_cfg
from utils.helpers import fix_seed, get_device

logger = logging.getLogger(__name__)

WRITER_NUMS = 496
CRITICAL_WIDTH = 512


# ---------------------------------------------------------------------------
# Style image loading helpers
# ---------------------------------------------------------------------------

def load_style_image(path: str) -> torch.Tensor:
    """Load a single grayscale image as a [1, H, W] float tensor."""
    img = cv2.imread(path, flags=0)
    if img is None:
        raise FileNotFoundError(f"Could not read style image: {path}")
    return torch.from_numpy(img.astype("float32") / 255.0).unsqueeze(0)


def load_style_from_writer(style_root: str, writer_id: str) -> torch.Tensor:
    """
    Load the best-quality style crop from a writer's folder.
    Prefers images wider than CRITICAL_WIDTH.
    """
    writer_dir = os.path.join(style_root, writer_id)
    if not os.path.isdir(writer_dir):
        raise FileNotFoundError(f"Writer directory not found: {writer_dir}")

    files = [f for f in os.listdir(writer_dir) if not f.startswith(".")]
    if not files:
        raise RuntimeError(f"No files in writer directory: {writer_dir}")

    random.shuffle(files)
    best_w, best_img = -1, None
    for fname in files:
        img = cv2.imread(os.path.join(writer_dir, fname), flags=0)
        if img is None:
            continue
        if img.shape[1] > CRITICAL_WIDTH:
            return torch.from_numpy(img.astype("float32") / 255.0).unsqueeze(0)
        if img.shape[1] > best_w:
            best_w, best_img = img.shape[1], img

    if best_img is None:
        raise RuntimeError(f"No readable images in {writer_dir}")

    logger.warning("Writer %r: no image wider than %d px; using widest (%d px)",
                   writer_id, CRITICAL_WIDTH, best_w)
    return torch.from_numpy(best_img.astype("float32") / 255.0).unsqueeze(0)


def load_random_style(style_root: str) -> torch.Tensor:
    """Pick a random writer and load a style image."""
    writers = [
        d for d in os.listdir(style_root)
        if os.path.isdir(os.path.join(style_root, d)) and not d.startswith(".")
    ]
    if not writers:
        raise RuntimeError(f"No writer subdirectories under {style_root!r}")
    wid = random.choice(writers)
    logger.debug("Random style: writer %r", wid)
    return load_style_from_writer(style_root, wid)


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def read_csv_rows(csv_path: str, allowed: set):
    """Yield (filename, cleaned_text) from a CSV with Images,Text columns."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = (row.get("Images") or row.get("images") or "").strip()
            raw_text = (row.get("Text") or row.get("text") or "").strip()
            cleaned = "".join(c for c in raw_text if c in allowed)
            if not cleaned:
                logger.warning("Skipping row with no valid chars: %r", row)
                continue
            if not filename:
                logger.warning("Skipping row with empty filename: %r", row)
                continue
            yield filename, cleaned


def pad_text(text: str, target_min: int = 35, target_max: int = 61) -> str:
    """Repeat a short text phrase to reach a minimum target length."""
    text = text.strip()
    if len(text) >= target_min:
        return text
    unit = " " + text
    out = text
    while len(out) + len(unit) <= target_max:
        out += unit
    if len(out) < target_min:
        out += unit
    return out


def crop_to_original_width(image: Image.Image, original_text: str,
                            padded_text: str, margin: float = 0.1,
                            min_extra: int = 16) -> Image.Image:
    """Crop the right side of a padded image to approximately the original text width."""
    w, h = image.size
    if not padded_text or len(original_text) >= len(padded_text):
        return image
    frac = len(original_text) / len(padded_text)
    crop_w = int(round(w * frac * (1 + margin))) + min_extra
    crop_w = min(w, max(1, crop_w))
    return image.crop((0, 0, crop_w, h))


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate(args):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg_from_file(args.cfg)
    assert_and_infer_cfg()
    fix_seed(cfg.TRAIN.SEED)

    device = get_device(args.device)
    logger.info("Device: %s", device)

    # Build UNet
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

    # Load checkpoint
    ckpt = torch.load(args.pretrained_model, map_location=device)
    # Support both raw state_dict and full checkpoint dicts
    state = ckpt.get("unet", ckpt) if isinstance(ckpt, dict) else ckpt
    unet.load_state_dict(state)
    logger.info("Loaded model from %s", args.pretrained_model)
    unet.eval()

    # Frozen VAE
    vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder="vae")
    vae = vae.to(device)
    vae.requires_grad_(False)
    vae.eval()

    diffusion = Diffusion(noise_steps=1000, device=device)

    # Dataset for glyph encoding (and optionally style)
    if os.path.isdir(cfg.TEST.STYLE_PATH):
        dataset = IAMGenerateDataset(cfg.TEST.STYLE_PATH, "test", ref_num=1)
    else:
        if not args.style_image:
            raise FileNotFoundError(
                f"STYLE_PATH {cfg.TEST.STYLE_PATH!r} is not a directory. "
                "Use --style_image to provide a direct style file."
            )
        # Minimal wrapper for content-only usage
        class _ContentOnly:
            letters = LETTERS
            fixed_len = FIXED_LEN
            letter2index = {c: i for i, c in enumerate(LETTERS)}

            def __init__(self):
                import pickle
                with open("files/unifont.pickle", "rb") as f:
                    syms = pickle.load(f)
                import numpy as np
                sym_map = {s["idx"][0]: s["mat"].astype(np.float32) for s in syms}
                import torch as _t
                contents = []
                for ch in LETTERS:
                    contents.append(_t.from_numpy(sym_map[ord(ch)]).float())
                contents.append(_t.zeros_like(contents[0]))
                self.con_symbols = _t.stack(contents)

            def get_content(self, label: str) -> torch.Tensor:
                idx = [self.letter2index[c] for c in label]
                ref = self.con_symbols[idx]
                return (1.0 - ref).unsqueeze(0)

        dataset = _ContentOnly()

    # Fixed style (if specified)
    use_fixed_style = bool(args.style_image or args.writer_id)
    if use_fixed_style:
        if args.style_image:
            style_ref = load_style_image(args.style_image).unsqueeze(0).to(device)
        else:
            style_ref = load_style_from_writer(cfg.TEST.STYLE_PATH, args.writer_id)
            style_ref = style_ref.unsqueeze(0).to(device)
    else:
        style_ref = None

    # Read CSV rows
    allowed = set(dataset.letters)
    rows = list(read_csv_rows(args.csv_path, allowed))
    rows = [(fn, t, pad_text(t)) for fn, t in rows]

    os.makedirs(args.save_dir, exist_ok=True)

    with torch.no_grad():
        for filename, original_label, padded_text in tqdm(rows, desc="generating"):
            if not use_fixed_style:
                s = load_random_style(cfg.TEST.STYLE_PATH)
                style_ref = s.unsqueeze(0).to(device)

            text_ref = dataset.get_content(padded_text).to(device)

            # Initial noise in latent space: [1, 4, H/8, W/8]
            h_lat = style_ref.shape[2] // 8
            w_lat = FIXED_LEN // 8
            x = torch.randn(1, 4, h_lat, w_lat, device=device)

            sampled = diffusion.ddim_sample(
                unet, vae, 1, x, style_ref, text_ref,
                sampling_timesteps=args.sampling_timesteps,
                eta=args.eta,
            )

            pil_img = torchvision.transforms.ToPILImage()(sampled[0]).convert("L")

            if not args.no_crop:
                pil_img = crop_to_original_width(pil_img, original_label, padded_text)

            pil_img.save(os.path.join(args.save_dir, filename))

    logger.info("Saved %d images to %s", len(rows), args.save_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate handwriting images")
    parser.add_argument("--cfg", dest="cfg", default="configs/iam.yml")
    parser.add_argument("--pretrained_model", type=str, required=True,
                        help="Path to UNet checkpoint (.pt)")
    parser.add_argument("--vae_path", type=str,
                        default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--csv_path", type=str, default="samples.csv",
                        help="CSV with Images,Text columns")
    parser.add_argument("--save_dir", type=str, default="Generated/")
    parser.add_argument("--style_image", type=str, default="",
                        help="Grayscale style reference image (highest priority)")
    parser.add_argument("--writer_id", type=str, default="",
                        help="Writer folder name under STYLE_PATH")
    parser.add_argument("--sampling_timesteps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0,
                        help="DDIM stochasticity (0=deterministic)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no_crop", action="store_true",
                        help="Do not crop to original text width")
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
