# handwriting_gen

A clean reimplementation of [DiffBrush](https://arxiv.org/abs/2508.03256) (ICCV 2025) — a latent diffusion model for handwritten text-line generation with disentangled style-content conditioning.

## Overview

Given a **style reference** (a sample of someone's handwriting) and a **text string** (what to write), the model generates a realistic handwritten text-line image in the target style.

The architecture operates in the latent space of a frozen Stable Diffusion v1.5 VAE, using a UNet denoiser conditioned on style-content embeddings produced by the `Mix_TR` encoder.

## Architecture

```
Style Image + Text Content
        │
        ▼
   Mix_TR Encoder
   ├── Style: ResNet-18 → Dilated ResNet → 2D PE → Transformer (vertical + horizontal heads)
   └── Content: Glyph images → ResNet-18 → 1D PE → Cross-attention decoders
        │
        ▼
  Context Embeddings [B, T, 512]
        │
        ▼
  UNet Denoiser (ResBlocks + SpatialTransformer cross-attention)
        │
        ▼
  VAE Decoder (Stable Diffusion v1.5, frozen)
        │
        ▼
  Output Handwriting Image
```

## Installation

```bash
uv sync
```

You will also need:
- IAM Handwriting Database (request access at https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
- Stable Diffusion v1.5 VAE weights (download via `diffusers`)
- Unifont glyph pickle file (`files/unifont.pickle`)

## Training

```bash
python train.py --cfg configs/iam.yml \
    --image_path /path/to/IAM/images \
    --style_path /path/to/IAM/style_crops \
    --label_path /path/to/iam_labels.txt \
    --vae_path runwayml/stable-diffusion-v1-5
```

## Generation

```bash
python generate.py \
    --cfg configs/iam.yml \
    --pretrained_model checkpoints/model_epoch_100.pt \
    --csv_path samples.csv \
    --style_image /path/to/style.png \
    --save_dir Generated/
```

## Key Differences from Original DiffBrush

- Working `train.py` (the original repo does not include a training script)
- Bug fixes: `binarize()` in loss.py, missing return in `get_style_ref`
- MPS support for Apple Silicon (no forced CUDA/gloo distributed)
- Works on a single device without mandatory multi-GPU setup
- Cleaner config system and merged dataset module
