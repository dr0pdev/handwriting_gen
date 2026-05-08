# handwriting_gen

A clean, faithful reimplementation of [DiffBrush](https://arxiv.org/abs/2508.03256) (ICCV 2025) — a latent diffusion model for **handwritten text-line generation** with disentangled style-content conditioning.

Given a **style reference** (a sample of someone's handwriting) and a **text string**, the model generates a realistic handwritten text-line image in the target writer's style.

---

## Architecture

```
Style Image (grayscale)    Text String
        │                      │
        ▼                      ▼
  ResNet-18 (1-ch)      Unifont Glyph Images
  + Dilated ResNet            │
  + 2D Positional Enc   ResNet-18 trunk
        │               + 1D Positional Enc
        ▼                      │
  Transformer Encoder          │
  ├── Vertical Head            │
  └── Horizontal Head          │
        │                      │
        └──── Cross-Attention Decoders ────┘
                     │
               Context [B, T, 512]
                     │
                     ▼
            UNet Denoiser
      (ResBlocks + SpatialTransformer)
                     │
                     ▼
          VAE Decoder (SD v1.5, frozen)
                     │
                     ▼
           Output Handwriting Image
```

The model runs in the **latent space** of the Stable Diffusion v1.5 VAE (4-channel, 8x spatial downsampling). Inference uses **DDIM** (default 50 steps).

Style is explicitly split into **vertical** (per-letter height, baseline) and **horizontal** (spacing, ligatures, word gaps) components, each supervised by a **Proxy Anchor loss** during training.

---

## Project Structure

```
handwriting_gen/
  configs/
    iam.yml              # IAM dataset config (LR, epochs, model dims)
  data_loader/
    dataset.py           # IAMDataset (training) + IAMGenerateDataset (inference)
  models/
    transformer.py       # Encoder/decoder stacks, 1D/2D positional encodings
    resnet_dilation.py   # Dilated ResNet-18 tail for style features
    loss.py              # Proxy Anchor loss
    encoder.py           # Mix_TR: style + content encoders + fusion decoders
    unet.py              # UNet denoiser with SpatialTransformer cross-attention
    diffusion.py         # Noise schedule, DDIM/DDPM sampling, EMA
  utils/
    config.py            # YAML config with AttrDict and merge helpers
    helpers.py           # Seed fixing, device auto-detection, checkpoint I/O
  train.py               # Training loop
  generate.py            # Inference / batch generation
  files/                 # Unifont glyph pickle (not included, see below)
```

---

## Requirements

- Python 3.10+
- PyTorch 2.2+ (CUDA, MPS on Apple Silicon, or CPU)
- [uv](https://docs.astral.sh/uv/) for dependency management

### Install

```bash
uv sync
```

### External assets

1. **IAM Handwriting Database** — request access at
   https://fki.tic.heia-fr.ch/databases/iam-handwriting-database

2. **Stable Diffusion v1.5 VAE** — downloaded automatically on first run from
   `runwayml/stable-diffusion-v1-5` via HuggingFace Diffusers, or supply a local path.

3. **Unifont glyph pickle** — place `unifont.pickle` in the `files/` directory.
   This file encodes each character as a binary glyph matrix and can be generated
   from [GNU Unifont](https://unifoundry.com/unifont/) or obtained from the
   original DiffBrush repository.

---

## Training

```bash
python train.py \
    --cfg configs/iam.yml \
    --image_path /path/to/IAM/images \
    --style_path /path/to/IAM/style_crops \
    --label_path /path/to/iam_labels.txt \
    --vae_path runwayml/stable-diffusion-v1-5 \
    --save_dir checkpoints/ \
    --save_every 10
```

**Label file format** (one entry per line):

```
writer_id,image_id transcription text here
```

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |
| `--save_every` | `10` | Checkpoint every N epochs |
| `--resume` | — | Path to a checkpoint to resume from |
| `--wandb` | off | Enable W&B experiment logging |

**Training loss:**

```
L = MSE(noise_pred, eps) + 0.1 * (vertical_proxy_loss + horizontal_proxy_loss)
```

---

## Generation

Prepare a CSV with `Images` and `Text` columns:

```csv
Images,Text
output_001.png,Hello world
output_002.png,The quick brown fox
```

Then run:

```bash
python generate.py \
    --cfg configs/iam.yml \
    --pretrained_model checkpoints/checkpoint_epoch_0099.pt \
    --csv_path samples.csv \
    --save_dir Generated/ \
    --style_image /path/to/handwriting_sample.png \
    --sampling_timesteps 50
```

**Style input options (in priority order):**

1. `--style_image <path>` — use a specific grayscale image
2. `--writer_id <folder>` — use a specific writer's crops from `STYLE_PATH`
3. *(default)* — pick a new random writer for each row

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--sampling_timesteps` | `50` | Number of DDIM denoising steps |
| `--eta` | `0.0` | DDIM stochasticity (0=deterministic) |
| `--no_crop` | off | Keep full padded line width |

## Citation

```bibtex
@inproceedings{diffbrush2025,
  title     = {Beyond Isolated Words: Diffusion Brush for Handwritten Text-Line Generation},
  booktitle = {ICCV},
  year      = {2025},
}
```
