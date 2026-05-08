"""
IAM Handwriting Dataset.

Provides:
  IAMDataset       -- for training, returns (image, style, content, wid, ...)
  IAMGenerateDataset -- for inference, provides style refs and get_content()

Data layout expected on disk:
  IMAGE_PATH / writer_id / image_name.png   (RGB text-line images)
  STYLE_PATH / writer_id / *.png            (grayscale style crops)
  LABEL_PATH                                 (text file, one sample per line:
                                              "writer_id,image_id transcription")

Character set covers IAM printable characters.
fixed_len = 1024 (max padded line width in pixels).
"""

import os
import random
import math
import pickle
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

LETTERS = " _!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
FIXED_LEN = 1024
CRITICAL_WIDTH = 512

_NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


# ---------------------------------------------------------------------------
# Glyph utilities
# ---------------------------------------------------------------------------

def load_glyph_symbols(content_type: str = "unifont") -> torch.Tensor:
    """
    Load per-character Unifont glyph images from a pickle file.

    Returns a tensor of shape [len(LETTERS)+1, H, W] where the last
    entry is a blank PAD token.
    """
    with open(f"files/{content_type}.pickle", "rb") as f:
        symbols = pickle.load(f)

    sym_map = {sym["idx"][0]: sym["mat"].astype(np.float32) for sym in symbols}
    contents = []
    for char in LETTERS:
        mat = torch.from_numpy(sym_map[ord(char)]).float()
        contents.append(mat)
    contents.append(torch.zeros_like(contents[0]))  # PAD token
    return torch.stack(contents)


def get_content_from_label(label: str, letter2index: Dict[str, int],
                            con_symbols: torch.Tensor) -> torch.Tensor:
    """
    Convert a text string to a stacked tensor of glyph images.

    Returns:
        [1, T, H, W]  (batch dim prepended)
    """
    indices = [letter2index[c] for c in label]
    content_ref = con_symbols[indices]
    content_ref = 1.0 - content_ref   # invert: black text on white -> white text on black
    return content_ref.unsqueeze(0)


# ---------------------------------------------------------------------------
# IAMDataset (training)
# ---------------------------------------------------------------------------

class IAMDataset(Dataset):
    """
    IAM Handwriting Dataset for training.

    Loads handwriting line images, style reference crops, and per-character
    glyph content tensors.
    """

    def __init__(self, image_path: str, style_path: str, text_path: str,
                 split: str = "train", content_type: str = "unifont"):
        self.image_path = image_path
        self.style_path = style_path
        self.split = split
        self.fixed_len = FIXED_LEN
        self.letters = LETTERS

        self.letter2index = {c: i for i, c in enumerate(LETTERS)}
        self.con_symbols = load_glyph_symbols(content_type)

        # Token encoding for sequence targets
        _special = ["[GO]", "[END]", "[PAD]"]
        self.character = _special + list(LETTERS)
        self.char2idx = {c: i for i, c in enumerate(self.character)}

        self.data = self._load_labels(text_path)

    def _load_labels(self, path: str) -> List[Dict]:
        records = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                head, transcription = line.split(" ", 1)
                writer_id, img_id = head.split(",")
                records.append({
                    "image": img_id + ".png",
                    "wid": writer_id,
                    "label": transcription,
                })
        return records

    def __len__(self) -> int:
        return len(self.data)

    def _load_style_ref(self, writer_id: str) -> np.ndarray:
        """Load a random style reference image for the given writer."""
        style_dir = os.path.join(self.style_path, writer_id)
        files = [f for f in os.listdir(style_dir) if not f.startswith(".")]
        if not files:
            raise RuntimeError(f"No style images found for writer {writer_id}")
        random.shuffle(files)
        for fname in files:
            img = cv2.imread(os.path.join(style_dir, fname), flags=0)
            if img is not None:
                return img.astype(np.float32) / 255.0
        raise RuntimeError(f"Could not read any style image for writer {writer_id}")

    def _concat_short_img(self, img: np.ndarray,
                           transcr: str) -> Tuple[np.ndarray, str]:
        """Repeat a short line image horizontally to reach near FIXED_LEN width."""
        h, w = img.shape[:2]
        space_w = max(1, w // len(transcr))
        repeat_width = space_w + w
        lower = math.ceil(max(0, CRITICAL_WIDTH - w) / repeat_width)
        upper = (FIXED_LEN - w) // repeat_width

        space = np.full((h, space_w, 3), 255, dtype=np.uint8)
        unit = cv2.hconcat([space, img])
        unit_text = " " + transcr

        if upper <= 0:
            return img, transcr

        times = random.randint(max(1, lower), max(1, upper))
        cat = img.copy()
        out_text = transcr
        for _ in range(times):
            cat = cv2.hconcat([cat, unit])
            out_text += unit_text
            if cat.shape[1] >= FIXED_LEN:
                break

        return cat, out_text

    def __getitem__(self, idx: int) -> Dict:
        record = self.data[idx]
        wid = record["wid"]
        transcr = record["label"]

        # Load image
        img_path = os.path.join(self.image_path, wid, record["image"])
        image = Image.open(img_path).convert("RGB")

        if image.size[0] < CRITICAL_WIDTH and self.split == "train":
            img_arr = np.array(image)
            img_arr, transcr = self._concat_short_img(img_arr, transcr)
            image = Image.fromarray(img_arr)

        image = _NORMALIZE(image)  # [3, H, W], values in [-1, 1]

        # Style reference
        style_arr = self._load_style_ref(wid)   # [H, W] float32
        style_ref = torch.from_numpy(style_arr).unsqueeze(0)  # [1, H, W]

        # Content: per-character glyph sequence, concatenated then resized
        char_glyphs = []
        for ch in transcr:
            ci = self.letter2index[ch]
            glyph = self.con_symbols[ci].numpy()
            # Crop to non-empty column extent
            mask = glyph == 1.0
            cols = np.where(mask.sum(axis=0))[0]
            if len(cols) > 0:
                glyph = glyph[:, max(0, cols[0] - 2): cols[-1] + 3]
            else:
                glyph = glyph[:, 2:14]
            char_glyphs.append(torch.from_numpy(glyph))

        concat_glyph = torch.cat(char_glyphs, dim=-1).numpy()
        glyph_resized = cv2.resize(concat_glyph, (image.shape[-1], 64))
        glyph_resized = 1.0 - glyph_resized
        glyph_rgb = np.stack([glyph_resized] * 3, axis=2).astype(np.float32)
        glyph_line = _NORMALIZE(glyph_rgb)  # [3, 64, W]

        return {
            "img": image,
            "style": style_ref,
            "content": transcr,
            "wid": int(wid),
            "transcr": transcr,
            "image_name": record["image"],
            "glyph_line": glyph_line,
        }

    def collate_fn(self, batch: List[Dict]) -> Dict:
        """Pad images and styles to uniform width within the batch."""
        s_widths = [item["style"].shape[2] for item in batch]
        max_s_w = min(max(s_widths), FIXED_LEN)

        imgs = torch.full(
            [len(batch), batch[0]["img"].shape[0], batch[0]["img"].shape[1], FIXED_LEN],
            fill_value=1.0, dtype=torch.float32,
        )
        style_ref = torch.ones(
            [len(batch), 1, batch[0]["style"].shape[1], max_s_w], dtype=torch.float32
        )
        glyph_lines = torch.ones(
            [len(batch), 3, 64, FIXED_LEN], dtype=torch.float32
        )
        wids = torch.tensor([item["wid"] for item in batch])
        transcrs = [item["transcr"] for item in batch]

        # Per-character glyph content tensors  [B, T, 16, 16]
        max_t = max(len(item["content"]) for item in batch)
        content_ref = torch.zeros([len(batch), max_t, 16, 16], dtype=torch.float32)

        for i, item in enumerate(batch):
            w_img = item["img"].shape[2]
            imgs[i, :, :, :w_img] = item["img"]

            w_sty = min(item["style"].shape[2], FIXED_LEN)
            style_ref[i, :, :, :w_sty] = item["style"][:, :, :w_sty]

            w_gl = item["glyph_line"].shape[2]
            glyph_lines[i, :, :, :w_gl] = item["glyph_line"]

            for ti, ch in enumerate(item["content"]):
                ci = self.letter2index[ch]
                content_ref[i, ti] = self.con_symbols[ci]

        content_ref = 1.0 - content_ref  # invert

        return {
            "img": imgs,
            "style": style_ref,
            "content": content_ref,
            "wid": wids,
            "transcr": transcrs,
            "glyph_line": glyph_lines,
        }

    def get_content(self, label: str) -> torch.Tensor:
        """Return [1, T, 16, 16] content tensor for a text string."""
        return get_content_from_label(label, self.letter2index, self.con_symbols)


# ---------------------------------------------------------------------------
# IAMGenerateDataset (inference)
# ---------------------------------------------------------------------------

class IAMGenerateDataset(Dataset):
    """
    Minimal dataset for inference: provides style images and content encoding.
    Does not require a label file.
    """

    def __init__(self, style_path: str, split: str = "test",
                 ref_num: int = 1, content_type: str = "unifont"):
        self.style_path = style_path
        self.fixed_len = FIXED_LEN
        self.letters = LETTERS
        self.letter2index = {c: i for i, c in enumerate(LETTERS)}
        self.con_symbols = load_glyph_symbols(content_type)
        self.author_ids = [
            d for d in os.listdir(style_path)
            if os.path.isdir(os.path.join(style_path, d)) and not d.startswith(".")
        ]
        self.ref_num = ref_num

    def __len__(self) -> int:
        return self.ref_num

    def __getitem__(self, _) -> Dict:
        batch = []
        for wid in self.author_ids:
            style_img = self._load_style_ref(wid)
            style_t = torch.from_numpy(style_img).unsqueeze(0).float()  # [1, H, W]
            batch.append({"style": style_t, "wid": wid})

        max_w = min(max(item["style"].shape[2] for item in batch), FIXED_LEN)
        style_ref = torch.ones(
            [len(batch), 1, batch[0]["style"].shape[1], max_w], dtype=torch.float32
        )
        wids = []
        for i, item in enumerate(batch):
            w = min(item["style"].shape[2], FIXED_LEN)
            style_ref[i, :, :, :w] = item["style"][:, :, :w]
            wids.append(item["wid"])

        return {"style": style_ref, "wid": wids}

    def _load_style_ref(self, writer_id: str) -> np.ndarray:
        style_dir = os.path.join(self.style_path, writer_id)
        files = [f for f in os.listdir(style_dir) if not f.startswith(".")]
        random.shuffle(files)
        best_w, best_img = -1, None
        for fname in files:
            img = cv2.imread(os.path.join(style_dir, fname), flags=0)
            if img is None:
                continue
            if img.shape[1] > CRITICAL_WIDTH:
                return img.astype(np.float32) / 255.0
            if img.shape[1] > best_w:
                best_w, best_img = img.shape[1], img
        if best_img is None:
            raise RuntimeError(f"No style images for writer {writer_id}")
        return best_img.astype(np.float32) / 255.0

    def get_content(self, label: str) -> torch.Tensor:
        """Return [1, T, 16, 16] content tensor for a text string."""
        return get_content_from_label(label, self.letter2index, self.con_symbols)
