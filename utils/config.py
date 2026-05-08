"""
YAML-based configuration system using EasyDict for dot-access.
Provides a global `cfg` object and helpers for loading and merging configs.
"""

import copy
import os
import yaml
from ast import literal_eval
from easydict import EasyDict


class AttrDict(EasyDict):
    """EasyDict subclass with immutability support."""

    _IMMUTABLE = "__immutable__"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__[AttrDict._IMMUTABLE] = False

    def immutable(self, is_immutable: bool):
        self.__dict__[AttrDict._IMMUTABLE] = is_immutable
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)
        for v in self.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)

    def is_immutable(self) -> bool:
        return self.__dict__[AttrDict._IMMUTABLE]

    def __setattr__(self, name, value):
        if self.__dict__.get(AttrDict._IMMUTABLE, False):
            raise AttributeError(f"Attempted to set {name} on immutable AttrDict")
        super().__setattr__(name, value)


# ---------------------------------------------------------------------------
# Global config defaults
# ---------------------------------------------------------------------------

_C = AttrDict()
cfg = _C

# Training
_C.TRAIN = AttrDict()
_C.TRAIN.TYPE = "train"
_C.TRAIN.IMG_H = 64
_C.TRAIN.IMG_W = 64
_C.TRAIN.IMS_PER_BATCH = 8
_C.TRAIN.SNAPSHOT_ITERS = 3000
_C.TRAIN.SNAPSHOT_BEGIN = 0
_C.TRAIN.SEED = 1001
_C.TRAIN.IMAGE_PATH = ""
_C.TRAIN.STYLE_PATH = ""
_C.TRAIN.LABEL_PATH = ""

# Data loader
_C.DATA_LOADER = AttrDict()
_C.DATA_LOADER.NUM_THREADS = 4

# Test / inference
_C.TEST = AttrDict()
_C.TEST.TYPE = "test"
_C.TEST.IMG_H = 64
_C.TEST.IMG_W = 64
_C.TEST.IMS_PER_BATCH = 8
_C.TEST.IMAGE_PATH = ""
_C.TEST.STYLE_PATH = ""
_C.TEST.LABEL_PATH = ""

# Model
_C.MODEL = AttrDict()
_C.MODEL.STYLE_ENCODER_LAYERS = 3
_C.MODEL.NUM_IMGS = 15
_C.MODEL.IN_CHANNELS = 4
_C.MODEL.OUT_CHANNELS = 4
_C.MODEL.NUM_RES_BLOCKS = 1
_C.MODEL.NUM_HEADS = 4
_C.MODEL.EMB_DIM = 512

# Solver
_C.SOLVER = AttrDict()
_C.SOLVER.TYPE = "AdamW"
_C.SOLVER.BASE_LR = 1e-4
_C.SOLVER.EPOCHS = 2000
_C.SOLVER.WARMUP_ITERS = 20000
_C.SOLVER.GRAD_L2_CLIP = 5.0

# Misc
_C.NUM_GPUS = 1
_C.OUTPUT_DIR = "checkpoints"
_C.DATASET = ""


# ---------------------------------------------------------------------------
# Config loading helpers
# ---------------------------------------------------------------------------

def cfg_from_file(filename: str):
    """Load a YAML config file and merge it into the global cfg."""
    with open(filename, "r") as f:
        yaml_cfg = AttrDict(yaml.full_load(f))
    _merge_into(yaml_cfg, _C)


def cfg_from_list(cfg_list):
    """Merge key=value pairs from a list (e.g. command-line overrides) into cfg."""
    assert len(cfg_list) % 2 == 0, "cfg_list must have an even number of elements"
    for key, value in zip(cfg_list[0::2], cfg_list[1::2]):
        key_parts = key.split(".")
        d = _C
        for part in key_parts[:-1]:
            assert part in d, f"Non-existent config key: {key}"
            d = d[part]
        leaf = key_parts[-1]
        assert leaf in d, f"Non-existent config key: {key}"
        d[leaf] = _decode_value(value)


def assert_and_infer_cfg(make_immutable: bool = True):
    """Call after all cfg merges are done to lock the config."""
    if make_immutable:
        cfg.immutable(True)


def _merge_into(src: AttrDict, dst: AttrDict, _stack=None):
    for k, v in src.items():
        full_key = ".".join(_stack + [k]) if _stack else k
        if k not in dst:
            raise KeyError(f"Non-existent config key: {full_key}")
        v = copy.deepcopy(v)
        v = _decode_value(v)
        if isinstance(v, dict):
            v = AttrDict(v)
        if isinstance(v, AttrDict) and isinstance(dst[k], AttrDict):
            _merge_into(v, dst[k], _stack=([k] if _stack is None else _stack + [k]))
        else:
            dst[k] = v


def _decode_value(v):
    if isinstance(v, dict):
        return AttrDict(v)
    if not isinstance(v, str):
        return v
    try:
        return literal_eval(v)
    except (ValueError, SyntaxError):
        return v
