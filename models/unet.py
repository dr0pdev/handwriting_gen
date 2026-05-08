"""
UNet denoiser for latent diffusion.

Operates on 4-channel latent tensors from the Stable Diffusion VAE.
Context embeddings from Mix_TR are injected at every resolution level
via SpatialTransformer cross-attention blocks.

Key components:
  - ResBlock: residual block conditioned on timestep embedding
  - SpatialTransformer: self-attention + cross-attention to context
  - UNetModel: full encoder-bottleneck-decoder with skip connections
"""

from abc import abstractmethod
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from inspect import isfunction

from models.encoder import MixTR


# ---------------------------------------------------------------------------
# Gradient checkpointing
# ---------------------------------------------------------------------------

def checkpoint(func, inputs, params, flag):
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [
            x.float().detach().requires_grad_(True) for x in ctx.input_tensors
        ]
        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors, ctx.input_params, output_tensors
        return (None, None) + input_grads


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels: int) -> GroupNorm32:
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps: torch.Tensor, dim: int,
                        max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int = None,
                 heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor = None,
                mask: torch.Tensor = None) -> torch.Tensor:
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h).contiguous(),
                      (q, k, v))
        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j").contiguous()
            max_neg = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg)

        attn = sim.softmax(dim=-1)
        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h).contiguous()
        return self.to_out(out)


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: int = None, mult: int = 4,
                 glu: bool = False, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )
        self.net = nn.Sequential(project_in, nn.Dropout(dropout),
                                 nn.Linear(inner_dim, dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, d_head: int, dropout: float = 0.0,
                 context_dim: int = None, gated_ff: bool = True,
                 use_checkpoint: bool = True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head,
                                    dropout=dropout)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        return checkpoint(self._forward, (x, context), self.parameters(),
                          self.use_checkpoint)

    def _forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer for 2D feature maps.
    Projects [B, C, H, W] -> tokens -> BasicTransformerBlock -> back to [B, C, H, W].
    """

    def __init__(self, in_channels: int, n_heads: int, d_head: int,
                 depth: int = 1, dropout: float = 0.1, context_dim: int = None):
        super().__init__()
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1)
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout,
                                  context_dim=context_dim)
            for _ in range(depth)
        ])
        self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1))

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        x = self.proj_out(x)
        return x + x_in


# ---------------------------------------------------------------------------
# QKV self-attention (bottleneck)
# ---------------------------------------------------------------------------

class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        bs, width, length = qkv.shape
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        return torch.einsum("bts,bcs->bct", weight, v).reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 1,
                 use_checkpoint: bool = False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.attention = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(nn.Conv2d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


# ---------------------------------------------------------------------------
# Upsampling / Downsampling
# ---------------------------------------------------------------------------

class Upsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, out_channels: int = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, out_channels: int = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(self.channels, self.out_channels, 3, stride=2, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool2d(2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.channels
        return self.op(x)


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x: torch.Tensor, emb: torch.Tensor,
                context: torch.Tensor = None) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class ResBlock(TimestepBlock):
    def __init__(self, channels: int, emb_channels: int, dropout: float,
                 out_channels: int = None, use_conv: bool = False,
                 use_scale_shift_norm: bool = False, dims: int = 2,
                 use_checkpoint: bool = False, up: bool = False, down: bool = False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h


# ---------------------------------------------------------------------------
# UNet model
# ---------------------------------------------------------------------------

class UNetModel(nn.Module):
    """
    Conditional UNet denoiser operating on 4-channel latent tensors.

    Conditioned on:
      - Diffusion timestep (sinusoidal embedding -> MLP)
      - Style+content context from Mix_TR (cross-attention in SpatialTransformer)

    Args:
        in_channels:          Latent input channels (4 for SD VAE)
        model_channels:       Base channel count
        out_channels:         Predicted noise channels (4 for SD VAE)
        num_res_blocks:       ResBlocks per resolution level
        attention_resolutions: Downsample ratios at which attention is applied
        dropout:              Dropout probability
        channel_mult:         Channel multipliers per resolution level
        conv_resample:        Use learned conv for up/downsampling
        num_heads:            Number of attention heads
        context_dim:          Cross-attention context dimension (matches Mix_TR d_model)
        transformer_depth:    SpatialTransformer depth per attention block
        nb_classes:           Number of writer IDs for Proxy Anchor proxies
    """

    def __init__(
        self,
        in_channels: int = 4,
        model_channels: int = 512,
        out_channels: int = 4,
        num_res_blocks: int = 1,
        attention_resolutions: tuple = (1, 1),
        dropout: float = 0.1,
        channel_mult: tuple = (1, 1),
        conv_resample: bool = True,
        dims: int = 2,
        use_checkpoint: bool = False,
        num_heads: int = 4,
        context_dim: int = 512,
        transformer_depth: int = 1,
        nb_classes: int = 496,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.mix_net = MixTR(nb_classes=nb_classes, d_model=context_dim)

        # Input blocks
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, 3, padding=1))
        ])
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                dim_head = ch // num_heads
                layers = [
                    ResBlock(ch, time_embed_dim, dropout,
                             out_channels=mult * model_channels,
                             use_checkpoint=use_checkpoint),
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append(
                        SpatialTransformer(ch, num_heads, dim_head,
                                           depth=transformer_depth,
                                           context_dim=context_dim)
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample))
                )
                input_block_chans.append(ch)
                ds *= 2

        # Middle block
        dim_head = ch // num_heads
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, use_checkpoint=use_checkpoint),
            SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth,
                               context_dim=context_dim),
            ResBlock(ch, time_embed_dim, dropout, use_checkpoint=use_checkpoint),
        )

        # Output blocks
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(ch + ich, time_embed_dim, dropout,
                             out_channels=model_channels * mult,
                             use_checkpoint=use_checkpoint),
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append(
                        SpatialTransformer(ch, num_heads, dim_head,
                                           depth=transformer_depth,
                                           context_dim=context_dim)
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(model_channels, out_channels, 3, padding=1)),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor,
                style: torch.Tensor, content: torch.Tensor,
                wid: torch.Tensor = None, tag: str = "test"):
        """
        Args:
            x:         [B, 4, H, W] noisy latent
            timesteps: [B] integer diffusion timesteps
            style:     [B, 1, H_s, W_s] grayscale style image
            content:   [B, T, 16, 16] per-character glyph images
            wid:       [B] writer IDs (training only)
            tag:       'train' or 'test'

        Returns (train):  (predicted_noise, vertical_proxy_loss, horizontal_proxy_loss)
        Returns (test):   predicted_noise
        """
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_emb)

        if tag == "train":
            context, v_loss, h_loss = self.mix_net(style, content, wid)
        else:
            context = self.mix_net.generate(style, content)

        h = x.float()

        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)

        h = self.middle_block(h, emb, context)

        for module in self.output_blocks:
            h_skip = hs.pop()
            # Crop width to handle variable-length sequences
            h = h[:, :, :, : h_skip.shape[3]]
            h = torch.cat([h, h_skip], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        noise_pred = self.out(h)

        if tag == "train":
            return noise_pred, v_loss, h_loss
        return noise_pred
