"""
Mix_TR: Style-content conditioning encoder for handwriting generation.

Architecture overview:
  Style path:
    Grayscale line image
    -> ResNet-18 (1-channel, truncated before global pool)
    -> Reshape to [B, 256, 4, W]
    -> Dilated ResNet conv5_x  ->  [B, 512, 4, W]
    -> 2D positional encoding
    -> Base Transformer Encoder (2 layers)
    -> Vertical Head (1 layer)  -- captures per-letter height/baseline info
    -> Horizontal Head (1 layer) -- captures spacing/ligature/word-gap info

  Content path:
    Per-character Unifont glyph images [B, T, 16, 16]
    -> ResNet-18 trunk (1-channel)
    -> 1D positional encoding

  Fusion (cross-attention decoders):
    Content queries x Horizontal style  ->  Horizontal Decoder (3 layers)
    Result           x Vertical style   ->  Vertical Decoder   (3 layers)
    Output: context  [B, T, d_model]

  During training:
    Random horizontal/vertical subsampling of style features
    -> MLP projection -> Proxy Anchor loss per writer ID
    (vertical_proxy_loss, horizontal_proxy_loss)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from einops import rearrange

from models.transformer import (
    TransformerEncoderLayer,
    TransformerEncoder,
    TransformerDecoderLayer,
    TransformerDecoder,
    PositionalEncoding,
    PositionalEncoding2D,
)
from models.loss import ProxyAnchorLoss
from models.resnet_dilation import resnet18 as resnet18_dilation


class MixTR(nn.Module):
    """
    Style-content conditioning encoder.

    Args:
        nb_classes: number of writer identities (for Proxy Anchor proxies)
        d_model: internal feature dimension (must match UNet context_dim)
        nhead: number of attention heads
        num_encoder_layers: base encoder depth
        num_head_layers: depth of each style head (vertical/horizontal)
        num_decoder_layers: depth of each content-style fusion decoder
        dim_feedforward: FFN width inside transformer layers
        dropout: dropout probability
    """

    def __init__(
        self,
        nb_classes: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        num_head_layers: int = 1,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        normalize_before: bool = True,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation="relu", normalize_before=normalize_before,
        )

        self.base_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, norm=None)

        vertical_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.vertical_head = TransformerEncoder(encoder_layer, num_head_layers, norm=vertical_norm)

        horizontal_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.horizontal_head = TransformerEncoder(encoder_layer, num_head_layers, norm=horizontal_norm)

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation="relu", normalize_before=normalize_before,
        )

        vertical_dec_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.vertical_decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, norm=vertical_dec_norm,
            return_intermediate=False,
        )

        horizontal_dec_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.horizontal_decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, norm=horizontal_dec_norm,
            return_intermediate=False,
        )

        self.pos_enc_1d = PositionalEncoding(dim=d_model, dropout=dropout)
        self.pos_enc_2d = PositionalEncoding2D(d_model=d_model, dropout=dropout)

        # Proxy projection MLPs (style embedding -> proxy loss input)
        self.vertical_pro_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.horizontal_pro_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )

        # Style encoder: pretrained ResNet-18, modified for 1-channel input
        self.feat_encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feat_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feat_encoder.layer4 = nn.Identity()
        self.feat_encoder.fc = nn.Identity()
        self.feat_encoder.avgpool = nn.Identity()

        # Dilated conv5_x applied on top of ResNet features
        self.style_dilation_layer = resnet18_dilation().conv5_x

        # Content encoder: 1-channel ResNet-18 trunk (conv layers only, no FC/pool)
        self.content_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            *list(models.resnet18(weights=models.ResNet18_Weights.DEFAULT).children())[1:-2],
        )

        # Proxy losses (one per axis)
        self.vertical_proxy = ProxyAnchorLoss(nb_classes=nb_classes, sz_embed=d_model)
        self.horizontal_proxy = ProxyAnchorLoss(nb_classes=nb_classes, sz_embed=d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------
    # Random sampling helpers for proxy loss
    # ------------------------------------------------------------------

    def _random_horizontal_sample(self, x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
        """
        Sample a random subset of columns from the style feature map.
        Encourages the vertical head to learn row-invariant (height) features.

        x: [(H*W), B, D]  (H=4)
        returns: [(H * W_sampled), B, D]
        """
        x = rearrange(x, "(H W) B D -> B H W D", H=4)
        B, H, W, D = x.shape
        noise = torch.rand(B, W, device=x.device)
        ids = torch.argsort(noise, dim=1)
        n_keep = int(W * ratio)
        x_sampled = torch.gather(
            x, dim=2,
            index=ids[:, :n_keep].unsqueeze(1).unsqueeze(-1).expand(B, H, n_keep, D)
        )
        return rearrange(x_sampled, "B H W D -> (H W) B D")

    def _random_vertical_sample(self, x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
        """
        Sample a random subset of rows from the style feature map.
        Encourages the horizontal head to learn column-invariant (spacing) features.

        x: [(H*W), B, D]  (H=4)
        returns: [(H_sampled * W), B, D]
        """
        x = rearrange(x, "(H W) B D -> B H W D", H=4)
        B, H, W, D = x.shape
        noise = torch.rand(B, H, device=x.device)
        ids = torch.argsort(noise, dim=1)
        n_keep = int(H * ratio)
        x_sampled = torch.gather(
            x, dim=1,
            index=ids[:, :n_keep].unsqueeze(-1).unsqueeze(-1).expand(B, n_keep, W, D)
        )
        return rearrange(x_sampled, "B H W D -> (H W) B D")

    # ------------------------------------------------------------------
    # Core style encoding (shared between train and generate)
    # ------------------------------------------------------------------

    def _encode_style(self, style: torch.Tensor):
        """
        Encode a grayscale style image into vertical and horizontal feature sequences.

        Args:
            style: [B, 1, H, W] grayscale style image

        Returns:
            vertical_style:   [(4*W'), B, d_model]
            horizontal_style: [(4*W'), B, d_model]
        """
        # ResNet-18 feature extraction (output is flattened due to removed avgpool)
        feats = self.feat_encoder(style)                          # [B, C*4*W']
        feats = rearrange(feats, "n (c h w) -> n c h w", c=256, h=4)  # [B, 256, 4, W']
        feats = self.style_dilation_layer(feats)                  # [B, 512, 4, W']
        feats = self.pos_enc_2d(feats)                            # 2D PE added
        feats = rearrange(feats, "n c h w -> (h w) n c")          # [(4*W'), B, 512]

        base = self.base_encoder(feats)                           # [(4*W'), B, 512]
        vertical = self.vertical_head(base)
        horizontal = self.horizontal_head(base)
        return vertical, horizontal

    # ------------------------------------------------------------------
    # Content encoding (shared)
    # ------------------------------------------------------------------

    def _encode_content(self, content: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Encode per-character glyph images into a sequence of content embeddings.

        Args:
            content: [B, T, 16, 16] per-character glyph images
            batch_size: B

        Returns:
            [T, B, d_model]
        """
        x = rearrange(content, "n t h w -> (n t) 1 h w")    # [(B*T), 1, H, W]
        x = self.content_encoder(x)                           # [(B*T), C, h', w']
        x = rearrange(x, "(n t) c h w -> t n (c h w)", n=batch_size)  # [T, B, D]
        x = self.pos_enc_1d(x)
        return x

    # ------------------------------------------------------------------
    # Forward (training) -- returns context + proxy losses
    # ------------------------------------------------------------------

    def forward(self, style: torch.Tensor, content: torch.Tensor, wid: torch.Tensor):
        """
        Training forward pass.

        Args:
            style:   [B, 1, H, W] grayscale style image
            content: [B, T, 16, 16] per-character glyph images
            wid:     [B] integer writer IDs

        Returns:
            context:               [B, T, d_model]
            vertical_proxy_loss:   scalar
            horizontal_proxy_loss: scalar
        """
        B = style.shape[0]

        vertical_style, horizontal_style = self._encode_style(style)

        # Proxy losses via random spatial subsampling
        vert_proxy_feat = self._random_horizontal_sample(vertical_style)   # columns dropped
        vert_proxy_feat = self.vertical_pro_mlp(vert_proxy_feat)
        vert_proxy_feat = vert_proxy_feat.mean(dim=0)                       # [B, d_model]
        vertical_loss = self.vertical_proxy(vert_proxy_feat, wid)

        horiz_proxy_feat = self._random_vertical_sample(horizontal_style)  # rows dropped
        horiz_proxy_feat = self.horizontal_pro_mlp(horiz_proxy_feat)
        horiz_proxy_feat = horiz_proxy_feat.mean(dim=0)                     # [B, d_model]
        horizontal_loss = self.horizontal_proxy(horiz_proxy_feat, wid)

        # Content encoding
        content_seq = self._encode_content(content, B)   # [T, B, d_model]

        # Cross-attention fusion
        style_hs = self.horizontal_decoder(content_seq, horizontal_style, tgt_mask=None)
        hs = self.vertical_decoder(style_hs[0], vertical_style, tgt_mask=None)

        context = hs[0].permute(1, 0, 2).contiguous()   # [B, T, d_model]
        return context, vertical_loss, horizontal_loss

    # ------------------------------------------------------------------
    # Generate (inference) -- no proxy losses
    # ------------------------------------------------------------------

    def generate(self, style: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        """
        Inference forward pass.

        Args:
            style:   [B, 1, H, W] grayscale style image
            content: [B, T, 16, 16] per-character glyph images

        Returns:
            context: [B, T, d_model]
        """
        B = style.shape[0]
        vertical_style, horizontal_style = self._encode_style(style)
        content_seq = self._encode_content(content, B)

        style_hs = self.horizontal_decoder(content_seq, horizontal_style, tgt_mask=None)
        hs = self.vertical_decoder(style_hs[0], vertical_style, tgt_mask=None)

        return hs[0].permute(1, 0, 2).contiguous()   # [B, T, d_model]
