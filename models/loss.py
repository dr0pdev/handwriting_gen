"""
Loss functions used during training.

Proxy Anchor Loss:
    Writer-identity metric learning loss. Each writer has a learnable proxy
    embedding. Style embeddings are pushed toward their writer's proxy and
    away from all others using a margin-based formulation.

    Reference: "Proxy Anchor Loss for Deep Metric Learning" (CVPR 2020)
    https://arxiv.org/abs/2003.13911
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    """L2-normalize along the last dimension."""
    norm = x.pow(2).sum(dim=-1, keepdim=True).add(1e-12).sqrt()
    return x / norm


def binarize(T: torch.Tensor, nb_classes: int) -> torch.Tensor:
    """
    One-hot encode a tensor of integer class labels.

    Args:
        T: [N] integer class labels
        nb_classes: total number of classes

    Returns:
        [N, nb_classes] float one-hot tensor on the same device as T
    """
    device = T.device
    T_np = T.cpu().numpy()
    import sklearn.preprocessing
    T_bin = sklearn.preprocessing.label_binarize(T_np, classes=list(range(nb_classes)))
    return torch.FloatTensor(T_bin).to(device)


class ProxyAnchorLoss(nn.Module):
    """
    Proxy Anchor Loss for writer-style metric learning.

    Each writer ID maps to a learnable proxy vector. The loss encourages
    style embeddings to be close to their writer's proxy and far from others.

    Args:
        nb_classes: number of writer identities (proxies)
        sz_embed: embedding dimension
        mrg: margin (default 0.1)
        alpha: scale factor (default 32)
    """

    def __init__(self, nb_classes: int, sz_embed: int,
                 mrg: float = 0.1, alpha: float = 32.0):
        super().__init__()
        self.proxies = nn.Parameter(torch.randn(nb_classes, sz_embed))
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

    def forward(self, X: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: [N, sz_embed] style embeddings (already mean-pooled per sample)
            T: [N] integer writer IDs

        Returns:
            scalar loss
        """
        P = self.proxies  # [nb_classes, sz_embed]
        cos = F.linear(l2_norm(X), l2_norm(P))  # [N, nb_classes]

        P_one_hot = binarize(T, self.nb_classes)   # [N, nb_classes]
        N_one_hot = 1.0 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        # Only count proxies that have at least one positive sample in this batch
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)
        num_valid_proxies = len(with_pos_proxies)
        if num_valid_proxies == 0:
            return torch.tensor(0.0, device=X.device, requires_grad=True)

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes

        return pos_term + neg_term
