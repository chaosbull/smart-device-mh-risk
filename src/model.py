# Author: ZengWenquan
# https://github.com/chaosbull
# License: Apache-2.0

"""DRG-MDN-U.

44 typed nodes, a stress-gated sparse graph, then a point head and a
3-Gaussian mixture. The mixture means sit on the point prediction.
Sigma gets a per-row scale, but that scale is skipped while training.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    CALIBRATOR_MIN_SCALE,
    EMBED_COLS,
    EMBED_DIM,
    GCN_DIM,
    GCN_LAYERS,
    GRAPH_ATTN_DIM,
    GRAPH_HEADS,
    GRAPH_TOPK,
    INPUT_DIM,
    LOW_RANK_CROSS,
    MDN_COMPONENTS,
    MDN_DROPOUT,
    MDN_HIDDEN,
    NODE_EMBED_DIM,
    POINT_HIDDEN,
    REFINE_DIM,
    STRESS_FEATURE_IDX,
    USE_ADAPTIVE_CALIBRATOR,
    CALIBRATE_AT_INFERENCE,
    USE_FUSION,
    USE_MULTISCALE_READOUT,
    USE_MULTITASK_WEIGHTS,
    USE_POINT_HEAD,
    USE_POINT_SKIP,
    USE_PREDICTIVE_FUSION,
    USE_REPRESENTATION_REFINE,
)


class EntityEmbedding(nn.Module):
    def __init__(self, cardinalities: Dict[str, int], embed_dim: int = 8):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(n, embed_dim) for col, n in cardinalities.items()
        })

    def forward(self, cat_indices: torch.Tensor) -> torch.Tensor:
        parts = [self.embeddings[col](cat_indices[:, i]) for i, col in enumerate(EMBED_COLS)]
        return torch.cat(parts, dim=-1)


class SemanticNodeEncoder(nn.Module):
    def __init__(self, n_nodes: int, node_embed_dim: int, out_dim: int):
        super().__init__()
        self.node_type = nn.Parameter(torch.randn(n_nodes, node_embed_dim) * 0.02)
        self.proj = nn.Sequential(
            nn.Linear(node_embed_dim + 1, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, _ = z.shape
        values = z.unsqueeze(-1)
        types = self.node_type.unsqueeze(0).expand(b, -1, -1)
        return self.proj(torch.cat([types, values], dim=-1))


class StressGatedMultiHeadGraphLearner(nn.Module):
    def __init__(self, n_nodes: int, d_g: int, n_heads: int = 4, top_k: int = 5, stress_idx: int = 0):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_heads = n_heads
        self.top_k = top_k
        self.stress_idx = stress_idx
        self.d_head = max(d_g // n_heads, 4)
        self.inner_dim = self.d_head * n_heads
        self.q_proj = nn.Linear(d_g, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(d_g, self.inner_dim, bias=False)
        self.stress_gate = nn.Sequential(nn.Linear(1, n_heads), nn.Sigmoid())
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, node_h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        b, n, _ = node_h.shape
        q = self.q_proj(node_h).view(b, n, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = self.k_proj(node_h).view(b, n, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        scores = self.leaky(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head))
        stress = z[:, self.stress_idx].unsqueeze(-1)
        gate = self.stress_gate(stress).view(b, self.n_heads, 1, 1)
        scores = scores * (0.5 + 0.5 * gate)
        adj = F.softmax(scores, dim=-1).mean(dim=1)
        if self.top_k < self.n_nodes:
            _, topk_idx = torch.topk(adj, self.top_k, dim=-1)
            mask = torch.zeros_like(adj).scatter(-1, topk_idx, 1.0)
            adj = adj * mask
            adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-8)
        return adj


class GraphConvStack(nn.Module):
    def __init__(self, hidden_dim: int, n_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        self.weights = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(n_layers)
        ])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(hidden_dim)) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

    @staticmethod
    def _normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        d_inv_sqrt = deg.pow(-0.5)
        return d_inv_sqrt * adj * d_inv_sqrt.transpose(-2, -1)

    def forward(self, node_h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(node_h)
        adj_norm = self._normalize_adj(adj)
        for w, b, norm in zip(self.weights, self.biases, self.norms):
            h_new = torch.matmul(adj_norm, h)
            h_new = norm(F.gelu(w(h_new) + b))
            h = h + h_new
        return h


class AttentionGraphReadout(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.global_query = nn.Linear(hidden_dim, hidden_dim)
        self.node_key = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        context = h.mean(dim=1)
        q = self.global_query(context).unsqueeze(1)
        k = self.node_key(h)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / self.scale, dim=-1)
        return torch.matmul(attn, h).squeeze(1)


class MultiScaleGraphReadout(nn.Module):
    """Attention pool and mean pool, concatenated."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn_readout = AttentionGraphReadout(hidden_dim)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        attn_pool = self.attn_readout(h)
        mean_pool = h.mean(dim=1)
        return self.proj(torch.cat([attn_pool, mean_pool], dim=-1))


class RepresentationRefinementBlock(nn.Module):
    """Extra residual MLP. Off in the reported config."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class LowRankInteractionTower(nn.Module):
    def __init__(self, input_dim: int, rank: int = 8):
        super().__init__()
        self.u = nn.Parameter(torch.randn(input_dim, rank) * 0.02)
        self.v = nn.Parameter(torch.randn(input_dim, rank) * 0.02)
        self.out = nn.Linear(rank, input_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        cross = (z @ self.u) * (z @ self.v)
        return self.out(cross)


class GatedDualStreamFusion(nn.Module):
    def __init__(self, graph_dim: int, input_dim: int):
        super().__init__()
        total = graph_dim + input_dim * 2
        self.gate = nn.Sequential(nn.Linear(total, graph_dim), nn.Sigmoid())
        self.proj = nn.Sequential(nn.Linear(total, graph_dim), nn.GELU(), nn.LayerNorm(graph_dim))

    def forward(self, graph_repr: torch.Tensor, z: torch.Tensor, interact: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([graph_repr, z, interact], dim=-1)
        fused = self.proj(cat)
        gate = self.gate(cat)
        return gate * fused + (1.0 - gate) * graph_repr


class PointPredictionHead(nn.Module):
    """Scalar head. The skip from z is on in the reported run."""

    def __init__(self, in_dim: int, hidden: int, skip_dim: int = 0):
        super().__init__()
        head_in = in_dim + skip_dim if skip_dim > 0 else in_dim
        self.net = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, f: torch.Tensor, z_skip: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.cat([f, z_skip], dim=-1) if z_skip is not None else f
        return self.net(x).squeeze(-1)


class PredictiveFusionGate(nn.Module):
    """Blend point head and mixture mean. Left off; the paper uses the mixture mean."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim, 16), nn.GELU(), nn.Linear(16, 1),
        )
        nn.init.constant_(self.gate[-1].bias, 1.5)

    def forward(self, f: torch.Tensor, y_point: torch.Tensor, mix_mean: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.gate(f)).squeeze(-1)
        return alpha * y_point + (1.0 - alpha) * mix_mean


class AdaptiveUncertaintyCalibrator(nn.Module):
    """s = softplus(mlp(f)) + min_scale. Applied to every mixture component."""

    def __init__(self, in_dim: int, min_scale: float = 0.85):
        super().__init__()
        self.min_scale = min_scale
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(f)).squeeze(-1) + self.min_scale


class MixtureDensityHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_components: int, skip_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.pi_head = nn.Linear(hidden, n_components)
        self.mu_head = nn.Linear(hidden, n_components)
        self.sigma_head = nn.Linear(hidden, n_components)
        self.residual = nn.Linear(skip_dim, n_components)
        self.residual_mode = True

    def forward(self, f: torch.Tensor, z_skip: torch.Tensor, y_anchor: torch.Tensor | None = None):
        h = self.net(f)
        pi = F.softmax(self.pi_head(h), dim=-1)
        delta = self.mu_head(h) + 0.10 * self.residual(z_skip)
        sigma = F.softplus(self.sigma_head(h)) + 1e-6
        if self.residual_mode and y_anchor is not None:
            mu = y_anchor.unsqueeze(-1) + delta
        else:
            mu = delta
        return pi, mu, sigma, delta


class MultiTaskUncertaintyWeights(nn.Module):
    """Kendall-style loss weights. Not used for the numbers in the paper."""

    def __init__(self, n_tasks: int = 3, init_log_vars: list[float] | None = None):
        super().__init__()
        init = init_log_vars if init_log_vars is not None else [0.0] * n_tasks
        self.log_vars = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    def loss(self, task_idx: int, task_loss: torch.Tensor) -> torch.Tensor:
        precision = torch.exp(-self.log_vars[task_idx])
        return precision * task_loss + self.log_vars[task_idx]


class DRGMDN(nn.Module):

    def __init__(self, cardinalities: Dict[str, int]):
        super().__init__()
        self.entity_emb = EntityEmbedding(cardinalities, EMBED_DIM)
        self.input_norm = nn.LayerNorm(INPUT_DIM)

        self.node_encoder = SemanticNodeEncoder(INPUT_DIM, NODE_EMBED_DIM, GRAPH_ATTN_DIM)
        self.graph_learner = StressGatedMultiHeadGraphLearner(
            INPUT_DIM, GRAPH_ATTN_DIM, GRAPH_HEADS, GRAPH_TOPK, STRESS_FEATURE_IDX
        )
        self.graph_conv = GraphConvStack(GRAPH_ATTN_DIM, GCN_LAYERS)
        readout_cls = MultiScaleGraphReadout if USE_MULTISCALE_READOUT else AttentionGraphReadout
        self.readout = readout_cls(GRAPH_ATTN_DIM)
        self.graph_proj = nn.Sequential(
            nn.Linear(GRAPH_ATTN_DIM, GCN_DIM), nn.GELU(), nn.LayerNorm(GCN_DIM),
        )
        self.interaction = LowRankInteractionTower(INPUT_DIM, LOW_RANK_CROSS)
        self.fusion = GatedDualStreamFusion(GCN_DIM, INPUT_DIM) if USE_FUSION else None
        self.refine = RepresentationRefinementBlock(GCN_DIM) if USE_REPRESENTATION_REFINE else nn.Identity()
        self.z_proj = nn.Linear(INPUT_DIM, GCN_DIM) if USE_POINT_SKIP else None

        self.mdn = MixtureDensityHead(GCN_DIM, MDN_HIDDEN, MDN_COMPONENTS, INPUT_DIM, MDN_DROPOUT)
        skip_dim = GCN_DIM if USE_POINT_SKIP else 0
        self.point_head = PointPredictionHead(GCN_DIM, POINT_HIDDEN, skip_dim) if USE_POINT_HEAD else None
        self.fusion_gate = PredictiveFusionGate(GCN_DIM) if USE_PREDICTIVE_FUSION else None
        self.calibrator = AdaptiveUncertaintyCalibrator(GCN_DIM, CALIBRATOR_MIN_SCALE) if USE_ADAPTIVE_CALIBRATOR else None
        self.task_weights = (
            MultiTaskUncertaintyWeights(3, init_log_vars=[0.0, -0.45, 0.08])
            if USE_MULTITASK_WEIGHTS else None
        )

    def encode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        node_h = self.node_encoder(z)
        adj = self.graph_learner(node_h, z)
        h = self.graph_conv(node_h, adj)
        g = self.graph_proj(self.readout(h))
        interact = self.interaction(z)
        if self.fusion is not None:
            f = self.fusion(g, z, interact)
        else:
            f = g
        f = self.refine(f)
        return f, adj

    def _point_skip(self, z: torch.Tensor) -> torch.Tensor | None:
        return self.z_proj(z) if self.z_proj is not None else None

    def build_z(self, cat: torch.Tensor, cont: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.entity_emb(cat), cont], dim=-1)
        return self.input_norm(z)

    def forward_from_z(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        f, adj = self.encode(z)

        if self.point_head is not None:
            y_point = self.point_head(f, self._point_skip(z))
        else:
            y_point = None

        pi, mu, sigma, delta = self.mdn(f, z, y_anchor=y_point)

        if self.calibrator is not None:
            cal = self.calibrator(f)
            # keep the scale out of the training graph; it only widens sigma at eval
            if not (CALIBRATE_AT_INFERENCE and self.training):
                sigma = sigma * cal.unsqueeze(-1)
        else:
            cal = torch.ones(pi.shape[0], device=pi.device)

        mix_mean = (pi * mu).sum(dim=-1)
        if y_point is None:
            y_point = mix_mean

        if self.fusion_gate is not None and USE_PREDICTIVE_FUSION and self.point_head is not None:
            y_pred = self.fusion_gate(f, y_point, mix_mean)
        else:
            y_pred = mix_mean

        return {
            "pi": pi, "mu": mu, "sigma": sigma, "adj": adj,
            "f": f, "y_pred": y_pred, "y_point": y_point, "mix_mean": mix_mean,
            "cal_scale": cal, "delta_mu": delta,
        }

    def forward(self, cat: torch.Tensor, cont: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.forward_from_z(self.build_z(cat, cont))
        return out["pi"], out["mu"], out["sigma"]

    def forward_with_z(self, cat: torch.Tensor, cont: torch.Tensor):
        z = self.build_z(cat, cont)
        out = self.forward_from_z(z)
        return out["pi"], out["mu"], out["sigma"], z, out["adj"]

    def predict(self, cat: torch.Tensor, cont: torch.Tensor) -> torch.Tensor:
        return self.forward_from_z(self.build_z(cat, cont))["y_pred"]

    @staticmethod
    def point_predict(pi: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        return (pi * mu).sum(dim=-1)

    def point_predict_batch(self, cat: torch.Tensor, cont: torch.Tensor) -> torch.Tensor:
        return self.forward_from_z(self.build_z(cat, cont))["y_pred"]

    @staticmethod
    def mixture_variance(pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        mean = (pi * mu).sum(dim=-1)
        second = (pi * (sigma ** 2 + mu ** 2)).sum(dim=-1)
        return (second - mean ** 2).clamp(min=1e-8)

    @staticmethod
    def mixture_entropy(pi: torch.Tensor) -> torch.Tensor:
        return -(pi * torch.log(pi + 1e-8)).sum(dim=-1)


def gaussian_pdf(y, mu, sigma):
    var = sigma ** 2
    log_norm = -0.5 * (math.log(2 * math.pi) + torch.log(var) + (y.unsqueeze(-1) - mu) ** 2 / var)
    return torch.exp(log_norm)


def mdn_nll(y, pi, mu, sigma):
    pdf = gaussian_pdf(y, mu, sigma)
    return -torch.log((pi * pdf).sum(dim=-1).clamp(min=1e-8))


def mdn_entropy_penalty(pi):
    return -(pi * torch.log(pi + 1e-8)).sum(dim=-1).mean()


def gaussian_kl(mu1, sig1, mu2, sig2):
    var1, var2 = sig1 ** 2, sig2 ** 2
    return 0.5 * (torch.log(var2 / var1) + (var1 + (mu1 - mu2) ** 2) / var2 - 1)


def mdn_kl(pi_p, mu_p, sig_p, pi_q, mu_q, sig_q):
    kl_pi = (pi_p * (torch.log(pi_p + 1e-8) - torch.log(pi_q + 1e-8))).sum(-1)
    return kl_pi + (pi_p * gaussian_kl(mu_p, sig_p, mu_q, sig_q)).sum(-1)


def calibration_coverage_loss(y, y_pred, pi, mu, sigma, target=0.90):
    std = torch.sqrt(DRGMDN.mixture_variance(pi, mu, sigma))
    margin = 1.645 * std
    soft_in = torch.sigmoid((margin - (y - y_pred).abs()) * 8.0)
    return (soft_in.mean() - target) ** 2


def vat_loss(model: DRGMDN, cat, cont, y, eps: float):
    pi, mu, sigma, z, _ = model.forward_with_z(cat, cont)
    nll = mdn_nll(y, pi, mu, sigma).mean()
    grad_z = torch.autograd.grad(nll, z, retain_graph=True, create_graph=True)[0]
    out_adv = model.forward_from_z(z + eps * grad_z.sign())
    return mdn_kl(
        pi.detach(), mu.detach(), sigma.detach(),
        out_adv["pi"], out_adv["mu"], out_adv["sigma"],
    ).mean()
