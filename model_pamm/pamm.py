from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextProtoBank(nn.Module):
    def __init__(self, num_contexts: int, d_model: int):
        super().__init__()
        self.context_prototypes = nn.Parameter(torch.randn(num_contexts, d_model))
        self.context_selector = nn.Linear(d_model, num_contexts)

    def forward(self, global_feat: torch.Tensor, tau: float = 1.0, hard: bool = False) -> tuple:
        selector_logits = self.context_selector(global_feat)
        soft_weights = F.gumbel_softmax(selector_logits, tau=tau, hard=hard, dim=-1)
        selected_context_protos = torch.einsum("bk,kd->bd", soft_weights, self.context_prototypes)

        dot_product = (global_feat * selected_context_protos).sum(dim=-1)
        norm_global_feat = torch.norm(global_feat, dim=-1)
        norm_selected_protos = torch.norm(selected_context_protos, dim=-1)
        cosine_sim = dot_product / ((norm_global_feat * norm_selected_protos) + 1e-8)
        deviation_score = 1 - cosine_sim

        return deviation_score, selected_context_protos, soft_weights


class RevIN(nn.Module):
    """
    RevIN: Reversible Instance Normalization.

    这里按“每个样本、每个通道”对时间维做归一化，目的是消除当前窗口内部的
    均值/方差漂移，让模型更关注形状和相对变化。推理后再用同一窗口的统计量
    反归一化，把趋势重建结果映射回趋势值空间。
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("mean", torch.zeros(1, 1, num_features), persistent=False)
        self.register_buffer("std", torch.ones(1, 1, num_features), persistent=False)
        self._stats_ready = False

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            self._stats_ready = True
            x = (x - self.mean) / self.std
            return x * self.affine_weight + self.affine_bias
        if mode == "denorm":
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            if not self._stats_ready:
                mean = torch.zeros(1, 1, self.num_features, device=x.device, dtype=x.dtype)
                std = torch.ones(1, 1, self.num_features, device=x.device, dtype=x.dtype)
            else:
                mean = self.mean.to(device=x.device, dtype=x.dtype)
                std = self.std.to(device=x.device, dtype=x.dtype)
            return x * std + mean
        raise ValueError(f"Unsupported RevIN mode: {mode}")


class PatchEmbed(nn.Module):
    """
    将点级时间序列切成重叠 patch，并映射到 token 空间。

    输入:
        x: [B, L, C]
    输出:
        tokens: [B, N, D]
        patch_num: N
    """

    def __init__(self, input_dim: int, patch_size: int, patch_stride: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.proj = nn.Linear(patch_size * input_dim, d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        batch, _, channels = x.shape
        patches = x.unfold(dimension=1, size=self.patch_size, step=self.patch_stride)
        patch_num = patches.shape[1]
        patches = patches.contiguous().view(batch, patch_num, self.patch_size * channels)
        return self.proj(patches), patch_num


class LocalityBiasedAttention(nn.Module):
    """
    多头自注意力 + 局部偏置。
    保持原有 AdaptiveLocalAttention 流程，额外在 scores 上加入距离惩罚。
    """

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 locality_strength: float = 0.2):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.locality_strength = locality_strength

        # QKV 和输出线性层
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape

        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # q,k,v -> [B, T, H, Dh]
        q = q.transpose(1, 2)  # [B, H, T, Dh]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B,H,T,T]

        idx = torch.arange(T, device=x.device)
        dist = (idx[None, :] - idx[:, None]).abs()  # [T,T]
        local_bias = -self.locality_strength * dist  # 距离惩罚
        scores = scores + local_bias.unsqueeze(0).unsqueeze(0)  # broadcast B,H,T,T

        attn = self.attn_softmax(scores)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B,H,T,Dh]
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # 拼接多头
        out = self.out(out)

        return out, attn


class MHAttention(nn.Module):
    """
    Standard multi-head self-attention without proximity bias.
    """

    def __init__(
            self,
            d_model: int,
            num_heads: int,
            dropout: float,
            locality_strength: float,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens, _ = x.shape
        qkv = self.qkv(x).view(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, tokens, self.d_model)
        output = self.out(output)
        return output, attn


class MultivariatePatchTokenGate(nn.Module):
    """Apply feature-wise gates to multivariate patch tokens."""

    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.attn_layer = nn.Linear(in_size, out_size)
        self.attn_softmax = nn.Softmax(dim=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        attn_weight = self.attn_softmax(self.attn_layer(inputs))
        return inputs * attn_weight


class TemporalMLPEncoderBlock(nn.Module):
    """Residual MLP encoder with local token mixing and gated channel mixing."""

    def __init__(self, d_model: int, dropout: float, ff_mult: int = 2):
        super().__init__()
        hidden_dim = d_model * ff_mult
        token_hidden_dim = hidden_dim

        self.token_norm = nn.LayerNorm(d_model * 3)
        self.token_mlp = nn.Sequential(
            nn.Linear(d_model * 3, token_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.channel_norm = nn.LayerNorm(d_model)
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.global_norm = nn.LayerNorm(d_model)
        self.global_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        left = torch.roll(x, shifts=1, dims=1)
        right = torch.roll(x, shifts=-1, dims=1)
        left[:, 0, :] = x[:, 0, :]
        right[:, -1, :] = x[:, -1, :]

        token_context = torch.cat([left, x, right], dim=-1)
        x = x + self.token_mlp(self.token_norm(token_context))
        x = x + self.channel_mlp(self.channel_norm(x))

        global_context = self.global_norm(x.mean(dim=1, keepdim=True))
        x = x + self.global_mlp(global_context).expand_as(x)
        return self.out_norm(x), None


@dataclass
class ModelOutput:
    loss: torch.Tensor
    recon: torch.Tensor
    recon_loss: torch.Tensor
    contrast_loss: torch.Tensor
    anomaly_score: Optional[torch.Tensor] = None
    point_recon_error: Optional[torch.Tensor] = None
    point_channel_error: Optional[torch.Tensor] = None
    patch_anomaly_score: Optional[torch.Tensor] = None
    patch_recon_error: Optional[torch.Tensor] = None
    patch_channel_error: Optional[torch.Tensor] = None


class PAMM(nn.Module):
    """
    PAMM backbone.

    The current version reconstructs the original input window directly and
    uses random disjoint patch masking.
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        loss_cfg = config["loss"]
        score_cfg = config.get("score", {})

        self.input_dim = int(model_cfg["input_dim"])
        self.patch_size = int(model_cfg["patch_size"])
        self.patch_stride = int(model_cfg["patch_stride"])
        self.d_model = int(model_cfg["d_model"])
        self.masking_ratio = float(model_cfg["masking_ratio"])
        self.num_mask_views = max(1, int(model_cfg.get("num_mask_views", 5)))
        self.mask_ratio_per_view = float(
            model_cfg.get("mask_ratio_per_view", self.masking_ratio / max(self.num_mask_views, 1))
        )
        self.temperature = float(model_cfg["temperature"])
        self.use_revin = bool(model_cfg["use_revin"])

        self.revin = RevIN(self.input_dim) if self.use_revin else None
        self.patch_embed = PatchEmbed(self.input_dim, self.patch_size, self.patch_stride, self.d_model)
        self.position = nn.Parameter(torch.randn(1, 512, self.d_model) * 0.02)
        self.multivariate_token_gate = (
            MultivariatePatchTokenGate(self.d_model, self.d_model)
            if self.input_dim > 1
            else nn.Identity()
        )

        self.use_locality_bias = bool(model_cfg.get("use_locality_bias", True))
        attention_cls = LocalityBiasedAttention if self.use_locality_bias else MHAttention
        self.importance_norm = nn.LayerNorm(self.d_model)
        self.importance_attention = attention_cls(
            d_model=self.d_model,
            num_heads=int(model_cfg["num_heads"]),
            dropout=float(model_cfg["dropout"]),
            locality_strength=float(model_cfg["locality_strength"]),
        )

        self.encoder = nn.ModuleList(
            [
                TemporalMLPEncoderBlock(
                    d_model=self.d_model,
                    dropout=float(model_cfg["dropout"]),
                    ff_mult=int(model_cfg.get("ff_mult", 2)),
                )
                for _ in range(int(model_cfg["num_layers"]))
            ]
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        decoder_hidden = self.d_model * int(model_cfg.get("decoder_hidden_multiplier", 2))

        self.token_reconstructor = nn.Sequential(
            nn.LayerNorm(self.d_model * 2),
            nn.Linear(self.d_model * 2, decoder_hidden),
            nn.GELU(),
            nn.Dropout(float(model_cfg["dropout"])),
            nn.Linear(decoder_hidden, self.d_model),
        )

        self.contrast_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )

        self.patch_decoder = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.patch_size * self.input_dim),
        )

        self.recon_weight = float(loss_cfg["recon_weight"])
        self.contrast_weight = float(loss_cfg["contrast_weight"])

        self.patch_channel_topk_weight = float(score_cfg.get("patch_channel_topk_weight", 0.2))
        self.patch_channel_topk_ratio = float(score_cfg.get("patch_channel_topk_ratio", 0.2))
        self.point_aggregate_mode = str(score_cfg.get("point_aggregate_mode", "center_weighted")).lower()
        self.point_center_power = float(score_cfg.get("point_center_power", 0.5))
        self.point_gaussian_sigma = float(score_cfg.get("point_gaussian_sigma", 0.0))
        self.global_context = ContextProtoBank(10, self.d_model)

    def _attention_reference(
            self,
            tokens: torch.Tensor,
            norm: nn.LayerNorm,
            attention: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_input = norm(tokens)
        attn_out, attn_map = attention(attn_input)
        return tokens + attn_out, attn_map

    def _make_disjoint_masks(
            self,
            batch_size: int,
            num_tokens: int,
            device: torch.device,
            dtype: torch.dtype,
            use_masking: bool = True,
    ) -> list[torch.Tensor]:
        masks = [torch.zeros(batch_size, num_tokens, device=device, dtype=dtype) for _ in range(self.num_mask_views)]
        if (not use_masking) or self.mask_ratio_per_view <= 0:
            return masks

        target_total = int(round(num_tokens * self.mask_ratio_per_view * self.num_mask_views))
        target_total = max(self.num_mask_views, min(num_tokens, target_total))
        base_size = target_total // self.num_mask_views
        remainder = target_total % self.num_mask_views
        view_sizes = [base_size + (1 if idx < remainder else 0) for idx in range(self.num_mask_views)]

        for batch_idx in range(batch_size):
            candidate_idx = torch.randperm(num_tokens, device=device)[:target_total]
            cursor = 0
            for view_idx, view_size in enumerate(view_sizes):
                if view_size <= 0:
                    continue
                current_idx = candidate_idx[cursor: cursor + view_size]
                masks[view_idx][batch_idx].scatter_(0, current_idx, 1.0)
                cursor += view_size
        return masks

    def _reconstruct_masked_tokens(
            self,
            attended_tokens: torch.Tensor,
            patch_masks: list[torch.Tensor],
            attn_map: torch.Tensor,
            mask_token: nn.Parameter,
            token_reconstructor: nn.Module,
    ) -> torch.Tensor:
        batch, num_tokens, _ = attended_tokens.shape
        output_tokens = attended_tokens.clone()
        mask_token = mask_token.expand(batch, num_tokens, -1)
        attn_weights = attn_map.mean(dim=1)

        for patch_mask in patch_masks:
            masked_tokens = torch.where(patch_mask.unsqueeze(-1).bool(), mask_token, attended_tokens)
            context_tokens = torch.matmul(attn_weights, masked_tokens)
            recon_input = torch.cat([masked_tokens, context_tokens], dim=-1)
            reconstructed_tokens = token_reconstructor(recon_input)
            hidden_tokens = torch.where(patch_mask.unsqueeze(-1).bool(), reconstructed_tokens, attended_tokens)
            output_tokens = torch.where(patch_mask.unsqueeze(-1).bool(), hidden_tokens, output_tokens)

        return output_tokens

    def _decode_patch_values(self, tokens: torch.Tensor, patch_decoder: nn.Module) -> torch.Tensor:
        batch, patch_num, _ = tokens.shape
        return patch_decoder(tokens).view(batch, patch_num, self.patch_size, self.input_dim)

    def _aggregate_decoded_patches(self, decoded: torch.Tensor, series_length: int) -> torch.Tensor:
        batch, patch_num, _, _ = decoded.shape
        recon = torch.zeros(batch, series_length, self.input_dim, device=decoded.device)
        count = torch.zeros(batch, series_length, self.input_dim, device=decoded.device)
        for idx in range(patch_num):
            start = idx * self.patch_stride
            end = start + self.patch_size
            recon[:, start:end, :] += decoded[:, idx]
            count[:, start:end, :] += 1
        return recon / count.clamp_min(1.0)

    def _extract_target_patches(self, raw: torch.Tensor) -> torch.Tensor:
        patches = raw.unfold(dimension=1, size=self.patch_size, step=self.patch_stride)
        return patches.contiguous().permute(0, 1, 3, 2)

    def _point_aggregate_weights(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.point_aggregate_mode == "center_weighted":
            positions = torch.arange(self.patch_size, device=device, dtype=dtype)
            center = (self.patch_size - 1) / 2.0
            if self.patch_size <= 1:
                patch_weights = torch.ones(1, device=device, dtype=dtype)
            else:
                patch_weights = 1.0 - (positions - center).abs() / (center + 1.0)
                if self.point_center_power != 1.0:
                    patch_weights = patch_weights.pow(self.point_center_power)
                patch_weights = patch_weights.clamp_min(1e-3)
        elif self.point_aggregate_mode in {"mean", "average", "uniform"}:
            patch_weights = torch.ones(self.patch_size, device=device, dtype=dtype)
        elif self.point_aggregate_mode in {"gaussian", "gaussian_weighted"}:
            positions = torch.arange(self.patch_size, device=device, dtype=dtype)
            center = (self.patch_size - 1) / 2.0
            sigma = self.point_gaussian_sigma if self.point_gaussian_sigma > 0 else max(self.patch_size / 6.0, 1.0)
            patch_weights = torch.exp(-0.5 * ((positions - center) / sigma).pow(2)).clamp_min(1e-3)
        else:
            raise ValueError(f"Unsupported point_aggregate_mode: {self.point_aggregate_mode}")
        return patch_weights

    def _patch_scores_to_point_scores(self, patch_scores: torch.Tensor, series_length: int) -> torch.Tensor:
        batch, patch_num = patch_scores.shape
        point_score = torch.zeros(batch, series_length, device=patch_scores.device, dtype=patch_scores.dtype)
        point_count = torch.zeros(batch, series_length, device=patch_scores.device, dtype=patch_scores.dtype)
        patch_weights = self._point_aggregate_weights(patch_scores.device, patch_scores.dtype)
        for idx in range(patch_num):
            start = idx * self.patch_stride
            end = start + self.patch_size
            weighted_scores = patch_scores[:, idx].unsqueeze(-1) * patch_weights.unsqueeze(0)
            point_score[:, start:end] += weighted_scores
            point_count[:, start:end] += patch_weights.unsqueeze(0)
        return point_score / point_count.clamp_min(1e-6)

    def _patch_channel_scores_to_point_scores(
            self,
            patch_channel_scores: torch.Tensor,
            series_length: int,
    ) -> torch.Tensor:
        batch, patch_num, channels = patch_channel_scores.shape
        point_score = torch.zeros(
            batch,
            series_length,
            channels,
            device=patch_channel_scores.device,
            dtype=patch_channel_scores.dtype,
        )
        point_count = torch.zeros(
            batch,
            series_length,
            1,
            device=patch_channel_scores.device,
            dtype=patch_channel_scores.dtype,
        )
        patch_weights = self._point_aggregate_weights(
            patch_channel_scores.device,
            patch_channel_scores.dtype,
        ).view(1, self.patch_size, 1)
        for idx in range(patch_num):
            start = idx * self.patch_stride
            end = start + self.patch_size
            weighted_scores = patch_channel_scores[:, idx, :].unsqueeze(1) * patch_weights
            point_score[:, start:end, :] += weighted_scores
            point_count[:, start:end, :] += patch_weights
        return point_score / point_count.clamp_min(1e-6)

    @staticmethod
    def _encode_tokens(tokens: torch.Tensor, encoder: nn.ModuleList) -> torch.Tensor:
        hidden = tokens
        for block in encoder:
            hidden, _ = block(hidden)
        return hidden

    def _masked_mean_pool(self, tokens: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        if token_mask is None:
            return tokens.mean(dim=1)
        weights = token_mask.unsqueeze(-1).to(dtype=tokens.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (tokens * weights).sum(dim=1) / denom

    def _contrastive_loss(
            self,
            clean_hidden: torch.Tensor,
            masked_hidden: torch.Tensor,
            contrast_proj: nn.Module,
            token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        clean = F.normalize(contrast_proj(self._masked_mean_pool(clean_hidden, token_mask)), dim=-1)
        masked = F.normalize(contrast_proj(self._masked_mean_pool(masked_hidden, token_mask)), dim=-1)
        logits_masked = torch.matmul(masked, clean.transpose(0, 1)) / self.temperature
        logits_clean = torch.matmul(clean, masked.transpose(0, 1)) / self.temperature
        labels = torch.arange(logits_masked.size(0), device=logits_masked.device)
        loss_masked = F.cross_entropy(logits_masked, labels)
        loss_clean = F.cross_entropy(logits_clean, labels)
        return 0.5 * (loss_masked + loss_clean)

    def _score_reconstruction(
            self,
            raw: torch.Tensor,
            recon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_patches = self._extract_target_patches(recon)
        target_patches = self._extract_target_patches(raw)
        patch_channel_error = (recon_patches - target_patches).pow(2).mean(dim=2)
        point_channel_error = self._patch_channel_scores_to_point_scores(patch_channel_error, raw.size(1))
        patch_recon_error = patch_channel_error.mean(dim=-1)

        if patch_channel_error.size(-1) > 1 and self.patch_channel_topk_weight > 0:
            ratio = min(max(self.patch_channel_topk_ratio, 0.0), 1.0)
            k = max(
                1,
                min(
                    patch_channel_error.size(-1),
                    int(round(patch_channel_error.size(-1) * ratio)),
                ),
            )
            patch_topk_error = patch_channel_error.topk(k, dim=-1).values.mean(dim=-1)
            patch_spike = (patch_topk_error - patch_recon_error).clamp_min(0.0)
            patch_anomaly_score = patch_recon_error + self.patch_channel_topk_weight * patch_spike
        else:
            patch_anomaly_score = patch_recon_error

        point_recon_error = self._patch_scores_to_point_scores(patch_anomaly_score, raw.size(1))
        return (
            point_recon_error,
            point_channel_error,
            patch_anomaly_score,
            patch_recon_error,
            patch_channel_error,
        )

    def forward(self, x: torch.Tensor, use_masking: bool = True) -> ModelOutput:
        raw = x
        if self.use_revin and self.revin is not None:
            x = self.revin(x, "norm")

        clean_tokens, patch_num = self.patch_embed(x)
        clean_tokens = clean_tokens + self.position[:, :patch_num, :]
        # clean_tokens = self.multivariate_token_gate(clean_tokens)
        attended_tokens, importance_attn = self._attention_reference(
            tokens=clean_tokens,
            norm=self.importance_norm,
            attention=self.importance_attention,
        )

        patch_masks = self._make_disjoint_masks(
            batch_size=attended_tokens.size(0),
            num_tokens=attended_tokens.size(1),
            device=attended_tokens.device,
            dtype=attended_tokens.dtype,
            use_masking=use_masking,
        )
        output_tokens = self._reconstruct_masked_tokens(
            attended_tokens=attended_tokens,
            patch_masks=patch_masks,
            attn_map=importance_attn,
            mask_token=self.mask_token,
            token_reconstructor=self.token_reconstructor,
        )
        hidden = self._encode_tokens(output_tokens, self.encoder)
        decoded_patches = self._decode_patch_values(hidden, self.patch_decoder)
        recon = self._aggregate_decoded_patches(decoded_patches, x.size(1))

        if self.use_revin and self.revin is not None:
            recon = self.revin(recon, "denorm")

        recon_error = (recon - raw) ** 2
        recon_loss = recon_error.mean()
        if self.contrast_weight > 0 and any(patch_mask.sum() > 0 for patch_mask in patch_masks):
            union_mask = torch.stack(patch_masks, dim=0).sum(dim=0).clamp_max(1.0)
            contrast_loss = self._contrastive_loss(
                clean_hidden=clean_tokens,
                masked_hidden=output_tokens,
                contrast_proj=self.contrast_proj,
                token_mask=union_mask,
            )
        else:
            contrast_loss = recon_loss.new_zeros(())

        (
            point_recon_error,
            point_channel_error,
            patch_anomaly_score,
            patch_recon_error,
            patch_channel_error,
        ) = self._score_reconstruction(
            raw=raw,
            recon=recon,
        )

        anomaly_score = point_recon_error

        total_loss = (
                self.recon_weight * recon_loss
                + self.contrast_weight * contrast_loss
        )

        return ModelOutput(
            loss=total_loss,
            recon=recon,
            recon_loss=recon_loss,
            contrast_loss=contrast_loss,
            anomaly_score=anomaly_score,
            point_recon_error=point_recon_error,
            point_channel_error=point_channel_error,
            patch_anomaly_score=patch_anomaly_score,
            patch_recon_error=patch_recon_error,
            patch_channel_error=patch_channel_error,
        )
