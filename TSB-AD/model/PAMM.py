import copy
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torchinfo
import tqdm
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model_pamm import PAMM as PAMMBackbone

from ..utils.dataset import ReconstructDataset
from ..utils.torch_utility import EarlyStoppingTorch, get_gpu


class PerChannelTemporalCNNAutoEncoder(nn.Module):
    """Learn a compact temporal-pattern embedding for each channel window."""

    def __init__(
        self,
        win_size: int,
        hidden_dim: int = 32,
        embedding_dim: int = 16,
        kernel_size: int = 5,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.win_size = int(win_size)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.to_embedding = nn.Linear(hidden_dim, embedding_dim)
        self.decoder = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.win_size),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, length, channels = x.shape
        channel_series = x.permute(0, 2, 1).reshape(batch * channels, 1, length)
        hidden = self.encoder(channel_series).squeeze(-1)
        embedding = self.to_embedding(hidden)
        recon = self.decoder(embedding).view(batch, channels, length).permute(0, 2, 1)
        return embedding.view(batch, channels, -1), recon


class PerChannelContextProtoBank(nn.Module):
    """ContextProtoBank-style prototype selector for per-channel CNN embeddings."""

    def __init__(self, num_channels: int, num_contexts: int, embedding_dim: int):
        super().__init__()
        self.num_channels = int(num_channels)
        self.num_contexts = int(num_contexts)
        self.embedding_dim = int(embedding_dim)
        self.context_prototypes = nn.Parameter(
            torch.randn(self.num_channels, self.num_contexts, self.embedding_dim) * 0.02
        )
        self.context_selector = nn.Linear(self.embedding_dim, self.num_contexts)

    def forward(
        self,
        embeddings: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selector_logits = self.context_selector(embeddings)
        if self.training:
            soft_weights = F.gumbel_softmax(selector_logits, tau=tau, hard=hard, dim=-1)
        else:
            soft_weights = torch.softmax(selector_logits / max(float(tau), 1e-6), dim=-1)
        selected_context_protos = torch.einsum("bck,ckd->bcd", soft_weights, self.context_prototypes)

        cosine_sim = F.cosine_similarity(embeddings, selected_context_protos, dim=-1, eps=1e-8)
        deviation_score = 1.0 - cosine_sim
        return deviation_score, selected_context_protos, soft_weights


class PAMM:
    """
    TSB-AD adapter for the standalone PAMM backbone.

    Fairness constraints for benchmark integration:
    1. fit only on the provided normal prefix (`data_train`);
    2. normalize all splits with train-only statistics;
    3. project window scores back to the full series with an explicit mode.
    """

    def __init__(
        self,
        win_size: int = 128,
        input_c: int = 1,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-4,
        patience: int = 3,
        validation_size: float = 0.2,
        patch_size: int = 32,
        patch_stride: Optional[int] = 4,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        ff_mult: int = 2,
        dropout: float = 0.1,
        locality_strength: float = 1.0,
        use_locality_bias: bool = True,
        masking_ratio: float = 0.3,
        temperature: float = 0.2,
        contrast_weight: float = 0.01,
        use_revin: bool = True,
        external_standardization_enabled: bool = False,
        point_aggregate_mode: str = "mean",
        point_center_power: float = 1.5,
        point_gaussian_sigma: float = 0.0,
        score_projection_mode: str = "mean",
        score_projection_center_power: float = 1.0,
        diff_prejudge_enabled: bool = True,
        diff_prejudge_quantile: float = 0.95,
        diff_prejudge_cosine_quantile: float = 0.05,
        diff_prejudge_margin: float = 1.0,
        diff_prejudge_suppression: float = 0.0,
        patch_channel_topk_weight: float = 0.2,
        patch_channel_topk_ratio: float = 0.2,
        cnn_pattern_enabled: bool = True,
        cnn_pattern_score_weight: float = 0.2,
        cnn_pattern_hidden_dim: int = 32,
        cnn_pattern_embedding_dim: int = 16,
        cnn_pattern_num_contexts: int = 10,
        cnn_pattern_proto_tau: float = 1.0,
        cnn_pattern_proto_loss_weight: float = 0.1,
        zero_drop_rule_enabled: bool = False,
        zero_drop_rule_weight: float = 0.0,
        zero_drop_rule_radius: int = 2,
        zero_drop_zero_frac: float = 0.1,
        use_cuda: bool = True,
        gpu_id: Optional[int] = 2,
        seed: int = 2024,
        save_proto_analysis: bool = True,
        proto_analysis_dir: Optional[str] = None,
    ):
        super().__init__()

        self.win_size = int(win_size)
        self.input_c = int(input_c)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.patience = int(patience)
        self.validation_size = float(validation_size)
        self.patch_size = max(2, min(int(patch_size), self.win_size))
        self.patch_stride = int(patch_stride) if patch_stride is not None else 2
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.ff_mult = int(ff_mult)
        self.dropout = float(dropout)
        self.locality_strength = float(locality_strength)
        self.use_locality_bias = bool(use_locality_bias)
        self.masking_ratio = float(masking_ratio)
        self.temperature = float(temperature)
        self.contrast_weight = float(contrast_weight)
        self.use_revin = bool(use_revin)
        self.external_standardization_enabled = bool(external_standardization_enabled)
        self.point_aggregate_mode = str(point_aggregate_mode)
        self.point_center_power = float(point_center_power)
        self.point_gaussian_sigma = float(point_gaussian_sigma)
        self.score_projection_mode = str(score_projection_mode).lower()
        self.score_projection_center_power = float(score_projection_center_power)
        valid_projection_modes = {
            "last",
            "center",
            "mean",
            "average",
            "uniform",
            "center_weighted",
            "gaussian",
            "gaussian_weighted",
        }
        if self.score_projection_mode not in valid_projection_modes:
            raise ValueError(f"Unsupported score_projection_mode: {self.score_projection_mode}")
        self._projection_mode_logged = False
        self.diff_prejudge_enabled = bool(diff_prejudge_enabled)
        self.diff_prejudge_quantile = min(max(float(diff_prejudge_quantile), 0.0), 1.0)
        self.diff_prejudge_cosine_quantile = min(max(float(diff_prejudge_cosine_quantile), 0.0), 1.0)
        self.diff_prejudge_margin = max(float(diff_prejudge_margin), 0.0)
        self.diff_prejudge_suppression = min(max(float(diff_prejudge_suppression), 0.0), 1.0)
        self.diff_prejudge_threshold = None
        self.diff_prejudge_cosine_ref = None
        self.diff_prejudge_cosine_threshold = None
        self.patch_channel_topk_weight = max(float(patch_channel_topk_weight), 0.0)
        self.patch_channel_topk_ratio = min(max(float(patch_channel_topk_ratio), 0.0), 1.0)
        self.cnn_pattern_enabled = bool(cnn_pattern_enabled)
        self.cnn_pattern_score_weight = max(float(cnn_pattern_score_weight), 0.0)
        self.cnn_pattern_hidden_dim = max(1, int(cnn_pattern_hidden_dim))
        self.cnn_pattern_embedding_dim = max(1, int(cnn_pattern_embedding_dim))
        self.cnn_pattern_num_contexts = max(1, int(cnn_pattern_num_contexts))
        self.cnn_pattern_proto_tau = max(float(cnn_pattern_proto_tau), 1e-6)
        self.cnn_pattern_proto_loss_weight = max(float(cnn_pattern_proto_loss_weight), 0.0)
        self.zero_drop_rule_enabled = bool(zero_drop_rule_enabled)
        self.zero_drop_rule_weight = float(zero_drop_rule_weight)
        self.zero_drop_rule_radius = max(1, int(zero_drop_rule_radius))
        self.zero_drop_zero_frac = float(zero_drop_zero_frac)
        self.gpu_id = gpu_id
        self.seed = int(seed)
        self.save_proto_analysis_enabled = bool(save_proto_analysis)
        self.proto_analysis_dir = proto_analysis_dir
        self.__proto_analysis = None

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        self.cuda = use_cuda and torch.cuda.is_available()
        if self.cuda and self.gpu_id is not None:
            device_count = torch.cuda.device_count()
            resolved_gpu_id = self.gpu_id
            if resolved_gpu_id < 0 or resolved_gpu_id >= device_count:
                print(
                    f"----- Requested GPU {resolved_gpu_id} is unavailable; "
                    f"falling back to GPU 0 out of {device_count} visible GPU(s) -----"
                )
                resolved_gpu_id = 0
            self.device = torch.device(f"cuda:{resolved_gpu_id}")
            torch.cuda.set_device(self.device)
            print(f"----- Using GPU {resolved_gpu_id}: {torch.cuda.get_device_name(self.device)} -----")
        else:
            self.device = get_gpu(self.cuda)

        self.train_mean = None
        self.train_std = None
        self.train_abs_q95 = None
        self.train_recon_score_median = None
        self.train_recon_score_q95 = None
        self.train_channel_recon_score_q95 = None
        self.train_cnn_pattern_score_q95 = None
        self.recon_background_quantile = 1.0
        self.__anomaly_score = None
        self.__channel_anomaly_score = None
        self.__recon_channel_anomaly_score = None
        self.__cnn_pattern_score = None
        self.__cnn_pattern_contribution_score = None
        self._reset_proto_analysis_cache()

        self.model = PAMMBackbone(self._build_config()).float().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.cnn_pattern_model = None
        self.cnn_pattern_proto_bank = None
        self.cnn_pattern_optimizer = None
        if self.cnn_pattern_enabled and self.cnn_pattern_score_weight > 0:
            self.cnn_pattern_model = PerChannelTemporalCNNAutoEncoder(
                win_size=self.win_size,
                hidden_dim=self.cnn_pattern_hidden_dim,
                embedding_dim=self.cnn_pattern_embedding_dim,
            ).float().to(self.device)
            self.cnn_pattern_proto_bank = PerChannelContextProtoBank(
                num_channels=self.input_c,
                num_contexts=self.cnn_pattern_num_contexts,
                embedding_dim=self.cnn_pattern_embedding_dim,
            ).float().to(self.device)
            self.cnn_pattern_optimizer = optim.Adam(
                list(self.cnn_pattern_model.parameters()) + list(self.cnn_pattern_proto_bank.parameters()),
                lr=self.lr,
            )
        self.early_stopping = EarlyStoppingTorch(None, patience=self.patience)
        self.input_shape = (self.batch_size, self.win_size, self.input_c)

    def _build_config(self) -> dict:
        return {
            "model": {
                "input_dim": self.input_c,
                "patch_size": self.patch_size,
                "patch_stride": self.patch_stride,
                "d_model": self.d_model,
                "masking_ratio": 1.0,
                "mask_ratio_per_view": 0.2,
                "temperature": self.temperature,
                "use_revin": self.use_revin,
                "num_mask_views": 5,
                "num_heads": self.num_heads,
                "ff_mult": self.ff_mult,
                "dropout": self.dropout,
                "locality_strength": self.locality_strength,
                "use_locality_bias": self.use_locality_bias,
                "num_layers": self.num_layers,
            },
            "loss": {
                "recon_weight": 1.0,
                "contrast_weight": self.contrast_weight,
            },
            "score": {
                "patch_channel_topk_weight": self.patch_channel_topk_weight,
                "patch_channel_topk_ratio": self.patch_channel_topk_ratio,
                "point_aggregate_mode": self.point_aggregate_mode,
                "point_center_power": self.point_center_power,
                "point_gaussian_sigma": self.point_gaussian_sigma,
            },
        }

    def _normalize_train(self, data: np.ndarray) -> np.ndarray:
        if not self.external_standardization_enabled:
            self.train_mean = None
            self.train_std = None
            return np.asarray(data, dtype=np.float32)
        self.train_mean = np.mean(data, axis=0)
        self.train_std = np.std(data, axis=0)
        self.train_std = np.where(self.train_std < 1e-8, 1.0, self.train_std)
        return (data - self.train_mean) / self.train_std

    def _normalize_eval(self, data: np.ndarray) -> np.ndarray:
        if not self.external_standardization_enabled:
            return np.asarray(data, dtype=np.float32)
        if self.train_mean is None or self.train_std is None:
            raise RuntimeError("PAMM must be fitted before calling decision_function.")
        return (data - self.train_mean) / self.train_std

    def _prepare_zero_drop_stats(self, data: np.ndarray):
        abs_data = np.abs(data)
        self.train_abs_q95 = np.quantile(abs_data, 0.95, axis=0)

    @staticmethod
    def _window_diff_energy_from_windows(windows: np.ndarray) -> np.ndarray:
        windows = np.asarray(windows, dtype=np.float32)
        if windows.ndim == 2:
            windows = windows[:, :, None]
        if windows.shape[1] <= 1:
            return np.zeros(windows.shape[0], dtype=np.float32)
        diff = np.diff(windows, axis=1)
        return np.mean(np.square(diff), axis=(1, 2)).astype(np.float32)

    @staticmethod
    def _flatten_windows(windows: np.ndarray) -> np.ndarray:
        windows = np.asarray(windows, dtype=np.float32)
        if windows.ndim == 2:
            windows = windows[:, :, None]
        return windows.reshape(windows.shape[0], -1)

    @staticmethod
    def _cosine_similarity_to_ref(flat_windows: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        ref_norm = np.linalg.norm(ref)
        window_norms = np.linalg.norm(flat_windows, axis=1)
        denom = np.maximum(window_norms * ref_norm, eps)
        return np.matmul(flat_windows, ref) / denom

    def _calibrate_diff_prejudge(self, data: np.ndarray):
        if not self.diff_prejudge_enabled:
            self.diff_prejudge_threshold = None
            self.diff_prejudge_cosine_ref = None
            self.diff_prejudge_cosine_threshold = None
            return
        if len(data) < self.win_size:
            self.diff_prejudge_threshold = None
            self.diff_prejudge_cosine_ref = None
            self.diff_prejudge_cosine_threshold = None
            return

        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        windows = np.lib.stride_tricks.sliding_window_view(data, window_shape=self.win_size, axis=0)
        windows = np.moveaxis(windows, -1, 1)
        diff_energy = self._window_diff_energy_from_windows(windows)
        base_threshold = float(np.quantile(diff_energy, self.diff_prejudge_quantile))
        self.diff_prejudge_threshold = base_threshold * self.diff_prejudge_margin

        flat_windows = self._flatten_windows(windows)
        cosine_ref = flat_windows.mean(axis=0).astype(np.float32)
        if np.linalg.norm(cosine_ref) <= 1e-8:
            self.diff_prejudge_cosine_ref = None
            self.diff_prejudge_cosine_threshold = None
            cosine_threshold_msg = "disabled"
        else:
            train_cosine = self._cosine_similarity_to_ref(flat_windows, cosine_ref)
            self.diff_prejudge_cosine_ref = cosine_ref
            self.diff_prejudge_cosine_threshold = float(
                np.quantile(train_cosine, self.diff_prejudge_cosine_quantile)
            )
            cosine_threshold_msg = f"{self.diff_prejudge_cosine_threshold:.6f}"
        print(
            f"PAMM diff+cosine pre-judge: diff_q={self.diff_prejudge_quantile:.3f}, "
            f"diff_threshold={self.diff_prejudge_threshold:.6f}, "
            f"cos_q={self.diff_prejudge_cosine_quantile:.3f}, "
            f"cos_threshold={cosine_threshold_msg}, "
            f"suppression={self.diff_prejudge_suppression:.3f}"
        )

    def _diff_prejudge_gate(self, batch_windows: torch.Tensor) -> torch.Tensor:
        if (not self.diff_prejudge_enabled) or self.diff_prejudge_threshold is None:
            return torch.ones(batch_windows.size(0), device=batch_windows.device, dtype=batch_windows.dtype)
        if batch_windows.size(1) <= 1:
            return torch.full(
                (batch_windows.size(0),),
                self.diff_prejudge_suppression,
                device=batch_windows.device,
                dtype=batch_windows.dtype,
            )

        diff = batch_windows[:, 1:, :] - batch_windows[:, :-1, :]
        diff_energy = diff.pow(2).mean(dim=(1, 2))
        diff_stable = diff_energy <= self.diff_prejudge_threshold
        if self.diff_prejudge_cosine_ref is None or self.diff_prejudge_cosine_threshold is None:
            cosine_similar = torch.ones_like(diff_stable, dtype=torch.bool)
        else:
            flat_windows = batch_windows.reshape(batch_windows.size(0), -1)
            ref = torch.as_tensor(
                self.diff_prejudge_cosine_ref,
                device=batch_windows.device,
                dtype=batch_windows.dtype,
            ).view(1, -1)
            numerator = (flat_windows * ref).sum(dim=1)
            denom = flat_windows.norm(dim=1) * ref.norm(dim=1).clamp_min(1e-8)
            cosine_score = numerator / denom.clamp_min(1e-8)
            cosine_similar = cosine_score >= self.diff_prejudge_cosine_threshold
        normal_window = diff_stable & cosine_similar
        keep_weight = torch.ones(batch_windows.size(0), device=batch_windows.device, dtype=batch_windows.dtype)
        suppress_weight = torch.full_like(keep_weight, self.diff_prejudge_suppression)
        return torch.where(normal_window, suppress_weight, keep_weight)

    def _zero_drop_rule_score(self, data: np.ndarray) -> np.ndarray:
        """
        规则项：短暂归零检测。

        思路是：
        1. 当前点自身绝对值很小，接近 0；
        2. 但邻域上下文绝对值明显更高；
        3. 则认为这更像“传感器短暂掉线/断连归零”，而不是正常低谷。
        """
        data = np.asarray(data, dtype=np.float32)
        if not self.zero_drop_rule_enabled:
            return np.zeros(len(data), dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        if len(data) == 0:
            return np.zeros(0, dtype=np.float32)

        abs_data = np.abs(data)
        radius = self.zero_drop_rule_radius
        padded = np.pad(abs_data, ((radius, radius), (0, 0)), mode="edge")
        local_ref = np.zeros_like(abs_data)
        neighbor_count = 0
        for offset in range(-radius, radius + 1):
            if offset == 0:
                continue
            start = radius + offset
            local_ref += padded[start : start + len(abs_data)]
            neighbor_count += 1
        local_ref /= max(neighbor_count, 1)

        train_scale = np.maximum(np.asarray(self.train_abs_q95, dtype=np.float32), 1e-6)
        zero_scale = np.maximum(train_scale * self.zero_drop_zero_frac, 1e-6)

        near_zero = np.exp(-abs_data / zero_scale)
        context_drop = np.clip((local_ref - abs_data) / (local_ref + 1e-6), 0.0, 1.0)
        context_strength = np.clip(local_ref / (train_scale + 1e-6), 0.0, 3.0)
        channel_scores = near_zero * context_drop * context_strength

        if channel_scores.shape[1] == 1:
            return channel_scores[:, 0].astype(np.float32)
        return channel_scores.max(axis=1).astype(np.float32)

    def _make_loader(self, data: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = ReconstructDataset(data, window_size=self.win_size, normalize=False)
        return DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0)

    def _cnn_pattern_active(self) -> bool:
        return (
            self.input_c > 1
            and
            self.cnn_pattern_enabled
            and self.cnn_pattern_score_weight > 0
            and self.cnn_pattern_model is not None
            and self.cnn_pattern_proto_bank is not None
        )

    def _train_cnn_pattern_batch(self, batch_x: torch.Tensor) -> Optional[float]:
        if not self._cnn_pattern_active() or self.cnn_pattern_optimizer is None:
            return None

        self.cnn_pattern_model.train()
        self.cnn_pattern_proto_bank.train()
        self.cnn_pattern_optimizer.zero_grad()
        embedding, cnn_recon = self.cnn_pattern_model(batch_x)
        deviation_score, _, _ = self.cnn_pattern_proto_bank(
            embedding,
            tau=self.cnn_pattern_proto_tau,
            hard=False,
        )
        cnn_loss = (
            (cnn_recon - batch_x).pow(2).mean()
            + self.cnn_pattern_proto_loss_weight * deviation_score.mean()
        )
        cnn_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.cnn_pattern_model.parameters()) + list(self.cnn_pattern_proto_bank.parameters()),
            max_norm=1.0,
        )
        self.cnn_pattern_optimizer.step()
        return float(cnn_loss.item())

    def _reset_proto_analysis_cache(self):
        """Clear cached prototype-analysis tensors from the latest inference pass."""
        self.__proto_analysis = None

    def _build_proto_analysis_payload(
            self,
            window_deviation_scores: list,
            window_soft_weights: list,
            window_selected_protos: list,
            window_embeddings: list,
            series_length: int,
            prefix: str,
    ) -> Optional[dict]:
        """
        Pack full-sequence prototype statistics into numpy arrays.

        Saved fields:
            window_deviation_scores: [N_w, C]
            window_soft_weights: [N_w, C, K]
            window_selected_protos: [N_w, C, D]
            window_embeddings: [N_w, C, D]
            point_deviation_scores: [T, C]
            hard_assignments: [N_w, C]
            usage_counts: [C, K]
            soft_usage: [C, K]
            context_prototypes: [C, K, D]
            context_selector_weight: [K, D]
            context_selector_bias: [K]
        """
        if not window_soft_weights:
            return None

        deviation = np.concatenate(window_deviation_scores, axis=0).astype(np.float32)
        soft_weights = np.concatenate(window_soft_weights, axis=0).astype(np.float32)
        selected_protos = np.concatenate(window_selected_protos, axis=0).astype(np.float32)
        embeddings = np.concatenate(window_embeddings, axis=0).astype(np.float32)

        hard_assignments = np.argmax(soft_weights, axis=-1).astype(np.int64)  # [N_w, C]

        num_windows, num_channels = hard_assignments.shape
        num_contexts = soft_weights.shape[-1]
        embedding_dim = embeddings.shape[-1]

        usage_counts = np.zeros((num_channels, num_contexts), dtype=np.int64)
        for c in range(num_channels):
            usage_counts[c] = np.bincount(
                hard_assignments[:, c],
                minlength=num_contexts,
            )[:num_contexts]

        # Soft usage is more meaningful than hard usage when the branch uses soft assignments.
        # Shape: [C, K]
        soft_usage = soft_weights.mean(axis=0).astype(np.float32)

        # Project window-level channel deviation back to point-level channel scores.
        # Shape: [T, C]
        point_deviation = self._project_window_channel_scores(
            window_channel_scores=deviation,
            series_length=series_length,
        ).astype(np.float32)

        # Save learned prototype bank and selector parameters.
        context_prototypes = None
        context_selector_weight = None
        context_selector_bias = None

        if self.cnn_pattern_proto_bank is not None:
            context_prototypes = (
                self.cnn_pattern_proto_bank.context_prototypes
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            context_selector_weight = (
                self.cnn_pattern_proto_bank.context_selector.weight
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            context_selector_bias = (
                self.cnn_pattern_proto_bank.context_selector.bias
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        payload = {
            # Metadata
            "prefix": str(prefix),
            "series_length": int(series_length),
            "win_size": int(self.win_size),
            "num_windows": int(num_windows),
            "num_channels": int(num_channels),
            "num_contexts": int(num_contexts),
            "embedding_dim": int(embedding_dim),
            "score_projection_mode": str(self.score_projection_mode),
            "cnn_pattern_proto_tau": float(self.cnn_pattern_proto_tau),
            "cnn_pattern_num_contexts": int(self.cnn_pattern_num_contexts),
            "cnn_pattern_embedding_dim": int(self.cnn_pattern_embedding_dim),

            # Window-level prototype statistics
            "window_deviation_scores": deviation,
            "window_soft_weights": soft_weights,
            "window_selected_protos": selected_protos,
            "window_embeddings": embeddings,
            "hard_assignments": hard_assignments,
            "usage_counts": usage_counts,
            "soft_usage": soft_usage,

            # Point-level projected prototype statistics
            "point_deviation_scores": point_deviation,

            # Learned prototype parameters
            "context_prototypes": context_prototypes,
            "context_selector_weight": context_selector_weight,
            "context_selector_bias": context_selector_bias,
        }

        return payload

    def _save_proto_analysis_payload(self, payload: Optional[dict], prefix: str):
        """
        Save prototype-analysis statistics to disk.

        Both .npz and .pt are saved:
            - .npz is convenient for numpy-based visualization.
            - .pt preserves the Python dictionary structure.
        """
        if payload is None:
            return

        if not self.save_proto_analysis_enabled:
            return

        if self.proto_analysis_dir is None:
            self.proto_analysis_dir = "./PAMM_proto_analysis"

        save_dir = Path(self.proto_analysis_dir).expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)

        npz_path = save_dir / f"{prefix}_proto_analysis.npz"
        np.savez_compressed(npz_path, **payload)

        pt_path = save_dir / f"{prefix}_proto_analysis.pt"
        torch.save(payload, pt_path)

        print(f"PAMM prototype analysis saved to: {npz_path}")
        print(f"PAMM prototype analysis saved to: {pt_path}")

        if "series_length" in payload:
            print(
                "PAMM prototype analysis summary: "
                f"prefix={payload.get('prefix')}, "
                f"series_length={payload.get('series_length')}, "
                f"num_windows={payload.get('num_windows')}, "
                f"num_channels={payload.get('num_channels')}, "
                f"num_contexts={payload.get('num_contexts')}, "
                f"embedding_dim={payload.get('embedding_dim')}"
            )

    def _cnn_pattern_window_deviation_scores_on_normalized(
            self,
            normalized_data: np.ndarray,
            collect_proto_analysis: bool = False,
            analysis_prefix: str = "eval",
    ) -> Optional[np.ndarray]:
        """
        Compute window-level CNN-pattern prototype deviation scores.

        If collect_proto_analysis=True, this function also saves full prototype
        analysis statistics for the exact input sequence `normalized_data`.
        Therefore, to obtain full-sequence prototype statistics, make sure
        decision_function(data) receives the full sequence.
        """
        if not self._cnn_pattern_active():
            return None

        loader = self._make_loader(normalized_data, shuffle=False)

        window_scores = []
        window_soft_weights = []
        window_selected_protos = []
        window_embeddings = []

        self.cnn_pattern_model.eval()
        self.cnn_pattern_proto_bank.eval()

        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.float().to(self.device)

                embedding, _ = self.cnn_pattern_model(batch_x)

                deviation_score, selected_context_protos, soft_weights = self.cnn_pattern_proto_bank(
                    embedding,
                    tau=self.cnn_pattern_proto_tau,
                    hard=False,
                )

                window_scores.append(deviation_score.detach().cpu().numpy())

                if collect_proto_analysis:
                    window_soft_weights.append(soft_weights.detach().cpu().numpy())
                    window_selected_protos.append(selected_context_protos.detach().cpu().numpy())
                    window_embeddings.append(embedding.detach().cpu().numpy())

        if not window_scores:
            return None

        if collect_proto_analysis:
            payload = self._build_proto_analysis_payload(
                window_deviation_scores=window_scores,
                window_soft_weights=window_soft_weights,
                window_selected_protos=window_selected_protos,
                window_embeddings=window_embeddings,
                series_length=len(normalized_data),
                prefix=analysis_prefix,
            )

            self.__proto_analysis = payload

            if payload is not None:
                print(
                    "PAMM prototype analysis collected: "
                    f"prefix={analysis_prefix}, "
                    f"series_length={payload['series_length']}, "
                    f"num_windows={payload['num_windows']}, "
                    f"num_channels={payload['num_channels']}, "
                    f"num_contexts={payload['num_contexts']}, "
                    f"embedding_dim={payload['embedding_dim']}"
                )

            self._save_proto_analysis_payload(payload, prefix=analysis_prefix)

        return np.concatenate(window_scores, axis=0).astype(np.float32)

    def _project_window_channel_scores(
        self,
        window_channel_scores: np.ndarray,
        series_length: int,
    ) -> np.ndarray:
        window_channel_scores = np.asarray(window_channel_scores, dtype=np.float32)
        if window_channel_scores.ndim == 1:
            window_channel_scores = window_channel_scores[:, None]

        if self.score_projection_mode == "last":
            return self._pad_single_point_projection(
                scores=window_channel_scores,
                series_length=series_length,
                offset=self.win_size - 1,
            )
        if self.score_projection_mode == "center":
            return self._pad_single_point_projection(
                scores=window_channel_scores,
                series_length=series_length,
                offset=self.win_size // 2,
            )

        weights = self._projection_weights(torch.device("cpu"), torch.float32).cpu().numpy()
        projected_scores = np.zeros((series_length, window_channel_scores.shape[1]), dtype=np.float32)
        projected_counts = np.zeros((series_length, 1), dtype=np.float32)
        for idx, score in enumerate(window_channel_scores):
            start = idx
            end = start + self.win_size
            projected_scores[start:end] += score.reshape(1, -1) * weights[:, None]
            projected_counts[start:end] += weights[:, None]
        return projected_scores / np.maximum(projected_counts, 1e-6)

    def _calibrate_cnn_pattern_proto_scores(self, normalized_data: np.ndarray):
        """
        Calibrate channel-wise CNN-pattern prototype deviation thresholds
        on the training prefix.

        Important:
            collect_proto_analysis=False here, because calibration data are
            usually only the normal training prefix. Prototype analysis for
            visualization should be collected during decision_function(data)
            on the full evaluation sequence.
        """
        self.train_cnn_pattern_score_q95 = None

        window_deviation_scores = self._cnn_pattern_window_deviation_scores_on_normalized(
            normalized_data,
            collect_proto_analysis=False,
            analysis_prefix="calibration",
        )

        if window_deviation_scores is None:
            return

        train_pattern_scores = self._project_window_channel_scores(
            window_channel_scores=window_deviation_scores,
            series_length=len(normalized_data),
        )

        self.train_cnn_pattern_score_q95 = np.quantile(
            train_pattern_scores,
            self.recon_background_quantile,
            axis=0,
        ).astype(np.float32)

        print(
            "PAMM CNN-pattern calibration: "
            f"weight={self.cnn_pattern_score_weight:.6f}, "
            f"embedding_dim={self.cnn_pattern_embedding_dim}, "
            f"contexts={self.cnn_pattern_num_contexts}, "
            f"background_quantile={self.recon_background_quantile:.2f}, "
            f"mean_threshold={float(np.mean(self.train_cnn_pattern_score_q95)):.6f}, "
            f"max_threshold={float(np.max(self.train_cnn_pattern_score_q95)):.6f}"
        )

    def _cnn_pattern_scores_on_normalized(
            self,
            normalized_data: np.ndarray,
            collect_proto_analysis: bool = False,
            analysis_prefix: str = "eval",
    ) -> Optional[np.ndarray]:
        if (
                not self._cnn_pattern_active()
                or self.train_cnn_pattern_score_q95 is None
        ):
            return None

        window_deviation_scores = self._cnn_pattern_window_deviation_scores_on_normalized(
            normalized_data,
            collect_proto_analysis=collect_proto_analysis,
            analysis_prefix=analysis_prefix,
        )

        if window_deviation_scores is None:
            return None

        pattern_scores = self._project_window_channel_scores(
            window_channel_scores=window_deviation_scores,
            series_length=len(normalized_data),
        )
        return np.maximum(
            pattern_scores - self.train_cnn_pattern_score_q95.reshape(1, -1),
            0.0,
        ).astype(np.float32)

    def _projection_weights(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.score_projection_mode in {"mean", "average", "uniform"}:
            return torch.ones(self.win_size, device=device, dtype=dtype)
        if self.score_projection_mode == "center_weighted":
            positions = torch.arange(self.win_size, device=device, dtype=dtype)
            center = (self.win_size - 1) / 2.0
            if self.win_size <= 1:
                return torch.ones(1, device=device, dtype=dtype)
            weights = 1.0 - (positions - center).abs() / (center + 1.0)
            if self.score_projection_center_power != 1.0:
                weights = weights.pow(self.score_projection_center_power)
            return weights.clamp_min(1e-3)
        if self.score_projection_mode in {"gaussian", "gaussian_weighted"}:
            positions = torch.arange(self.win_size, device=device, dtype=dtype)
            center = (self.win_size - 1) / 2.0
            sigma = max(self.win_size / 6.0, 1.0)
            return torch.exp(-0.5 * ((positions - center) / sigma).pow(2)).clamp_min(1e-3)
        raise ValueError(f"Unsupported score_projection_mode: {self.score_projection_mode}")

    def _center_window_score(self, window_scores: torch.Tensor) -> torch.Tensor:
        center_idx = self.win_size // 2
        if self.win_size % 2 == 1:
            return window_scores[:, center_idx]
        left_center = center_idx - 1
        return 0.5 * (window_scores[:, left_center] + window_scores[:, center_idx])

    def _pad_single_point_projection(self, scores: np.ndarray, series_length: int, offset: int) -> np.ndarray:
        if scores.shape[0] >= series_length:
            return scores[:series_length]

        left_pad = min(max(offset, 0), series_length - scores.shape[0])
        right_pad = series_length - scores.shape[0] - left_pad
        left_values = np.broadcast_to(scores[0], (left_pad,) + scores.shape[1:]).copy()
        right_values = np.broadcast_to(scores[-1], (right_pad,) + scores.shape[1:]).copy()
        return np.concatenate(
            [
                left_values.astype(scores.dtype, copy=False),
                scores,
                right_values.astype(scores.dtype, copy=False),
            ]
        )

    def _fuse_channel_scores(self, channel_scores: np.ndarray) -> np.ndarray:
        channel_scores = np.asarray(channel_scores, dtype=np.float32)
        if channel_scores.ndim == 1:
            return channel_scores
        if channel_scores.shape[1] == 1:
            return channel_scores[:, 0]
        mean_score = channel_scores.mean(axis=1)
        if self.patch_channel_topk_weight <= 0:
            return mean_score.astype(np.float32)

        ratio = min(max(self.patch_channel_topk_ratio, 0.0), 1.0)
        k = max(1, min(channel_scores.shape[1], int(round(channel_scores.shape[1] * ratio))))
        topk_score = np.partition(channel_scores, -k, axis=1)[:, -k:].mean(axis=1)
        spike_score = np.maximum(topk_score - mean_score, 0.0)
        return (mean_score + self.patch_channel_topk_weight * spike_score).astype(np.float32)

    def _split_train_valid(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(data) < 2 * self.win_size:
            return data, data

        split_idx = int((1.0 - self.validation_size) * len(data))
        split_idx = max(self.win_size, min(split_idx, len(data) - self.win_size))
        return data[:split_idx], data[split_idx:]

    def fit(self, data: np.ndarray):
        if len(data) < self.win_size:
            raise ValueError(
                f"PAMM requires at least win_size={self.win_size} points, got {len(data)}."
            )

        raw_train_data = np.asarray(data, dtype=np.float32)
        self._prepare_zero_drop_stats(raw_train_data)
        train_data = self._normalize_train(raw_train_data)
        self._calibrate_diff_prejudge(train_data)
        ts_train, ts_valid = self._split_train_valid(train_data)

        train_loader = self._make_loader(ts_train, shuffle=True)
        valid_loader = self._make_loader(ts_valid, shuffle=False)
        if len(train_loader.dataset) == 0:
            raise ValueError("PAMM received an empty training window set.")

        best_state = None
        best_cnn_pattern_state = None
        best_val = float("inf")
        self.model.contrast_weight = self.contrast_weight
        print(f"----- PAMM training: masking=True, contrast_weight={self.contrast_weight:.6f} -----")
        if self._cnn_pattern_active():
            print(
                "----- PAMM CNN-pattern branch: "
                f"enabled=True, weight={self.cnn_pattern_score_weight:.6f}, "
                f"hidden_dim={self.cnn_pattern_hidden_dim}, "
                f"embedding_dim={self.cnn_pattern_embedding_dim}, "
                f"contexts={self.cnn_pattern_num_contexts}, "
                f"proto_loss_weight={self.cnn_pattern_proto_loss_weight:.6f} -----"
            )

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_losses = []
            cnn_train_losses = []

            loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
            for _, (batch_x, _) in loop:
                batch_x = batch_x.float().to(self.device)

                self.optimizer.zero_grad()
                output = self.model(batch_x, use_masking=True)
                output.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                cnn_loss = self._train_cnn_pattern_batch(batch_x)
                if cnn_loss is not None:
                    cnn_train_losses.append(cnn_loss)

                train_losses.append(output.loss.item())
                loop.set_description(f"Training Epoch [{epoch}/{self.epochs}]")
                postfix = {
                    "loss": output.loss.item(),
                    "avg_loss": float(np.mean(train_losses)),
                }
                if cnn_train_losses:
                    postfix["cnn_loss"] = float(np.mean(cnn_train_losses))
                loop.set_postfix(**postfix)

            self.model.eval()
            valid_losses = []
            loop = tqdm.tqdm(enumerate(valid_loader), total=len(valid_loader), leave=True)
            with torch.no_grad():
                for _, (batch_x, _) in loop:
                    batch_x = batch_x.float().to(self.device)
                    output = self.model(batch_x, use_masking=False)
                    valid_losses.append(output.loss.item())
                    loop.set_description(f"Valid Epoch [{epoch}/{self.epochs}]")
                    loop.set_postfix(loss=output.loss.item())

            valid_loss = float(np.mean(valid_losses)) if valid_losses else float(np.mean(train_losses))
            if valid_loss < best_val:
                best_val = valid_loss
                best_state = copy.deepcopy(self.model.state_dict())
                if self._cnn_pattern_active():
                    best_cnn_pattern_state = {
                        "model": copy.deepcopy(self.cnn_pattern_model.state_dict()),
                        "proto_bank": copy.deepcopy(self.cnn_pattern_proto_bank.state_dict()),
                    }

            self.early_stopping(valid_loss, self.model)
            print(
                f"Epoch {epoch}: train_loss={float(np.mean(train_losses)):.6f}, "
                f"valid_loss={valid_loss:.6f}"
            )

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        if best_cnn_pattern_state is not None and self._cnn_pattern_active():
            self.cnn_pattern_model.load_state_dict(best_cnn_pattern_state["model"])
            self.cnn_pattern_proto_bank.load_state_dict(best_cnn_pattern_state["proto_bank"])

        self._calibrate_cnn_pattern_proto_scores(train_data)
        if self.input_c > 1:
            train_channel_recon_scores = self._raw_decision_components_on_normalized(
                train_data,
                return_channel_scores=True,
            )
            self.train_channel_recon_score_q95 = np.quantile(
                train_channel_recon_scores,
                self.recon_background_quantile,
                axis=0,
            ).astype(np.float32)
            train_calibrated_channel_scores = np.maximum(
                train_channel_recon_scores - self.train_channel_recon_score_q95.reshape(1, -1),
                0.0,
            )
            train_cnn_pattern_scores = self._cnn_pattern_scores_on_normalized(train_data)
            if train_cnn_pattern_scores is not None:
                train_calibrated_channel_scores = (
                    train_calibrated_channel_scores
                    + self.cnn_pattern_score_weight
                    * train_cnn_pattern_scores
                )
            train_recon_scores = self._fuse_channel_scores(train_calibrated_channel_scores)
            print(
                "PAMM channel recon-score calibration: "
                f"channels={self.input_c}, "
                f"background_quantile={self.recon_background_quantile:.2f}, "
                f"mean_threshold={float(np.mean(self.train_channel_recon_score_q95)):.6f}, "
                f"max_threshold={float(np.max(self.train_channel_recon_score_q95)):.6f}"
            )
        else:
            self.train_channel_recon_score_q95 = None
            train_recon_scores = self._raw_decision_components_on_normalized(train_data)
            train_cnn_pattern_scores = self._cnn_pattern_scores_on_normalized(train_data)
            if train_cnn_pattern_scores is not None:
                train_cnn_pattern_scores = np.asarray(train_cnn_pattern_scores, dtype=np.float32).reshape(len(train_data), -1)
                train_recon_scores = (
                    train_recon_scores
                    + self.cnn_pattern_score_weight
                    * train_cnn_pattern_scores[:, 0]
                )
        self.train_recon_score_median = float(np.median(train_recon_scores))
        self.train_recon_score_q95 = float(np.quantile(train_recon_scores, self.recon_background_quantile))
        print(
            f"PAMM recon-score calibration: median={self.train_recon_score_median:.6f}, "
            f"background_quantile={self.recon_background_quantile:.2f}, "
            f"threshold={self.train_recon_score_q95:.6f}"
        )

        return self

    def _raw_decision_components_on_normalized(
        self,
        normalized_data: np.ndarray,
        return_channel_scores: bool = False,
    ) -> np.ndarray:
        if len(normalized_data) < self.win_size:
            raise ValueError(
                f"PAMM requires at least win_size={self.win_size} points, got {len(normalized_data)}."
            )

        if not self._projection_mode_logged:
            print(f"----- PAMM score_projection_mode={self.score_projection_mode} -----")
            self._projection_mode_logged = True

        test_loader = self._make_loader(normalized_data, shuffle=False)
        self.model.eval()
        recon_scores = []
        if return_channel_scores:
            projected_scores = np.zeros((len(normalized_data), self.input_c), dtype=np.float32)
            projected_counts = np.zeros((len(normalized_data), 1), dtype=np.float32)
        else:
            projected_scores = np.zeros(len(normalized_data), dtype=np.float32)
            projected_counts = np.zeros(len(normalized_data), dtype=np.float32)
        window_start = 0

        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
        with torch.no_grad():
            for _, (batch_x, _) in loop:
                batch_x = batch_x.float().to(self.device)
                output = self.model(batch_x, use_masking=False)

                if return_channel_scores:
                    if output.point_channel_error is None:
                        raise RuntimeError("PAMM backbone did not return point_channel_error.")
                    window_scores = output.point_channel_error
                    window_scores = window_scores * self._diff_prejudge_gate(batch_x).view(-1, 1, 1)
                else:
                    window_scores = output.anomaly_score
                    window_scores = window_scores * self._diff_prejudge_gate(batch_x).unsqueeze(-1)

                if self.score_projection_mode == "last":
                    recon_scores.append(window_scores[:, -1].detach().cpu().numpy())
                elif self.score_projection_mode == "center":
                    recon_scores.append(self._center_window_score(window_scores).detach().cpu().numpy())
                else:
                    weights = self._projection_weights(window_scores.device, window_scores.dtype)
                    if return_channel_scores:
                        weighted_scores = (window_scores * weights.view(1, -1, 1)).detach().cpu().numpy()
                        weight_values = weights.detach().cpu().numpy()[:, None]
                    else:
                        weighted_scores = (window_scores * weights.unsqueeze(0)).detach().cpu().numpy()
                        weight_values = weights.detach().cpu().numpy()
                    batch_size = weighted_scores.shape[0]
                    for batch_idx in range(batch_size):
                        start = window_start + batch_idx
                        end = start + self.win_size
                        projected_scores[start:end] += weighted_scores[batch_idx]
                        projected_counts[start:end] += weight_values
                    window_start += batch_size
                loop.set_description("Testing Phase")

        if self.score_projection_mode in {"last", "center"}:
            point_recon_scores = np.concatenate(recon_scores, axis=0).astype(np.float32)
            if not return_channel_scores:
                point_recon_scores = point_recon_scores.reshape(-1)
        else:
            point_recon_scores = projected_scores / np.maximum(projected_counts, 1e-6)
        if self.score_projection_mode == "last":
            point_recon_scores = self._pad_single_point_projection(
                scores=point_recon_scores,
                series_length=len(normalized_data),
                offset=self.win_size - 1,
            )
        elif self.score_projection_mode == "center":
            point_recon_scores = self._pad_single_point_projection(
                scores=point_recon_scores,
                series_length=len(normalized_data),
                offset=self.win_size // 2,
            )
        return point_recon_scores

    def decision_function(self, data: np.ndarray) -> np.ndarray:
        if len(data) < self.win_size:
            raise ValueError(
                f"PAMM requires at least win_size={self.win_size} points, got {len(data)}."
            )

        raw_data = np.asarray(data, dtype=np.float32)
        test_data = self._normalize_eval(raw_data)
        self.__channel_anomaly_score = None
        self.__recon_channel_anomaly_score = None
        self.__cnn_pattern_score = None
        self.__cnn_pattern_contribution_score = None
        self._reset_proto_analysis_cache()
        if self.input_c > 1 and self.train_channel_recon_score_q95 is not None:
            channel_recon_scores = self._raw_decision_components_on_normalized(
                test_data,
                return_channel_scores=True,
            )
            calibrated_channel_scores = np.maximum(
                channel_recon_scores - self.train_channel_recon_score_q95.reshape(1, -1),
                0.0,
            )
            self.__recon_channel_anomaly_score = np.asarray(calibrated_channel_scores, dtype=np.float32)
            cnn_pattern_scores = self._cnn_pattern_scores_on_normalized(
                test_data,
                collect_proto_analysis=True,
                analysis_prefix="decision",
            )
            if cnn_pattern_scores is not None:
                cnn_pattern_scores = np.asarray(cnn_pattern_scores, dtype=np.float32)
                self.__cnn_pattern_score = cnn_pattern_scores
                cnn_pattern_contribution_scores = (
                    self.cnn_pattern_score_weight
                    * cnn_pattern_scores
                ).astype(np.float32)
                self.__cnn_pattern_contribution_score = cnn_pattern_contribution_scores
                calibrated_channel_scores = (
                    calibrated_channel_scores
                    + cnn_pattern_contribution_scores
                )
            self.__channel_anomaly_score = np.asarray(calibrated_channel_scores, dtype=np.float32)
            calibrated_recon_scores = self._fuse_channel_scores(calibrated_channel_scores)
        else:
            recon_scores = self._raw_decision_components_on_normalized(test_data)
            calibrated_recon_scores = recon_scores
            cnn_pattern_scores = self._cnn_pattern_scores_on_normalized(
                test_data,
                collect_proto_analysis=True,
                analysis_prefix="decision",
            )
            if cnn_pattern_scores is not None:
                cnn_pattern_scores = np.asarray(cnn_pattern_scores, dtype=np.float32).reshape(len(test_data), -1)
                self.__cnn_pattern_score = cnn_pattern_scores
                calibrated_cnn_scores = (
                    self.cnn_pattern_score_weight
                    * cnn_pattern_scores[:, 0]
                ).astype(np.float32)
                self.__cnn_pattern_contribution_score = calibrated_cnn_scores.reshape(-1, 1)
                self.__channel_anomaly_score = cnn_pattern_scores.astype(np.float32)
                calibrated_recon_scores = calibrated_recon_scores + calibrated_cnn_scores
        if (
            self.input_c > 1
            and self.train_recon_score_q95 is not None
            and not (self.input_c > 1 and self.train_channel_recon_score_q95 is not None)
        ):
            calibrated_recon_scores = np.maximum(recon_scores - self.train_recon_score_q95, 0.0)
        zero_drop_scores = self._zero_drop_rule_score(raw_data)
        combined_score = (
            calibrated_recon_scores
            + self.zero_drop_rule_weight * zero_drop_scores
        )
        self.__anomaly_score = np.asarray(combined_score, dtype=np.float32).reshape(-1)

        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def channel_anomaly_score(self) -> Optional[np.ndarray]:
        return self.__channel_anomaly_score

    def recon_channel_anomaly_score(self) -> Optional[np.ndarray]:
        return self.__recon_channel_anomaly_score

    def cnn_pattern_score(self) -> Optional[np.ndarray]:
        return self.__cnn_pattern_score

    def cnn_pattern_contribution_score(self) -> Optional[np.ndarray]:
        return self.__cnn_pattern_contribution_score

    def proto_analysis(self) -> Optional[dict]:
        """Return cached prototype-analysis statistics from the latest decision_function call."""
        return self.__proto_analysis

    def save_proto_analysis(self, save_dir: str, prefix: str = "manual") -> Optional[dict]:
        """Save cached prototype-analysis statistics to .npz and .pt files.

        Call decision_function(data) before using this method.
        """
        if self.__proto_analysis is None:
            print("No prototype-analysis cache found. Please call decision_function(data) first.")
            return None

        previous_dir = self.proto_analysis_dir
        previous_flag = self.save_proto_analysis_enabled
        self.proto_analysis_dir = save_dir
        self.save_proto_analysis_enabled = True
        self._save_proto_analysis_payload(self.__proto_analysis, prefix=prefix)
        self.proto_analysis_dir = previous_dir
        self.save_proto_analysis_enabled = previous_flag
        return self.__proto_analysis


    def param_statistic(self, save_file: str):
        model_stats = torchinfo.summary(self.model, input_size=self.input_shape, verbose=0)
        with open(save_file, "w") as f:
            f.write(str(model_stats))
