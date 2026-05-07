import argparse
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Default this test runner to physical GPU 2 while still allowing an external
# override if CUDA_VISIBLE_DEVICES is already set before launch.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import torch
from sklearn.exceptions import UndefinedMetricWarning

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.model_wrapper import (
    Semisupervise_AD_Pool,
    Unsupervise_AD_Pool,
    get_last_pamm_channel_scores,
    get_last_pamm_cnn_pattern_contribution_scores,
    get_last_pamm_cnn_pattern_scores,
    get_last_pamm_recon_channel_scores,
    run_Semisupervise_AD,
    run_Unsupervise_AD,
)
from TSB_AD.utils.slidingWindows import find_length_rank

DEFAULT_DATASETS = ['Genesis']
# 'MSL', 'SMAP', 'SMD', 'NAB', 'WSD', 'Stock', 'MGAB', 'TAO', 'UCR', 'YAHOO', 'CATSv2',
# 'Daphnet', 'Exathlon', 'IOPS', 'LTDB', 'MGAB', 'MITDB', 'NEK', 'OPPORTUNITY', 'Power',
# 'SED', 'SVDB', 'SWaT', 'TODS'
DEFAULT_MODELS = ['AnomalyTransformer',"LSTMAD", "OmniAnomaly","AutoEncoder", "FITS","TimesNet"]   #"IForest", "KMeansAD", "LOF", "LSTMAD", "OmniAnomaly", "AnomalyTransformer", "AutoEncoder", "FITS", "TranAD", "TimesNet", "PAMM"
METRIC_COLUMNS = [
    "AUC-PR",
    "AUC-ROC",
    "VUS-PR",
    "VUS-ROC",
    "Precision",
    "Recall",
    "F1-score",
    "Standard-F1",
    "PA-F1",
    "Event-based-F1",
    "R-based-F1",
    "Affiliation-F",
]
SEED = 2024


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def silence_expected_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message="Precision is ill-defined and being set to 0.0 due to no predicted samples.*",
        category=UndefinedMetricWarning,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run representative anomaly detectors on selected TSB-AD splits "
            "using the repository default hyperparameters."
        )
    )
    parser.add_argument(
        "--split",
        type=str,
        default="M",
        choices=["M", "U"],
        help="Benchmark split to run. Use U for univariate MSL/SMAP/SMD tests.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="eval",
        choices=["tuning", "eval"],
        help=(
            "Benchmark phase to run. Use tuning for *-Tuning.csv when selecting or "
            "sanity-checking hyperparameters; use eval for final *-Eva.csv reporting."
        ),
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=None,
        help="Directory containing split csv files. Defaults to the selected benchmark split.",
    )
    parser.add_argument(
        "--file_list",
        type=Path,
        default=None,
        help=(
            "Explicit benchmark file list. If omitted, defaults to the selected "
            "split/phase official TSB-AD file list."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset keywords to keep from the file list.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Representative multivariate models to run.",
    )
    parser.add_argument(
        "--score_dir",
        type=Path,
        default=ROOT / "test" / "scores",
        help="Base output directory for anomaly scores.",
    )
    parser.add_argument(
        "--metrics_dir",
        type=Path,
        default=ROOT / "test" / "metrics",
        help="Base output directory for detailed metrics and summaries.",
    )
    parser.add_argument(
        "--limit_per_dataset",
        type=int,
        default=None,
        help="Optional cap on the number of files kept for each dataset.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional cap on the total number of filtered files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run files even if the score .npy already exists.",
    )
    parser.add_argument(
        "--no_save_scores",
        action="store_true",
        help="Disable saving .npy score files.",
    )
    parser.add_argument(
        "--pamm_win_size",
        type=int,
        default=None,
        help="Optional PAMM-only override for the sliding window size.",
    )
    parser.add_argument(
        "--pamm_lr",
        type=float,
        default=None,
        help="Optional PAMM-only override for the learning rate.",
    )
    parser.add_argument(
        "--pamm_batch_size",
        type=int,
        default=None,
        help="Optional PAMM-only override for the batch size.",
    )
    parser.add_argument(
        "--pamm_epochs",
        type=int,
        default=None,
        help="Optional PAMM-only override for the number of training epochs.",
    )
    parser.add_argument(
        "--pamm_d_model",
        type=int,
        default=None,
        help="Optional PAMM-only override for the token hidden dimension.",
    )
    parser.add_argument(
        "--pamm_patch_size",
        type=int,
        default=None,
        help="Optional PAMM-only override for the patch size.",
    )
    parser.add_argument(
        "--pamm_patch_stride",
        type=int,
        default=None,
        help="Optional PAMM-only override for the patch stride.",
    )
    parser.add_argument(
        "--pamm_use_locality_bias",
        type=int,
        default=None,
        choices=[0, 1],
        help="Optional PAMM-only switch for proximity-biased attention. Use 0 for vanilla MHAttention.",
    )
    parser.add_argument(
        "--pamm_contrast_weight",
        type=float,
        default=None,
        help="Optional PAMM-only contrast loss weight.",
    )
    parser.add_argument(
        "--pamm_use_revin",
        type=int,
        default=None,
        choices=[0, 1],
        help="Optional PAMM-only RevIN switch. Use 0 to disable per-window RevIN.",
    )
    parser.add_argument(
        "--pamm_point_aggregate_mode",
        type=str,
        default=None,
        choices=["center_weighted", "mean", "average", "uniform", "gaussian", "gaussian_weighted"],
        help="Optional PAMM-only override for patch-score back-projection.",
    )
    parser.add_argument(
        "--pamm_point_center_power",
        type=float,
        default=None,
        help=(
            "Optional PAMM-only center weighting power for center_weighted "
            "patch-score back-projection."
        ),
    )
    parser.add_argument(
        "--pamm_point_gaussian_sigma",
        type=float,
        default=None,
        help="Optional PAMM Gaussian back-projection sigma. Use 0 for patch_size / 6.",
    )
    parser.add_argument(
        "--pamm_score_projection_mode",
        type=str,
        default=None,
        choices=["last", "center", "mean", "average", "uniform", "center_weighted", "gaussian", "gaussian_weighted"],
        help="Optional PAMM-only override for projecting window scores back to the full series.",
    )
    parser.add_argument(
        "--pamm_score_projection_center_power",
        type=float,
        default=None,
        help=(
            "Optional PAMM-only center weighting power for center_weighted "
            "window-score projection."
        ),
    )
    parser.add_argument(
        "--pamm_diff_prejudge_enabled",
        type=int,
        default=None,
        choices=[0, 1],
        help="Optional PAMM-only diff-stability pre-judge switch.",
    )
    parser.add_argument(
        "--pamm_diff_prejudge_quantile",
        type=float,
        default=None,
        help="Optional PAMM-only train diff-energy quantile used as the normal threshold.",
    )
    parser.add_argument(
        "--pamm_diff_prejudge_cosine_quantile",
        type=float,
        default=None,
        help="Optional PAMM-only low cosine-similarity quantile used as the normal-pattern threshold.",
    )
    parser.add_argument(
        "--pamm_diff_prejudge_margin",
        type=float,
        default=None,
        help="Optional PAMM-only multiplier applied to the diff pre-judge threshold.",
    )
    parser.add_argument(
        "--pamm_diff_prejudge_suppression",
        type=float,
        default=None,
        help="Optional PAMM-only score multiplier for diff-stable windows.",
    )
    parser.add_argument(
        "--pamm_patch_channel_topk_weight",
        type=float,
        default=None,
        help="Optional PAMM-only top-k channel spike weight after robust channel aggregation.",
    )
    parser.add_argument(
        "--pamm_patch_channel_topk_ratio",
        type=float,
        default=None,
        help="Optional PAMM-only ratio of channels used by top-k channel spike aggregation.",
    )
    parser.add_argument(
        "--pamm_cnn_pattern_enabled",
        type=int,
        default=None,
        choices=[0, 1],
        help="Optional PAMM-only CNN pattern prototype branch switch.",
    )
    parser.add_argument(
        "--pamm_cnn_pattern_score_weight",
        type=float,
        default=None,
        help="Optional PAMM-only weight for the CNN pattern prototype score.",
    )
    parser.add_argument(
        "--pamm_cnn_pattern_hidden_dim",
        type=int,
        default=None,
        help="Optional PAMM-only hidden dimension for the CNN pattern branch.",
    )
    parser.add_argument(
        "--pamm_cnn_pattern_embedding_dim",
        type=int,
        default=None,
        help="Optional PAMM-only embedding dimension for the CNN pattern prototype bank.",
    )
    parser.add_argument(
        "--pamm_cnn_pattern_num_contexts",
        type=int,
        default=None,
        help="Optional PAMM-only number of learnable context prototypes per channel.",
    )
    parser.add_argument(
        "--pamm_cnn_pattern_proto_tau",
        type=float,
        default=None,
        help="Optional PAMM-only soft prototype-selection temperature.",
    )
    parser.add_argument(
        "--pamm_cnn_pattern_proto_loss_weight",
        type=float,
        default=None,
        help="Optional PAMM-only weight for CNN embedding-to-prototype training loss.",
    )
    return parser.parse_args()


def dataset_name_from_file(filename: str, dataset_names: List[str]) -> str:
    for dataset_name in dataset_names:
        if f"_{dataset_name}_" in filename:
            return dataset_name
    return "Unknown"


def load_target_files(
        file_list_path: Path,
        dataset_names: List[str],
        limit_per_dataset: Optional[int] = None,
        max_files: Optional[int] = None,
) -> List[str]:
    file_list = pd.read_csv(file_list_path)["file_name"].tolist()
    selected_files = []
    dataset_counts = {name: 0 for name in dataset_names}

    for filename in file_list:
        dataset_name = dataset_name_from_file(filename, dataset_names)
        if dataset_name == "Unknown":
            continue

        if limit_per_dataset is not None and dataset_counts[dataset_name] >= limit_per_dataset:
            continue

        selected_files.append(filename)
        dataset_counts[dataset_name] += 1

        if max_files is not None and len(selected_files) >= max_files:
            break

    return selected_files


def validate_models(models: List[str]) -> None:
    missing = [model for model in models if model not in Optimal_Multi_algo_HP_dict]
    if missing:
        raise ValueError(
            "These models do not have default multivariate hyperparameters in "
            f"Optimal_Multi_algo_HP_dict: {missing}"
        )


def get_split_paths(split: str, phase: str) -> Tuple[Path, Path]:
    list_suffix = "Tuning" if phase == "tuning" else "Eva"
    if split == "U":
        return (
            ROOT / "Datasets" / "TSB-AD-U",
            ROOT / "Datasets" / "File_List" / f"TSB-AD-U-{list_suffix}.csv",
        )
    return (
        ROOT / "Datasets" / "TSB-AD-M",
        ROOT / "Datasets" / "File_List" / f"TSB-AD-M-{list_suffix}.csv",
    )


def infer_phase_from_file_list(file_list_path: Path) -> Optional[str]:
    name = file_list_path.name.lower()
    if "tuning" in name:
        return "tuning"
    if "eva" in name or "eval" in name:
        return "eval"
    return None


def validate_phase_file_list(phase: str, file_list_path: Path) -> None:
    inferred_phase = infer_phase_from_file_list(file_list_path)
    if inferred_phase is not None and inferred_phase != phase:
        raise ValueError(
            f"--phase {phase!r} conflicts with --file_list {str(file_list_path)!r}, "
            f"which looks like a {inferred_phase!r} file list. Run tuning and eval "
            "as separate phases to avoid using evaluation results for tuning."
        )


def get_hp_dict(split: str) -> Dict[str, dict]:
    return Optimal_Uni_algo_HP_dict if split == "U" else Optimal_Multi_algo_HP_dict


def validate_models_for_split(models: List[str], split: str) -> None:
    hp_dict = get_hp_dict(split)
    mapped_models = [map_model_name_for_split(model, split) for model in models]
    missing = [model for model, mapped in zip(models, mapped_models) if mapped not in hp_dict]
    if missing:
        raise ValueError(
            f"These models do not have default hyperparameters for split {split}: {missing}"
        )


def map_model_name_for_split(model_name: str, split: str) -> str:
    if split == "U" and model_name == "KMeansAD":
        return "KMeansAD_U"
    return model_name


def run_model(
        model_name: str,
        split: str,
        data_train: np.ndarray,
        data: np.ndarray,
        hp_overrides: Optional[Dict[str, dict]] = None,
) -> np.ndarray:
    actual_model_name = map_model_name_for_split(model_name, split)
    hyper_params = dict(get_hp_dict(split)[actual_model_name])
    if hp_overrides and actual_model_name in hp_overrides:
        hyper_params.update(hp_overrides[actual_model_name])
    if actual_model_name in Semisupervise_AD_Pool:
        output = run_Semisupervise_AD(actual_model_name, data_train, data, **hyper_params)
    elif actual_model_name in Unsupervise_AD_Pool:
        output = run_Unsupervise_AD(actual_model_name, data, **hyper_params)
    else:
        raise ValueError(f"{actual_model_name} is not defined in the model pools.")

    if not isinstance(output, np.ndarray):
        raise RuntimeError(str(output))

    return output


def load_file_data(
        filename: str,
        dataset_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, int]:
    file_path = dataset_dir / filename
    df = pd.read_csv(file_path).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df["Label"].astype(int).to_numpy()
    sliding_window = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
    return data, label, sliding_window


def run_single_file(
        model_name: str,
        split: str,
        filename: str,
        dataset_dir: Path,
        hp_overrides: Optional[Dict[str, dict]] = None,
) -> Tuple[np.ndarray, Dict[str, float], float]:
    data, label, sliding_window = load_file_data(filename, dataset_dir)
    train_index = int(filename.split(".")[0].split("_")[-3])
    data_train = data[:train_index, :]

    start_time = time.time()
    output = run_model(model_name, split, data_train, data, hp_overrides=hp_overrides)
    runtime = time.time() - start_time
    metrics = get_metrics(output, label, slidingWindow=sliding_window)
    return output, metrics, runtime


def evaluate_saved_score(
        filename: str,
        dataset_dir: Path,
        score_path: Path,
) -> Dict[str, float]:
    _, label, sliding_window = load_file_data(filename, dataset_dir)
    output = np.load(score_path)
    return get_metrics(output, label, slidingWindow=sliding_window)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_runtime_lookup(detail_path: Path) -> Dict[str, float]:
    if not detail_path.exists():
        return {}

    detail_df = pd.read_csv(detail_path)
    if "file" not in detail_df.columns or "runtime" not in detail_df.columns:
        return {}

    runtime_lookup = {}
    for _, row in detail_df.iterrows():
        runtime_lookup[str(row["file"])] = row["runtime"]
    return runtime_lookup


def save_results(records: List[dict], metrics_dir: Path) -> None:
    ensure_dir(metrics_dir)
    detail_df = pd.DataFrame(records)
    if detail_df.empty:
        raise ValueError("No records were produced. Nothing to save.")
    detail_path = metrics_dir / "detailed_metrics.csv"
    detail_df.to_csv(detail_path, index=False)

    success_df = detail_df[detail_df["status"].isin(["success", "loaded_existing_score"])].copy()
    if success_df.empty:
        raise ValueError("All runs failed. No summary metrics can be generated.")

    summary_df = (
        success_df.groupby(["model", "dataset"], as_index=False)[["runtime"] + METRIC_COLUMNS]
        .mean()
        .sort_values(["dataset", "model"])
    )

    counts_df = (
        detail_df.groupby(["model", "dataset"], as_index=False)["status"]
        .agg(
            files_total="count",
            files_success=lambda s: int(s.isin(["success", "loaded_existing_score"]).sum()),
            files_failed=lambda s: int((~s.isin(["success", "loaded_existing_score"])).sum()),
        )
        .sort_values(["dataset", "model"])
    )

    model_dataset_summary_df = counts_df.merge(summary_df, on=["model", "dataset"], how="left")
    model_dataset_summary_df.to_csv(metrics_dir / "model_dataset_summary.csv", index=False)
    model_dataset_summary_df.to_csv(metrics_dir / "summary_by_dataset.csv", index=False)

    overall_df = (
        success_df.groupby("model", as_index=False)[["runtime"] + METRIC_COLUMNS]
        .mean()
        .sort_values("model")
    )
    overall_counts_df = (
        detail_df.groupby("model", as_index=False)["status"]
        .agg(
            files_total="count",
            files_success=lambda s: int(s.isin(["success", "loaded_existing_score"]).sum()),
            files_failed=lambda s: int((~s.isin(["success", "loaded_existing_score"])).sum()),
        )
        .sort_values("model")
    )
    overall_summary_df = overall_counts_df.merge(overall_df, on="model", how="left")
    overall_summary_df.to_csv(metrics_dir / "summary_overall.csv", index=False)

    pivot_dir = metrics_dir / "pivot_tables"
    ensure_dir(pivot_dir)
    for metric in METRIC_COLUMNS:
        pivot_df = (
            summary_df.pivot(index="dataset", columns="model", values=metric)
            .sort_index()
            .reset_index()
        )
        safe_metric_name = metric.replace("-", "_").replace(" ", "_")
        pivot_df.to_csv(pivot_dir / f"{safe_metric_name}_by_dataset.csv", index=False)


def build_hp_overrides(args: argparse.Namespace) -> Optional[Dict[str, dict]]:
    pamm_overrides = {}
    scalar_overrides = {
        "win_size": args.pamm_win_size,
        "lr": args.pamm_lr,
        "batch_size": args.pamm_batch_size,
        "epochs": args.pamm_epochs,
        "d_model": args.pamm_d_model,
        "patch_size": args.pamm_patch_size,
        "patch_stride": args.pamm_patch_stride,
        "contrast_weight": args.pamm_contrast_weight,
    }
    for name, value in scalar_overrides.items():
        if value is not None:
            pamm_overrides[name] = value
    if args.pamm_use_revin is not None:
        pamm_overrides["use_revin"] = bool(args.pamm_use_revin)
    if args.pamm_use_locality_bias is not None:
        pamm_overrides["use_locality_bias"] = bool(args.pamm_use_locality_bias)
    if args.pamm_point_aggregate_mode is not None:
        pamm_overrides["point_aggregate_mode"] = args.pamm_point_aggregate_mode
    if args.pamm_point_center_power is not None:
        pamm_overrides["point_center_power"] = args.pamm_point_center_power
    if args.pamm_point_gaussian_sigma is not None:
        pamm_overrides["point_gaussian_sigma"] = args.pamm_point_gaussian_sigma
    if args.pamm_score_projection_mode is not None:
        pamm_overrides["score_projection_mode"] = args.pamm_score_projection_mode
    if args.pamm_score_projection_center_power is not None:
        pamm_overrides["score_projection_center_power"] = args.pamm_score_projection_center_power
    if args.pamm_diff_prejudge_enabled is not None:
        pamm_overrides["diff_prejudge_enabled"] = bool(args.pamm_diff_prejudge_enabled)
    if args.pamm_diff_prejudge_quantile is not None:
        pamm_overrides["diff_prejudge_quantile"] = args.pamm_diff_prejudge_quantile
    if args.pamm_diff_prejudge_cosine_quantile is not None:
        pamm_overrides["diff_prejudge_cosine_quantile"] = args.pamm_diff_prejudge_cosine_quantile
    if args.pamm_diff_prejudge_margin is not None:
        pamm_overrides["diff_prejudge_margin"] = args.pamm_diff_prejudge_margin
    if args.pamm_diff_prejudge_suppression is not None:
        pamm_overrides["diff_prejudge_suppression"] = args.pamm_diff_prejudge_suppression
    if args.pamm_patch_channel_topk_weight is not None:
        pamm_overrides["patch_channel_topk_weight"] = args.pamm_patch_channel_topk_weight
    if args.pamm_patch_channel_topk_ratio is not None:
        pamm_overrides["patch_channel_topk_ratio"] = args.pamm_patch_channel_topk_ratio
    if args.pamm_cnn_pattern_enabled is not None:
        pamm_overrides["cnn_pattern_enabled"] = bool(args.pamm_cnn_pattern_enabled)
    if args.pamm_cnn_pattern_score_weight is not None:
        pamm_overrides["cnn_pattern_score_weight"] = args.pamm_cnn_pattern_score_weight
    if args.pamm_cnn_pattern_hidden_dim is not None:
        pamm_overrides["cnn_pattern_hidden_dim"] = args.pamm_cnn_pattern_hidden_dim
    if args.pamm_cnn_pattern_embedding_dim is not None:
        pamm_overrides["cnn_pattern_embedding_dim"] = args.pamm_cnn_pattern_embedding_dim
    if args.pamm_cnn_pattern_num_contexts is not None:
        pamm_overrides["cnn_pattern_num_contexts"] = args.pamm_cnn_pattern_num_contexts
    if args.pamm_cnn_pattern_proto_tau is not None:
        pamm_overrides["cnn_pattern_proto_tau"] = args.pamm_cnn_pattern_proto_tau
    if args.pamm_cnn_pattern_proto_loss_weight is not None:
        pamm_overrides["cnn_pattern_proto_loss_weight"] = args.pamm_cnn_pattern_proto_loss_weight
    if not pamm_overrides:
        return None
    return {"PAMM": pamm_overrides}


def merge_model_hp_overrides(
        hp_overrides: Optional[Dict[str, dict]],
        model_name: str,
        updates: Dict[str, object],
) -> Dict[str, dict]:
    merged = {name: dict(params) for name, params in (hp_overrides or {}).items()}
    merged.setdefault(model_name, {}).update(updates)
    return merged


def main() -> None:
    print("Runner script:", Path(__file__).resolve())
    args = parse_args()
    set_seed(SEED)
    silence_expected_warnings()
    validate_models_for_split(args.models, args.split)
    hp_overrides = build_hp_overrides(args)

    default_dataset_dir, default_file_list = get_split_paths(args.split, args.phase)
    args.dataset_dir = args.dataset_dir or default_dataset_dir
    args.file_list = args.file_list or default_file_list
    validate_phase_file_list(args.phase, args.file_list)

    split_suffix = "uni" if args.split == "U" else "multi"
    args.score_dir = args.score_dir / split_suffix / args.phase
    args.metrics_dir = args.metrics_dir / split_suffix / args.phase

    target_files = load_target_files(
        file_list_path=args.file_list,
        dataset_names=args.datasets,
        limit_per_dataset=args.limit_per_dataset,
        max_files=args.max_files,
    )
    if not target_files:
        raise ValueError("No files matched the dataset filters.")

    print("CUDA available:", torch.cuda.is_available())
    print("cuDNN version:", torch.backends.cudnn.version())
    print("Split:", args.split)
    print("Phase:", args.phase)
    print("File list:", args.file_list)
    print("Datasets:", ", ".join(args.datasets))
    print("Models:", ", ".join(args.models))
    if hp_overrides:
        print("PAMM HP overrides:", hp_overrides["PAMM"])
    print("Files selected:", len(target_files))

    all_records = []
    ensure_dir(args.metrics_dir)
    if not args.no_save_scores:
        ensure_dir(args.score_dir)

    for model_name in args.models:
        model_score_dir = args.score_dir / model_name
        model_channel_score_dir = args.score_dir / f"{model_name}_channel_scores"
        model_recon_channel_score_dir = args.score_dir / f"{model_name}_recon_channel_scores"
        model_cnn_pattern_score_dir = args.score_dir / f"{model_name}_cnn_pattern_scores"
        model_cnn_pattern_contribution_dir = args.score_dir / f"{model_name}_cnn_pattern_contribution_scores"
        detail_path = args.metrics_dir / f"{model_name}_details.csv"
        runtime_lookup = load_runtime_lookup(detail_path)
        if not args.no_save_scores:
            ensure_dir(model_score_dir)
            if model_name == "PAMM":
                ensure_dir(model_channel_score_dir)
                ensure_dir(model_recon_channel_score_dir)
                ensure_dir(model_cnn_pattern_score_dir)
                ensure_dir(model_cnn_pattern_contribution_dir)

        print(f"\n=== Running {model_name} ===")
        model_records = []

        for file_index, filename in enumerate(target_files, start=1):
            dataset_name = dataset_name_from_file(filename, args.datasets)
            score_path = model_score_dir / f"{Path(filename).stem}.npy"
            recon_channel_score_path = model_recon_channel_score_dir / f"{Path(filename).stem}.npy"
            cnn_pattern_score_path = model_cnn_pattern_score_dir / f"{Path(filename).stem}.npy"
            cnn_pattern_contribution_path = model_cnn_pattern_contribution_dir / f"{Path(filename).stem}.npy"
            proto_analysis_dir = None
            if model_name == "PAMM":
                proto_analysis_dir = (
                    args.score_dir
                    / "PAMM_proto_analysis"
                    / dataset_name
                    / Path(filename).stem
                )

            pamm_aux_missing = False
            if model_name == "PAMM" and args.split == "M":
                pamm_aux_missing = (
                    not recon_channel_score_path.exists()
                    or not cnn_pattern_score_path.exists()
                    or not cnn_pattern_contribution_path.exists()
                )

            if (
                    not args.no_save_scores
                    and score_path.exists()
                    and not args.overwrite
                    and not pamm_aux_missing
            ):
                print(f"[{file_index}/{len(target_files)}] Load existing score: {filename}")
                metrics = evaluate_saved_score(
                    filename=filename,
                    dataset_dir=args.dataset_dir,
                    score_path=score_path,
                )
                model_records.append(
                    {
                        "model": model_name,
                        "split": args.split,
                        "phase": args.phase,
                        "file_list": str(args.file_list),
                        "dataset": dataset_name,
                        "file": filename,
                        "runtime": runtime_lookup.get(filename, np.nan),
                        "status": "loaded_existing_score",
                        "proto_analysis_dir": str(proto_analysis_dir) if proto_analysis_dir else "",
                        **metrics,
                    }
                )
                continue

            print(f"[{file_index}/{len(target_files)}] {dataset_name} | {filename}")

            try:
                file_hp_overrides = hp_overrides
                if proto_analysis_dir is not None:
                    file_hp_overrides = merge_model_hp_overrides(
                        hp_overrides,
                        "PAMM",
                        {"proto_analysis_dir": str(proto_analysis_dir)},
                    )
                output, metrics, runtime = run_single_file(
                    model_name=model_name,
                    split=args.split,
                    filename=filename,
                    dataset_dir=args.dataset_dir,
                    hp_overrides=file_hp_overrides,
                )
            except Exception as exc:
                print(f"Failed on {filename} with {model_name}: {exc}")
                model_records.append(
                    {
                        "model": model_name,
                        "split": args.split,
                        "phase": args.phase,
                        "file_list": str(args.file_list),
                        "dataset": dataset_name,
                        "file": filename,
                        "runtime": np.nan,
                        "status": f"failed: {exc}",
                        "proto_analysis_dir": str(proto_analysis_dir) if proto_analysis_dir else "",
                    }
                )
                continue

            if not args.no_save_scores:
                np.save(score_path, output)
                if model_name == "PAMM":
                    channel_scores = get_last_pamm_channel_scores()
                    if channel_scores is not None:
                        channel_scores = np.asarray(channel_scores, dtype=np.float32)
                        channel_score_path = model_channel_score_dir / f"{Path(filename).stem}.npy"
                        channel_score_csv_path = model_channel_score_dir / f"{Path(filename).stem}.csv"
                        np.save(channel_score_path, channel_scores)
                        pd.DataFrame(
                            channel_scores,
                            columns=[f"channel_{idx}" for idx in range(channel_scores.shape[1])],
                        ).to_csv(channel_score_csv_path, index=False)
                    recon_channel_scores = get_last_pamm_recon_channel_scores()
                    if recon_channel_scores is not None:
                        recon_channel_scores = np.asarray(recon_channel_scores, dtype=np.float32)
                        recon_channel_score_path = model_recon_channel_score_dir / f"{Path(filename).stem}.npy"
                        recon_channel_score_csv_path = model_recon_channel_score_dir / f"{Path(filename).stem}.csv"
                        np.save(recon_channel_score_path, recon_channel_scores)
                        pd.DataFrame(
                            recon_channel_scores,
                            columns=[f"channel_{idx}" for idx in range(recon_channel_scores.shape[1])],
                        ).to_csv(recon_channel_score_csv_path, index=False)
                    cnn_pattern_scores = get_last_pamm_cnn_pattern_scores()
                    if cnn_pattern_scores is not None:
                        cnn_pattern_scores = np.asarray(cnn_pattern_scores, dtype=np.float32)
                        cnn_pattern_score_path = model_cnn_pattern_score_dir / f"{Path(filename).stem}.npy"
                        cnn_pattern_score_csv_path = model_cnn_pattern_score_dir / f"{Path(filename).stem}.csv"
                        np.save(cnn_pattern_score_path, cnn_pattern_scores)
                        pd.DataFrame(
                            cnn_pattern_scores,
                            columns=[f"channel_{idx}" for idx in range(cnn_pattern_scores.shape[1])],
                        ).to_csv(cnn_pattern_score_csv_path, index=False)
                    cnn_pattern_contribution_scores = get_last_pamm_cnn_pattern_contribution_scores()
                    if cnn_pattern_contribution_scores is not None:
                        cnn_pattern_contribution_scores = np.asarray(cnn_pattern_contribution_scores, dtype=np.float32)
                        cnn_pattern_contribution_path = model_cnn_pattern_contribution_dir / f"{Path(filename).stem}.npy"
                        cnn_pattern_contribution_csv_path = model_cnn_pattern_contribution_dir / f"{Path(filename).stem}.csv"
                        np.save(cnn_pattern_contribution_path, cnn_pattern_contribution_scores)
                        pd.DataFrame(
                            cnn_pattern_contribution_scores,
                            columns=[f"channel_{idx}" for idx in range(cnn_pattern_contribution_scores.shape[1])],
                        ).to_csv(cnn_pattern_contribution_csv_path, index=False)

            row = {
                "model": model_name,
                "split": args.split,
                "phase": args.phase,
                "file_list": str(args.file_list),
                "dataset": dataset_name,
                "file": filename,
                "runtime": runtime,
                "status": "success",
                "proto_analysis_dir": str(proto_analysis_dir) if proto_analysis_dir else "",
            }
            row.update(metrics)
            model_records.append(row)
            print(
                "  "
                f"VUS-PR={metrics['VUS-PR']:.4f}, "
                f"AUC-PR={metrics['AUC-PR']:.4f}, "
                f"runtime={runtime:.2f}s"
            )

        all_records.extend(model_records)
        pd.DataFrame(model_records).to_csv(detail_path, index=False)

    save_results(all_records, args.metrics_dir)
    print(f"\nDetailed metrics saved to: {args.metrics_dir}")
    print(f"Model-dataset summary saved to: {args.metrics_dir / 'model_dataset_summary.csv'}")
    if not args.no_save_scores:
        print(f"Scores saved to: {args.score_dir}")


if __name__ == "__main__":
    main()
