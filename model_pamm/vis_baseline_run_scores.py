import argparse
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency fallback
    def tqdm(iterable, total=None, desc=None):
        return iterable


DEFAULT_EXCLUDED_MODELS = {
    "PAMM",
    "PAMM_channel_scores",
    "PAMM_cnn_pattern_scores",
    "PAMM_cnn_pattern_contribution_scores",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize baseline anomaly scores from test/scores together with "
            "data overview and ground-truth labels."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root that contains Datasets/ and test/.",
    )
    parser.add_argument(
        "--scores-root",
        type=Path,
        default=None,
        help="Score root. Defaults to <root>/test/scores.",
    )
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=None,
        help="Metrics root. Defaults to <root>/test/metrics.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output directory. Defaults to <root>/test/visualizations/baselines.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["M", "U"],
        choices=["M", "U", "multi", "uni"],
        help="Benchmark splits to visualize.",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["eval"],
        choices=["tuning", "eval"],
        help="Benchmark phases to visualize.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional model filter. By default all non-PAMM baseline models are visualized.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset family filter, e.g. MSL SMAP SMD.",
    )
    parser.add_argument(
        "--max-files-per-model",
        type=int,
        default=None,
        help="Optional cap per split/phase/model after filtering.",
    )
    parser.add_argument(
        "--include-pamm",
        action="store_true",
        help="Include PAMM rows if they exist in test/metrics.",
    )
    parser.add_argument(
        "--normalize-score",
        action="store_true",
        help="Normalize each anomaly score to [0, 1] before plotting.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Figure DPI.",
    )
    return parser.parse_args()


def split_to_dir(split: str) -> str:
    return "multi" if split in {"M", "multi"} else "uni"


def split_to_dataset_suffix(split: str) -> str:
    return "M" if split in {"M", "multi"} else "U"


def contiguous_ranges(label: np.ndarray):
    label = np.asarray(label).astype(int)
    ranges = []
    start = None
    for idx, value in enumerate(label):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            ranges.append((start, idx - 1))
            start = None
    if start is not None:
        ranges.append((start, len(label) - 1))
    return ranges


def add_common_annotations(ax, anomaly_ranges, train_index, series_length):
    for start, end in anomaly_ranges:
        ax.axvspan(start, end, color="#ef4444", alpha=0.26, linewidth=0, zorder=0)
    if train_index is not None and 0 < train_index < series_length:
        ax.axvline(train_index, color="#111827", linestyle="--", linewidth=1.05, alpha=0.82, zorder=2)


def style_score_axis(ax):
    ax.set_facecolor("#fbfbf7")
    ax.grid(color="#9ca3af", alpha=0.22, linewidth=0.65)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#6b7280")
        ax.spines[side].set_linewidth(0.85)
    ax.tick_params(colors="#111827", labelsize=9)
    ax.yaxis.label.set_color("#111827")
    ax.xaxis.label.set_color("#111827")


def compute_zscore(data: np.ndarray) -> np.ndarray:
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (data - mean) / std


def normalize_score(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    low = float(np.nanmin(score))
    high = float(np.nanmax(score))
    if high - low < 1e-12:
        return np.zeros_like(score)
    return (score - low) / (high - low)


def load_series(csv_path: Path):
    df = pd.read_csv(csv_path).dropna()
    if "Label" not in df.columns:
        raise ValueError(f"{csv_path} does not contain a Label column.")
    data = df.iloc[:, :-1].to_numpy(dtype=float)
    label = df["Label"].to_numpy(dtype=int)
    try:
        train_index = int(csv_path.stem.split("_")[-3])
    except Exception:
        train_index = None
    return data, label, train_index


def load_metrics(metrics_root: Path, split_dir: str, phase: str) -> pd.DataFrame:
    phase_dir = metrics_root / split_dir / phase
    detailed_path = phase_dir / "detailed_metrics.csv"
    if detailed_path.exists():
        frame = pd.read_csv(detailed_path)
    else:
        detail_files = sorted(phase_dir.glob("*_details.csv"))
        if not detail_files:
            raise FileNotFoundError(f"No detailed metrics found under {phase_dir}.")
        frame = pd.concat([pd.read_csv(path) for path in detail_files], ignore_index=True)
    frame["split_dir"] = split_dir
    frame["phase"] = phase
    return frame


def resolve_score_path(scores_root: Path, split_dir: str, phase: str, model: str, file_name: str) -> Path:
    score_path = scores_root / split_dir / phase / model / f"{Path(file_name).stem}.npy"
    if not score_path.exists():
        raise FileNotFoundError(f"Missing score file: {score_path}")
    return score_path


def sanitize_score(score: np.ndarray, expected_length: int) -> np.ndarray:
    score = np.asarray(score)
    if score.ndim > 1:
        score = np.squeeze(score)
    if score.ndim > 1:
        score = score.reshape(score.shape[0], -1).mean(axis=1)
    score = score.reshape(-1).astype(float)
    if len(score) != expected_length:
        raise ValueError(f"Score length mismatch: score={len(score)} label={expected_length}.")
    return score


def plot_sequence(
    data: np.ndarray,
    label: np.ndarray,
    score: np.ndarray,
    train_index: Optional[int],
    title: str,
    save_path: Path,
    normalize_score_flag: bool,
    dpi: int,
):
    series_length, n_features = data.shape
    anomaly_ranges = contiguous_ranges(label)
    score_to_plot = normalize_score(score) if normalize_score_flag else score
    score_label = "Anomaly Score"

    fig, ax = plt.subplots(1, 1, figsize=(18, 4.2), facecolor="white")
    ax.plot(score_to_plot, color="#1d4ed8", linewidth=1.35, zorder=3)
    add_common_annotations(ax, anomaly_ranges, train_index, series_length)
    ax.set_ylabel(score_label)
    ax.set_xlabel("Time Index")
    ax.set_title(title)
    style_score_axis(ax)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    root = args.root.resolve()
    scores_root = (args.scores_root or root / "test" / "scores").resolve()
    metrics_root = (args.metrics_root or root / "test" / "metrics").resolve()
    output_root = (args.output_root or root / "test" / "visualizations" / "baselines").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    split_dirs = [split_to_dir(split) for split in args.splits]
    frames = []
    for split_dir in dict.fromkeys(split_dirs):
        for phase in args.phases:
            frames.append(load_metrics(metrics_root, split_dir, phase))
    metrics_df = pd.concat(frames, ignore_index=True)

    if "status" in metrics_df.columns:
        metrics_df = metrics_df[metrics_df["status"].astype(str).str.startswith(("success", "loaded_existing_score"))].copy()
    if not args.include_pamm:
        metrics_df = metrics_df[~metrics_df["model"].isin(DEFAULT_EXCLUDED_MODELS)].copy()
    if args.models:
        metrics_df = metrics_df[metrics_df["model"].isin(args.models)].copy()
    if args.datasets:
        metrics_df = metrics_df[metrics_df["dataset"].isin(args.datasets)].copy()
    if metrics_df.empty:
        raise ValueError("No metric rows matched the requested filters.")

    if args.max_files_per_model is not None:
        metrics_df = (
            metrics_df.sort_values(["split_dir", "phase", "model", "dataset", "file"])
            .groupby(["split_dir", "phase", "model"], as_index=False, group_keys=False)
            .head(args.max_files_per_model)
        )

    manifest_rows = []
    for _, row in tqdm(metrics_df.iterrows(), total=len(metrics_df), desc="Visualizing baseline scores"):
        row = row.to_dict()
        split_dir = row["split_dir"]
        split_suffix = split_to_dataset_suffix(split_dir)
        phase = row["phase"]
        model_name = str(row["model"])
        dataset = str(row["dataset"])
        file_name = str(row["file"])

        csv_path = root / "Datasets" / f"TSB-AD-{split_suffix}" / file_name
        score_path = resolve_score_path(scores_root, split_dir, phase, model_name, file_name)
        data, label, train_index = load_series(csv_path)
        score = sanitize_score(np.load(score_path), expected_length=len(label))

        save_dir = output_root / split_dir / phase / model_name / dataset
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{Path(file_name).stem}.svg"

        title = (
            f"{Path(file_name).stem} | model={model_name}"
        )
        plot_sequence(
            data=data,
            label=label,
            score=score,
            train_index=train_index,
            title=title,
            save_path=save_path,
            normalize_score_flag=args.normalize_score,
            dpi=args.dpi,
        )

        manifest_rows.append(
            {
                "model": model_name,
                "split": split_dir,
                "phase": phase,
                "dataset": dataset,
                "file": file_name,
                "score_path": str(score_path),
                "figure_path": str(save_path),
                "AUC-PR": row.get("AUC-PR", ""),
                "AUC-ROC": row.get("AUC-ROC", ""),
                "VUS-PR": row.get("VUS-PR", ""),
                "VUS-ROC": row.get("VUS-ROC", ""),
            }
        )

    manifest_path = output_root / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"Saved baseline score visualizations to: {output_root}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
