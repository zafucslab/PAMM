import argparse
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, total=None, desc=None):
        return iterable


# ============================================================
# 1. Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize selected time-series channels and PAMM anomaly scores "
            "with ground-truth anomaly regions."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root that contains Datasets/ and model_pamm/.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="PAMM experiment directory under model_pamm/results_tsb_ad_runs/.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        choices=["M", "U"],
        help="Optional split filter.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset family filter, e.g. MSL SMAP SMD.",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=None,
        choices=["tuning", "eval"],
        help="Optional benchmark phase filter.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["PAMM"],
        help="Model names to visualize.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        type=int,
        default=[0,5,11,12, 19, 39,43,54],
        help=(
            "Selected data column indices to visualize. "
            "The Label column is excluded. Example: --columns 0 1 2 10"
        ),
    )
    parser.add_argument(
        "--max-channels",
        type=int,
        default=8,
        help=(
            "Maximum number of channels to visualize when --columns is not provided. "
            "Default: 8."
        ),
    )
    parser.add_argument(
        "--normalize-channels",
        action="store_true",
        default=False,
        help="Z-score normalize each selected channel before plotting.",
    )
    parser.add_argument(
        "--normalize-score",
        action="store_true",
        default=True,
        help="Normalize anomaly score to [0, 1] per sequence before plotting. Enabled by default.",
    )
    parser.add_argument(
        "--no-normalize-score",
        dest="normalize_score",
        action="store_false",
        help="Plot raw anomaly scores instead of normalized scores.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="svg",
        choices=["svg", "pdf", "png"],
        help="Output figure format.",
    )
    return parser.parse_args()


# ============================================================
# 2. Utility functions
# ============================================================

def contiguous_ranges(label: np.ndarray):
    label = np.asarray(label).astype(int)
    ranges = []
    start = None

    for idx, value in enumerate(label):
        if value == 1 and start is None:
            start = idx
        elif value != 1 and start is not None:
            ranges.append((start, idx - 1))
            start = None

    if start is not None:
        ranges.append((start, len(label) - 1))

    return ranges


def add_anomaly_regions(ax, anomaly_ranges, alpha: float = 0.14):
    """
    Add light red anomaly regions.
    """
    first = True
    for start, end in anomaly_ranges:
        ax.axvspan(
            start,
            end + 1,
            color="#E74C3C",
            alpha=alpha,
            linewidth=0,
            zorder=0,
            label="Ground-truth anomaly" if first else None,
        )
        first = False


def add_train_split(ax, train_index: Optional[int], series_length: int):
    """
    Add train/test split line.
    """
    if train_index is not None and 0 < train_index < series_length:
        ax.axvline(
            train_index,
            color="#111827",
            linestyle="--",
            linewidth=1.05,
            alpha=0.78,
            zorder=4,
            label="Train/test split",
        )


def style_axis(ax, is_last: bool = False):
    ax.set_facecolor("white")

    ax.grid(
        True,
        axis="y",
        linestyle="--",
        linewidth=0.58,
        alpha=0.30,
        color="#9CA3AF",
        zorder=1,
    )
    ax.grid(
        True,
        axis="x",
        linestyle=":",
        linewidth=0.42,
        alpha=0.16,
        color="#9CA3AF",
        zorder=1,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)

    ax.tick_params(axis="both", colors="#374151", labelsize=9, length=3, width=0.7)

    if not is_last:
        ax.tick_params(axis="x", labelbottom=False)


def compute_zscore(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (data - mean) / std


def normalize_score(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float32)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    low = float(score.min())
    high = float(score.max())

    if high - low < 1e-12:
        return np.zeros_like(score)

    return (score - low) / (high - low)


def sanitize_score(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score)

    if score.ndim > 1:
        score = np.squeeze(score)

    if score.ndim > 1:
        score = score.reshape(score.shape[0], -1).mean(axis=1)

    return score.reshape(-1).astype(np.float32)


def infer_train_index_from_name(file_name: str) -> Optional[int]:
    """
    Example:
        009_MSL_id_8_Sensor_tr_714_1st_1390.csv -> 714
    """
    stem = Path(file_name).stem
    parts = stem.split("_")

    for i, token in enumerate(parts):
        if token == "tr" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None

    return None


def load_series(csv_path: Path):
    df = pd.read_csv(csv_path).dropna()

    if "Label" not in df.columns:
        raise ValueError(f"{csv_path} does not contain a Label column.")

    data_df = df.drop(columns=["Label"]).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    data = data_df.to_numpy(dtype=np.float32)

    label = pd.to_numeric(df["Label"], errors="coerce").fillna(0).to_numpy(dtype=int)

    train_index = infer_train_index_from_name(csv_path.name)

    return data, label, train_index, list(data_df.columns)


def select_columns(
    n_features: int,
    requested_columns: Optional[Sequence[int]],
    max_channels: int,
):
    if requested_columns is not None:
        selected = [int(c) for c in requested_columns if 0 <= int(c) < n_features]
    else:
        selected = list(range(min(n_features, max_channels)))

    if not selected:
        raise ValueError("No valid channels selected for visualization.")

    return selected


# ============================================================
# 3. Metrics and score path helpers
# ============================================================

def normalize_metrics_columns(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}

    if "dataset_family" not in metrics_df.columns and "dataset" in metrics_df.columns:
        rename_map["dataset"] = "dataset_family"

    if "file_name" not in metrics_df.columns and "file" in metrics_df.columns:
        rename_map["file"] = "file_name"

    return metrics_df.rename(columns=rename_map)


def infer_split_from_text(text: str) -> Optional[str]:
    text = str(text)
    normalized = text.replace("\\", "/")

    if "TSB-AD-U" in normalized or "/uni/" in normalized:
        return "U"

    if "TSB-AD-M" in normalized or "/multi/" in normalized:
        return "M"

    return None


def find_metrics_files(run_dir: Path) -> list[Path]:
    old_metrics_path = run_dir / "metrics" / "pamm_msl_smap_smd_file_metrics.csv"
    if old_metrics_path.exists():
        return [old_metrics_path]

    metrics_root = run_dir / "metrics"

    candidates = sorted(metrics_root.glob("*/*/detailed_metrics.csv"))
    if candidates:
        return candidates

    flat_metrics_path = metrics_root / "detailed_metrics.csv"
    if flat_metrics_path.exists():
        return [flat_metrics_path]

    raise FileNotFoundError(
        f"Cannot find metrics file under {metrics_root}. Expected old "
        "pamm_msl_smap_smd_file_metrics.csv or new */*/detailed_metrics.csv."
    )


def load_metrics(run_dir: Path) -> pd.DataFrame:
    frames = []

    for metrics_path in find_metrics_files(run_dir):
        frame = normalize_metrics_columns(pd.read_csv(metrics_path))
        frame["_metrics_path"] = str(metrics_path)

        if "split" not in frame.columns:
            inferred_split = infer_split_from_text(metrics_path)

            if inferred_split is None and "file_list" in frame.columns and not frame.empty:
                inferred_split = infer_split_from_text(str(frame["file_list"].iloc[0]))

            frame["split"] = inferred_split

        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def split_dir_name(split: str) -> str:
    return "uni" if split == "U" else "multi"


def resolve_score_path(run_dir: Path, row: dict) -> Path:
    split = row["split"]
    dataset_family = row["dataset_family"]
    stem = Path(row["file_name"]).stem

    model_name = str(row.get("model", "PAMM"))
    phase = row.get("phase")

    if phase:
        local_path = run_dir / "scores" / split_dir_name(split) / str(phase) / model_name / f"{stem}.npy"
        if local_path.exists():
            return local_path

    local_path = run_dir / "scores" / split / dataset_family / f"{stem}.npy"
    if local_path.exists():
        return local_path

    score_path = Path(str(row.get("score_path", "")))
    if score_path.exists():
        return score_path

    raise FileNotFoundError(f"Cannot find score file for {row['file_name']}.")


# ============================================================
# 4. Plotting
# ============================================================

def plot_sequence(
    data: np.ndarray,
    label: np.ndarray,
    score: np.ndarray,
    train_index: Optional[int],
    title: str,
    save_path: Path,
    normalize_score_flag: bool,
    selected_columns: Sequence[int],
    normalize_channels: bool,
    dpi: int,
):
    series_length, n_features = data.shape

    anomaly_ranges = contiguous_ranges(label)
    time = np.arange(series_length)

    selected_data = data[:, selected_columns].astype(np.float32)
    if normalize_channels:
        selected_data = compute_zscore(selected_data)

    score_to_plot = normalize_score(score) if normalize_score_flag else np.asarray(score, dtype=np.float32)

    n_channels = len(selected_columns)
    n_rows = n_channels + 1

    fig_height = max(3.0, 1.25 * n_rows)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(16.5, fig_height),
        sharex=True,
        constrained_layout=False,
    )

    if n_rows == 1:
        axes = [axes]

    fig.patch.set_facecolor("white")

    # -------------------------
    # Channel subplots
    # -------------------------
    for idx, col_idx in enumerate(selected_columns):
        ax = axes[idx]
        y = selected_data[:, idx]

        add_anomaly_regions(ax, anomaly_ranges, alpha=0.13)
        add_train_split(ax, train_index, series_length)

        ax.plot(
            time,
            y,
            linewidth=1.15,
            color="#1F2937",
            zorder=3,
        )

        y_min = float(np.nanmin(y))
        y_max = float(np.nanmax(y))

        if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
            ax.fill_between(
                time,
                y,
                y_min,
                color="#1F2937",
                alpha=0.045,
                linewidth=0,
                zorder=2,
            )
            margin = 0.08 * (y_max - y_min)
            ax.set_ylim(y_min - margin, y_max + margin)

        ax.set_ylabel(
            f"ch{col_idx}",
            rotation=0,
            labelpad=28,
            va="center",
            ha="right",
            fontsize=10.5,
            fontweight="bold",
            color="#111827",
        )

        style_axis(ax, is_last=False)

    # -------------------------
    # Score subplot
    # -------------------------
    ax_score = axes[-1]

    add_anomaly_regions(ax_score, anomaly_ranges, alpha=0.13)
    add_train_split(ax_score, train_index, series_length)

    ax_score.plot(
        time,
        score_to_plot,
        linewidth=1.35,
        color="#2563EB",
        zorder=3,
        label="Anomaly score",
    )

    ax_score.fill_between(
        time,
        score_to_plot,
        np.nanmin(score_to_plot),
        color="#2563EB",
        alpha=0.055,
        linewidth=0,
        zorder=2,
    )

    score_ylabel = "score\n(norm.)" if normalize_score_flag else "score"
    ax_score.set_ylabel(
        score_ylabel,
        rotation=0,
        labelpad=36,
        va="center",
        ha="right",
        fontsize=10.5,
        fontweight="bold",
        color="#111827",
    )

    ax_score.set_xlabel("time", fontsize=12, fontweight="bold", color="#111827")
    style_axis(ax_score, is_last=True)

    # -------------------------
    # Title and legend
    # -------------------------
    fig.suptitle(
        title,
        fontsize=15.5,
        fontweight="bold",
        y=0.992,
        color="#111827",
    )

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_items = [
        Line2D([0], [0], color="#1F2937", linewidth=1.2, label="Selected channel"),
        Line2D([0], [0], color="#2563EB", linewidth=1.4, label="Anomaly score"),
        Patch(facecolor="#E74C3C", alpha=0.13, label="Ground-truth anomaly"),
    ]

    if train_index is not None and 0 < train_index < series_length:
        legend_items.append(
            Line2D(
                [0],
                [0],
                color="#111827",
                linewidth=1.05,
                linestyle="--",
                label="Train/test split",
            )
        )

    fig.legend(
        handles=legend_items,
        loc="upper right",
        bbox_to_anchor=(0.987, 0.985),
        frameon=True,
        framealpha=0.96,
        edgecolor="#E5E7EB",
        facecolor="white",
        fontsize=9.5,
    )

    plt.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.93,
        bottom=0.065,
        hspace=0.16,
    )

    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 5. Main
# ============================================================

def main():
    args = parse_args()

    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir).resolve()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.linewidth": 0.85,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": args.dpi,
    })

    metrics_df = load_metrics(run_dir)
    metrics_df = metrics_df[metrics_df["status"] == "success"].copy()

    if "model" in metrics_df.columns and args.models:
        metrics_df = metrics_df[metrics_df["model"].isin(args.models)].copy()

    if args.splits:
        metrics_df = metrics_df[metrics_df["split"].isin(args.splits)].copy()

    if args.phases and "phase" in metrics_df.columns:
        metrics_df = metrics_df[metrics_df["phase"].isin(args.phases)].copy()

    if args.datasets:
        metrics_df = metrics_df[metrics_df["dataset_family"].isin(args.datasets)].copy()

    if metrics_df["split"].isna().any():
        missing_count = int(metrics_df["split"].isna().sum())
        raise ValueError(f"Could not infer split for {missing_count} metric rows.")

    if metrics_df.empty:
        raise ValueError("No successful metric rows matched the requested filters.")

    output_root = run_dir / "visualizations"
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for _, row in tqdm(metrics_df.iterrows(), total=len(metrics_df), desc="Visualizing scores"):
        row = row.to_dict()

        split = row["split"]
        dataset_family = row["dataset_family"]
        file_name = row["file_name"]
        model_name = str(row.get("model", "PAMM"))
        phase = row.get("phase", "legacy")

        csv_path = root / "Datasets" / f"TSB-AD-{split}" / file_name
        score_path = resolve_score_path(run_dir, row)

        data, label, train_index, data_columns = load_series(csv_path)
        score = sanitize_score(np.load(score_path, allow_pickle=False))

        if len(score) != len(label):
            raise ValueError(
                f"Score length mismatch for {file_name}: score={len(score)} label={len(label)}."
            )

        selected_columns = select_columns(
            n_features=data.shape[1],
            requested_columns=args.columns,
            max_channels=args.max_channels,
        )

        save_dir = output_root / model_name / split / str(phase) / dataset_family
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / f"{Path(file_name).stem}.{args.format}"

        title = (
            f"{Path(file_name).stem} | model=PAMM | "
            f"channels={selected_columns}"
        )

        plot_sequence(
            data=data,
            label=label,
            score=score,
            train_index=train_index,
            title=title,
            save_path=save_path,
            normalize_score_flag=args.normalize_score,
            selected_columns=selected_columns,
            normalize_channels=args.normalize_channels,
            dpi=args.dpi,
        )

        manifest_rows.append(
            {
                "model": model_name,
                "split": split,
                "phase": phase,
                "dataset_family": dataset_family,
                "file_name": file_name,
                "score_path": str(score_path),
                "figure_path": str(save_path),
                "selected_columns": ",".join(map(str, selected_columns)),
                "normalize_channels": bool(args.normalize_channels),
                "normalize_score": bool(args.normalize_score),
                "metrics_path": str(row.get("_metrics_path", "")),
            }
        )

    pd.DataFrame(manifest_rows).to_csv(output_root / "manifest.csv", index=False)

    print(f"Saved visualizations to: {output_root}")
    print(f"Manifest: {output_root / 'manifest.csv'}")


if __name__ == "__main__":
    main()