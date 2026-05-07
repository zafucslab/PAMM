from __future__ import annotations

import argparse
import csv
import gc
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

np = None
torch = None
Optimal_Multi_algo_HP_dict = None


DEFAULT_DATASETS = ["MSL", "SMAP", "SMD", "Genesis", "CATSv2"]
DEFAULT_MODELS = [
    "AnomalyTransformer",
    "AutoEncoder",
    "FITS",
    "LSTMAD",
    "OmniAnomaly",
    "PatchTST",
    "TimesNet",
    "TranAD",
    "PAMM",
]
SEED = 2024


def load_runtime_modules() -> None:
    global np
    global torch
    global Optimal_Multi_algo_HP_dict

    import numpy as numpy_module
    import torch as torch_module

    from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict as hp_dict

    np = numpy_module
    torch = torch_module
    Optimal_Multi_algo_HP_dict = hp_dict


class DataLoaderIterationTimer:
    def __init__(self):
        self.times: List[float] = []
        self._original_iter = None

    def reset(self) -> None:
        self.times.clear()

    def __enter__(self):
        import torch.utils.data

        self._original_iter = torch.utils.data.DataLoader.__iter__
        timer = self

        def timed_iter(loader):
            iterator = timer._original_iter(loader)
            for batch in iterator:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                yield batch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timer.times.append(time.perf_counter() - start)

        torch.utils.data.DataLoader.__iter__ = timed_iter
        return self

    def __exit__(self, exc_type, exc, tb):
        import torch.utils.data

        if self._original_iter is not None:
            torch.utils.data.DataLoader.__iter__ = self._original_iter
        self._original_iter = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile parameter count, runtime, train/test time, peak GPU memory, "
            "and average DataLoader iteration time for multivariate TSB-AD models."
        )
    )
    parser.add_argument("--split", type=str, default="M", choices=["M"])
    parser.add_argument("--phase", type=str, default="eval", choices=["eval", "tuning"])
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=None,
        help="Directory containing split csv files. Defaults to Datasets/TSB-AD-M.",
    )
    parser.add_argument(
        "--file_list",
        type=Path,
        default=None,
        help="Explicit benchmark file list. Defaults to the selected multivariate phase list.",
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--limit_per_dataset", type=int, default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "test" / "metrics" / "multi" / "eval" / "model_resource_profile_epoch2.csv",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def dataset_name_from_file(filename: str, dataset_names: List[str]) -> str:
    for dataset_name in dataset_names:
        if f"_{dataset_name}_" in filename:
            return dataset_name
    return "Unknown"


def get_split_paths(phase: str) -> Tuple[Path, Path]:
    list_suffix = "Tuning" if phase == "tuning" else "Eva"
    return (
        ROOT / "Datasets" / "TSB-AD-M",
        ROOT / "Datasets" / "File_List" / f"TSB-AD-M-{list_suffix}.csv",
    )


def read_file_list(file_list_path: Path) -> List[str]:
    with file_list_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row["file_name"] for row in reader]


def load_target_files(
    file_list_path: Path,
    dataset_names: List[str],
    limit_per_dataset: Optional[int] = None,
    max_files: Optional[int] = None,
) -> List[str]:
    selected_files = []
    dataset_counts = {name: 0 for name in dataset_names}
    for filename in read_file_list(file_list_path):
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


def load_file_data(filename: str, dataset_dir: Path) -> np.ndarray:
    file_path = dataset_dir / filename
    data_rows = []
    with file_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "Label" not in reader.fieldnames:
            raise ValueError(f"{file_path} does not contain a Label column.")
        data_columns = [name for name in reader.fieldnames if name != "Label"]
        for row in reader:
            try:
                values = [float(row[column]) for column in data_columns]
                float(row["Label"])
            except (TypeError, ValueError):
                continue
            if any(np.isnan(value) for value in values):
                continue
            data_rows.append(values)
    return np.asarray(data_rows, dtype=float)


def validate_models(models: List[str]) -> None:
    supported = set(DEFAULT_MODELS)
    missing = [model for model in models if model not in supported]
    if missing:
        raise ValueError(f"Unsupported models for resource profiling: {missing}")


def make_estimator(model_name: str, data: np.ndarray, epochs: int):
    hp = dict(Optimal_Multi_algo_HP_dict[model_name])
    input_c = data.shape[1]

    if model_name == "AutoEncoder":
        from TSB_AD.models.AE import AutoEncoder

        return AutoEncoder(
            slidingWindow=hp.get("window_size", 100),
            hidden_neurons=hp.get("hidden_neurons", [64, 32]),
            batch_size=128,
            epochs=epochs,
        )
    if model_name == "LSTMAD":
        from TSB_AD.models.LSTMAD import LSTMAD

        return LSTMAD(
            window_size=hp.get("window_size", 100),
            pred_len=1,
            lr=hp.get("lr", 0.0008),
            feats=input_c,
            batch_size=128,
            epochs=epochs,
        )
    if model_name == "TranAD":
        from TSB_AD.models.TranAD import TranAD

        return TranAD(
            win_size=hp.get("win_size", 10),
            feats=input_c,
            lr=hp.get("lr", 1e-3),
            epochs=epochs,
        )
    if model_name == "AnomalyTransformer":
        from TSB_AD.models.AnomalyTransformer import AnomalyTransformer

        return AnomalyTransformer(
            win_size=hp.get("win_size", 100),
            input_c=input_c,
            lr=hp.get("lr", 1e-4),
            batch_size=hp.get("batch_size", 128),
            num_epochs=epochs,
        )
    if model_name == "PatchTST":
        from TSB_AD.models.PatchTST import PatchTST

        return PatchTST(
            win_size=hp.get("win_size", 100),
            input_c=input_c,
            lr=hp.get("lr", 1e-4),
            batch_size=hp.get("batch_size", 128),
            epochs=epochs,
        )
    if model_name == "OmniAnomaly":
        from TSB_AD.models.OmniAnomaly import OmniAnomaly

        return OmniAnomaly(
            win_size=hp.get("win_size", 100),
            feats=input_c,
            lr=hp.get("lr", 0.002),
            epochs=epochs,
        )
    if model_name == "TimesNet":
        from TSB_AD.models.TimesNet import TimesNet

        return TimesNet(
            win_size=hp.get("win_size", 96),
            enc_in=input_c,
            lr=hp.get("lr", 1e-4),
            epochs=epochs,
        )
    if model_name == "FITS":
        from TSB_AD.models.FITS import FITS

        return FITS(
            win_size=hp.get("win_size", 100),
            input_c=input_c,
            lr=hp.get("lr", 1e-3),
            batch_size=hp.get("batch_size", 128),
            epochs=epochs,
        )
    if model_name == "PAMM":
        import inspect

        from TSB_AD.models.PAMM import PAMM

        params = dict(hp)
        params.update(
            {
                "input_c": input_c,
                "epochs": epochs,
                "save_proto_analysis": False,
                "proto_analysis_dir": None,
            }
        )
        signature = inspect.signature(PAMM)
        supported = {name for name in signature.parameters if name != "self"}
        return PAMM(**{key: value for key, value in params.items() if key in supported})

    raise ValueError(f"Unsupported model: {model_name}")


def count_parameters(estimator) -> int:
    modules = []
    if isinstance(estimator, torch.nn.Module):
        modules.append(estimator)
    for value in vars(estimator).values():
        if isinstance(value, torch.nn.Module):
            modules.append(value)

    seen = set()
    total = 0
    for module in modules:
        for param in module.parameters():
            ident = id(param)
            if ident in seen:
                continue
            seen.add(ident)
            total += param.numel()
    return int(total)


def reset_cuda_peak() -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def current_peak_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return float(torch.cuda.max_memory_reserved() / (1024 ** 2))


def profile_file(
    model_name: str,
    filename: str,
    dataset_dir: Path,
    epochs: int,
    timer: DataLoaderIterationTimer,
) -> Tuple[int, float, List[float], float, float, float]:
    data = load_file_data(filename, dataset_dir)
    train_index = int(filename.split(".")[0].split("_")[-3])
    data_train = data[:train_index, :]

    timer.reset()
    reset_cuda_peak()

    estimator = make_estimator(model_name, data, epochs=epochs)

    train_start = time.perf_counter()
    estimator.fit(data_train)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_time_s = time.perf_counter() - train_start

    param_count = count_parameters(estimator)

    test_start = time.perf_counter()
    output = estimator.decision_function(data)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    test_time_s = time.perf_counter() - test_start

    runtime_s = train_time_s + test_time_s
    peak_memory_mb = current_peak_memory_mb()

    del output
    del estimator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return param_count, peak_memory_mb, list(timer.times), runtime_s, train_time_s, test_time_s


def write_results(output_path: Path, rows: List[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "dataset",
                "param_count",
                "peak_gpu_memory_mb",
                "avg_itr_time_s",
                "runtime_s",
                "train_time_s",
                "test_time_s",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    load_runtime_modules()
    set_seed(SEED)
    validate_models(args.models)

    default_dataset_dir, default_file_list = get_split_paths(args.phase)
    dataset_dir = args.dataset_dir or default_dataset_dir
    file_list = args.file_list or default_file_list
    target_files = load_target_files(
        file_list,
        args.datasets,
        limit_per_dataset=args.limit_per_dataset,
        max_files=args.max_files,
    )
    if not target_files:
        raise ValueError("No files matched the dataset filters.")

    files_by_dataset: Dict[str, List[str]] = {dataset: [] for dataset in args.datasets}
    for filename in target_files:
        dataset_name = dataset_name_from_file(filename, args.datasets)
        if dataset_name != "Unknown":
            files_by_dataset[dataset_name].append(filename)

    print("CUDA available:", torch.cuda.is_available())
    print("Datasets:", ", ".join(args.datasets))
    print("Models:", ", ".join(args.models))
    print("Epochs:", args.epochs)
    print("Output:", args.output)

    rows = []
    with DataLoaderIterationTimer() as timer:
        for model_name in args.models:
            print(f"\n=== Profiling {model_name} ===")
            for dataset_name in args.datasets:
                param_counts = []
                peak_values = []
                iteration_times = []
                runtime_values = []
                train_time_values = []
                test_time_values = []
                for filename in files_by_dataset.get(dataset_name, []):
                    print(f"[{model_name}] {dataset_name} | {filename}")
                    try:
                        (
                            param_count,
                            peak_memory_mb,
                            times,
                            runtime_s,
                            train_time_s,
                            test_time_s,
                        ) = profile_file(
                            model_name=model_name,
                            filename=filename,
                            dataset_dir=dataset_dir,
                            epochs=args.epochs,
                            timer=timer,
                        )
                    except Exception as exc:
                        print(f"Failed on {filename} with {model_name}: {exc}")
                        continue
                    param_counts.append(param_count)
                    peak_values.append(peak_memory_mb)
                    iteration_times.extend(times)
                    runtime_values.append(runtime_s)
                    train_time_values.append(train_time_s)
                    test_time_values.append(test_time_s)

                param_count = max(param_counts) if param_counts else ""
                peak_gpu_memory_mb = max(peak_values) if peak_values else ""
                avg_itr_time_s = (
                    float(sum(iteration_times) / len(iteration_times))
                    if iteration_times
                    else ""
                )
                runtime_s = float(sum(runtime_values)) if runtime_values else ""
                train_time_s = float(sum(train_time_values)) if train_time_values else ""
                test_time_s = float(sum(test_time_values)) if test_time_values else ""
                rows.append(
                    {
                        "model": model_name,
                        "dataset": dataset_name,
                        "param_count": param_count,
                        "peak_gpu_memory_mb": peak_gpu_memory_mb,
                        "avg_itr_time_s": avg_itr_time_s,
                        "runtime_s": runtime_s,
                        "train_time_s": train_time_s,
                        "test_time_s": test_time_s,
                    }
                )
                write_results(args.output, rows)

    write_results(args.output, rows)
    print(f"\nResource profile saved to: {args.output}")


if __name__ == "__main__":
    main()
