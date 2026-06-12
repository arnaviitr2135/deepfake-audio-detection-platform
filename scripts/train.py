from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deepfake_audio_detector.audio_features import extract_features_from_wav
from deepfake_audio_detector.dataset import LABEL_NAMES, find_dataset_zip, list_examples, read_zip_bytes
from deepfake_audio_detector.metrics import classification_report_dict, equal_error_rate
from deepfake_audio_detector.pipeline import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline deepfake audio detector.")
    parser.add_argument("--dataset", type=Path, default=None, help="Path to dataset zip. Defaults to the first zip in root.")
    parser.add_argument("--model-out", type=Path, default=ROOT / "models" / "deepfake_audio_model.joblib")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "features_cache.joblib")
    parser.add_argument("--limit-per-class", type=int, default=None, help="Fast smoke-test limit per split/class.")
    parser.add_argument("--force-features", action="store_true", help="Rebuild cached features.")
    parser.add_argument(
        "--include-validation-in-training",
        action="store_true",
        help="Train the final model on training+validation after feature extraction.",
    )
    parser.add_argument(
        "--operating-threshold",
        type=float,
        default=None,
        help="Threshold stored with the model. Defaults to validation EER threshold, or 0.5 when validation is included in training.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Parallel feature extraction workers.",
    )
    return parser.parse_args()


def filtered_examples(examples, limit_per_class: int | None):
    if limit_per_class is None:
        return examples
    counts: Counter[tuple[str, str]] = Counter()
    kept = []
    for example in examples:
        key = (example.split, example.label_name)
        if counts[key] < limit_per_class:
            kept.append(example)
            counts[key] += 1
    return kept


def build_feature_table(
    dataset_zip: Path,
    cache_path: Path,
    limit_per_class: int | None,
    force: bool,
    workers: int,
):
    cache_key = {
        "dataset": str(dataset_zip.resolve()),
        "limit_per_class": limit_per_class,
        "feature_version": 5,
    }
    if cache_path.exists() and not force:
        cached = joblib.load(cache_path)
        if cached.get("cache_key") == cache_key:
            print(f"Loaded feature cache: {cache_path}")
            return cached

    examples = filtered_examples(list_examples(dataset_zip), limit_per_class)
    if not examples:
        raise RuntimeError("No WAV examples found in dataset archive.")

    rows = []
    features = []
    feature_names = None
    started = time.time()

    def consume_result(example, vector, names):
        nonlocal feature_names
        if feature_names is None:
            feature_names = names
        features.append(vector)
        rows.append(
            {
                "split": example.split,
                "label": example.label,
                "label_name": example.label_name,
                "path": example.path,
            }
        )

    def log_progress(done: int) -> None:
        if done % 500 == 0 or done == len(examples):
            elapsed = time.time() - started
            print(f"Extracted {done}/{len(examples)} files in {elapsed:.1f}s")

    if workers <= 1:
        with zipfile.ZipFile(dataset_zip) as zf:
            for idx, example in enumerate(examples, start=1):
                wav_bytes = zf.read(example.path)
                vector, names = extract_features_from_wav(wav_bytes)
                consume_result(example, vector, names)
                log_progress(idx)
    else:
        lock = threading.Lock()
        with zipfile.ZipFile(dataset_zip) as zf:
            def extract_one(example):
                with lock:
                    wav_bytes = zf.read(example.path)
                vector, names = extract_features_from_wav(wav_bytes)
                return example, vector, names

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(extract_one, example) for example in examples]
                for idx, future in enumerate(as_completed(futures), start=1):
                    consume_result(*future.result())
                    log_progress(idx)

    table = pd.DataFrame(rows)
    payload = {
        "cache_key": cache_key,
        "X": np.vstack(features),
        "metadata": table,
        "feature_names": feature_names,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, cache_path)
    print(f"Saved feature cache: {cache_path}")
    return payload


def save_confusion_matrix(cm: list[list[int]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    arr = np.array(cm)
    image = ax.imshow(arr, cmap="Blues")
    ax.set_xticks([0, 1], ["Predicted Real", "Predicted Fake"])
    ax.set_yticks([0, 1], ["Actual Real", "Actual Fake"])
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            ax.text(x, y, str(arr[y, x]), ha="center", va="center", color="black")
    ax.set_title("Confusion Matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_zip = args.dataset or find_dataset_zip(ROOT)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    payload = build_feature_table(dataset_zip, args.cache, args.limit_per_class, args.force_features, args.workers)
    X = payload["X"]
    meta = payload["metadata"]

    if args.include_validation_in_training:
        train_mask = meta["split"].isin(["training", "validation"])
    else:
        train_mask = meta["split"].eq("training")
    val_mask = meta["split"].eq("validation")
    test_mask = meta["split"].eq("testing")

    model = build_model()
    print("Training model...")
    model.fit(X[train_mask], meta.loc[train_mask, "label"].to_numpy())

    if args.operating_threshold is not None:
        tuned_threshold = args.operating_threshold
    elif args.include_validation_in_training:
        tuned_threshold = 0.5
    else:
        val_scores = model.predict_proba(X[val_mask])[:, 1]
        _, tuned_threshold = equal_error_rate(meta.loc[val_mask, "label"].to_numpy(), val_scores)

    package = {
        "model": model,
        "threshold": float(tuned_threshold),
        "feature_names": payload["feature_names"],
        "label_names": LABEL_NAMES,
        "dataset": str(dataset_zip),
    }
    joblib.dump(package, args.model_out)
    print(f"Saved model: {args.model_out}")

    reports = {}
    for split_name, mask in (("validation", val_mask), ("testing", test_mask)):
        scores = model.predict_proba(X[mask])[:, 1]
        y_true = meta.loc[mask, "label"].to_numpy()
        reports[split_name] = classification_report_dict(y_true, scores, threshold=tuned_threshold)
        eer_threshold = reports[split_name]["eer_threshold"]
        reports[split_name]["at_eer_threshold"] = classification_report_dict(
            y_true, scores, threshold=eer_threshold
        )

    metrics_path = args.report_dir / "metrics.json"
    metrics_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    save_confusion_matrix(reports["testing"]["confusion_matrix"], args.report_dir / "confusion_matrix.png")

    rows = []
    for split, report in reports.items():
        rows.append(
            {
                "split": split,
                "accuracy": report["accuracy"],
                "eer": report["eer"],
                "f1_macro": report["f1_macro"],
                "real_accuracy": report["per_class_accuracy"]["real"],
                "fake_accuracy": report["per_class_accuracy"]["fake"],
                "accuracy_at_eer_threshold": report["at_eer_threshold"]["accuracy"],
                "f1_at_eer_threshold": report["at_eer_threshold"]["f1_macro"],
                "threshold": report["threshold"],
                "eer_threshold": report["eer_threshold"],
            }
        )
    pd.DataFrame(rows).to_csv(args.report_dir / "summary.csv", index=False)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
