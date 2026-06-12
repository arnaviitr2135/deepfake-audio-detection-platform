from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deepfake_audio_detector.audio_features import extract_features_from_wav


def decision_confidence(fake_score: float, threshold: float) -> float:
    if fake_score >= threshold:
        return 0.5 + 0.5 * ((fake_score - threshold) / max(1.0 - threshold, 1e-8))
    return 0.5 + 0.5 * ((threshold - fake_score) / max(threshold, 1e-8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict whether a WAV file is genuine or deepfake.")
    parser.add_argument("audio", type=Path, help="Path to a WAV file.")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "deepfake_audio_model.joblib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package = joblib.load(args.model)
    vector, _ = extract_features_from_wav(args.audio)
    fake_score = float(package["model"].predict_proba([vector])[0, 1])
    threshold = float(package.get("threshold", 0.5))
    label = "Deepfake (AI-Generated)" if fake_score >= threshold else "Genuine (Human)"
    confidence = decision_confidence(fake_score, threshold)
    print(
        json.dumps(
            {
                "file": str(args.audio),
                "prediction": label,
                "fake_probability": fake_score,
                "confidence": confidence,
                "threshold": threshold,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
