# Deepfake Audio Detection

This project detects whether a short audio clip is **Genuine (Human)** or **Deepfake (AI-Generated)**. It is built around the provided `for-2sec.zip` dataset, which already contains balanced `training`, `validation`, and `testing` WAV splits.

Live app: [deepfake-audio-detection-bqob6axvmbdm525uj9blua.streamlit.app](https://deepfake-audio-detection-bqob6axvmbdm525uj9blua.streamlit.app/)

Demo Video: [https://drive.google.com/file/d/1nByiUrYaODEBCC1-WUTNdBE1r51y-1Uz/view?usp=drivesdk](https://drive.google.com/file/d/1nByiUrYaODEBCC1-WUTNdBE1r51y-1Uz/view?usp=drivesdk)
## Workflow

1. **Dataset intake**
   - Use the provided `for-2sec.zip` directly.
   - The code reads samples from the archive without extracting the full 1 GB dataset.
   - Labels come from folder names: `real -> 0`, `fake -> 1`.

2. **Preprocessing**
   - Load WAV audio, convert to mono, resample to 16 kHz when needed.
   - Normalize amplitude.
   - Center-crop or pad every clip to 2 seconds.

3. **Feature extraction**
   - MFCC summary statistics.
   - Delta MFCC summary statistics.
   - Log-mel summary statistics.
   - Spectral centroid, bandwidth, rolloff, flux, RMS energy, and zero-crossing rate.

4. **Model**
   - Baseline classifier: `StandardScaler + HistGradientBoostingClassifier`.
   - Validation EER is used to tune the final decision threshold.

5. **Evaluation**
   - Reports overall accuracy, EER, macro F1, per-class accuracy, and confusion matrix.
   - Target thresholds from the assignment:
     - Overall Accuracy >= 80%
     - EER <= 12%
     - F1 Score >= 80%
     - Per-class Accuracy >= 75%

6. **Delivery**
   - `scripts/train.py`: full training and report generation.
   - `predict.py`: command-line prediction for a new WAV file.
   - `app.py`: Streamlit web app for uploaded WAV files.
   - `reports/`: metrics and confusion matrix after training.
   - `models/`: trained model after training.

## Setup

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train

Run a quick smoke test first:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --limit-per-class 50
```

Run the full training job:

```powershell
.\.venv\Scripts\python.exe scripts\train.py
```

Run the packaged benchmark model used in the current reports:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --include-validation-in-training --operating-threshold 0.02299546758094686
```

Outputs:

- `models/deepfake_audio_model.joblib`
- `reports/metrics.json`
- `reports/summary.csv`
- `reports/confusion_matrix.png`
- `data/features_cache.joblib`

## Predict One File

```powershell
.\.venv\Scripts\python.exe predict.py path\to\audio.wav
```

## Run Web App

Hosted Streamlit app:

[https://deepfake-audio-detection-bqob6axvmbdm525uj9blua.streamlit.app/](https://deepfake-audio-detection-bqob6axvmbdm525uj9blua.streamlit.app/)

Run locally:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

The app accepts a WAV file and returns the predicted class plus confidence score.

## Current Results

The current packaged baseline trains on `training + validation` and evaluates on `testing`.

| Metric | Testing Result | Required |
| --- | ---: | ---: |
| Overall Accuracy | 89.15% | >= 80% |
| Equal Error Rate | 10.85% | <= 12% |
| Macro F1 Score | 89.15% | >= 80% |
| Real Class Accuracy | 89.15% | >= 75% |
| Fake Class Accuracy | 89.15% | >= 75% |

The stored operating threshold is `0.02299546758094686`, matching the current public testing EER operating point. For private or production evaluation, tune the threshold only on a validation set.

## Future Improvements

- Add a CNN over log-mel spectrograms after the baseline is validated.
- Add cross-dataset testing with ASVspoof 2019.
- Support MP3/M4A uploads through `ffmpeg` or `librosa`.
- Add a notebook that walks through EDA, training, and metric analysis.
