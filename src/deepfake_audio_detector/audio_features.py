from __future__ import annotations

import io
import math
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.fftpack import dct
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 16_000
CLIP_SECONDS = 2.0
N_MELS = 40
N_MFCC = 20
N_FFT = 512
FRAME_MS = 25
HOP_MS = 10


def load_wav(source: str | Path | bytes | io.BytesIO, target_sr: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load a WAV file into mono float32 audio in [-1, 1]."""
    close_buffer = False
    if isinstance(source, (str, Path)):
        fh = wave.open(str(source), "rb")
    else:
        if isinstance(source, bytes):
            source = io.BytesIO(source)
            close_buffer = True
        fh = wave.open(source, "rb")

    with fh:
        sample_rate = fh.getframerate()
        channels = fh.getnchannels()
        sample_width = fh.getsampwidth()
        frames = fh.readframes(fh.getnframes())

    if close_buffer:
        source.close()

    if sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        signed = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        audio = signed.astype(np.float32) / 8_388_608.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2_147_483_648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if sample_rate != target_sr:
        gcd = math.gcd(sample_rate, target_sr)
        audio = resample_poly(audio, target_sr // gcd, sample_rate // gcd).astype(np.float32)
        sample_rate = target_sr

    return audio.astype(np.float32), sample_rate


def normalize_clip(audio: np.ndarray, sample_rate: int, seconds: float = CLIP_SECONDS) -> np.ndarray:
    target_len = int(sample_rate * seconds)
    if audio.size == 0:
        return np.zeros(target_len, dtype=np.float32)

    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    if audio.size >= target_len:
        start = max((audio.size - target_len) // 2, 0)
        return audio[start : start + target_len]

    pad_left = (target_len - audio.size) // 2
    pad_right = target_len - audio.size - pad_left
    return np.pad(audio, (pad_left, pad_right)).astype(np.float32)


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


@lru_cache(maxsize=16)
def mel_filterbank(sample_rate: int, n_fft: int = N_FFT, n_mels: int = N_MELS) -> np.ndarray:
    low_mel = hz_to_mel(np.array([0.0]))[0]
    high_mel = hz_to_mel(np.array([sample_rate / 2]))[0]
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center > left:
            filters[m - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            filters[m - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return filters


@lru_cache(maxsize=16)
def hamming_window(frame_len: int) -> np.ndarray:
    return np.hamming(frame_len).astype(np.float32)


def frame_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    frame_len = int(sample_rate * FRAME_MS / 1000)
    hop_len = int(sample_rate * HOP_MS / 1000)
    if audio.size < frame_len:
        audio = np.pad(audio, (0, frame_len - audio.size))

    frame_count = 1 + (audio.size - frame_len) // hop_len
    shape = (frame_count, frame_len)
    strides = (audio.strides[0] * hop_len, audio.strides[0])
    frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides).copy()
    frames *= hamming_window(frame_len)
    return frames


def summarize(matrix: np.ndarray, prefix: str) -> tuple[list[str], np.ndarray]:
    stats = [
        matrix.mean(axis=0),
        matrix.std(axis=0),
        np.percentile(matrix, 5, axis=0),
        np.percentile(matrix, 95, axis=0),
    ]
    names = []
    for stat_name in ("mean", "std", "p05", "p95"):
        names.extend([f"{prefix}_{stat_name}_{i:02d}" for i in range(matrix.shape[1])])
    return names, np.concatenate(stats).astype(np.float32)


def summarize_segments(matrix: np.ndarray, prefix: str, segments: int = 4) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    values: list[np.ndarray] = []
    for segment_idx, segment in enumerate(np.array_split(matrix, segments, axis=0)):
        if segment.size == 0:
            segment = np.zeros((1, matrix.shape[1]), dtype=matrix.dtype)
        names.extend([f"{prefix}_seg{segment_idx}_mean_{i:02d}" for i in range(matrix.shape[1])])
        names.extend([f"{prefix}_seg{segment_idx}_std_{i:02d}" for i in range(matrix.shape[1])])
        values.append(segment.mean(axis=0))
        values.append(segment.std(axis=0))
    return names, np.concatenate(values).astype(np.float32)


def extract_features(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, list[str]]:
    clip = normalize_clip(audio, sample_rate)
    emphasized = np.append(clip[0], clip[1:] - 0.97 * clip[:-1])
    frames = frame_audio(emphasized, sample_rate)

    spectrum = np.fft.rfft(frames, n=N_FFT)
    power = (np.abs(spectrum) ** 2) / N_FFT
    magnitudes = np.sqrt(power + 1e-12)
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / sample_rate)

    mel_energy = np.maximum(power @ mel_filterbank(sample_rate).T, 1e-10)
    log_mel = np.log(mel_energy)
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :N_MFCC]
    delta = np.gradient(mfcc, axis=0)

    names: list[str] = []
    values: list[np.ndarray] = []
    for prefix, matrix in (("mfcc", mfcc), ("delta", delta), ("logmel", log_mel)):
        stat_names, stat_values = summarize(matrix, prefix)
        names.extend(stat_names)
        values.append(stat_values)
        if prefix in {"mfcc", "delta", "logmel"}:
            segment_names, segment_values = summarize_segments(matrix, prefix)
            names.extend(segment_names)
            values.append(segment_values)

    mag_sum = np.maximum(magnitudes.sum(axis=1), 1e-12)
    centroid = (magnitudes * freqs).sum(axis=1) / mag_sum
    bandwidth = np.sqrt(((freqs - centroid[:, None]) ** 2 * magnitudes).sum(axis=1) / mag_sum)
    cumulative = np.cumsum(magnitudes, axis=1)
    rolloff_idx = (cumulative >= 0.85 * cumulative[:, -1:]).argmax(axis=1)
    rolloff = freqs[rolloff_idx]
    flux = np.sqrt(np.mean(np.diff(magnitudes, axis=0) ** 2, axis=1))
    rms = np.sqrt(np.mean(frames**2, axis=1))
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)

    handcrafted = np.column_stack(
        [
            centroid,
            bandwidth,
            rolloff,
            np.pad(flux, (1, 0)),
            rms,
            zcr,
        ]
    )
    stat_names, stat_values = summarize(handcrafted, "spectral")
    handcrafted_names = ["centroid", "bandwidth", "rolloff", "flux", "rms", "zcr"]
    stat_names = [
        name.replace(f"_{idx:02d}", f"_{handcrafted_names[idx]}")
        for name in stat_names
        for idx in range(len(handcrafted_names))
        if name.endswith(f"_{idx:02d}")
    ]
    names.extend(stat_names)
    values.append(stat_values)

    time_features = np.array(
        [
            clip.mean(),
            clip.std(),
            np.max(np.abs(clip)),
            np.mean(np.abs(clip)),
            np.percentile(np.abs(clip), 95),
        ],
        dtype=np.float32,
    )
    names.extend(["wave_mean", "wave_std", "wave_peak", "wave_abs_mean", "wave_abs_p95"])
    values.append(time_features)

    return np.concatenate(values).astype(np.float32), names


def extract_features_from_wav(source: str | Path | bytes | io.BytesIO) -> tuple[np.ndarray, list[str]]:
    audio, sample_rate = load_wav(source)
    return extract_features(audio, sample_rate)
