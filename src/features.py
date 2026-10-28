"""Speaker-robust audio features for spoken-digit recognition.

Pipeline per clip: DC removal -> peak normalisation -> pre-emphasis -> 32 ms Hamming frames ->
energy VAD (keep the speech part) -> 13 MFCCs -> summary statistics over time.
Then every feature is z-scored inside each speaker, which removes speaker / microphone offsets.
"""
import os
from math import gcd
import numpy as np
import pandas as pd
from scipy.fft import dct, rfft
from scipy.io import wavfile
from scipy.signal import resample_poly

SR = 8000
NFFT = 256       # 32 ms frames
HOP = 80         # 10 ms hop
N_MEL = 26
N_MFCC = 13
N_SEG = 5        # time segments per word
VAD_DB = 35      # keep frames within 35 dB of the loudest frame


def load_audio(clip_id, data_dir):
    """(sample_rate, signal in [-1, 1]) of one clip."""
    sr, x = wavfile.read(os.path.join(data_dir, "audio", f"{clip_id}.wav"))
    return sr, x.astype(np.float64) / 32768.0


def audio_stats(df, data_dir):
    """Simple numbers per clip: length, loudness, DC offset, clipping, noise floor, silence."""
    rows = []
    for c in df["clip_id"]:
        sr, x = load_audio(c, data_dir)
        frames = x[:len(x) // 80 * 80].reshape(-1, 80)                      # 10 ms frames
        level = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)   # level of each frame in dB
        rows.append({"sr": sr,
                     "dur": len(x) / sr,
                     "peak": np.abs(x).max(),
                     "rms_db": 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-9),
                     "dc": x.mean(),
                     "clipped": np.abs(x).max() >= 0.99,
                     "floor_db": np.percentile(level, 10),
                     "dyn_db": level.max() - np.percentile(level, 10),
                     "sil": np.mean(np.abs(x) < 0.1 * np.abs(x).max())})
    return pd.DataFrame(rows, index=df.index)


def mel_filterbank():
    hz = np.linspace(0, SR / 2, NFFT // 2 + 1)
    mel_max = 2595 * np.log10(1 + (SR / 2) / 700)
    pts = 700 * (10 ** (np.linspace(0, mel_max, N_MEL + 2) / 2595) - 1)
    fb = np.zeros((N_MEL, len(hz)))
    for i in range(N_MEL):
        up = (hz - pts[i]) / (pts[i + 1] - pts[i])
        down = (pts[i + 2] - hz) / (pts[i + 2] - pts[i + 1])
        fb[i] = np.clip(np.minimum(up, down), 0, None)      # triangle filter
    return fb


FB = mel_filterbank()


def mfcc_feats(clip_id, data_dir):
    sr, x = load_audio(clip_id, data_dir)
    return mfcc_from_signal(x, sr)


def mfcc_from_signal(x, sr=SR):
    """91 numbers per clip: MFCC mean, std, delta-std and the mean of 5 time segments."""
    if sr != SR:                                  # e.g. 48 kHz microphone audio
        k = gcd(SR, sr)
        x = resample_poly(x, SR // k, sr // k)
    x = x - x.mean()                              # remove DC offset
    x = x / (np.abs(x).max() + 1e-9)              # same loudness for every clip
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])    # pre-emphasis
    if len(x) < NFFT:
        x = np.pad(x, (0, NFFT - len(x)))
    n = 1 + (len(x) - NFFT) // HOP
    frames = np.array([x[i * HOP:i * HOP + NFFT] for i in range(n)]) * np.hamming(NFFT)
    power = np.abs(rfft(frames, axis=1)) ** 2

    # keep only the speech: from the first to the last loud frame
    energy = 10 * np.log10(power.sum(axis=1) + 1e-10)
    loud = np.where(energy > energy.max() - VAD_DB)[0]
    power = power[loud[0]:loud[-1] + 1]

    mfcc = dct(np.log(power @ FB.T + 1e-10), norm="ortho", axis=1)[:, :N_MFCC]
    if len(mfcc) > 1:
        delta = np.gradient(mfcc, axis=0)
    else:
        delta = np.zeros_like(mfcc)
    # mean MFCC of 5 equal time segments = rough shape of the word over time
    if len(mfcc) >= N_SEG:
        segs = [s.mean(axis=0) for s in np.array_split(mfcc, N_SEG)]
    else:
        segs = [mfcc.mean(axis=0)] * N_SEG
    return np.concatenate([mfcc.mean(axis=0), mfcc.std(axis=0), delta.std(axis=0)] + segs)


def spk_norm(X, df):
    """z-score every feature inside each speaker (uses only the rows of df)."""
    X = pd.DataFrame(X, index=df.index)
    spk = df["g"].fillna("_") if "g" in df else pd.Series("_", index=df.index)
    out = X.copy()
    for s in spk.unique():
        rows = (spk == s).to_numpy()
        part = X[rows]
        out.loc[rows] = (part - part.mean()) / (part.std(ddof=0) + 1e-9)
    return out.to_numpy(dtype=float)


def make_features(df, data_dir):
    X = np.array([mfcc_feats(c, data_dir) for c in df["clip_id"]])
    return spk_norm(X, df)
