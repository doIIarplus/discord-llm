#!/usr/bin/env python3
"""Compare TTS parameter sweep samples against reference audio.

Uses speaker embeddings (resemblyzer) and prosodic features (librosa)
to rank which generated samples most closely match the reference voice.

Usage: python tts_voice_compare.py [--reference PATH] [--samples-dir DIR]
"""

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np


def load_audio(path: Path, sr: int = 16000) -> np.ndarray:
    """Load audio file, convert to mono, resample to target sr."""
    audio, _ = librosa.load(str(path), sr=sr, mono=True)
    return audio


def speaker_embedding_similarity(ref_audio: np.ndarray, sample_audio: np.ndarray) -> float:
    """Compute cosine similarity between speaker embeddings."""
    from resemblyzer import VoiceEncoder, preprocess_wav
    encoder = VoiceEncoder()
    ref_processed = preprocess_wav(ref_audio)
    sample_processed = preprocess_wav(sample_audio)
    ref_embed = encoder.embed_utterance(ref_processed)
    sample_embed = encoder.embed_utterance(sample_processed)
    # Cosine similarity
    return float(np.dot(ref_embed, sample_embed) /
                 (np.linalg.norm(ref_embed) * np.linalg.norm(sample_embed)))


def mfcc_similarity(ref_audio: np.ndarray, sample_audio: np.ndarray, sr: int = 16000) -> float:
    """Compare MFCC distributions using mean/std of each coefficient."""
    ref_mfcc = librosa.feature.mfcc(y=ref_audio, sr=sr, n_mfcc=20)
    sample_mfcc = librosa.feature.mfcc(y=sample_audio, sr=sr, n_mfcc=20)
    # Compare mean and std of each MFCC coefficient
    ref_stats = np.concatenate([ref_mfcc.mean(axis=1), ref_mfcc.std(axis=1)])
    sample_stats = np.concatenate([sample_mfcc.mean(axis=1), sample_mfcc.std(axis=1)])
    # Cosine similarity of the stats vectors
    return float(np.dot(ref_stats, sample_stats) /
                 (np.linalg.norm(ref_stats) * np.linalg.norm(sample_stats)))


def pitch_similarity(ref_audio: np.ndarray, sample_audio: np.ndarray, sr: int = 16000) -> float:
    """Compare pitch (F0) distributions — mean, std, and range."""
    def pitch_stats(audio):
        f0, voiced, _ = librosa.pyin(audio, fmin=50, fmax=500, sr=sr)
        f0_voiced = f0[voiced & ~np.isnan(f0)]
        if len(f0_voiced) < 5:
            return np.array([150.0, 30.0, 100.0, 200.0])
        return np.array([
            np.mean(f0_voiced),
            np.std(f0_voiced),
            np.percentile(f0_voiced, 10),
            np.percentile(f0_voiced, 90),
        ])

    ref_p = pitch_stats(ref_audio)
    sample_p = pitch_stats(sample_audio)
    # Normalized L2 distance → similarity (1 = identical)
    diff = np.abs(ref_p - sample_p) / (ref_p + 1e-6)
    return float(1.0 - np.clip(np.mean(diff), 0, 1))


def spectral_similarity(ref_audio: np.ndarray, sample_audio: np.ndarray, sr: int = 16000) -> float:
    """Compare spectral shape: centroid, bandwidth, rolloff distributions."""
    def spectral_stats(audio):
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        return np.array([
            np.mean(centroid), np.std(centroid),
            np.mean(bandwidth), np.std(bandwidth),
            np.mean(rolloff), np.std(rolloff),
        ])

    ref_s = spectral_stats(ref_audio)
    sample_s = spectral_stats(sample_audio)
    return float(np.dot(ref_s, sample_s) /
                 (np.linalg.norm(ref_s) * np.linalg.norm(sample_s)))


def speaking_rate_similarity(ref_audio: np.ndarray, sample_audio: np.ndarray, sr: int = 16000) -> float:
    """Compare speaking rate via onset density (onsets per second)."""
    def onset_rate(audio):
        onsets = librosa.onset.onset_detect(y=audio, sr=sr, units="time")
        duration = len(audio) / sr
        return len(onsets) / duration if duration > 0 else 0

    ref_rate = onset_rate(ref_audio)
    sample_rate = onset_rate(sample_audio)
    if ref_rate == 0:
        return 0.5
    ratio = sample_rate / ref_rate
    # Penalize deviation from 1.0
    return float(1.0 - min(abs(ratio - 1.0), 1.0))


def energy_dynamics_similarity(ref_audio: np.ndarray, sample_audio: np.ndarray, sr: int = 16000) -> float:
    """Compare energy envelope dynamics (how volume varies over time)."""
    def energy_stats(audio):
        rms = librosa.feature.rms(y=audio)[0]
        return np.array([
            np.mean(rms), np.std(rms),
            np.max(rms) / (np.mean(rms) + 1e-8),  # peak-to-mean ratio
            np.std(np.diff(rms)),  # energy variability
        ])

    ref_e = energy_stats(ref_audio)
    sample_e = energy_stats(sample_audio)
    diff = np.abs(ref_e - sample_e) / (ref_e + 1e-6)
    return float(1.0 - np.clip(np.mean(diff), 0, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="voices/trump.mp3",
                        help="Path to reference voice audio")
    parser.add_argument("--samples-dir", default="api_out/tts_test",
                        help="Directory containing generated samples")
    args = parser.parse_args()

    ref_path = Path(args.reference)
    samples_dir = Path(args.samples_dir)

    if not ref_path.exists():
        print(f"Reference not found: {ref_path}", file=sys.stderr)
        sys.exit(1)

    samples = sorted(samples_dir.glob("*.wav"))
    if not samples:
        print(f"No WAV files in {samples_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Reference: {ref_path}")
    print(f"Samples:   {len(samples)} files in {samples_dir}/")
    print(f"\nLoading reference audio...")

    sr = 16000
    ref_audio = load_audio(ref_path, sr=sr)
    print(f"Reference: {len(ref_audio)/sr:.1f}s")

    # Weights for the composite score
    weights = {
        "speaker_embed": 0.35,  # voice identity (most important)
        "mfcc": 0.20,           # timbral quality
        "pitch": 0.15,          # intonation/F0
        "spectral": 0.10,       # frequency shape
        "speaking_rate": 0.10,  # pacing
        "energy": 0.10,         # dynamics
    }

    print(f"\nAnalyzing {len(samples)} samples...\n")
    print(f"{'File':<40} {'Speaker':>8} {'MFCC':>6} {'Pitch':>6} {'Spec':>6} "
          f"{'Rate':>6} {'Energy':>6} {'TOTAL':>7}")
    print(f"{'-'*40} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")

    results = []
    for sample_path in samples:
        sample_audio = load_audio(sample_path, sr=sr)

        scores = {
            "speaker_embed": speaker_embedding_similarity(ref_audio, sample_audio),
            "mfcc": mfcc_similarity(ref_audio, sample_audio, sr),
            "pitch": pitch_similarity(ref_audio, sample_audio, sr),
            "spectral": spectral_similarity(ref_audio, sample_audio, sr),
            "speaking_rate": speaking_rate_similarity(ref_audio, sample_audio, sr),
            "energy": energy_dynamics_similarity(ref_audio, sample_audio, sr),
        }
        total = sum(scores[k] * weights[k] for k in weights)

        results.append({"file": sample_path.name, "scores": scores, "total": total})
        print(f"{sample_path.name:<40} {scores['speaker_embed']:>8.4f} "
              f"{scores['mfcc']:>6.4f} {scores['pitch']:>6.4f} "
              f"{scores['spectral']:>6.4f} {scores['speaking_rate']:>6.4f} "
              f"{scores['energy']:>6.4f} {total:>7.4f}")

    # Sort by total score descending
    results.sort(key=lambda x: x["total"], reverse=True)

    print(f"\n{'='*80}")
    print("RANKED RESULTS (best → worst)")
    print(f"{'='*80}")
    for i, r in enumerate(results, 1):
        # Extract params from filename
        parts = r["file"].replace(".wav", "").split("_")
        exag = parts[1].replace("exag", "")
        cfg = parts[2].replace("cfg", "")
        print(f"  {i:>2}. {r['file']:<40} score={r['total']:.4f}  "
              f"(exag={exag}, cfg={cfg})")

    # Top recommendation
    best = results[0]
    parts = best["file"].replace(".wav", "").split("_")
    exag = parts[1].replace("exag", "")
    cfg = parts[2].replace("cfg", "")
    print(f"\n{'='*80}")
    print(f"RECOMMENDED: exaggeration={exag}, cfg_weight={cfg}  (score={best['total']:.4f})")
    print(f"{'='*80}")

    # Also show top 3 with their per-metric breakdown
    print(f"\nTop 3 detailed breakdown:")
    for i, r in enumerate(results[:3], 1):
        print(f"\n  #{i}: {r['file']}")
        for metric, score in r["scores"].items():
            w = weights[metric]
            print(f"      {metric:<16} = {score:.4f}  (weight {w:.2f}, contribution {score*w:.4f})")
        print(f"      {'TOTAL':<16} = {r['total']:.4f}")


if __name__ == "__main__":
    main()
