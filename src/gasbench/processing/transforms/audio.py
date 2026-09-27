"""Audio transforms for benchmark inputs."""

import math

import numpy as np


def apply_audio_robustness_augmentations(
    waveform,
    target_sr=16000,
    seed=None,
    gain_db=-6.0,
    snr_db=30.0,
    downsample_sr=8000,
):
    """Fixed robustness suite for preprocessed mono, floating-point PCM audio.

    Reduce bandwidth with a downsample/upsample round trip, change gain, then
    add white noise relative to the resulting signal RMS. The default suite
    preserves the model's sample rate and window length. Silence stays silent.
    Set downsample_sr or snr_db to None to disable that operation, or gain_db
    to zero for unity gain. Final samples are clipped to [-1, 1].

    Like the image/video suites, return (array, None, suite_name, params).
    A per-call RNG makes seeded noise independent of worker scheduling. Bump
    the audio augmentation cache version when changing the default suite.
    """
    from scipy.signal import resample_poly

    audio = np.array(waveform, dtype=np.float32, copy=True)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("Audio waveform must be a nonempty, finite mono array")
    if not isinstance(target_sr, int) or target_sr <= 0:
        raise ValueError("target_sr must be a positive integer")
    if downsample_sr is not None and (
        not isinstance(downsample_sr, int) or not 0 < downsample_sr <= target_sr
    ):
        raise ValueError("downsample_sr must be a positive integer <= target_sr")
    if not np.isfinite(gain_db) or (snr_db is not None and not np.isfinite(snr_db)):
        raise ValueError("gain_db and snr_db must be finite")

    if downsample_sr is not None and downsample_sr < target_sr:
        divisor = math.gcd(target_sr, downsample_sr)
        down, up = downsample_sr // divisor, target_sr // divisor
        reduced = resample_poly(audio, down, up)
        # Rational resampling rounds up; trim back to the original window.
        audio = resample_poly(reduced, up, down)[: audio.size]

    audio *= 10.0 ** (gain_db / 20.0)
    if snr_db is not None:
        rms = np.sqrt(np.mean(np.square(audio, dtype=np.float64)))
        if rms > 0:
            rng = np.random.default_rng(seed)
            noise = rng.normal(0.0, rms * 10.0 ** (-snr_db / 20.0), audio.shape)
            audio += noise.astype(np.float32)

    params = {
        "gain_db": gain_db,
        "snr_db": snr_db,
        "downsample_sr": downsample_sr,
        "target_sr": target_sr,
    }
    return np.clip(audio, -1.0, 1.0), None, "robustness_audio", params
