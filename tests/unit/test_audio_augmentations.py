"""Audio transform contracts: signal effects, input preservation, and local RNG."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from gasbench.processing.transforms import apply_audio_robustness_augmentations


def test_gain_uses_amplitude_decibels_and_clips_pcm():
    # Catches power-ratio gain (10 instead of 20) and gain being normalized away.
    waveform = np.array([-0.9, -0.1, 0.0, 0.1, 0.9], dtype=np.float32)
    output, *_ = apply_audio_robustness_augmentations(
        waveform, gain_db=6.0, snr_db=None, downsample_sr=None,
    )
    np.testing.assert_allclose(output, np.clip(waveform * 10 ** (6 / 20), -1, 1))


def test_noise_is_scaled_to_signal_rms():
    # Catches absolute-amplitude noise and using a power ratio for noise amplitude.
    waveform = np.full(100_000, 0.1, dtype=np.float32)
    output, *_ = apply_audio_robustness_augmentations(
        waveform, gain_db=0, snr_db=20, downsample_sr=None, seed=123,
    )
    noise = output - waveform
    measured_snr = 20 * np.log10(np.sqrt(np.mean(waveform**2) / np.mean(noise**2)))
    assert measured_snr == pytest.approx(20, abs=0.1)


def test_resampling_removes_high_frequencies_without_changing_duration():
    # Catches decimation without anti-aliasing or forgetting the upsample step.
    sample_rate = 16000
    time = np.arange(sample_rate) / sample_rate
    low = 0.1 * np.sin(2 * np.pi * 500 * time)
    high = 0.1 * np.sin(2 * np.pi * 6000 * time)
    output, *_ = apply_audio_robustness_augmentations(
        low + high, target_sr=sample_rate, downsample_sr=8000,
        gain_db=0, snr_db=None,
    )
    assert output.shape == time.shape
    # Ignore filter transients at the window boundaries.
    assert np.sqrt(np.mean((output[100:-100] - low[100:-100]) ** 2)) < 0.002


@pytest.mark.parametrize("length", [1, 17, 96000])
@pytest.mark.parametrize("silent", [False, True])
def test_suite_preserves_input_shape_dtype_and_finite_pcm(length, silent):
    # Odd/short windows expose rounding errors in the resampling round trip.
    waveform = np.zeros(length) if silent else np.linspace(-0.9, 0.9, length)
    original = waveform.copy()
    output, *_ = apply_audio_robustness_augmentations(waveform, seed=17)
    np.testing.assert_array_equal(waveform, original)
    assert not np.shares_memory(output, waveform)
    assert output.shape == waveform.shape
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert np.max(np.abs(output)) <= 1
    if silent:
        assert not np.any(output)
    else:
        assert not np.array_equal(output, waveform)


def test_noise_is_reproducible_without_changing_global_rng():
    waveform = np.sin(np.arange(3001) / 20).astype(np.float32) * 0.2

    def augment(seed):
        return apply_audio_robustness_augmentations(waveform, seed=seed)[0]

    global_state = np.random.get_state()
    seeds = list(range(12))
    serial = [augment(seed) for seed in seeds]
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(augment, seeds))
    for expected, actual in zip(serial, concurrent):
        np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(serial[0], serial[1])
    current_state = np.random.get_state()
    assert global_state[0] == current_state[0]
    np.testing.assert_array_equal(global_state[1], current_state[1])
    assert global_state[2:] == current_state[2:]


@pytest.mark.parametrize("waveform", [[], [[0.1, 0.2]], [np.nan], [np.inf]])
def test_rejects_invalid_waveforms(waveform):
    with pytest.raises(ValueError, match="mono array"):
        apply_audio_robustness_augmentations(waveform)
