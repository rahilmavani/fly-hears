"""Convert audio into 32 frequency bands and fixed sensory-neuron inputs."""

import hashlib

import librosa
import numpy as np

SR = 8000
N_BANDS = 32
HOP_MS = 10
FMIN = 100
FMAX = 4000
COMPRESS = 2
STEPS = 300
P_MAX = 0.2
FLOOR_DB = -45
NOISE_PCT = 20
CONDITIONS = ('clean', 'train_noise', 'test_noise')


def filterbank(audio, sample_rate=SR):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError('Audio must be a finite, mono waveform.')
    if audio.size == 0 or not np.any(audio):
        return np.zeros((N_BANDS, 1), dtype=np.float32)
    if sample_rate != SR:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SR)
    audio, _ = librosa.effects.trim(audio, top_db=25)
    power = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=256, hop_length=SR * HOP_MS // 1000,
        n_mels=N_BANDS, fmin=FMIN, fmax=FMAX, power=2.0,
    )
    decibels = np.clip(librosa.power_to_db(power, ref=np.max), FLOOR_DB, 0)
    noise_floor = np.percentile(decibels, NOISE_PCT, axis=1, keepdims=True)
    energy = np.maximum(decibels - noise_floor, 0)
    return (energy / (energy.max() + 1e-9)).astype(np.float32)


def to_steps(bands):
    """Use 2x compression, fitting longer utterances instead of cutting off their ends."""
    repeated = np.repeat(bands, HOP_MS // COMPRESS, axis=1)
    if repeated.shape[1] > STEPS:
        positions = np.linspace(0, repeated.shape[1] - 1, STEPS)
        repeated = np.stack([
            np.interp(positions, np.arange(len(band)), band) for band in repeated
        ])
    padding = max(0, STEPS - repeated.shape[1])
    return np.pad(repeated, ((0, 0), (0, padding))).astype(np.float32)


def drive_matrix(energy, jo_a, jo_b, jo_other=None):
    half = N_BANDS // 2
    indices = [jo_b, jo_a]
    band_numbers = [np.arange(len(jo_b)) % half, half + np.arange(len(jo_a)) % half]
    if jo_other is not None and len(jo_other):
        indices.append(jo_other)
        band_numbers.append(np.arange(len(jo_other)) % N_BANDS)
    inputs = np.concatenate(indices)
    probabilities = energy[np.concatenate(band_numbers)] * P_MAX
    return inputs, probabilities.astype(np.float32)


def augment(audio, rng):
    """Synthetic noise and padding; this is not a substitute for real microphone tests."""
    rms = float(np.sqrt(np.mean(audio ** 2)))
    signal_to_noise_db = rng.uniform(10, 30)
    noise_scale = rms / 10 ** (signal_to_noise_db / 20)
    padding = rng.integers(0, SR // 4, size=2)
    padded = np.pad(audio, tuple(padding))
    noisy = padded + rng.normal(0, noise_scale, len(padded))
    return noisy.astype(np.float32)


def encode_file(path, condition='clean'):
    if condition not in CONDITIONS:
        raise ValueError(f'Unknown audio condition: {condition}')
    audio, sample_rate = librosa.load(path, sr=SR)
    if condition != 'clean':
        key = f'{path.name}:{condition}'.encode()
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little')
        audio = augment(audio, np.random.default_rng(seed))
    return to_steps(filterbank(audio, sample_rate))
