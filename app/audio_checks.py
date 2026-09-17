"""Reject recordings without enough speech before the digit reader sees them."""

import numpy as np
import webrtcvad

FRAME_MS = 20
MIN_ACTIVE_MS = 60
MIN_VOICED_MS = 80
MIN_FRAME_RMS = 10 ** (-50 / 20)


class NoSpeechError(ValueError):
    pass


def require_speech(audio, sample_rate):
    if sample_rate not in (8000, 16000, 32000, 48000):
        raise ValueError('Unsupported sample rate for speech detection.')
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError('Audio must be finite and mono.')
    if samples.size == 0:
        raise NoSpeechError('No speech detected. Say one digit clearly and try again.')

    # Analysis uses a copy; accepted recordings reach the trained encoder unchanged.
    samples = samples - samples.mean()
    frame_size = sample_rate * FRAME_MS // 1000
    samples = np.pad(samples, (0, (-len(samples)) % frame_size))
    frames = samples.reshape(-1, frame_size)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    background = float(np.percentile(rms, 20))
    active = rms > max(MIN_FRAME_RMS, background * 2)
    if int(active.sum()) * FRAME_MS < MIN_ACTIVE_MS:
        raise NoSpeechError('No clear speech above the background sound. Say one digit closer to the mic.')

    detector = webrtcvad.Vad(2)
    voiced = []
    for frame in frames:
        pcm = (np.clip(frame, -1, 1) * 32767).astype('<i2').tobytes()
        voiced.append(detector.is_speech(pcm, sample_rate))
    voiced_ms = int(np.count_nonzero(voiced)) * FRAME_MS
    if voiced_ms < MIN_VOICED_MS:
        raise NoSpeechError('No speech detected. Say one digit clearly and try again.')
    return {'voiced_ms': voiced_ms, 'active_ms': int(active.sum()) * FRAME_MS}
