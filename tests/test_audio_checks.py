import asyncio
import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import soundfile as sf

from app.audio_checks import NoSpeechError, require_speech
from app.server import create_app, decode_audio


def voiced_fixture():
    time = np.arange(4000) / 8000
    envelope = np.sin(np.linspace(0, np.pi, len(time))) ** 2
    tone = sum(np.sin(2 * np.pi * frequency * time) / (index + 1)
               for index, frequency in enumerate([160, 480, 800, 1120]))
    return np.pad((tone * envelope * 0.15).astype(np.float32), (1600, 1600))


def wav_bytes(audio):
    buffer = io.BytesIO()
    sf.write(buffer, audio, 8000, format='WAV', subtype='FLOAT')
    return buffer.getvalue()


async def post_audio(application, audio):
    body = (b'--clip\r\nContent-Disposition: form-data; name="file"; filename="mic.wav"\r\n'
            b'Content-Type: audio/wav\r\n\r\n' + wav_bytes(audio) + b'\r\n--clip--\r\n')
    sent = False
    messages = []

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {'type': 'http.request', 'body': body, 'more_body': False}
        await asyncio.Future()

    async def send(message):
        messages.append(message)

    scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.4'},
             'http_version': '1.1', 'method': 'POST', 'scheme': 'http',
             'path': '/predict', 'raw_path': b'/predict', 'query_string': b'',
             'headers': [(b'content-type', b'multipart/form-data; boundary=clip')],
             'server': ('localhost', 8002), 'client': ('localhost', 1234)}
    await application(scope, receive, send)
    start = next(message for message in messages if message['type'] == 'http.response.start')
    response = b''.join(message.get('body', b'') for message in messages if message['type'] == 'http.response.body')
    return start['status'], json.loads(response)


class AudioCheckTests(unittest.TestCase):
    def test_silence_and_stationary_noise_are_rejected_before_normalization(self):
        with self.assertRaises(NoSpeechError):
            require_speech(np.zeros(16000), 8000)
        for seed in range(20):
            for amplitude in [0.00005, 0.001, 0.01, 0.1]:
                noise = np.random.default_rng(seed).normal(0, amplitude, 16000)
                with self.subTest(seed=seed, amplitude=amplitude):
                    with self.assertRaises(NoSpeechError):
                        require_speech(noise, 8000)

    def test_hum_dc_and_single_click_are_rejected(self):
        cases = [np.ones(16000) * 0.05,
                 np.sin(2 * np.pi * 100 * np.arange(16000) / 8000) * 0.05,
                 np.pad(np.ones(24) * 0.5, (5000, 10976))]
        for audio in cases:
            with self.assertRaises(NoSpeechError):
                require_speech(audio, 8000)

    def test_accepted_audio_reaches_the_encoder_unchanged(self):
        audio = voiced_fixture()
        original = audio.copy()
        stats = require_speech(audio, 8000)
        self.assertGreaterEqual(stats['voiced_ms'], 80)
        np.testing.assert_array_equal(audio, original)
        decoded, sample_rate = decode_audio(wav_bytes(audio))
        self.assertEqual(sample_rate, 8000)
        np.testing.assert_array_equal(decoded, original)

    def test_api_rejects_noise_without_running_the_digit_reader(self):
        application = create_app(Path('/tmp/unused-fly-audio-test'))
        predictor = Mock()
        application.state.predictor = predictor
        for audio in [np.zeros(16000), np.random.default_rng(21).normal(0, 0.001, 16000)]:
            status, result = asyncio.run(post_audio(application, audio))
            self.assertEqual(status, 400)
            self.assertEqual(result['code'], 'no_speech')
            self.assertNotIn('digit', result)
        predictor.predict.assert_not_called()

    def test_api_passes_accepted_recording_to_the_existing_reader(self):
        application = create_app(Path('/tmp/unused-fly-audio-test'))
        audio = voiced_fixture()
        predict = Mock(return_value={'digit': 7})
        application.state.predictor = SimpleNamespace(predict=predict)
        status, result = asyncio.run(post_audio(application, audio))
        self.assertEqual((status, result['digit']), (200, 7))
        predict.assert_called_once()
        np.testing.assert_array_equal(predict.call_args.args[0], audio)


if __name__ == '__main__':
    unittest.main()
