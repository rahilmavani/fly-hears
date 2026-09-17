import asyncio
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import soundfile as sf

from app.inference import Predictor
from app.server import create_app
from scripts import artifacts, encode, simulate, train
from tests.test_samples import get


class TrainingWorkflowTests(unittest.TestCase):
    def test_tiny_experiment_from_recordings_to_compatible_demo(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / 'data'
            recordings = data_dir / 'fsdd' / 'recordings'
            recordings.mkdir(parents=True)
            run_dir = data_dir / 'runs' / 'test'
            graph_path = data_dir / 'auditory.npz'
            weights = sp.csr_matrix(([40., -20., 30., 50.], ([2, 3, 4, 5], [0, 1, 2, 3])), shape=(6, 6))
            np.savez_compressed(
                graph_path, n=6, W_data=weights.data, W_indices=weights.indices,
                W_indptr=weights.indptr, jo_a=np.array([0]), jo_b=np.array([1]),
                jo_other=np.array([], dtype=int), hop=np.array([0, 0, 1, 1, 2, 2]),
                types=np.array(['JO-A', 'JO-B', 'a', 'b', 'c', 'd']),
            )
            for digit in range(10):
                for speaker in ['alice', 'bob']:
                    time = np.arange(1600) / encode.SR
                    audio = np.sin(2 * np.pi * (150 + digit * 120) * time) * np.sin(np.linspace(0, np.pi, len(time)))
                    sf.write(recordings / f'{digit}_{speaker}_0.wav', audio, encode.SR)
            with (patch.object(artifacts, 'DATA', data_dir),
                  patch.object(artifacts, 'GRAPH_PATH', graph_path),
                  contextlib.redirect_stdout(io.StringIO())):
                for mode in [[], ['--rewired']]:
                    argv = ['simulate', '--run', str(run_dir), '--batch-size', '7', *mode]
                    with patch('sys.argv', argv):
                        simulate.main()
                    before = {path.name: path.stat().st_mtime_ns for path in run_dir.glob('*.npz')}
                    with patch('sys.argv', argv):
                        simulate.main()
                    self.assertEqual(before, {path.name: path.stat().st_mtime_ns for path in run_dir.glob('*.npz')})
                with patch('sys.argv', ['train', '--run', str(run_dir)]):
                    train.main()
                report = json.loads((run_dir / 'results.json').read_text())
                for mode in ['real', 'rewired', 'no_brain']:
                    self.assertEqual(len(report[mode]['per_speaker']), 2)
                    self.assertEqual(np.asarray(report[mode]['clean']['confusion']).sum(), 20)
                    self.assertEqual(np.asarray(report[mode]['test_noise']['confusion']).sum(), 20)
                self.assertEqual(report['model_sha256'], artifacts.file_hash(run_dir / 'model.npz'))
                predictor = Predictor(run_dir)
                audio, sample_rate = sf.read(recordings / '4_alice_0.wav')
                result = predictor.predict(audio, sample_rate)
                self.assertIn(result['digit'], range(10))
                self.assertEqual(result['replay']['cumulative_spikes'][-1], result['n_spikes'])
                application = create_app(run_dir)
                async def check_status():
                    async with application.router.lifespan_context(application):
                        status, _, body = await get('/status', application)
                        self.assertEqual(status, 200)
                        self.assertTrue(json.loads(body)['ready'])
                asyncio.run(check_status())
                with graph_path.open('ab') as file:
                    file.write(b'changed')
                with self.assertRaisesRegex(ValueError, 'do not match'):
                    Predictor(run_dir)

    def test_untrained_demo_reports_unavailable_without_guessing(self):
        with tempfile.TemporaryDirectory() as directory:
            application = create_app(Path(directory))
            with contextlib.redirect_stdout(io.StringIO()):
                async def check_status():
                    async with application.router.lifespan_context(application):
                        status, _, body = await get('/status', application)
                        self.assertEqual(status, 200)
                        payload = json.loads(body)
                        self.assertFalse(payload['ready'])
                        self.assertIn('No trained model', payload['error'])
                asyncio.run(check_status())


if __name__ == '__main__':
    unittest.main()
