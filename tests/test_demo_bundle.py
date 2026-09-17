import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.inference import Predictor
from app.server import decode_audio, sample_recordings
from scripts import artifacts
from scripts.setup_demo import BUNDLE, verify_bundle


class BundledDemoTests(unittest.TestCase):
    def test_bundled_training_examples_remain_compatible_with_the_live_pipeline(self):
        verify_bundle(BUNDLE)
        with patch.object(artifacts, 'GRAPH_PATH', BUNDLE / 'auditory.npz'):
            predictor = Predictor(BUNDLE)
        self.assertEqual(predictor.results['n_clips'], 3000)
        paths = sorted((BUNDLE / 'recordings').glob('*.wav'))
        self.assertEqual([int(path.name[0]) for path in paths], list(range(10)))
        for path in paths:
            with self.subTest(recording=path.name):
                audio, sample_rate = decode_audio(path.read_bytes())
                result = predictor.predict(audio, sample_rate)
                self.assertEqual(result['digit'], int(path.name[0]))
                self.assertEqual(result['replay']['cumulative_spikes'][-1], result['n_spikes'])

    def test_fresh_checkout_can_serve_samples_without_the_full_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(artifacts, 'DATA', Path(directory)):
                paths = sample_recordings()
        self.assertEqual([int(path.name[0]) for path in paths], list(range(10)))
        self.assertTrue(all(path.parent == BUNDLE / 'recordings' for path in paths))


if __name__ == '__main__':
    unittest.main()
