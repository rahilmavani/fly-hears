import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from scripts import artifacts, controls, encode, features, readout, train


class PipelineTests(unittest.TestCase):
    def test_single_clip_matches_every_batch_position(self):
        graph = features.Graph(
            sp.csr_matrix(([250., -50., 150.], ([2, 3, 3], [0, 1, 2])), shape=(4, 4)),
            np.array([1]), np.array([0]), np.array([], dtype=int),
            np.array([0, 0, 1, 2]), np.array(['a', 'b', 'c', 'd']),
        )
        energy = np.random.default_rng(2).random((3, 32, encode.STEPS), dtype=np.float32)
        batch, _, _ = features.simulate_batch(graph, energy)
        for index in range(len(energy)):
            single, counts, frames = features.simulate_batch(graph, energy[index:index + 1], record=True)
            np.testing.assert_array_equal(single[0], batch[index])
            self.assertEqual(sum(map(len, frames)), int(counts.sum()))
        reversed_batch, _, _ = features.simulate_batch(graph, energy[::-1])
        np.testing.assert_array_equal(batch, reversed_batch[::-1])

    def test_control_preserves_degrees_signs_and_outgoing_weight_sets(self):
        rng = np.random.default_rng(4)
        size = 30
        matrix = rng.integers(1, 10, size=(size, size)).astype(np.float32)
        matrix[rng.random((size, size)) > 0.15] = 0
        np.fill_diagonal(matrix, 0)
        matrix[:, ::2] *= -1
        original = sp.csr_matrix(matrix)
        scrambled, stats = controls.rewire(original)
        self.assertGreater(stats['accepted_swaps'], 0)
        self.assertGreater((original != scrambled).nnz, 0)
        self.assertEqual(original.nnz, scrambled.nnz)
        for axis in [0, 1]:
            np.testing.assert_array_equal(original.getnnz(axis=axis), scrambled.getnnz(axis=axis))
        for column in range(size):
            np.testing.assert_array_equal(np.sort(original[:, column].data), np.sort(scrambled[:, column].data))
        for sign in [1, -1]:
            np.testing.assert_array_equal((original.sign() == sign).sum(axis=1),
                                          (scrambled.sign() == sign).sum(axis=1))
        repeated, _ = controls.rewire(original)
        self.assertEqual((scrambled != repeated).nnz, 0)

    def test_silence_and_long_recording_handling(self):
        self.assertFalse(encode.to_steps(encode.filterbank(np.zeros(8000))).any())
        bands = np.zeros((32, 100), dtype=np.float32)
        bands[:, -10:] = 1
        fitted = encode.to_steps(bands)
        self.assertEqual(fitted.shape, (32, encode.STEPS))
        self.assertTrue(np.all(fitted[:, -1] == 1))
        with self.assertRaises(ValueError):
            encode.filterbank(np.array([np.nan]))

    def test_training_and_test_noise_use_different_reproducible_recordings(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '4_speaker_0.wav'
            time = np.arange(4000) / encode.SR
            audio = (np.sin(2 * np.pi * 400 * time) * np.sin(np.pi * time / time[-1])).astype(np.float32)
            sf.write(path, audio, encode.SR)
            train_noise = encode.encode_file(path, 'train_noise')
            test_noise = encode.encode_file(path, 'test_noise')
            np.testing.assert_array_equal(train_noise, encode.encode_file(path, 'train_noise'))
            self.assertFalse(np.array_equal(train_noise, test_noise))
            self.assertFalse(np.array_equal(train_noise, encode.encode_file(path, 'clean')))

    def test_held_out_speakers_and_noise_copies_do_not_enter_training(self):
        speakers = np.repeat(['a', 'b', 'c'], 10)
        clean = np.arange(30, dtype=np.float32)[:, None]
        data = {'clean': {'audio': clean, 'labels': np.tile(np.arange(10), 3)},
                'train_noise': {'audio': clean + 1000},
                'test_noise': {'audio': clean + 2000}}
        for held_out, training, testing in train.speaker_splits(speakers):
            self.assertTrue(np.all(speakers[training] != held_out))
            self.assertTrue(np.all(speakers[testing] == held_out))
            values, labels = train.training_rows(data, training, 'audio')
            self.assertEqual(len(values), 2 * len(training))
            self.assertFalse(np.any(values >= 2000))
            self.assertFalse(set((values[:, 0] % 1000).astype(int)) & set(testing))
            self.assertEqual(len(labels), len(values))

    def test_artifact_round_trip_and_stale_model_rejection(self):
        expected = {'format': 2, 'encoder': 'current'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.npz'
            artifacts.save_npz(path, {'signature': expected}, value=np.array([1, 2]))
            with np.load(path, allow_pickle=False) as data:
                metadata = artifacts.read_metadata(data)
                artifacts.require_signature(metadata, expected)
                np.testing.assert_array_equal(data['value'], [1, 2])
                with self.assertRaises(ValueError):
                    artifacts.require_signature(metadata, {'encoder': 'different'})
            with self.assertRaises(ValueError):
                artifacts.read_metadata({})

    def test_saved_readout_keeps_class_order_and_predictions(self):
        rng = np.random.default_rng(0)
        labels = np.repeat(np.arange(10), 8)
        values = sp.csr_matrix(np.eye(10)[labels] * 20 + rng.random((80, 10)))
        columns = readout.select_columns(values, n_neurons=10, bins=1, limit=10)
        model = readout.fit(values, labels, columns)
        before = readout.probabilities(model, values)
        self.assertGreater(np.mean(readout.predict(model, values) == labels), 0.95)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'toy-model.npz'
            artifacts.save_npz(path, {}, **model)
            with np.load(path, allow_pickle=False) as saved:
                after = readout.probabilities(saved, values)
            np.testing.assert_allclose(after, before)


if __name__ == '__main__':
    unittest.main()
