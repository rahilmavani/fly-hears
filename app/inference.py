"""Load a compatible trained reader and run one recording through the network."""

import json
import time
from pathlib import Path

import numpy as np

from app.replay import build_replay
from scripts import artifacts, encode, features, fly_lif, readout


class Predictor:
    def __init__(self, run_dir):
        run_dir = Path(run_dir)
        model_path = run_dir / 'model.npz'
        if not model_path.exists():
            raise ValueError(f'No trained model in {run_dir}. Complete the training commands in README.md.')
        with np.load(model_path, allow_pickle=False) as data:
            metadata = artifacts.read_metadata(data)
            artifacts.require_signature(metadata, artifacts.signature())
            self.model = {key: data[key] for key in ['cols', 'mean', 'scale', 'coef', 'intercept', 'classes']}
        self.graph = features.load_graph(artifacts.GRAPH_PATH)
        if metadata['n_neurons'] != self.graph.size or metadata['bins'] != features.BINS:
            raise ValueError('Model feature dimensions do not match the network.')
        if not np.array_equal(self.model['classes'], np.arange(10)):
            raise ValueError('The reader must contain all ten digit classes.')
        results_path = run_dir / 'results.json'
        self.results = json.loads(results_path.read_text()) if results_path.exists() else {}
        if self.results.get('model_sha256') != artifacts.file_hash(model_path):
            self.results = {}

    def predict(self, audio, sample_rate):
        started = time.monotonic()
        energy = encode.to_steps(encode.filterbank(audio, sample_rate))
        if not np.any(energy):
            raise ValueError('No usable sound detected. Try speaking a little closer to the mic.')
        values, counts, frames = features.simulate_batch(self.graph, energy[None, ...], record=True)
        scores = readout.probabilities(self.model, values)[0]
        replay = build_replay(
            self.graph.weights, self.graph.hop, self.graph.types, frames, counts[:, 0],
            dt_ms=fly_lif.DT, delay_ms=fly_lif.DELAY * fly_lif.DT,
        )
        return {
            'digit': int(self.model['classes'][scores.argmax()]),
            'probs': scores.tolist(), 'replay': replay,
            'n_active': int(np.count_nonzero(counts)), 'n_spikes': int(counts.sum()),
            'sim_ms': encode.STEPS * fly_lif.DT,
            'cochleogram': energy[:, ::5].round(3).tolist(),
            'wall_ms': int((time.monotonic() - started) * 1000),
        }
