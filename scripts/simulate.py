"""Generate clean and noisy features without changing any fly-network weights."""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from scripts import artifacts, controls, encode, features


def recordings(directory):
    paths = sorted(Path(directory).glob('*.wav'))
    if not paths:
        raise ValueError(f'No WAV recordings found in {directory}')
    labels, speakers = [], []
    for path in paths:
        match = re.fullmatch(r'(\d)_([a-z]+)_(\d+)\.wav', path.name)
        if match is None:
            raise ValueError(f'Unexpected recording filename: {path.name}')
        labels.append(int(match[1]))
        speakers.append(match[2])
    return paths, np.array(labels), np.array(speakers)


def dataset_hash(paths):
    manifest = [(path.name, artifacts.file_hash(path)) for path in paths]
    return hashlib.sha256(json.dumps(manifest).encode()).hexdigest()


def feature_path(run_dir, mode, condition):
    return Path(run_dir) / f'features_{mode}_{condition}.npz'


def generate(graph, paths, condition, batch_size):
    blocks, audio_blocks = [], []
    started = time.monotonic()
    for offset in range(0, len(paths), batch_size):
        batch = paths[offset:offset + batch_size]
        energy = np.stack([encode.encode_file(path, condition) for path in batch])
        spike_counts, _, _ = features.simulate_batch(graph, energy)
        blocks.append(sp.csr_matrix(spike_counts))
        audio_blocks.append(features.audio_features(energy))
        elapsed = time.monotonic() - started
        finished = offset + len(batch)
        remaining = elapsed / finished * (len(paths) - finished)
        print(f'  {condition}: {finished}/{len(paths)} clips; '
              f'{elapsed / 60:.1f} min elapsed, ~{remaining / 60:.1f} min left', flush=True)
    return sp.vstack(blocks).tocsr(), np.concatenate(audio_blocks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path, help='Directory for this experiment')
    parser.add_argument('--rewired', action='store_true', help='Generate the scrambled-wiring control')
    parser.add_argument('--batch-size', type=int, default=32, help='Lower this if memory is limited')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')

    mode = 'rewired' if args.rewired else 'real'
    paths, labels, speakers = recordings(artifacts.DATA / 'fsdd' / 'recordings')
    current_signature = artifacts.signature()
    source_hash = dataset_hash(paths)
    pending = []
    for condition in encode.CONDITIONS:
        path = feature_path(args.run, mode, condition)
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                metadata = artifacts.read_metadata(data)
            artifacts.require_signature(metadata, current_signature)
            if (metadata['dataset'] != source_hash or metadata['mode'] != mode
                    or metadata['condition'] != condition):
                raise ValueError(f'{path} belongs to a different experiment. Choose a new --run directory.')
            if mode == 'rewired' and metadata.get('control_code') != artifacts.file_hash(artifacts.ROOT / 'scripts' / 'controls.py'):
                raise ValueError('The scrambling code changed. Choose a new --run directory.')
            print(f'Already complete: {path}', flush=True)
        else:
            pending.append(condition)
    if not pending:
        return

    graph = features.load_graph(artifacts.GRAPH_PATH)
    control = None
    if args.rewired:
        print('Scrambling connections while preserving degrees and sender signs…', flush=True)
        graph.weights, control = controls.rewire(graph.weights)
        print(f"Completed {control['accepted_swaps']:,} edge swaps.", flush=True)
    print(f'{mode}: {graph.size:,} neurons, {graph.weights.nnz:,} edges, {len(paths):,} recordings', flush=True)

    for condition in pending:
        spike_counts, audio = generate(graph, paths, condition, args.batch_size)
        metadata = {
            'signature': current_signature, 'dataset': source_hash,
            'mode': mode, 'condition': condition, 'n_neurons': graph.size,
            'bins': features.BINS, 'n_clips': len(paths), 'control': control,
            'control_code': artifacts.file_hash(artifacts.ROOT / 'scripts' / 'controls.py') if control else None,
        }
        path = feature_path(args.run, mode, condition)
        artifacts.save_npz(
            path, metadata,
            X_data=spike_counts.data, X_indices=spike_counts.indices,
            X_indptr=spike_counts.indptr, X_shape=spike_counts.shape,
            audio=audio, labels=labels, speakers=speakers,
            files=np.array([path.name for path in paths]),
        )
        print(f'Saved {path}', flush=True)


if __name__ == '__main__':
    main()
