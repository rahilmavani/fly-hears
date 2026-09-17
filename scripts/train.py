"""Benchmark on held-out speakers, then fit the final reader for the demo."""

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

from scripts import artifacts, encode, features, readout
from scripts.simulate import feature_path


def load_features(path, expected_signature):
    if not path.exists():
        raise ValueError(f'Missing {path}. Run both simulation commands first.')
    with np.load(path, allow_pickle=False) as data:
        metadata = artifacts.read_metadata(data)
        artifacts.require_signature(metadata, expected_signature)
        matrix = sp.csr_matrix(
            (data['X_data'], data['X_indices'], data['X_indptr']),
            shape=tuple(data['X_shape']),
        )
        return {
            'spikes': matrix, 'audio': data['audio'], 'labels': data['labels'],
            'speakers': data['speakers'], 'files': data['files'], 'metadata': metadata,
        }


def load_mode(run_dir, mode, expected_signature):
    data = {condition: load_features(feature_path(run_dir, mode, condition), expected_signature)
            for condition in encode.CONDITIONS}
    clean = data['clean']
    for condition, entry in data.items():
        metadata = entry['metadata']
        if metadata['mode'] != mode or metadata['condition'] != condition:
            raise ValueError('A feature file has the wrong mode or audio condition.')
        if metadata['bins'] != features.BINS or metadata['n_neurons'] != clean['metadata']['n_neurons']:
            raise ValueError('Feature dimensions do not agree.')
        if metadata['dataset'] != clean['metadata']['dataset']:
            raise ValueError('Feature files were generated from different recordings.')
        for key in ['files', 'labels', 'speakers']:
            if not np.array_equal(entry[key], clean[key]):
                raise ValueError(f'Mismatched {key} in the {mode} features.')
        if mode == 'rewired' and metadata['control_code'] != artifacts.file_hash(artifacts.ROOT / 'scripts' / 'controls.py'):
            raise ValueError('The scrambling code changed. Regenerate the control features.')
    return data


def speaker_splits(speakers):
    for speaker in np.unique(speakers):
        yield str(speaker), np.flatnonzero(speakers != speaker), np.flatnonzero(speakers == speaker)


def training_rows(data, indices, kind):
    clean = data['clean'][kind][indices]
    noisy = data['train_noise'][kind][indices]
    values = sp.vstack([clean, noisy]).tocsr() if kind == 'spikes' else np.concatenate([clean, noisy])
    labels = np.tile(data['clean']['labels'][indices], 2)
    return values, labels


def evaluate(data, kind, name):
    clean = data['clean']
    labels, speakers = clean['labels'], clean['speakers']
    predictions = {condition: np.full(len(labels), -1, dtype=int) for condition in ['clean', 'test_noise']}
    per_speaker = []
    for speaker, train_indices, test_indices in speaker_splits(speakers):
        values, targets = training_rows(data, train_indices, kind)
        if not np.array_equal(np.unique(targets), np.arange(10)):
            raise ValueError(f'Training without {speaker} does not contain all ten digits.')
        columns = readout.select_columns(values, clean['metadata']['n_neurons'], features.BINS) if kind == 'spikes' else None
        model = readout.fit(values, targets, columns)
        scores = {'speaker': speaker, 'n_test': len(test_indices)}
        for condition in predictions:
            predicted = readout.predict(model, data[condition][kind][test_indices])
            predictions[condition][test_indices] = predicted
            scores[condition] = float(np.mean(predicted == labels[test_indices]))
        per_speaker.append(scores)
        print(f"{name}, held-out {speaker}: clean {scores['clean']:.1%}, noisy {scores['test_noise']:.1%}", flush=True)

    result = {'per_speaker': per_speaker}
    for condition, predicted in predictions.items():
        result[condition] = {
            'accuracy': float(np.mean(predicted == labels)),
            'speaker_mean_accuracy': float(np.mean([row[condition] for row in per_speaker])),
            'confusion': confusion_matrix(labels, predicted, labels=np.arange(10)).tolist(),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    model_path = args.run / 'model.npz'
    if model_path.exists():
        parser.error('This run already has a model. Use a new run directory for a new experiment.')

    started = time.monotonic()
    current_signature = artifacts.signature()
    real = load_mode(args.run, 'real', current_signature)
    rewired = load_mode(args.run, 'rewired', current_signature)
    for condition in encode.CONDITIONS:
        for key in ['files', 'labels', 'speakers', 'audio']:
            if not np.array_equal(real[condition][key], rewired[condition][key]):
                raise ValueError(f'Real and rewired runs do not have identical {key}.')
        if real[condition]['metadata']['dataset'] != rewired[condition]['metadata']['dataset']:
            raise ValueError('Real and rewired runs use different datasets.')
    if len(np.unique(real['clean']['speakers'])) < 2:
        raise ValueError('Benchmarking needs at least two speakers.')

    report = {
        'signature': current_signature,
        'protocol': 'leave-one-speaker-out; train on clean plus train_noise; evaluate clean and independent test_noise',
        'noise': 'Synthetic 10–30 dB SNR white noise and padding; not a real-microphone benchmark.',
        'n_clips': len(real['clean']['labels']),
    }
    with threadpool_limits(limits=4):
        report['real'] = evaluate(real, 'spikes', 'Fly wiring')
        report['no_brain'] = evaluate(real, 'audio', 'Audio only')
        report['rewired'] = evaluate(rewired, 'spikes', 'Scrambled wiring')
        del rewired
        print('Benchmarks complete. Fitting the demo reader on all speakers…', flush=True)
        all_indices = np.arange(len(real['clean']['labels']))
        values, labels = training_rows(real, all_indices, 'spikes')
        columns = readout.select_columns(values, real['clean']['metadata']['n_neurons'], features.BINS)
        model = readout.fit(values, labels, columns)

    metadata = {
        'signature': current_signature, 'dataset': real['clean']['metadata']['dataset'],
        'n_neurons': real['clean']['metadata']['n_neurons'], 'bins': features.BINS,
        'training_conditions': ['clean', 'train_noise'],
        'benchmark_protocol': report['protocol'],
    }
    artifacts.save_npz(model_path, metadata, **model)
    report['model_sha256'] = artifacts.file_hash(model_path)
    report['elapsed_minutes'] = (time.monotonic() - started) / 60
    artifacts.save_json(args.run / 'results.json', report)
    for key in ['real', 'no_brain', 'rewired']:
        print(f"{key}: clean {report[key]['clean']['accuracy']:.1%}, "
              f"noisy {report[key]['test_noise']['accuracy']:.1%}")
    print(f'Saved {model_path} and {args.run / "results.json"}', flush=True)


if __name__ == '__main__':
    main()
