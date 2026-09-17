"""Evaluate labelled, unseen microphone recordings separately for each device."""

import argparse
import re
from collections import defaultdict
from pathlib import Path

from app.inference import Predictor
from app.server import decode_audio
from scripts import artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--recordings', required=True, type=Path,
                        help='WAV files named digit_device_take.wav, e.g. 4_earpods_01.wav')
    args = parser.parse_args()
    paths = sorted(args.recordings.glob('*.wav'))
    if not paths:
        parser.error('No WAV files found in the recording directory.')
    cases = []
    for path in paths:
        match = re.fullmatch(r'(\d)_([A-Za-z0-9-]+)_(\d+)\.wav', path.name)
        if match is None:
            parser.error(f'Use digit_device_take.wav filenames: {path.name}')
        cases.append((path, int(match[1]), match[2]))
    predictor = Predictor(args.run)
    devices = defaultdict(list)
    for path, label, device in cases:
        try:
            audio, sample_rate = decode_audio(path.read_bytes())
            prediction = predictor.predict(audio, sample_rate)['digit']
            error = None
        except (ValueError, RuntimeError) as problem:
            prediction, error = None, str(problem)
        result = {'file': path.name, 'expected': label, 'predicted': prediction, 'error': error}
        devices[device].append(result)
        print(f'{path.name}: expected {label}, predicted {prediction}', flush=True)
    report = {}
    for device, rows in devices.items():
        accuracy = sum(row['expected'] == row['predicted'] for row in rows) / len(rows)
        report[device] = {'n_clips': len(rows), 'accuracy': accuracy, 'recordings': rows}
        print(f'{device}: {accuracy:.1%} on {len(rows)} recordings')
    output = args.recordings / 'results.json'
    artifacts.save_json(output, report)
    print(f'Saved {output}. This command does not train on these recordings.')


if __name__ == '__main__':
    main()
