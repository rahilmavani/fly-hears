# Training and evaluation

[Back to the project](../README.md)

Training changes the reader that labels activity. It never changes the fly circuit’s connections.
Run commands from the repository root. Full training needs more memory and disk than the demo;
allow several GB of free disk space for dependencies, recordings, and feature files. A CPU is sufficient.

## 1. Install and prepare the graph

```bash
uv sync --locked
uv run python -m scripts.setup_demo
```

This uses the exact graph from the published `v2` run. If you have a different local graph,
use a separate checkout or keep that graph and train a new run against it. Do not replace an
experiment’s graph while its features are being generated.

## 2. Download the full recording dataset

The ten bundled examples are not enough for training. Download the full FSDD and select the
revision used for the published run:

```bash
git clone https://github.com/Jakobovski/free-spoken-digit-dataset data/fsdd
git -C data/fsdd checkout --detach 26eb9aaf76e81b692f806f9140c2d2777410d7a1
uv run python -m scripts.check
```

Skip the clone if `data/fsdd` already contains your dataset. The published revision has 3,000
recordings, ten digits, and six speakers. The check reads data and metadata; it does not simulate
or train. The training scripts expect filenames such as `4_jackson_12.wav`.

## 3. Generate activity using the fly wiring

```bash
uv run python -m scripts.simulate --run data/runs/my-run
```

For each recording, this computes the activity in three conditions:

| Saved file | Contents |
|---|---|
| `features_real_clean.npz` | Original recordings |
| `features_real_train_noise.npz` | Reproducible noisy copies for training |
| `features_real_test_noise.npz` | Independently generated noisy copies for evaluation |

The names identify augmentation conditions, not a split of speakers. Speakers are separated
later, before any model fitting. The original 3,000 recordings produce three versions each;
these are not 9,000 independently collected recordings.

Completed conditions are saved atomically. If interrupted, rerun the same command. Compatible
completed files are skipped; the unfinished condition starts again. Use `--batch-size 16` to
reduce batch memory. Batch size and ordering do not change a recording’s input random sequence.

## 4. Generate the scrambled control

```bash
uv run python -m scripts.simulate --run data/runs/my-run --rewired
```

This creates three `features_rewired_*.npz` files using the same recordings and noise. Directed
edge swaps preserve incoming/outgoing edge counts, each sender’s signs and outgoing weight
values, and incoming positive/negative edge counts. New duplicate edges and self-connections
are rejected. The published run used seed 0 and accepted 6,607,240 swaps.

One control graph is generated. Stronger claims about wiring would require multiple controls,
additional seeds, and independent evaluations.

## 5. Benchmark, then fit the final reader

```bash
uv run python -m scripts.train --run data/runs/my-run
```

For each speaker, the script trains on clean and noisy recordings from the other five speakers.
Feature selection and scaling are fitted only on those training rows. It tests on the excluded
speaker’s clean recordings and independent noisy copies. It repeats this for real wiring,
scrambled wiring, and an audio-only baseline.

After evaluation, the final demo reader is fitted on all six speakers. Outputs are:

- `model.npz`: selected features, scaling values, classifier weights, and compatibility metadata.
- `results.json`: benchmark protocol, clean/noisy scores, per-speaker results, confusion matrices,
  and the model hash. Confusion-matrix rows are spoken digits; columns are predictions.

The script rejects incompatible files, missing classes, and optimizer convergence failures.
It refuses to overwrite an existing model. Use a new `--run` directory for another experiment.

## 6. Start your trained demo

```bash
uv run python -m app.server --run data/runs/my-run --port 8001
```

Open [localhost:8001](http://localhost:8001). If that port is already in use, stop the previous
server with Ctrl+C in its terminal or choose another port.

All feature/model files record hashes of their graph and five core source files, plus the core
package versions. Changing even a comment in a signature-tracked file invalidates those artifacts.
Regenerate features and train a new run after changing the encoder, simulator, or reader. UI-only
changes do not require training.

## Evaluate your microphone

The demo records one digit for up to two seconds. **Listen back** plays the same WAV used for
prediction; **Save WAV** downloads it. The device name tells you which microphone was used.
Nothing is automatically saved by the server.

Collect new recordings for each device and rename them with the actual spoken digit, device,
and take number, for example `4_laptop_01.wav` or `4_earpods_01.wav`. Store them in `data/mic_eval/`.
Five fresh takes per digit per device is a useful first comparison, not a general accuracy claim.

```bash
uv run python -m scripts.evaluate_mics --run data/runs/my-run --recordings data/mic_eval
```

This runs the speech check and reader, reports each device separately, and saves
`data/mic_eval/results.json`. Rejected recordings count as incorrect. It does not train on these
recordings. If you later train on some microphone recordings, collect new takes for evaluation.

## Optional: rebuild from the original connectome tables

The bundled graph is enough for the demo and training. Rebuilding is for changes to graph
selection. Prefer a separate checkout because `scripts.subgraph` writes `data/auditory.npz`.

```bash
mkdir -p data
curl --fail --location --retry 3 -o data/annotations.feather https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather
curl --fail --location --retry 3 -o data/neurotransmitters.feather https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather
curl --fail --location --retry 3 -o data/weights.feather https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather
uv run python -m scripts.subgraph
```

The source tables total roughly 1.1 GB on disk; loading/filtering them takes additional memory.
Source SHA-256 hashes are recorded in [data-sources.json](data-sources.json).
The extractor keeps traced neurons, starts from JO-labelled sensory inputs, and expands by
connection strength to at most 30,000 neurons within three outward steps.

The current extractor adds original body IDs and uses stable tie ordering. A rebuilt graph is
not guaranteed to have the same selection or byte hash as the earlier bundled graph. Generate
new features and retrain against it; do not attach the published scores to a different graph.
