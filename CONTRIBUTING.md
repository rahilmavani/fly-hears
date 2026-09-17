# Contributing

Use Python 3.12 and the checked-in lockfile. Run commands from the repository root:

```bash
uv sync --locked
uv run python -m unittest discover -s tests -v
node --test tests/replay.test.mjs
```

Node.js 22+ is only needed for the frontend tests. The app has no JavaScript dependencies or
build step. Python tests use small synthetic fixtures plus the ten bundled recordings and do
not launch full training.

For interface changes, check recording, sample playback, pause/seek, silence rejection, and
the layout on a laptop and a narrow phone screen. Respect reduced-motion preferences. The
neuron animation must continue to use recorded spikes, and rewinding must hide the completed
prediction. Sample labels, playback URLs, and submitted audio must refer to the same recording.

For model changes, follow [the training guide](docs/training.md). Keep feature selection,
scaling, and noisy copies out of held-out speakers' training folds. Report real wiring,
scrambled wiring, and the audio-only baseline together. Do not treat demo examples as unseen tests.

Five source files are byte-hashed into trained artifacts: `encode.py`, `fly_lif.py`,
`features.py`, `readout.py`, and `simulate.py`. Any change to these files requires regenerating
features and the model. A changed `controls.py` requires regenerating control features.
Do not bypass compatibility checks to reuse stale results. Use a new run directory.

Keep local microphone audio, environment files, full datasets, feature caches, and session
notes out of commits. The only bundled recordings should be the attributed public FSDD examples.
If reporting a microphone problem, include the digit spoken, device/browser, and whether a
wrong digit or a speech-check rejection occurred. Share audio only if you intend to make it public.

Keep changes focused and include the checks you ran. Contributions to project code are under
the [MIT license](LICENSE); artifact and dataset licenses remain separate.
