# The Fly Hears: model card

## Intended use

A local, educational spoken-digit experiment and an activity-replay demo. The reader predicts
one English digit from 0 to 9. It is not a general transcription service or evidence that a
biological fly understands speech. The simulated network has no learned internal connections.

## Bundled model

| Property | Value |
|---|---|
| Training run | `v2` |
| Connectome source | MaleCNS v1.0 |
| Selected network | 30,000 neurons; 1,348,769 signed connections |
| Sensory inputs | 672 JO-labelled neurons: 50 JO-A, 88 JO-B, 534 other JO |
| Input | Mono 8 kHz audio; 32 log-mel bands, 100–4,000 Hz |
| Simulation | 300 steps of 1 ms; fixed leaky integrate-and-fire dynamics |
| Activity features | Four time bins; 6,000 neurons selected by training-set variance |
| Reader | 24,000 features; `log1p`, standard scaling, logistic regression |
| Learned classifier parameters | 240,000 coefficients and 10 intercepts |
| Training recordings | 3,000 FSDD clips from six speakers, plus synthetic noisy copies |
| Model license | CC BY-SA 4.0; graph separately CC BY 4.0 |

Hashes are in [the manifest](artifacts/demo/manifest.json). The model and results also contain
the graph hash, five core source hashes, dataset hash, and core package versions. The demo
refuses an incompatible model; it only displays benchmark results matching the loaded model's hash.

## Evaluation

The protocol is leave-one-speaker-out evaluation, not FSDD's suggested recording-index split.
There are six folds, each with 2,500 original training recordings and 500 test recordings.
Each training recording contributes its clean and `train_noise` versions. The held-out speaker's
clean and independent `test_noise` versions are evaluated separately.

Feature selection and scaling are fitted inside each fold on training speakers only. The
audio-only reader uses 32 frequency bands averaged into four time bins. It uses the same
speaker splits and augmentation conditions as the circuit-based readers.

| Reader | Clean correct / 3,000 | Clean accuracy | Noisy correct / 3,000 | Noisy accuracy |
|---|---:|---:|---:|---:|
| Fly wiring | 2,101 | 70.03% | 2,047 | 68.23% |
| Scrambled wiring | 2,092 | 69.73% | 2,099 | 69.97% |
| Audio only | 2,008 | 66.93% | 2,034 | 67.80% |

[Raw results](artifacts/demo/results.json) include all speaker scores and confusion matrices.
Clean fly-wiring accuracy varies from 52.0% to 82.4% by held-out speaker. The graph, dataset,
and model were used during project development; there is no separate untouched external test set.

The final demo reader is subsequently fitted on all six speakers. Its ten displayed examples
are from its training data. Correct predictions on those examples are compatibility checks,
not an additional generalization benchmark.

Synthetic noise is white noise at a reproducible 10–30 dB signal-to-noise ratio, with up to
0.25 seconds of padding per side. The noisy test does not establish performance on different
microphones, rooms, accents, or browser audio processing.

## Simulation assumptions

- The selected graph expands outward from JO inputs, subject to a 30,000-neuron cap. The
  bundled graph reaches that cap at the second outward step.
- Frequency-to-neuron assignment is engineered: JO-B cycles through the lower 16 bands,
  JO-A through the upper 16, and other JO through all 32. These assignments are not measured
  speech-frequency tuning from the connectome.
- Only traced neurons are used. Edges with fewer than five synapses and Kenyon-to-Kenyon
  edges are excluded. Connection signs are inferred from neurotransmitter annotations;
  unknown or modulatory transmitters have no fast effect in this model.
- The simulator uses a 2 ms connection delay and shared membrane parameters. Audio is
  compressed in time and longer inputs are interpolated to fit 300 steps.
- The same seeded random draws are used for each recording, independent of batch size.
  The reported run does not measure variability over multiple input random seeds.
- A single rewired control preserves degrees and sender signs/weights. Its near-tie with
  real wiring does not support a general biological advantage for speech decoding.

## Speech check and live use

The web server accepts short audio recordings and checks raw volume variation and WebRTC
voice activity before prediction. Accepted audio reaches the encoder unchanged. Quiet speech
may be rejected, and some noises may be accepted. The classifier has no class for silence,
unknown words, or multiple digits, so accepted out-of-scope audio can receive an incorrect digit.

The reported benchmark runs before this speech check. It is not end-to-end microphone accuracy.
No fresh user-microphone benchmark is included. Reader scores have not been calibrated as
probabilities of correctness for new speakers or microphones.

The page sends audio to the local Python server. The server does not automatically save it.
The browser retains the most recent microphone WAV for optional listening or download.
The CLI binds to `127.0.0.1`; this is a local demo, not an authenticated public hosting service.

## Replay

At most 144 representative active neurons are drawn: 32 sensory, 48 first-stage, and 64 deeper
neurons. Their flashes use recorded spike times; visible edges are a subset of actual directed
connections. Counters describe the full selected network, not just the visible nodes.

The layout is schematic and the illustrated fly is decorative. Travelling dots show signal
delivery after a source spike, not proof that one edge caused the next neuron to fire. The
prediction is calculated from completed activity and revealed at the end of playback.
