import { BrainReplay } from './replay.js';

const $ = id => document.getElementById(id);
let prediction = null, revealed = false, busy = false, capturing = false;
let modelReady = false;
let recorder = null, stream = null, stopTimer = null, sampleAudio = null, sampleUrl = null;
let recordingUrl = null, recordingAudio = null;
let spectrum = null;
const spectrumImage = document.createElement('canvas');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

function setFly(mood, status, quip) {
  $('fly-panel').dataset.mood = mood;
  $('fly-status').textContent = status;
  $('fly-quip').textContent = quip;
}

function setState(state) {
  const labels = { ready: 'Ready to replay', playing: 'Replaying activity', paused: 'Paused', complete: 'Replay complete' };
  $('replay-state').textContent = labels[state] || state;
  $('replay-state').classList.toggle('playing', state === 'playing');
  $('play').textContent = state === 'playing' ? 'Pause' : 'Play';
  if (state === 'complete') setFly('complete', 'Verdict delivered', 'I’m going with');
  else if (state === 'playing') setFly('thinking', 'Following the signal', 'A lot going on up here.');
  else if (state === 'paused') setFly('idle', 'Taking a breather', 'Take your time. I’ll wait.');
  else if (state === 'ready') setFly('idle', 'Ready when you are', 'Press play. Watch this.');
}

function showResult(show) {
  if (revealed === show && prediction) return;
  revealed = show;
  $('digit').textContent = show && prediction ? prediction.digit : '?';
  $('digit').classList.toggle('dim', !show);
  $('digit').setAttribute('aria-label', show && prediction ? `Predicted digit ${prediction.digit}` : 'Prediction not revealed');
  $('result-note').textContent = show && prediction
    ? `The reader's top guess. Computed in ${prediction.wall_ms.toLocaleString()} ms.`
    : 'The guess appears when the replay ends.';
  Array.from($('bars').children).forEach((bar, i) => {
    const probability = show && prediction ? prediction.probs[i] : 0;
    bar.style.height = `${Math.max(3, probability * 100)}%`;
    bar.classList.toggle('top', Boolean(show && prediction && i === prediction.digit));
    bar.setAttribute('aria-label', show && prediction ? `Digit ${i}, score ${(probability * 100).toFixed(1)} percent` : `Digit ${i}, score not revealed`);
  });
}

const replay = new BrainReplay($('network'), $('neuron-tooltip'), frame => {
  $('seek').value = frame.time;
  $('time-label').textContent = `${Math.floor(frame.time)} / ${frame.duration} ms`;
  $('active-count').textContent = frame.active.toLocaleString();
  $('spike-count').textContent = frame.spikes.toLocaleString();
  frame.stages.forEach((count, i) => { $(`stage-${i}-count`).textContent = `${count.toLocaleString()} spikes`; });
  drawSpectrum(frame.time / frame.duration);
  showResult(frame.complete);
}, setState);

for (let i = 0; i < 10; i++) {
  const bar = document.createElement('div'), label = document.createElement('span');
  label.textContent = i; bar.append(label); $('bars').append(bar);
}
showResult(false);

function setBusy(value) {
  busy = value;
  $('mic').disabled = value || !modelReady;
  $('samples').querySelectorAll('button').forEach(button => { button.disabled = value || capturing || !modelReady; });
  $('listen-recording').disabled = value || capturing || !recordingUrl;
}

function showError(message) {
  $('error').textContent = message;
  $('error').hidden = false;
  $('hint').textContent = 'Try another recording or choose a sample.';
  setFly('quiet', 'Let’s try again', 'A little help here?');
}

function prepare() {
  replay.clear();
  prediction = null;
  revealed = true;
  showResult(false);
  spectrum = null;
  drawSpectrum(0);
  $('error').hidden = true;
  $('play').disabled = $('restart').disabled = $('seek').disabled = true;
  $('seek').value = 0;
  $('time-label').textContent = '0 / 300 ms';
  $('active-count').textContent = $('spike-count').textContent = '0';
  ['Sensory neurons', 'First connections', 'Deeper activity'].forEach((text, i) => { $(`stage-${i}-count`).textContent = text; });
  $('network-meta').textContent = 'Recorded spikes · schematic layout';
  $('network-empty').hidden = false;
  setState('Running the simulation…');
  setFly('thinking', 'Working on it', '30,000 neurons. Please hold.');
}

async function processAudio(blob, kind) {
  setBusy(true);
  prepare();
  $('hint').textContent = 'Recording received. Following the signal…';
  try {
    const body = new FormData();
    body.append('file', blob, kind === 'mic' ? 'mic.wav' : 'clip.wav');
    const response = await fetch('/predict', { method: 'POST', body });
    if (!response.headers.get('content-type')?.includes('application/json')) throw new Error('The simulation could not finish. Please try again.');
    const result = await response.json();
    if (!response.ok || result.error) {
      const error = new Error(result.error || 'The recording could not be processed.');
      error.code = result.code;
      throw error;
    }
    if (!result.replay?.nodes?.length) throw new Error('No neuron activity was recorded. Try speaking a little closer.');
    prediction = result;
    spectrum = result.cochleogram;
    buildSpectrum();
    $('network-empty').hidden = true;
    $('network-meta').textContent = `${result.replay.shown_neurons} representative neurons shown · ${result.replay.active_neurons.toLocaleString()} active in the full network`;
    $('network').setAttribute('aria-label', `Recorded activity from ${result.replay.shown_neurons} representative neurons. Use the playback controls to explore the ${result.sim_ms} millisecond simulation.`);
    $('seek').max = result.replay.duration_ms;
    $('play').disabled = $('restart').disabled = $('seek').disabled = false;
    replay.load(result.replay);
    $('hint').textContent = reducedMotion.matches ? 'Recording ready. Press Play or move the timeline.' : 'Your recording is ready. Follow the lights.';
    if (!reducedMotion.matches) replay.play();
  } catch (error) {
    showError(error.message || 'Something went wrong while processing the recording.');
    setState(error.code === 'no_speech' ? 'No speech detected' : 'Try another recording');
    if (error.code === 'no_speech') {
      $('result-note').textContent = 'No digit predicted. No clear speech was detected.';
      setFly('quiet', 'Nothing to go on', 'The silent treatment. Classic.');
    }
  } finally {
    setBusy(false);
  }
}

function stopSample() {
  if (sampleAudio) { sampleAudio.pause(); sampleAudio.src = ''; sampleAudio = null; }
  if (sampleUrl) { URL.revokeObjectURL(sampleUrl); sampleUrl = null; }
}

async function chooseSample(sample) {
  if (busy || capturing || !modelReady) return;
  $('samples').querySelectorAll('button').forEach(button => {
    const selected = button.dataset.sample === sample.name;
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
  setBusy(true);
  stopSample();
  stopListening();
  prepare();
  $('error').hidden = true;
  try {
    const response = await fetch(sample.url, { cache: 'no-store' });
    if (!response.ok) throw new Error('The sample recording could not be loaded.');
    const blob = await response.blob();
    sampleUrl = URL.createObjectURL(blob);
    sampleAudio = new Audio(sampleUrl);
    sampleAudio.addEventListener('ended', stopSample, { once: true });
    sampleAudio.play().catch(() => {});
    await processAudio(blob, 'sample');
  } catch (error) {
    showError(error.message);
    setState('Try another recording');
  } finally { setBusy(false); }
}

async function loadSamples() {
  try {
    const response = await fetch('/samples', { cache: 'no-store' });
    if (!response.ok) throw new Error('Samples unavailable');
    const samples = await response.json();
    for (const sample of samples) {
      const button = document.createElement('button'); button.type = 'button';
      const digit = document.createElement('strong'), speaker = document.createElement('small');
      digit.textContent = sample.digit; speaker.textContent = sample.speaker;
      button.append(digit, speaker);
      button.dataset.sample = sample.name;
      button.setAttribute('aria-pressed', 'false');
      button.setAttribute('aria-label', `Try digit ${sample.digit}, spoken by ${sample.speaker}`);
      button.addEventListener('click', () => chooseSample(sample));
      button.disabled = busy || capturing || !modelReady;
      $('samples').append(button);
    }
  } catch { $('samples').textContent = 'Samples unavailable. You can still use the microphone.'; }
}

async function loadResults() {
  try {
    const response = await fetch('/results');
    if (!response.ok) return;
    const results = await response.json();
    for (const [key, label] of [['real', 'Fly wiring'], ['rewired', 'Rewired control'], ['no_brain', 'Audio only']]) {
      const block = document.createElement('div'), score = document.createElement('strong');
      const result = results[key];
      score.textContent = result ? `${(result.clean.accuracy * 100).toFixed(1)}% clean` : 'Not benchmarked';
      const noisy = result ? ` · ${(result.test_noise.accuracy * 100).toFixed(1)}% with synthetic noise` : '';
      block.append(score, document.createTextNode(label + noisy)); $('nums').append(block);
    }
  } catch { /* Benchmarks are optional; recording and replay remain available. */ }
}

async function loadStatus() {
  try {
    const response = await fetch('/status', { cache: 'no-store' });
    if (!response.ok) throw new Error('Restart the updated demo server using the command in README.md.');
    const status = await response.json();
    modelReady = status.ready;
    if (!modelReady) throw new Error(status.error || 'The model is not ready yet.');
  } catch (error) {
    modelReady = false;
    showError(error.message);
    setState('Model unavailable');
  } finally { setBusy(false); }
}

$('play').addEventListener('click', () => replay.playing ? replay.pause() : replay.play());
$('restart').addEventListener('click', () => { replay.pause(); replay.seek(0); replay.play(); });
$('speed').addEventListener('change', () => { replay.speed = Number($('speed').value); });
$('seek').addEventListener('input', () => { replay.pause(); replay.seek($('seek').value); });
document.addEventListener('visibilitychange', () => { if (document.hidden) replay.pause(); });
reducedMotion.addEventListener('change', event => { if (event.matches) replay.pause(); });

function stopRecording() {
  clearTimeout(stopTimer);
  if (recorder?.state === 'recording') recorder.stop();
}

function stopListening() {
  if (recordingAudio) { recordingAudio.pause(); recordingAudio = null; }
}

function rememberRecording(blob, device) {
  stopListening();
  if (recordingUrl) URL.revokeObjectURL(recordingUrl);
  recordingUrl = URL.createObjectURL(blob);
  const name = device.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'mic';
  $('save-recording').href = recordingUrl;
  $('save-recording').download = `recording_${name}_${Date.now()}.wav`;
  $('mic-device').textContent = device;
  $('mic-device').title = `This recording used: ${device}`;
  $('recording-review').hidden = false;
}

$('listen-recording').addEventListener('click', () => {
  if (!recordingUrl || busy || capturing) return;
  stopListening();
  stopSample();
  recordingAudio = new Audio(recordingUrl);
  recordingAudio.play().catch(() => showError('Playback was blocked. Save the WAV to listen to it.'));
});

$('mic').addEventListener('click', async () => {
  if (recorder?.state === 'recording') { stopRecording(); return; }
  if (busy || capturing || !modelReady) return;
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    showError('Microphone recording is unavailable in this browser. Try a sample recording.'); return;
  }
  setBusy(true);
  replay.pause();
  stopSample();
  stopListening();
  $('error').hidden = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const device = stream.getAudioTracks()[0]?.label || 'Default microphone';
    prepare();
    $('recording-review').hidden = true;
    $('samples').querySelectorAll('button').forEach(button => {
      button.classList.remove('selected'); button.setAttribute('aria-pressed', 'false');
    });
    recorder = new MediaRecorder(stream);
    const chunks = [];
    recorder.addEventListener('dataavailable', event => { if (event.data.size) chunks.push(event.data); });
    recorder.addEventListener('stop', async () => {
      clearTimeout(stopTimer);
      stream?.getTracks().forEach(track => track.stop());
      capturing = false;
      $('mic').classList.remove('recording'); $('mic-label').textContent = 'Record a digit';
      setBusy(true);
      try {
        const wav = await toWav(new Blob(chunks, { type: recorder.mimeType }));
        rememberRecording(wav, device);
        await processAudio(wav, 'mic');
      } catch (error) {
        showError(error.message || 'The microphone recording could not be decoded.');
        setState('Try another recording');
      }
      finally { recorder = null; stream = null; setBusy(false); }
    }, { once: true });
    recorder.start();
    capturing = true;
    setBusy(false);
    $('mic').classList.add('recording'); $('mic-label').textContent = 'Stop recording';
    setState('Recording…');
    setFly('listening', 'All antennae', 'Go on. I’m listening.');
    $('hint').textContent = 'Say one digit now. Stops after 2 seconds.';
    stopTimer = setTimeout(stopRecording, 2000);
  } catch {
    stream?.getTracks().forEach(track => track.stop());
    stream = null; recorder = null; capturing = false; setBusy(false);
    showError('Microphone access was blocked. Allow microphone access or try a sample recording.');
  }
});

async function toWav(blob) {
  const context = new (window.AudioContext || window.webkitAudioContext)();
  try {
    const buffer = await context.decodeAudioData(await blob.arrayBuffer());
    const offline = new OfflineAudioContext(1, Math.max(1, Math.ceil(buffer.duration * 8000)), 8000);
    const source = offline.createBufferSource(); source.buffer = buffer; source.connect(offline.destination); source.start();
    const samples = (await offline.startRendering()).getChannelData(0);
    const wav = new ArrayBuffer(44 + samples.length * 2), view = new DataView(wav);
    const text = (offset, value) => [...value].forEach((char, i) => view.setUint8(offset + i, char.charCodeAt(0)));
    text(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true); text(8, 'WAVE'); text(12, 'fmt ');
    view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
    view.setUint32(24, 8000, true); view.setUint32(28, 16000, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
    text(36, 'data'); view.setUint32(40, samples.length * 2, true);
    samples.forEach((sample, i) => view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, sample)) * 32767, true));
    return new Blob([wav], { type: 'audio/wav' });
  } finally { await context.close(); }
}

function buildSpectrum() {
  const rect = $('coch').getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  $('coch').width = spectrumImage.width = Math.max(1, Math.round(rect.width * ratio));
  $('coch').height = spectrumImage.height = Math.max(1, Math.round(rect.height * ratio));
  const context = spectrumImage.getContext('2d'), w = spectrumImage.width, h = spectrumImage.height;
  context.fillStyle = '#131813'; context.fillRect(0, 0, w, h);
  if (spectrum) {
    const columns = spectrum[0].length, rows = spectrum.length;
    spectrum.forEach((row, band) => row.forEach((energy, time) => {
      if (energy < 0.02) return;
      context.fillStyle = `hsl(${40 + energy * 4} 76% ${10 + energy * 58}%)`;
      context.fillRect(time * w / columns, h - (band + 1) * h / rows, w / columns + 0.5, h / rows + 0.5);
    }));
  }
  drawSpectrum(replay.data ? replay.time / replay.data.duration_ms : 0);
}

function drawSpectrum(progress) {
  const canvas = $('coch'), context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (!spectrum || !spectrumImage.width) return;
  context.drawImage(spectrumImage, 0, 0);
  const x = Math.min(canvas.width - 1, progress * canvas.width);
  context.fillStyle = '#121412aa'; context.fillRect(x, 0, canvas.width - x, canvas.height);
  context.fillStyle = '#83d5d0'; context.fillRect(x, 0, 2, canvas.height);
}

new ResizeObserver(buildSpectrum).observe($('coch'));
loadSamples();
loadResults();
loadStatus();
