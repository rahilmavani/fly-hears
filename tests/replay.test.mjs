import assert from 'node:assert/strict';
import test from 'node:test';
import { BrainReplay, lastSpikeAt } from '../app/static/replay.js';

// Supply drawing and scheduling surfaces without a browser. These checks exercise
// the real controller clock and callbacks, not visual layout or microphone access.
globalThis.window = { devicePixelRatio: 1 };
globalThis.ResizeObserver = class { observe() {} };
let nextId = 0;
const scheduled = new Map();
globalThis.requestAnimationFrame = callback => {
  scheduled.set(++nextId, callback);
  return nextId;
};
globalThis.cancelAnimationFrame = id => scheduled.delete(id);
const noop = () => {};
const context = new Proxy({}, { get: (_, key) => key === 'createRadialGradient'
  ? () => ({ addColorStop: noop }) : noop });

function fixture() {
  const frames = [], states = [];
  const canvas = { getContext: () => context,
    getBoundingClientRect: () => ({ width: 900, height: 300 }), addEventListener: noop };
  const replay = new BrainReplay(canvas, { hidden: true, style: {} },
    frame => frames.push(frame), state => states.push(state));
  replay.load({ duration_ms: 10, step_ms: 1, delay_ms: 2,
    nodes: [{ stage: 0, spikes: [2, 6], count: 2 }], edges: [],
    stages: [{ spikes: [0, 0, 1, 0, 0, 0, 1, 0, 0, 0] }],
    cumulative_spikes: [0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
    cumulative_active: [0, 0, 1, 1, 1, 1, 1, 1, 1, 1] });
  return { replay, frames, states };
}

function advance(replay, wallDelta) {
  const callback = scheduled.get(replay.raf);
  scheduled.delete(replay.raf);
  callback(replay.lastWall + wallDelta);
}

test('slower playback advances simulated time, pause stops it, completion happens once', () => {
  const { replay, frames, states } = fixture();
  replay.speed = 0.025;
  replay.play();
  advance(replay, 100);
  assert.equal(replay.time, 2.5);
  assert.equal(frames.at(-1).spikes, 1);
  assert.equal(frames.at(-1).complete, false);
  replay.pause();
  const time = replay.time;
  replay.tick(replay.lastWall + 1000);
  assert.equal(replay.time, time);
  assert.equal(replay.raf, 0);
  replay.play();
  advance(replay, 1000);
  assert.equal(replay.time, 10);
  assert.equal(frames.at(-1).complete, true);
  assert.equal(frames.at(-1).spikes, 2);
  assert.equal(states.at(-1), 'complete');
  assert.equal(replay.playing, false);
  assert.equal(scheduled.size, 0);
});

test('seeking backward and replaying immediately withdraw the completed result', () => {
  const { replay, frames } = fixture();
  replay.seek(100);
  assert.equal(frames.at(-1).complete, true);
  replay.seek(3);
  assert.equal(frames.at(-1).complete, false);
  assert.equal(frames.at(-1).spikes, 1);
  replay.seek(10);
  replay.play();
  assert.equal(frames.at(-1).time, 0);
  assert.equal(frames.at(-1).complete, false);
  assert.equal(frames.at(-1).spikes, 0);
  replay.clear();
  assert.equal(scheduled.size, 0);
  assert.equal(replay.data, null);
});

test('flashes only use spikes at or before the current replay time', () => {
  const times = [2, 6, 9];
  assert.equal(lastSpikeAt(times, 0), -Infinity);
  assert.equal(lastSpikeAt(times, 2), 2);
  assert.equal(lastSpikeAt(times, 5.9), 2);
  assert.equal(lastSpikeAt(times, 6), 6);
  assert.equal(lastSpikeAt(times, 10), 9);
  assert.equal(lastSpikeAt([], 10), -Infinity);
});
