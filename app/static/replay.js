const COLORS = ['#efbc70', '#83d5d0', '#b6c88c'];

// Recorded spike times are the only source of node flashes. Seeking simply
// evaluates the same recording at another time; it never generates new activity.
export function lastSpikeAt(times, time) {
  let left = 0, right = times.length;
  while (left < right) {
    const mid = (left + right) >>> 1;
    if (times[mid] <= time) left = mid + 1;
    else right = mid;
  }
  return left ? times[left - 1] : -Infinity;
}

export class BrainReplay {
  constructor(canvas, tooltip, onFrame, onState) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.tooltip = tooltip;
    this.onFrame = onFrame;
    this.onState = onState;
    this.data = null;
    this.time = 0;
    this.speed = 0.1;
    this.playing = false;
    this.raf = 0;
    this.positions = [];
    this.hover = -1;
    this.tick = this.tick.bind(this);
    this.resize = new ResizeObserver(() => this.layout());
    this.resize.observe(canvas);
    canvas.addEventListener('pointermove', event => this.point(event));
    canvas.addEventListener('pointerleave', () => {
      this.tooltip.hidden = true;
      this.hover = -1;
      if (!this.playing) this.draw();
    });
    this.layout();
  }

  load(data) {
    this.pause();
    this.data = data;
    this.time = 0;
    this.stageTotals = data.stages.map(stage => {
      let total = 0;
      return stage.spikes.map(count => (total += count));
    });
    this.layout();
    this.emit();
    this.onState('ready');
  }

  clear() {
    this.pause();
    this.data = null;
    this.time = 0;
    this.hover = -1;
    this.tooltip.hidden = true;
    this.layout();
  }

  play() {
    if (!this.data || this.playing) return;
    if (this.time >= this.data.duration_ms) this.time = 0;
    this.playing = true;
    this.lastWall = performance.now();
    this.draw();
    this.emit();
    this.onState('playing');
    this.raf = requestAnimationFrame(this.tick);
  }

  pause() {
    cancelAnimationFrame(this.raf);
    this.raf = 0;
    this.playing = false;
    if (this.data) this.onState(this.time >= this.data.duration_ms ? 'complete' : 'paused');
  }

  seek(time) {
    if (!this.data) return;
    this.time = Math.max(0, Math.min(this.data.duration_ms, Number(time)));
    this.lastWall = performance.now();
    this.draw();
    this.emit();
    if (!this.playing) this.onState(this.time >= this.data.duration_ms ? 'complete' : 'paused');
  }

  tick(wall) {
    if (!this.playing || !this.data) return;
    this.time = Math.min(this.data.duration_ms, this.time + (wall - this.lastWall) * this.speed);
    this.lastWall = wall;
    this.draw();
    this.emit();
    if (this.time >= this.data.duration_ms) this.pause();
    else this.raf = requestAnimationFrame(this.tick);
  }

  emit() {
    if (!this.data) return;
    const index = Math.min(this.data.cumulative_spikes.length - 1,
      Math.floor(this.time / this.data.step_ms));
    this.onFrame({ time: this.time, duration: this.data.duration_ms,
      complete: this.time >= this.data.duration_ms,
      spikes: this.data.cumulative_spikes[index] || 0,
      active: this.data.cumulative_active[index] || 0,
      stages: this.stageTotals.map(totals => totals[index] || 0) });
  }

  layout() {
    const rect = this.canvas.getBoundingClientRect();
    this.width = Math.max(1, rect.width);
    this.height = Math.max(1, rect.height);
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(this.width * ratio);
    this.canvas.height = Math.round(this.height * ratio);
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const nodes = this.data?.nodes || Array.from({ length: 66 }, (_, i) => ({ stage: Math.floor(i / 22), neuron: i }));
    const totals = [0, 0, 0], seen = [0, 0, 0];
    nodes.forEach(node => totals[node.stage]++);
    this.positions = nodes.map(node => {
      const stage = node.stage, index = seen[stage]++;
      const angle = index * 2.399963 + stage * 0.7;
      const radius = Math.sqrt((index + 0.6) / Math.max(1, totals[stage]));
      return { x: this.width * ((stage + 0.5) / 3) + Math.cos(angle) * radius * this.width * 0.127,
        y: this.height * 0.49 + Math.sin(angle) * radius * this.height * (stage ? 0.38 : 0.3) };
    });
    this.draw();
  }

  curve(edge) {
    const from = this.positions[edge.source], to = this.positions[edge.target];
    const bend = ((edge.source * 7 + edge.target * 3) % 9 - 4) * 4;
    return [from, { x: (from.x + to.x) / 2, y: (from.y + to.y) / 2 + bend }, to];
  }

  draw() {
    const g = this.ctx, w = this.width, h = this.height;
    g.clearRect(0, 0, w, h);
    for (let stage = 0; stage < 3; stage++) {
      const cx = w * ((stage + 0.5) / 3);
      const glow = g.createRadialGradient(cx, h / 2, 5, cx, h / 2, h * 0.48);
      glow.addColorStop(0, COLORS[stage] + '0b');
      glow.addColorStop(1, COLORS[stage] + '00');
      g.fillStyle = glow;
      g.fillRect(stage * w / 3, 0, w / 3, h);
      g.beginPath();
      g.ellipse(cx, h * 0.49, w * 0.145, h * (stage ? 0.43 : 0.36), 0, 0, Math.PI * 2);
      g.strokeStyle = '#687c59';
      g.globalAlpha = 0.35;
      g.lineWidth = 1;
      g.stroke();
    }
    g.globalAlpha = 1;
    if (!this.data) {
      this.positions.forEach((point, i) => {
        g.fillStyle = COLORS[Math.floor(i / 22)] + '36';
        g.beginPath(); g.arc(point.x, point.y, 1.8, 0, Math.PI * 2); g.fill();
      });
      return;
    }
    const latest = this.data.nodes.map(node => lastSpikeAt(node.spikes, this.time));
    for (const edge of this.data.edges) {
      const [from, control, to] = this.curve(edge);
      const age = this.time - latest[edge.source];
      const recent = Math.max(0, 1 - age / 16);
      g.beginPath(); g.moveTo(from.x, from.y);
      g.quadraticCurveTo(control.x, control.y, to.x, to.y);
      g.strokeStyle = recent ? COLORS[this.data.nodes[edge.source].stage] : '#607454';
      g.globalAlpha = recent ? 0.12 + recent * 0.32 : 0.14;
      g.lineWidth = recent ? 1.05 : 0.65;
      g.setLineDash(edge.excitatory ? [] : [2, 4]);
      g.stroke();
      g.setLineDash([]);
      // A travelling dot uses the simulator's actual synaptic delay. Node flashes
      // still depend exclusively on target spikes, independently of this dot.
      if (age >= 0 && age <= this.data.delay_ms) {
        const t = age / this.data.delay_ms, u = 1 - t;
        const x = u * u * from.x + 2 * u * t * control.x + t * t * to.x;
        const y = u * u * from.y + 2 * u * t * control.y + t * t * to.y;
        g.globalAlpha = 0.95;
        g.fillStyle = COLORS[this.data.nodes[edge.source].stage];
        g.beginPath(); g.arc(x, y, 1.8, 0, Math.PI * 2); g.fill();
      }
    }
    g.globalAlpha = 1;
    this.data.nodes.forEach((node, i) => {
      const point = this.positions[i], age = this.time - latest[i];
      const flash = Math.max(0, 1 - age / 14), fired = Number.isFinite(latest[i]);
      const color = COLORS[node.stage];
      if (flash > 0) {
        const glow = g.createRadialGradient(point.x, point.y, 0, point.x, point.y, 13);
        glow.addColorStop(0, color + '80'); glow.addColorStop(1, color + '00');
        g.globalAlpha = flash;
        g.fillStyle = glow; g.fillRect(point.x - 13, point.y - 13, 26, 26);
      }
      g.globalAlpha = fired ? 0.45 + flash * 0.55 : 0.2;
      g.fillStyle = flash > 0.75 ? '#f5fff9' : color;
      g.beginPath(); g.arc(point.x, point.y, 1.9 + flash * 1.7, 0, Math.PI * 2); g.fill();
      if (this.hover === i) {
        g.globalAlpha = 1; g.strokeStyle = color; g.lineWidth = 1;
        g.beginPath(); g.arc(point.x, point.y, 7, 0, Math.PI * 2); g.stroke();
      }
    });
    g.globalAlpha = 1;
  }

  point(event) {
    if (!this.data) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left, y = event.clientY - rect.top;
    let nearest = -1, distance = 10;
    this.positions.forEach((point, i) => {
      const d = Math.hypot(point.x - x, point.y - y);
      if (d < distance) { nearest = i; distance = d; }
    });
    this.hover = nearest;
    this.tooltip.hidden = nearest < 0;
    if (nearest >= 0) {
      const node = this.data.nodes[nearest];
      this.tooltip.textContent = `${node.type || 'Unlabelled neuron'} · ${node.count} recorded spikes`;
      this.tooltip.style.left = `${Math.max(4, Math.min(x + 12, this.width - 215))}px`;
      this.tooltip.style.top = `${Math.max(4, y - 38)}px`;
    }
    if (!this.playing) this.draw();
  }
}
