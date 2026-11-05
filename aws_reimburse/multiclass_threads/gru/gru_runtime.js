// Runs the thread GRU in plain JavaScript (no dependencies; works in a service worker / content script).
// Input : chronological per-line vectors (384-d, from the frozen MiniLM) of the last 3/5/7/10 messages.
// Output: probabilities over the 5 classes. Nothing leaves the browser.
//
// Line text must be prefixed BEFORE encoding: 'in: ' for the speaker of the LAST message,
// 'out: ' for the other speaker (the two alternate). Use tagLines() for that.

export const CLASSES = ['benign', 'friendly_sarcasm', 'hostile_abuse', 'sexual_harassment', 'predatory_grooming'];
export const MAX_LINES = 10;

export function tagLines(texts) {
  const n = texts.length;
  return texts.map((t, i) => ((n - 1 - i) % 2 === 0 ? 'in: ' : 'out: ') + String(t).trim());
}

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

export function loadGru(json) {
  const g = typeof json === 'string' ? JSON.parse(json) : json;
  const f = (a) => Float32Array.from(a);
  return { D: g.input_dim, H: g.hidden, classes: g.classes, Wih: f(g.W_ih), Whh: f(g.W_hh), bih: f(g.b_ih), bhh: f(g.b_hh), Wfc: f(g.W_fc), bfc: f(g.b_fc) };
}

export function logits(m, vectors) {
  const { D, H } = m;
  const vs = vectors.slice(-MAX_LINES);          // keep the last 10 lines at most
  let h = new Float32Array(H);
  const gi = new Float32Array(3 * H), gh = new Float32Array(3 * H);
  for (const x of vs) {                           // only real lines are visited, so there are no padded steps
    for (let r = 0; r < 3 * H; r++) {
      let a = m.bih[r], b = m.bhh[r];
      const oi = r * D, oh = r * H;
      for (let k = 0; k < D; k++) a += m.Wih[oi + k] * x[k];
      for (let k = 0; k < H; k++) b += m.Whh[oh + k] * h[k];
      gi[r] = a; gh[r] = b;
    }
    const hn = new Float32Array(H);
    for (let j = 0; j < H; j++) {
      const r = sigmoid(gi[j] + gh[j]);
      const z = sigmoid(gi[H + j] + gh[H + j]);
      const n = Math.tanh(gi[2 * H + j] + r * gh[2 * H + j]);
      hn[j] = (1 - z) * n + z * h[j];
    }
    h = hn;
  }
  const K = m.classes.length, out = new Float32Array(K);
  for (let c = 0; c < K; c++) {
    let s = m.bfc[c];
    for (let j = 0; j < H; j++) s += m.Wfc[c * H + j] * h[j];
    out[c] = s;
  }
  return out;
}

export function classify(m, vectors) {
  const lg = logits(m, vectors);
  const mx = Math.max(...lg);
  const e = Array.from(lg, (v) => Math.exp(v - mx));
  const s = e.reduce((a, b) => a + b, 0);
  const probs = e.map((v) => v / s);
  const top = probs.indexOf(Math.max(...probs));
  return { label: m.classes[top], probs: Object.fromEntries(m.classes.map((c, i) => [c, probs[i]])) };
}
