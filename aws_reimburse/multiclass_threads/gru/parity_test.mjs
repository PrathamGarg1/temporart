// Node check: JS GRU logits must match the PyTorch logits on the fixture threads.
import { readFileSync } from 'node:fs';
import { loadGru, logits } from './gru_runtime.js';
const m = loadGru(readFileSync(new URL('./out/gru.json', import.meta.url), 'utf8'));
const fx = JSON.parse(readFileSync(new URL('./out/parity_fixture.json', import.meta.url), 'utf8'));
let worst = 0, argmaxOk = 0;
const t0 = performance.now();
for (const t of fx) {
  const lg = logits(m, t.vectors);
  lg.forEach((v, i) => { worst = Math.max(worst, Math.abs(v - t.logits[i])); });
  if (lg.indexOf(Math.max(...lg)) === t.logits.indexOf(Math.max(...t.logits))) argmaxOk++;
}
const ms = (performance.now() - t0) / fx.length;
console.log(`threads=${fx.length} max|logit diff|=${worst.toExponential(2)} argmax match=${argmaxOk}/${fx.length} avg ${ms.toFixed(2)} ms/thread`);
if (worst > 1e-3 || argmaxOk !== fx.length) { console.error('PARITY FAIL'); process.exit(1); }
console.log('PARITY OK');
