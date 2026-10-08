import { readFileSync } from 'node:fs';
import { NarrativeCache, RollingWindow, iIndex, mIndex } from '../narrative_cache.js';

const weights = JSON.parse(readFileSync(new URL('../assets/narrative/kv_weights.json', import.meta.url), 'utf8'));
const cases = JSON.parse(readFileSync(new URL('../codemix_eval/narrative_parity_cases.json', import.meta.url), 'utf8'));

if (Math.abs(mIndex(['hi', 'en']) - 1) > 1e-9) throw new Error('M-index balanced case failed');
if (Math.abs(mIndex(['en', 'en']) ) > 1e-9) throw new Error('M-index monolingual case failed');
if (Math.abs(iIndex(['hi', 'en', 'hi']) - 1) > 1e-9) throw new Error('I-index failed');

const cache = new NarrativeCache(weights);
for (const row of cases) {
  const contact = `parity-${row.name}`;
  let got = null;
  for (const turn of row.history) got = cache.score(contact, turn);
  got = cache.score(contact, row.text);
  const delta = Math.abs(got.abusive - row.prob_abusive);
  if (delta > 1e-4) {
    throw new Error(`${row.name} prob ${got.abusive} != ${row.prob_abusive} (delta ${delta})`);
  }
  if (Math.abs(got.mIndex - row.m_index) > 1e-6) {
    throw new Error(`${row.name} M-index ${got.mIndex} != ${row.m_index}`);
  }
  if (got.slots !== row.slots) throw new Error(`${row.name} slots ${got.slots} != ${row.slots}`);
  console.log(`${row.name} ok prob=${got.abusive.toFixed(4)} slots=${got.slots}`);
}

const rolling = new RollingWindow({ maxSlots: 2, maxTokens: 96, shortTokens: 8 });
const first = rolling.observe('c', 'this is a long enough benign message for the gate');
if (first.useWindow) throw new Error('first turn must stay on the current message');
const second = rolling.observe('c', 'short');
if (!second.useWindow) throw new Error('short follow-up must open the rolling window');
if (second.slots !== 2) throw new Error(`expected 2 slots, got ${second.slots}`);
const third = rolling.observe('c', 'tail');
if (third.slots !== 2) throw new Error('slot cap failed');
if (!third.windowText.endsWith('tail')) throw new Error('window should keep the newest turn');
console.log('rolling window ok');
