/**
 * Rolling context window and a small dynamic key/value cache for on-device
 * Hinglish narrative scoring.
 *
 * The service worker keeps at most 8 turns and 96 content tokens per chat.
 * When the newest turn is a short fragment, the worker re-scores the joined
 * window with the INT8 encoder. When that encoder is unavailable, score()
 * runs the attention cache trained in aws_train/benchmark_narrative_kv.py.
 *
 * M-index and I-index follow Yang & Chai, CodeMixBench, EMNLP 2025, §3.4.
 */

const TOKEN_RE = /[A-Za-z]+(?:'[A-Za-z]+)?|[\u0900-\u097F]+|\d+|[^\s\w]/gu;
const DEVANAGARI_RE = /[\u0900-\u097F]/;

export function tokenize(text) {
  return String(text || '').match(TOKEN_RE) || [];
}

function isIndependent(tok) {
  if (!tok) return true;
  if (/^\d+$/.test(tok) || /^<number>$/i.test(tok)) return true;
  const cp = tok.codePointAt(0);
  const emoji =
    tok.length <= 2 &&
    ((cp >= 0x1f300 && cp <= 0x1f9ff) ||
      (cp >= 0x2700 && cp <= 0x27bf) ||
      (cp >= 0x1fa00 && cp <= 0x1faff));
  if (emoji) return true;
  if (/^[^\w\s]$/u.test(tok)) return true;
  if (tok.startsWith('@') || tok.startsWith('#') || tok.startsWith('http')) return true;
  if (/^[A-Z]{2,5}$/.test(tok)) return true;
  return false;
}

function contentTokens(text) {
  const out = [];
  for (const tok of tokenize(text)) {
    if (!isIndependent(tok)) out.push(tok.toLowerCase());
  }
  return out;
}

export function contentTokenCount(text) {
  return Math.max(contentTokens(text).length, text && String(text).trim() ? 1 : 0);
}

function langTag(tok, hindi, english) {
  if (isIndependent(tok)) return 'univ';
  if (DEVANAGARI_RE.test(tok)) return 'hi';
  const w = tok.toLowerCase();
  if (hindi.has(w)) return 'hi';
  if (english.has(w)) return 'en';
  return 'hi';
}

function languageTags(text, hindi, english) {
  return tokenize(text).map((tok) => langTag(tok, hindi, english));
}

export function mIndex(tags) {
  const langs = tags.filter((t) => t === 'hi' || t === 'en');
  if (!langs.length) return 0;
  const pHi = langs.filter((t) => t === 'hi').length / langs.length;
  const pEn = 1 - pHi;
  const sumSq = pHi * pHi + pEn * pEn;
  const denom = sumSq;
  if (denom <= 0) return 0;
  return (1 - sumSq) / denom;
}

export function iIndex(tags) {
  const langs = tags.filter((t) => t === 'hi' || t === 'en');
  if (langs.length < 2) return 0;
  let switches = 0;
  for (let i = 1; i < langs.length; i += 1) {
    if (langs[i] !== langs[i - 1]) switches += 1;
  }
  return switches / (langs.length - 1);
}

function stemHit(tokens, stem) {
  for (const tok of tokens) {
    if (tok === stem) return true;
    if (stem.length >= 4 && tok.startsWith(stem) && tok.length - stem.length <= 2) return true;
  }
  return false;
}

function featurize(text, weights) {
  const tags = languageTags(text, weights._hindi, weights._english);
  const toks = contentTokens(text);
  const n = Math.max(toks.length, 1);
  let hi = 0;
  let en = 0;
  let switches = 0;
  let prev = null;
  for (const tag of tags) {
    if (tag !== 'hi' && tag !== 'en') continue;
    if (tag === 'hi') hi += 1;
    else en += 1;
    if (prev && tag !== prev) switches += 1;
    prev = tag;
  }
  const langN = hi + en;
  const abuseHits = weights.abuseStems.reduce((sum, stem) => sum + (stemHit(toks, stem) ? 1 : 0), 0);
  const dev = toks.reduce((sum, tok) => sum + (DEVANAGARI_RE.test(tok) ? 1 : 0), 0);
  const raw = tokenize(text);
  const feats = [
    mIndex(tags),
    iIndex(tags),
    langN ? en / langN : 0,
    langN ? hi / langN : 0,
    dev / n,
    Math.min(abuseHits, 3) / 3,
    Math.log1p(toks.length) / Math.log(32),
    Math.min(switches, 8) / 8,
    raw.some((t) => t === '?') ? 1 : 0,
    raw.some((t) => t === '!') ? 1 : 0,
  ];
  for (const stem of weights.abuseStems) feats.push(stemHit(toks, stem) ? 1 : 0);
  for (const stem of weights.benignStems) feats.push(stemHit(toks, stem) ? 1 : 0);
  return feats;
}

function standardize(feats, weights) {
  return feats.map((value, i) => (value - weights.mean[i]) / Math.max(weights.std[i], 1e-6));
}

function tanhVec(matrix, vector) {
  const out = new Array(matrix.length);
  for (let r = 0; r < matrix.length; r += 1) {
    const row = matrix[r];
    let sum = 0;
    for (let c = 0; c < vector.length; c += 1) sum += row[c] * vector[c];
    out[r] = Math.tanh(sum);
  }
  return out;
}

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) sum += a[i] * b[i];
  return sum;
}

function sigmoid(logit) {
  const clipped = Math.max(-20, Math.min(20, logit));
  return 1 / (1 + Math.exp(-clipped));
}

function trim(slots, maxSlots, maxTokens) {
  const kept = slots.slice();
  while (kept.length > maxSlots) kept.shift();
  while (kept.length > 1 && kept.reduce((sum, slot) => sum + slot.tokens, 0) > maxTokens) kept.shift();
  return kept;
}

export class RollingWindow {
  constructor({ maxSlots = 8, maxTokens = 96, shortTokens = 8, maxContacts = 24 } = {}) {
    this.maxSlots = maxSlots;
    this.maxTokens = maxTokens;
    this.shortTokens = shortTokens;
    this.maxContacts = maxContacts;
    this.chats = new Map();
  }

  observe(contact, text) {
    const key = contact || 'unknown';
    let slots = this.chats.get(key);
    if (!slots) {
      slots = [];
      this.chats.set(key, slots);
    } else {
      this.chats.delete(key);
      this.chats.set(key, slots);
    }
    while (this.chats.size > this.maxContacts) {
      const oldest = this.chats.keys().next().value;
      this.chats.delete(oldest);
    }
    const tokens = contentTokenCount(text);
    const last = slots[slots.length - 1];
    if (!last || last.text !== text) {
      slots.push({ text, tokens });
    }
    const kept = trim(slots, this.maxSlots, this.maxTokens);
    this.chats.set(key, kept);
    const currentTokens = tokens;
    const useWindow = kept.length > 1 && currentTokens < this.shortTokens;
    return {
      useWindow,
      windowText: kept.map((slot) => slot.text).join(' '),
      slots: kept.length,
      currentTokens,
      windowTokens: kept.reduce((sum, slot) => sum + slot.tokens, 0),
    };
  }
}

export class NarrativeCache {
  constructor(weights) {
    this.weights = {
      ...weights,
      _hindi: new Set(weights.hindiClosed || []),
      _english: new Set(weights.enForce || []),
    };
    this.chats = new Map();
    this.maxContacts = 24;
  }

  score(contact, text) {
    const weights = this.weights;
    const key = contact || 'unknown';
    let slots = this.chats.get(key) || [];
    const feats = standardize(featurize(text, weights), weights);
    const tokens = contentTokenCount(text);
    const last = slots[slots.length - 1];
    if (!last || last.text !== text) {
      slots = slots.concat([
        {
          text,
          tokens,
          keyVec: tanhVec(weights.Wk, feats),
          valueVec: tanhVec(weights.Wv, feats),
          feats,
        },
      ]);
    }
    slots = trim(slots, weights.maxSlots, weights.maxWindowTokens);
    this.chats.delete(key);
    this.chats.set(key, slots);
    while (this.chats.size > this.maxContacts) {
      const oldest = this.chats.keys().next().value;
      this.chats.delete(oldest);
    }
    const current = slots[slots.length - 1];
    const query = tanhVec(weights.Wq, current.feats);
    const scale = Math.sqrt(weights.dim);
    const scores = slots.map((slot) => dot(slot.keyVec, query) / scale);
    const peak = Math.max(...scores);
    const exp = scores.map((score) => Math.exp(score - peak));
    const mass = exp.reduce((sum, value) => sum + value, 0);
    const alpha = exp.map((value) => value / mass);
    const ctx = new Array(weights.dim).fill(0);
    slots.forEach((slot, i) => {
      for (let d = 0; d < weights.dim; d += 1) ctx[d] += alpha[i] * slot.valueVec[d];
    });
    const hidden = current.feats.concat(ctx);
    let logit = weights.bias;
    for (let i = 0; i < weights.head.length; i += 1) logit += weights.head[i] * hidden[i];
    const tags = languageTags(text, weights._hindi, weights._english);
    return {
      abusive: sigmoid(logit),
      mIndex: mIndex(tags),
      iIndex: iIndex(tags),
      slots: slots.length,
      currentTokens: tokens,
      attnLast: alpha[alpha.length - 1],
    };
  }
}
