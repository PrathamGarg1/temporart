import { pipeline, env } from '@xenova/transformers';
import weights from './assets/narrative/kv_weights.json' with { type: 'json' };
import { NarrativeCache, RollingWindow } from './narrative_cache.js';
import { saveIncident } from './store.js';

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = chrome.runtime.getURL('assets/models/');
// CacheStorage rejects chrome-extension:// request URLs in Chrome.
env.useBrowserCache = false;
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.proxy = false;
env.backends.onnx.wasm.wasmPaths = chrome.runtime.getURL('assets/');

const DEBUG = true;
const DEBUG_LOG_MESSAGES = true;
const FLAG = 0.5;
const HIGH = 0.9;
const PREFIX = '[SurakshaNet]';

let classifier = null;
let modelPromise = null;
const queue = [];
let busy = false;
const rolling = new RollingWindow({
  maxSlots: weights.maxSlots,
  maxTokens: weights.maxWindowTokens,
  shortTokens: weights.shortTokens || 8,
});
const narrativeFallback = new NarrativeCache(weights);

function log(event, details = {}) {
  if (DEBUG) console.info(PREFIX, event, details);
}

function logError(event, error, details = {}) {
  console.error(PREFIX, event, {
    ...details,
    error: String(error?.stack || error?.message || error),
  });
}

function debugText(text) {
  return DEBUG_LOG_MESSAGES ? text : `[redacted:${text.length} chars]`;
}

log('worker:loaded', {
  model: 'custom-macd-model',
  localModels: env.allowLocalModels,
  remoteModels: env.allowRemoteModels,
  wasmThreads: env.backends.onnx.wasm.numThreads,
});

async function ensureModel() {
  if (classifier) return classifier;
  if (!modelPromise) {
    log('model:load:start', { modelPath: env.localModelPath });
    modelPromise = pipeline('text-classification', 'custom-macd-model', { top_k: null })
      .then((loaded) => {
        classifier = loaded;
        log('model:load:success');
        return loaded;
      })
      .catch((error) => {
        modelPromise = null;
        logError('model:load:failed', error, { modelPath: env.localModelPath });
        throw error;
      });
  }
  return modelPromise;
}

function abusiveScore(results) {
  const rows = Array.isArray(results) ? results : [results];
  for (const r of rows) {
    const l = String(r.label).toLowerCase();
    if (l === 'abusive' || l === 'label_0') return r.score;
  }
  for (const r of rows) {
    const l = String(r.label).toLowerCase();
    if (l === 'non-abusive' || l === 'label_1') return 1 - r.score;
  }
  return 0;
}

async function classify(text, contact) {
  const view = rolling.observe(contact, text);
  const fallback = narrativeFallback.score(contact, text);
  log('classify:start', {
    text: debugText(text),
    textLength: text.length,
    slots: view.slots,
    useWindow: view.useWindow,
    mIndex: Number(fallback.mIndex.toFixed(3)),
    iIndex: Number(fallback.iIndex.toFixed(3)),
  });
  let score;
  let mode;
  try {
    await ensureModel();
    score = abusiveScore(await classifier(text));
    mode = 'current';
    if (view.useWindow) {
      const windowScore = abusiveScore(await classifier(view.windowText));
      score = Math.max(score, windowScore);
      mode = 'rolling-window';
    }
  } catch (error) {
    logError('classify:onnx:fallback', error, { slots: view.slots });
    score = fallback.abusive;
    mode = 'kv-fallback';
  }
  const flagged = mode === 'kv-fallback'
    ? score >= (weights.precisionThreshold ?? weights.threshold ?? FLAG)
    : score > FLAG;
  const result = {
    flagged,
    score,
    severity: !flagged ? null : score >= HIGH ? 'high' : 'medium',
    narrative: {
      mode,
      slots: view.slots,
      currentTokens: view.currentTokens,
      mIndex: fallback.mIndex,
      iIndex: fallback.iIndex,
    },
  };
  log('classify:complete', {
    text: debugText(text),
    textLength: text.length,
    flagged: result.flagged,
    score: Number(result.score.toFixed(4)),
    severity: result.severity,
    mode: result.narrative.mode,
  });
  return result;
}

async function drain() {
  if (busy) return;
  busy = true;
  log('queue:drain:start', { pending: queue.length });
  try {
    while (queue.length) {
      const job = queue.shift();
      log('queue:job:start', {
        pending: queue.length,
        text: debugText(job.text),
        textLength: job.text.length,
        contact: job.contact,
      });
      try {
        const result = await classify(job.text, job.contact);
        if (result.flagged) {
          const saved = await saveIncident({
            text: job.text,
            contact: job.contact,
            score: result.score,
            severity: result.severity,
          });
          log('incident:save', {
            created: Boolean(saved),
            id: saved?.id?.slice(0, 12) || null,
            text: debugText(job.text),
            contact: job.contact,
          });
        }
        job.resolve(result);
        log('queue:job:success', { pending: queue.length, flagged: result.flagged });
      } catch (error) {
        logError('queue:job:failed', error, {
          textLength: job.text.length,
          contact: job.contact,
        });
        job.reject(error);
      }
    }
  } finally {
    busy = false;
    log('queue:drain:complete', { pending: queue.length });
  }
}

function enqueue(text, contact) {
  log('queue:enqueue', {
    text: debugText(text),
    pending: queue.length + 1,
    textLength: text.length,
    contact,
  });
  return new Promise((resolve, reject) => {
    queue.push({ text, contact, resolve, reject });
    drain();
  });
}

chrome.runtime.onInstalled.addListener((details) => {
  log('runtime:installed', { reason: details.reason });
  ensureModel().catch(() => {});
});
chrome.runtime.onStartup.addListener(() => {
  log('runtime:startup');
  ensureModel().catch(() => {});
});
chrome.runtime.onSuspend.addListener(() => log('runtime:suspend'));

chrome.runtime.onMessage.addListener((msg, _s, sendResponse) => {
  if (msg?.type !== 'ANALYZE') {
    log('message:ignored', { type: msg?.type || null });
    return false;
  }
  if (typeof msg.text !== 'string' || !msg.text.trim()) {
    log('message:invalid', { textType: typeof msg.text });
    sendResponse({ flagged: false, error: 'Message text is required.' });
    return false;
  }
  log('message:analyze', {
    text: debugText(msg.text),
    textLength: msg.text.length,
    contact: msg.contact,
  });
  enqueue(msg.text, msg.contact)
    .then((r) => sendResponse(r))
    .catch((error) => {
      logError('message:analyze:failed', error, {
        text: debugText(msg.text),
        textLength: msg.text.length,
        contact: msg.contact,
      });
      sendResponse({ flagged: false, error: String(error.message || error) });
    });
  return true;
});
