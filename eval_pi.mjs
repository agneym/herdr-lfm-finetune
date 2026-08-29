#!/usr/bin/env node
/**
 * eval_pi.mjs — score any pi-catalog model on the pinned 98-row Herdr
 * holdout, driven through pi's model layer (ModelRuntime), with the 25 Herdr
 * tools loaded from reference/herdr_schemas.json.
 *
 * One turn per row (planner mode), matching eval_lfm2.py's scoring:
 *   - exact-call accuracy (tool name + args, key-order-insensitive)
 *   - exact-call normalized (pane_split current=true invariant)
 *   - tool-selection accuracy (tool name only)
 *   - off-topic (no expected tool -> model must call nothing)
 *
 * Also records per-row + total token usage and cost (pi's own accounting).
 *
 * Usage:
 *   node eval_pi.mjs [--provider openrouter] [--model z-ai/glm-5.3-flash]
 *                    [--limit N] [--thinking off|low|high|max] [--max-tokens N]
 */
import { readFileSync, writeFileSync } from "node:fs";
import { ModelRuntime } from "/home/agney/.local/share/mise/installs/npm-earendil-works-pi-coding-agent/0.84.3/node_modules/.mise/@earendil-works+pi-coding-agent@0.84.3/node_modules/@earendil-works/pi-coding-agent/dist/bundle/index.js";

const DATA = "dataset.jsonl";
const HOLDOUT = "runs/eval_v5_holdout.json";
const SCHEMAS = "reference/herdr_schemas.json";

// --- CLI args ---------------------------------------------------------------
const args = process.argv.slice(2);
function argVal(name, dflt) {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] !== undefined ? args[i + 1] : dflt;
}
const PROVIDER = argVal("--provider", "deepseek");
const MODEL = argVal("--model", "deepseek-v4-flash-vision-exp");
const LIMIT = parseInt(argVal("--limit", "0"), 10) || 0;
const THINKING = argVal("--thinking", "high"); // off|low|high|max (medium unsupported)
const MAX_TOKENS = parseInt(argVal("--max-tokens", "512"), 10) || 512;

// --- JSON helpers (key-order-insensitive) -----------------------------------
function sortKeys(x) {
  if (Array.isArray(x)) return x.map(sortKeys);
  if (x && typeof x === "object") {
    const out = {};
    for (const k of Object.keys(x).sort()) out[k] = sortKeys(x[k]);
    return out;
  }
  return x;
}
const sortedJson = (obj) => JSON.stringify(sortKeys(obj == null ? {} : obj));

// --- Counters (matching eval_lfm2.py's collections.Counter) ------------------
function add(map, key) { map.set(key, (map.get(key) || 0) + 1); }
function countersEqual(a, b) {
  if (a.size !== b.size) return false;
  for (const [k, v] of a) if (b.get(k) !== v) return false;
  return true;
}
function normCalls(calls) {
  const m = new Map();
  for (const c of calls) add(m, `${c.name}\u0000${sortedJson(c.arguments)}`);
  return m;
}
function namesOf(calls) {
  const m = new Map();
  for (const c of calls) add(m, c.name);
  return m;
}
function argKey(name, calls) {
  const m = new Map();
  for (const c of calls) if (c.name === name) add(m, sortedJson(c.arguments));
  return m;
}
function normalizeCall(call) {
  if (call.name === "pane_split") {
    const args = { ...(call.arguments || {}) };
    if (!args.pane && !("current" in args)) args.current = true;
    return { name: call.name, arguments: args };
  }
  return call;
}

// --- Load data ---------------------------------------------------------------
const rows = readFileSync(DATA, "utf8").trim().split("\n").map((l) => JSON.parse(l));
const holdout = JSON.parse(readFileSync(HOLDOUT, "utf8"));
const schemas = JSON.parse(readFileSync(SCHEMAS, "utf8"));
const tools = schemas.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters }));

const querySet = new Set(holdout.queries);
const byQuery = new Map();
rows.forEach((r, i) => {
  const q = r.messages[1].content;
  if (!byQuery.has(q)) byQuery.set(q, []);
  byQuery.get(q).push(i);
});
const holdIdx = holdout.queries.flatMap((q) => byQuery.get(q) || []);
const evalIdx = LIMIT ? holdIdx.slice(0, LIMIT) : holdIdx;

console.log(`data rows: ${rows.length}   holdout rows: ${holdIdx.length}   evaluating: ${evalIdx.length}`);
console.log(`model: ${PROVIDER}/${MODEL}   thinking: ${THINKING}   maxTokens: ${MAX_TOKENS}`);
console.log(`tools: ${tools.length} herdr tools loaded`);

// --- Model runtime -----------------------------------------------------------
const rt = await ModelRuntime.create();
const model = rt.getModel(PROVIDER, MODEL);
if (!model) throw new Error(`model not found: ${PROVIDER}/${MODEL}`);
if (!rt.hasConfiguredAuth(PROVIDER)) throw new Error(`no auth for provider: ${PROVIDER}`);

const reasoningEffort = THINKING === "off" ? undefined : THINKING;

// --- Eval loop ---------------------------------------------------------------
let exact = 0, exactNorm = 0, toolOk = 0, nExpected = 0, nOfftopic = 0, offOk = 0;
const argCounts = new Map(); // name -> [ok, total]
const totals = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, reasoning: 0, totalTokens: 0, cost: 0 };
const detail = [];
const perRow = [];

const t0 = Date.now();
for (let n = 0; n < evalIdx.length; n++) {
  const i = evalIdx[n];
  const row = rows[i];
  const sys = row.messages[0].content;
  const user = row.messages[1].content;
  const expected = row.expected || [];

  let predicted = [];
  let usage = null;
  let err = null;
  try {
    const ctx = {
      systemPrompt: sys,
      messages: [{ role: "user", content: user, timestamp: Date.now() }],
      tools,
    };
    const out = await rt.complete(model, ctx, {
      maxTokens: MAX_TOKENS,
      reasoningEffort,
      temperature: 0,
    });
    usage = out.usage;
    predicted = out.content
      .filter((b) => b.type === "toolCall")
      .map((b) => ({ name: b.name, arguments: b.arguments || {} }));
  } catch (e) {
    err = String(e?.message || e);
    usage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, reasoning: 0, totalTokens: 0, cost: { total: 0 } };
  }

  // accumulate usage
  totals.input += usage.input || 0;
  totals.output += usage.output || 0;
  totals.cacheRead += usage.cacheRead || 0;
  totals.cacheWrite += usage.cacheWrite || 0;
  totals.reasoning += usage.reasoning || 0;
  totals.totalTokens += usage.totalTokens || 0;
  totals.cost += usage.cost?.total || 0;

  const normalized = predicted.map(normalizeCall);
  const pn = normCalls(predicted);
  const pnNorm = normCalls(normalized);
  const en = normCalls(expected);
  const pw = namesOf(predicted);
  const ew = namesOf(expected);

  let note = err ? `ERROR: ${err}` : "";
  if (en.size === 0) {
    nOfftopic++;
    if (predicted.length === 0) offOk++;
    else note = "OFF-TOPIC but called a tool";
  } else {
    nExpected++;
    if (countersEqual(pn, en)) exact++;
    if (countersEqual(pnNorm, en)) exactNorm++;
    if (countersEqual(pw, ew)) toolOk++;
    for (const name of ew.keys()) {
      const ok = countersEqual(argKey(name, predicted), argKey(name, expected));
      const rec = argCounts.get(name) || [0, 0];
      rec[0] += ok ? 1 : 0;
      rec[1] += 1;
      argCounts.set(name, rec);
    }
  }

  perRow.push({
    row: i, query: user, expected, predicted, normalized, ok: countersEqual(pnNorm, en),
    note,
    usage,
  });
  if (note || (en.size && !countersEqual(pn, en))) {
    detail.push({ row: i, query: user, expected, predicted, note });
  }
  if ((n + 1) % 10 === 0 || n === evalIdx.length - 1) {
    console.log(`  ${n + 1}/${evalIdx.length}  (${((Date.now() - t0) / 1000).toFixed(0)}s)  exact so far: ${exact}/${nExpected}`);
  }
}

// --- Report ------------------------------------------------------------------
console.log("\n== summary ==");
console.log(`  eval rows              : ${evalIdx.length}`);
if (nExpected) {
  console.log(`  exact-call accuracy    : ${exact}/${nExpected} = ${(100 * exact / nExpected).toFixed(1)}%`);
  console.log(`  exact (normalized)     : ${exactNorm}/${nExpected} = ${(100 * exactNorm / nExpected).toFixed(1)}%`);
  console.log(`  tool-selection acc     : ${toolOk}/${nExpected} = ${(100 * toolOk / nExpected).toFixed(1)}%`);
}
if (nOfftopic) {
  console.log(`  off-topic (no call)    : ${offOk}/${nOfftopic} = ${(100 * offOk / nOfftopic).toFixed(1)}%`);
}
console.log("\n== per-tool argument grounding ==");
for (const name of [...argCounts.keys()].sort()) {
  const [ok, total] = argCounts.get(name);
  console.log(`  ${name.padEnd(22)} ${ok}/${total} exact (${Math.round(100 * ok / total)}%)`);
}

console.log("\n== tokens & cost ==");
const m = 1e6;
console.log(`  input tokens     : ${totals.input}  ($${(totals.input / m * model.cost.input).toFixed(4)})`);
console.log(`  cacheRead tokens : ${totals.cacheRead}  ($${(totals.cacheRead / m * model.cost.cacheRead).toFixed(4)})`);
console.log(`  cacheWrite tokens: ${totals.cacheWrite}`);
console.log(`  output tokens    : ${totals.output}  ($${(totals.output / m * model.cost.output).toFixed(4)})`);
console.log(`  reasoning tokens : ${totals.reasoning}  (subset of output)`);
console.log(`  total tokens     : ${totals.totalTokens}`);
console.log(`  TOTAL COST       : $${totals.cost.toFixed(6)}`);
console.log(`  per-row cost     : $${(totals.cost / evalIdx.length).toFixed(6)}`);

if (detail.length) {
  console.log("\n== mismatches / errors (first 15) ==");
  for (const d of detail.slice(0, 15)) {
    console.log(`  row ${d.row}: ${d.query.slice(0, 80)}`);
    if (d.note) console.log(`    ${d.note}`);
    console.log(`    expected : ${JSON.stringify(d.expected).slice(0, 160)}`);
    console.log(`    predicted: ${JSON.stringify(d.predicted).slice(0, 160)}`);
  }
}

// persist
const stamp = new Date().toISOString().replace(/[:.]/g, "-");
const outJson = {
  model: `${PROVIDER}/${MODEL}`,
  provider: PROVIDER,
  modelId: MODEL,
  thinking: THINKING,
  maxTokens: MAX_TOKENS,
  evalIdx, totals, perRow,
  acc: { exact, exactNorm, toolOk, nExpected, offOk, nOfftopic },
};
const slug = `${PROVIDER}__${MODEL}`.replace(/[^a-zA-Z0-9_-]+/g, "-");
writeFileSync(`runs/eval_pi_${slug}_${stamp}.json`, JSON.stringify(outJson, null, 2));
console.log(`\nsaved: runs/eval_pi_${slug}_${stamp}.json`);
