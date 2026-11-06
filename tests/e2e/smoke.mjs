// Browser smoke test of the caddy catalogue app (Playwright + Chromium).
//   node tests/e2e/smoke.mjs           (uses .venv/bin/python; set PYTHON=... to use another)
//   SMOKE_URL=http://127.0.0.1:8000 node tests/e2e/smoke.mjs   → test an app that is already running
// Starts the app (unless SMOKE_URL is set), then for each catalogue family loads its sample through
// POST /api/reference, reloads the page, clicks every detail tab and the New screen, and finally replays
// the recorded bspt_nipple run. Fails on any page error or console error (failed Google Fonts requests are ignored:
// they are cosmetic and often blocked offline/in CI).
import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";

const require = createRequire(import.meta.url);
let chromium;
try { ({ chromium } = require("playwright")); }
catch { ({ chromium } = createRequire(path.join(process.env.NODE_PATH || "/usr/local/lib/node_modules", "x"))("playwright")); }

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const APP = ROOT;
const PORT = +(process.env.SMOKE_PORT || 8765);
const BASE = process.env.SMOKE_URL || `http://127.0.0.1:${PORT}`;
const PYTHON = process.env.PYTHON || path.join(ROOT, ".venv/bin/python");
const FAMILIES = ["bspt_nipple", "gripper", "roadster", "flying_car"];
const BUILD_MS = 5 * 60_000;          // the two cars take 1–2 min to build; allow 5

const t0 = Date.now();
const log = (...a) => console.log(`[${((Date.now() - t0) / 1000).toFixed(1).padStart(6)}s]`, ...a);
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

let server = null;
async function startServer() {
  if (process.env.SMOKE_URL) return;
  server = spawn(PYTHON, ["-m", "caddy", "--port", String(PORT)], { cwd: APP, stdio: ["ignore", "pipe", "pipe"] });
  let out = "";
  server.stdout.on("data", d => out += d); server.stderr.on("data", d => out += d);
  server.on("exit", c => { if (c !== null && c !== 0) { console.error(out); console.error(`app exited with ${c}`); } });
  for (let i = 0; i < 120; i++) {
    try { if ((await fetch(`${BASE}/api/state`)).ok) return log(`app up on ${BASE}`); } catch {}
    if (server.exitCode !== null) throw new Error(`app failed to start:\n${out}`);
    await sleep(500);
  }
  throw new Error(`app did not answer on ${BASE} within 60 s:\n${out}`);
}

const errors = [];
const ignorable = (url = "", text = "") => /fonts\.(googleapis|gstatic)\.com/.test(url + " " + text);

async function waitIdle(page, ms = BUILD_MS) {
  // wait until the server reports no run in progress
  const end = Date.now() + ms;
  while (Date.now() < end) {
    const st = await (await page.request.get(`${BASE}/api/state`, { timeout: ms })).json();
    if (!st.running) return st;
    await sleep(1000);
  }
  throw new Error("timed out waiting for the run to finish");
}

async function clickTabs(page, fam) {
  await page.waitForSelector("#dtabs", { state: "visible", timeout: 60_000 });
  const tabs = page.locator("#dtabs [data-tab]");
  const n = await tabs.count();
  const clicked = [];
  for (let i = 0; i < n; i++) {
    const b = tabs.nth(i);
    if (!(await b.isVisible())) continue;
    const name = (await b.textContent()).trim();
    await b.click();
    await page.waitForTimeout(name === "Aero" ? 4000 : 1500);   // let each tab render (Aero computes on the server)
    clicked.push(name);
  }
  log(`${fam}: tabs ${clicked.join(", ")}`);
  if (!clicked.includes("Feature tree") || !clicked.includes("Explainability")) throw new Error(`${fam}: expected tabs missing (${clicked})`);
  if (["roadster", "flying_car"].includes(fam) && !clicked.includes("Aero")) log(`${fam}: note, no Aero tab shown`);
}

async function openNew(page) {
  // "+ New" lives on the empty screen (#btn-new) or above the model viewer (#btn-new2)
  await page.locator("#btn-new:visible, #btn-new2:visible, #btn-new3:visible").first().click();
  await page.waitForSelector("#samples [data-go=replay]", { timeout: 30_000 });
  await page.waitForTimeout(500);
}

async function main() {
  await startServer();
  const browser = await chromium.launch({ args: ["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"] });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  await page.addInitScript(() => { try { localStorage.setItem("caddy.tour", "done"); } catch {} });   // the first-visit tour would cover the page
  page.setDefaultTimeout(60_000);
  page.on("pageerror", e => errors.push(`pageerror: ${e.message}`));
  page.on("console", m => {
    if (m.type() !== "error") return;
    if (ignorable(m.location()?.url, m.text())) return;
    errors.push(`console.error: ${m.text()} (${m.location()?.url || ""})`);
  });

  await page.goto(BASE, { waitUntil: "load" });
  await openNew(page);
  log("New screen ok");

  for (const fam of FAMILIES) {
    const ts = Date.now();
    const r = await page.request.post(`${BASE}/api/reference`, { data: { family: fam }, timeout: BUILD_MS });
    if (!r.ok()) throw new Error(`${fam}: POST /api/reference → ${r.status()} ${await r.text()}`);
    await waitIdle(page);
    log(`${fam}: sample built in ${((Date.now() - ts) / 1000).toFixed(1)} s`);
    await page.reload({ waitUntil: "load" });
    await clickTabs(page, fam);
    await openNew(page);
  }

  // ▶ Replay the recorded bspt_nipple run and wait for it to finish
  const ts = Date.now();
  await page.click("#samples [data-go=replay][data-f=bspt_nipple]");
  await page.waitForFunction(async () => (await (await fetch("/api/state")).json()).running, null, { timeout: 30_000, polling: 250 })
    .catch(() => {});   // a short replay may finish before we look
  await waitIdle(page);
  await page.waitForSelector("#dtabs", { state: "visible", timeout: 60_000 });
  log(`bspt_nipple: replay finished in ${((Date.now() - ts) / 1000).toFixed(1)} s`);
  await clickTabs(page, "bspt_nipple (replayed)");

  await browser.close();
}

let code = 0;
try { await main(); }
catch (e) { console.error(`FAIL: ${e.stack || e}`); code = 1; }
if (errors.length) { console.error(`FAIL: ${errors.length} browser error(s):\n  ` + errors.join("\n  ")); code = 1; }
if (!code) log("PASS: smoke test");
server?.kill();
process.exit(code);
