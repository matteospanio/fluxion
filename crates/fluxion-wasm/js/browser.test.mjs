// fluxion actually runs in a real AudioWorklet (roadmap W4).
//
// `worklet.test.mjs` answers "does a block cost an allocation" deterministically, which is the part
// that must never regress. This answers a different question that only a browser can: can an
// AudioWorklet host this at all — module registration, synchronous wasm instantiation on the audio
// thread, `process()` on the real 128-frame quantum, and messages from the page reaching it.
//
//   ./scripts/build-wasm.sh && node crates/fluxion-wasm/js/browser.test.mjs
//
// Needs a Chromium-family browser. Set CHROME_PATH, or have one on a usual path.
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { extname, join } from "node:path";
import { fileURLToPath } from "node:url";

import puppeteer from "puppeteer-core";

const ROOT = fileURLToPath(new URL(".", import.meta.url));

const BROWSERS = [
  process.env.CHROME_PATH,
  "/usr/bin/google-chrome",
  "/usr/bin/google-chrome-stable",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
  "/snap/bin/chromium",
].filter(Boolean);

const browserPath = BROWSERS.find((p) => existsSync(p));
if (!browserPath) {
  console.error(
    "no Chromium-family browser found. Set CHROME_PATH, or install one.\n" +
      `tried: ${BROWSERS.join(", ")}`,
  );
  process.exit(1);
}

const TYPES = {
  ".html": "text/html",
  ".mjs": "text/javascript",
  ".js": "text/javascript",
  ".wasm": "application/wasm",
};

// A real origin: `file://` blocks module imports and `WebAssembly.compileStreaming`.
const server = createServer(async (req, res) => {
  const path = join(ROOT, decodeURIComponent(req.url.split("?")[0]).replace(/^\/+/, ""));
  try {
    const body = await readFile(path);
    res.writeHead(200, { "content-type": TYPES[extname(path)] ?? "application/octet-stream" });
    res.end(body);
  } catch {
    res.writeHead(404).end("not found");
  }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const origin = `http://127.0.0.1:${server.address().port}`;

const browser = await puppeteer.launch({
  executablePath: browserPath,
  headless: true,
  args: [
    "--no-sandbox",
    // Audio must run without a click, and without real hardware in CI.
    "--autoplay-policy=no-user-gesture-required",
    "--use-fake-device-for-media-stream",
    "--mute-audio",
  ],
});

let failed = 0;
try {
  const page = await browser.newPage();
  const problems = [];
  page.on("pageerror", (e) => problems.push(`page error: ${e.message}`));
  // A missing favicon is the browser being a browser, not a fault in the code under test.
  page.on("console", (m) => {
    // The response handler above reports which URL failed; this would only repeat it.
    if (m.type() === "error" && !/favicon|Failed to load resource/.test(m.text())) {
      problems.push(`console: ${m.text()}`);
    }
  });
  page.on("response", (r) => {
    if (r.status() >= 400 && !/favicon/.test(r.url())) {
      problems.push(`${r.status()} for ${r.url()}`);
    }
  });

  await page.goto(`${origin}/demo.html`, { waitUntil: "load" });

  const chain = "highpass(80, 4) | peaking(1000, 6, 1.5) | gain(0.8)";
  const { frames, peak, worst } = await page.evaluate(
    async (chain) => await window.fluxionOfflineRender({ chain, seconds: 1 }),
    chain,
  );

  const say = (ok, name, detail) => {
    console.log(`  ${ok ? "ok  " : "FAIL"} ${name}${detail ? ` — ${detail}` : ""}`);
    if (!ok) failed++;
  };

  console.log(`fluxion AudioWorklet, in ${browserPath}`);
  // Reaching here at all means the processor registered, the wasm instantiated synchronously on
  // the audio thread, and `process()` ran on the real 128-frame quantum — every one of which is a
  // thing only a browser can answer.
  say(frames === 48_000, "the worklet rendered a full second", `${frames} frames`);
  say(peak > 0.05, "the output is audio, not silence", `peak ${peak.toFixed(3)}`);
  say(
    worst <= 1e-5,
    "the worklet renders what the offline path renders",
    `worst difference ${worst.toExponential(2)}`,
  );
  say(problems.length === 0, "no page or console errors", problems.join("; ") || "clean");
} finally {
  await browser.close();
  server.close();
}

if (failed > 0) {
  console.error(`\n${failed} check(s) failed`);
  process.exit(1);
}
console.log("browser playback OK");
