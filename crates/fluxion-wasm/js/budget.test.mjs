// The wasm build stays inside its budget (roadmap W7).
//
// Two limits, both hard, because both degrade quietly. A module that grows a megabyte still works
// — it just costs every visitor a second. A block that takes 90% of its deadline still plays —
// until the machine is a little busier and it does not.
//
//   ./scripts/build-wasm.sh && node crates/fluxion-wasm/js/budget.test.mjs
import { gzipSync } from "node:zlib";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import init, { Player, allocations } from "./index.mjs";

const url = (p) => fileURLToPath(new URL(p, import.meta.url));

// --- size -------------------------------------------------------------------------------------

/// What the roadmap sets. Gzipped, because that is what a browser downloads.
const SIZE_BUDGET = 1.5 * 1024 * 1024;

const bytes = await readFile(url("./pkg/fluxion_wasm_bg.wasm"));
const gzipped = gzipSync(bytes, { level: 9 }).length;

// --- speed ------------------------------------------------------------------------------------

const FS = 48_000;
const BLOCK = 128;

/// A 128-frame block at 48 kHz has this long to be produced before the audio thread misses it.
const DEADLINE_MS = (BLOCK / FS) * 1000;

/// The roadmap's limit. The margin exists for everything else on the machine: the browser's own
/// work, another tab, a laptop deciding to throttle.
const SPEED_BUDGET = 0.3;

await init({ module_or_path: bytes });

/// Time one block, over enough blocks that the number means something.
function costOfOneBlock(chain) {
  const player = new Player(chain, FS, BLOCK, BLOCK * 64);

  // Warm up: the first blocks pay for branch prediction and any lazy initialisation.
  for (let i = 0; i < 500; i++) player.render();

  const blocks = 20_000;
  const before = allocations();
  const start = process.hrtime.bigint();
  for (let i = 0; i < blocks; i++) player.render();
  const elapsedMs = Number(process.hrtime.bigint() - start) / 1e6;
  const allocated = allocations() - before;

  return { perBlockMs: elapsedMs / blocks, allocated };
}

// Something a mastering preview would actually run, not a single gain.
const CHAINS = [
  "gain(0.8)",
  "highpass(80, 4) | gain(-3dB)",
  "highpass(30, 2) | lowshelf(120, 3, 0.7) | peaking(1000, -4, 1.5) | highshelf(8000, 2, 0.7) | gain(-1dB)",
];

let failed = 0;
const say = (ok, name, detail) => {
  console.log(`  ${ok ? "ok  " : "FAIL"} ${name}${detail ? ` — ${detail}` : ""}`);
  if (!ok) failed++;
};

console.log("fluxion wasm budget");
say(
  gzipped <= SIZE_BUDGET,
  "module size",
  `${(gzipped / 1024).toFixed(0)} KB gzipped of ${(SIZE_BUDGET / 1024).toFixed(0)} KB ` +
    `(${((gzipped / SIZE_BUDGET) * 100).toFixed(1)}% of budget, ${(bytes.length / 1024).toFixed(0)} KB raw)`,
);

for (const chain of CHAINS) {
  const { perBlockMs, allocated } = costOfOneBlock(chain);
  const fraction = perBlockMs / DEADLINE_MS;
  say(
    fraction <= SPEED_BUDGET,
    `block cost: ${chain.length > 40 ? `${chain.slice(0, 37)}...` : chain}`,
    `${(perBlockMs * 1000).toFixed(1)} us = ${(fraction * 100).toFixed(2)}% of the ` +
      `${DEADLINE_MS.toFixed(2)} ms deadline`,
  );
  say(allocated === 0, "  and allocated nothing while doing it", `${allocated} allocations`);
}

if (failed > 0) {
  console.error(`\n${failed} budget check(s) failed`);
  process.exit(1);
}
console.log("within budget");
