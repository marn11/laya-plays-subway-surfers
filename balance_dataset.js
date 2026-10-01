/**
 * Downsamples the dominant "none" action so the dataset isn't 95% "do
 * nothing" -- training on the raw oracle output as-is would teach a model
 * to mostly predict "none" and look deceptively accurate while being
 * useless at the ticks that actually matter.
 *
 * Choice made here: keep every non-"none" row untouched (no duplication/
 * upsampling of the smallest classes, which would risk overfitting on a
 * handful of exact repeated frames), and randomly downsample "none" rows
 * down to the same total count as all non-"none" rows combined -- a ~50/50
 * split overall, with jump/duck/left/right keeping their natural relative
 * frequency among themselves. This is one defensible ratio, not the only
 * correct one -- revisit if the fine-tuned result is still "none"-biased.
 *
 * Usage: node balance_dataset.js data/oracle_dataset.jsonl data/oracle_dataset_balanced.jsonl
 */

import fs from "node:fs";

const SEED = 20261002;

function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle(arr, rng) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function main() {
  const [inputPath, outputPath] = process.argv.slice(2);
  if (!inputPath || !outputPath) {
    console.error("Usage: node balance_dataset.js <input.jsonl> <output.jsonl>");
    process.exit(1);
  }

  const lines = fs.readFileSync(inputPath, "utf8").split("\n").filter(Boolean);
  const rows = lines.map((l) => JSON.parse(l));

  const noneRows = rows.filter((r) => r.action === "none");
  const actionRows = rows.filter((r) => r.action !== "none");

  const rng = mulberry32(SEED);
  const noneSampled = shuffle([...noneRows], rng).slice(0, actionRows.length);

  const balanced = shuffle([...noneSampled, ...actionRows], rng);

  fs.writeFileSync(outputPath, balanced.map((r) => JSON.stringify(r)).join("\n") + "\n");

  const counts = {};
  for (const r of balanced) counts[r.action] = (counts[r.action] || 0) + 1;
  console.log(`${rows.length} rows -> ${balanced.length} rows (${noneRows.length} "none" downsampled to ${noneSampled.length})`);
  console.log("New distribution:", counts);
  console.log(`Wrote ${outputPath}`);
}

main();
