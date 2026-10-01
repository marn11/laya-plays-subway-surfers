/**
 * Runs the oracle policy (oracle.js) against game.html for one or more
 * episodes, using the __getState()/__action()/__restart() bridge -- no
 * vision, no simulated keyboard/touch events.
 *
 * Usage:
 *   npm install
 *   npx playwright install chromium   # one-time browser download
 *   node play_oracle.js                           # watch one run, headed
 *   node play_oracle.js --episodes 20 --headless   # play 20 runs, no window
 *   node play_oracle.js --record data/oracle_run.jsonl  # also log every tick
 */

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { decideAction } from "./oracle.js";

const DIR = path.dirname(fileURLToPath(import.meta.url));
const TICK_MS = 80;

function parseArgs(argv) {
  const args = { episodes: 1, headless: false, record: null, port: 8731 };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--episodes") args.episodes = parseInt(argv[++i], 10);
    else if (argv[i] === "--headless") args.headless = true;
    else if (argv[i] === "--record") args.record = argv[++i];
    else if (argv[i] === "--port") args.port = parseInt(argv[++i], 10);
  }
  return args;
}

function serveDir(dir, port) {
  const mime = { ".html": "text/html", ".js": "text/javascript" };
  const server = http.createServer((req, res) => {
    const filePath = path.join(dir, req.url === "/" ? "/game.html" : req.url);
    fs.readFile(filePath, (err, data) => {
      if (err) {
        res.writeHead(404);
        res.end("not found");
        return;
      }
      res.writeHead(200, { "Content-Type": mime[path.extname(filePath)] || "text/plain" });
      res.end(data);
    });
  });
  return new Promise((resolve) => server.listen(port, () => resolve(server)));
}

async function playOneEpisode(page, recordStream) {
  await page.evaluate(() => window.__restart());

  let ticks = 0;
  let state = await page.evaluate(() => window.__getState());

  while (state.isPlaying) {
    const action = decideAction(state);

    if (recordStream) {
      recordStream.write(JSON.stringify({ state, action }) + "\n");
    }
    if (action !== "none") {
      await page.evaluate((a) => window.__action(a), action);
    }

    await page.waitForTimeout(TICK_MS);
    ticks++;
    state = await page.evaluate(() => window.__getState());
  }

  return { ticks, score: state.score, distanceTraveled: state.distanceTraveled };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const server = await serveDir(DIR, args.port);

  let recordStream = null;
  if (args.record) {
    fs.mkdirSync(path.dirname(args.record), { recursive: true });
    recordStream = fs.createWriteStream(args.record, { flags: "a" });
  }

  const browser = await chromium.launch({ headless: args.headless });
  const page = await browser.newPage();
  await page.goto(`http://localhost:${args.port}/game.html`);

  const results = [];
  for (let ep = 1; ep <= args.episodes; ep++) {
    const result = await playOneEpisode(page, recordStream);
    results.push(result);
    console.log(
      `episode ${ep}/${args.episodes}: ${result.ticks} ticks, score ${result.score}, distance ${result.distanceTraveled.toFixed(1)}`
    );
  }

  const avgScore = results.reduce((s, r) => s + r.score, 0) / results.length;
  const avgDistance = results.reduce((s, r) => s + r.distanceTraveled, 0) / results.length;
  console.log(`\n${results.length} episode(s) -- avg score ${avgScore.toFixed(1)}, avg distance ${avgDistance.toFixed(1)}`);
  if (args.record) console.log(`Logged ticks to ${args.record}`);

  if (recordStream) recordStream.end();
  await browser.close();
  server.close();
}

main();
