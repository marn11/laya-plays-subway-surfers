/**
 * Runs the oracle policy (oracle.js) against game.html for one or more
 * episodes, using the __getState()/__action()/__restart() bridge -- no
 * vision, no simulated keyboard/touch events.
 *
 * The whole episode loop (read state, decide, act, repeat until game over)
 * runs INSIDE the browser via a single page.evaluate() call per episode --
 * not one Node<->browser round trip per tick. That round trip costs ~60-70ms
 * regardless of tickMs, which made per-tick evaluate() calls the real
 * bottleneck, not the game speed. oracle.js's source is read and injected
 * into the page so there's still one copy of the decision logic, not two.
 *
 * In headless mode, window.__fastMode also skips game.html's renderer.render()
 * call entirely (physics/collisions still run every frame) -- measured 3.6x
 * faster (26.7 -> 97.6 ticks/sec), CPU 612% -> 40%. Rendering a frame no one
 * will ever see was the real bottleneck, not game logic or round trips.
 *
 * Usage:
 *   npm install
 *   npx playwright install chromium   # one-time browser download
 *   node play_oracle.js --tick-ms 80            # watch one run, headed, human-followable pace
 *   node play_oracle.js --headless --max-ticks 50000 --record data/oracle_dataset.jsonl
 *     # fast, predictable-duration data-collection run, no window, logs every tick
 *     # (episode length varies too much for --episodes alone to give a
 *     # predictable run time -- use --max-ticks instead for a real collection run)
 */

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const DIR = path.dirname(fileURLToPath(import.meta.url));
const ORACLE_SRC = fs.readFileSync(path.join(DIR, "oracle.js"), "utf8").replace("export function", "function");

function parseArgs(argv) {
  // The game's own physics run on real elapsed time (requestAnimationFrame
  // delta), not on how often the oracle polls it -- so tickMs only controls
  // how fast data gets collected, not game correctness. Default is fast
  // (headless data collection); pass --tick-ms 80 or so to watch it at a
  // human-followable pace in headed mode.
  // maxTicks (if set) stops collection once the total tick budget is hit,
  // finishing the episode in progress first -- episode length varies too
  // much (485 to 5065 ticks seen in one 10-episode sample) for --episodes
  // alone to give a predictable run time. If --max-ticks is set without
  // --episodes, episodes becomes an effectively-unbounded safety cap.
  const args = { episodes: 1, episodesExplicit: false, headless: false, record: null, port: 8731, tickMs: 10, maxTicks: null };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--episodes") {
      args.episodes = parseInt(argv[++i], 10);
      args.episodesExplicit = true;
    } else if (argv[i] === "--headless") args.headless = true;
    else if (argv[i] === "--record") args.record = argv[++i];
    else if (argv[i] === "--port") args.port = parseInt(argv[++i], 10);
    else if (argv[i] === "--tick-ms") args.tickMs = parseInt(argv[++i], 10);
    else if (argv[i] === "--max-ticks") args.maxTicks = parseInt(argv[++i], 10);
  }
  if (args.maxTicks && !args.episodesExplicit) args.episodes = Infinity;
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

async function playOneEpisode(page, tickMs, fastMode) {
  return page.evaluate(
    ({ oracleSrc, tickMs, fastMode }) => {
      // eslint-disable-next-line no-eval -- injecting oracle.js's own source, not arbitrary input
      eval(oracleSrc); // defines decideAction(state) in this closure

      window.__fastMode = fastMode; // skip renderer.render() -- no one's watching, physics/collisions still run every frame

      return new Promise((resolve) => {
        window.__restart();
        const log = [];
        const timer = setInterval(() => {
          const state = window.__getState();
          if (!state.isPlaying) {
            clearInterval(timer);
            resolve({ log, finalState: state });
            return;
          }
          const action = decideAction(state);
          log.push({ state, action });
          if (action !== "none") window.__action(action);
        }, tickMs);
      });
    },
    { oracleSrc: ORACLE_SRC, tickMs, fastMode }
  );
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
  let totalTicks = 0;
  for (let ep = 1; ep <= args.episodes; ep++) {
    const { log, finalState } = await playOneEpisode(page, args.tickMs, args.headless);

    if (recordStream) {
      for (const entry of log) recordStream.write(JSON.stringify(entry) + "\n");
    }

    totalTicks += log.length;
    const result = { ticks: log.length, score: finalState.score, distanceTraveled: finalState.distanceTraveled };
    results.push(result);
    console.log(
      `episode ${ep}: ${result.ticks} ticks (${totalTicks} total), score ${result.score}, distance ${result.distanceTraveled.toFixed(1)}`
    );

    if (args.maxTicks && totalTicks >= args.maxTicks) break;
  }

  const avgScore = results.reduce((s, r) => s + r.score, 0) / results.length;
  const avgDistance = results.reduce((s, r) => s + r.distanceTraveled, 0) / results.length;
  console.log(
    `\n${results.length} episode(s), ${totalTicks} total ticks -- avg score ${avgScore.toFixed(1)}, avg distance ${avgDistance.toFixed(1)}`
  );
  if (args.record) console.log(`Logged ticks to ${args.record}`);

  if (recordStream) recordStream.end();
  await browser.close();
  server.close();
}

main();
