/**
 * Headless Chrome CDP screenshot helper.
 * Usage: node scripts/take_screenshots.mjs
 */
import { spawn, execSync } from "child_process";
import http from "http";
import { createRequire } from "module";
import fs from "fs";
import path from "path";

const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const DEBUG_PORT = 9223;
const OUT_DIR = path.resolve("docs/assets");

function cdpGet(path) {
  return new Promise((resolve, reject) => {
    http.get({ hostname: "localhost", port: DEBUG_PORT, path }, (res) => {
      let data = "";
      res.on("data", (c) => (data += c));
      res.on("end", () => resolve(JSON.parse(data)));
    }).on("error", reject);
  });
}

// Minimal CDP over raw HTTP via /json/new + /json/activate + fetch for commands
// Use simpler approach: just launch chrome --screenshot for each scenario
async function shot(url, outFile, width = 1440, height = 900, delay = 1500) {
  return new Promise((resolve, reject) => {
    const args = [
      "--headless=new",
      `--screenshot=${outFile}`,
      `--window-size=${width},${height}`,
      "--hide-scrollbars",
      "--no-sandbox",
      "--force-device-scale-factor=1",
    ];
    // For virtual time, use --run-all-compositor-stages-before-draw
    const proc = spawn(CHROME, [...args, url], { stdio: "pipe" });
    let out = "";
    proc.stderr.on("data", (d) => (out += d));
    proc.on("close", (code) => {
      if (fs.existsSync(outFile)) resolve(outFile);
      else reject(new Error(`Screenshot failed: ${out}`));
    });
    setTimeout(() => {}, delay); // let it run
  });
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });

  // 1. Homepage (cold start)
  console.log("📸 Homepage...");
  await shot("http://localhost:5173", path.join(OUT_DIR, "screenshot_home.png"), 1440, 900);
  console.log("   ✓ screenshot_home.png");

  // 2. Narrow / tablet view showing sidebar + hero
  console.log("📸 Tablet (sidebar visible)...");
  await shot("http://localhost:5173", path.join(OUT_DIR, "screenshot_tablet.png"), 900, 700);
  console.log("   ✓ screenshot_tablet.png");

  // 3. Mobile view
  console.log("📸 Mobile...");
  await shot("http://localhost:5173", path.join(OUT_DIR, "screenshot_mobile.png"), 390, 844);
  console.log("   ✓ screenshot_mobile.png");

  console.log("\nDone. Files in docs/assets/");
}

main().catch((e) => { console.error(e); process.exit(1); });
