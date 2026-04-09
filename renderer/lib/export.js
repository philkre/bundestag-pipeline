// lib/export.js
import puppeteer from 'puppeteer';
import { writeFileSync, unlinkSync, existsSync } from 'node:fs';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const execFileAsync = promisify(execFile);

// Try rsvg-convert first (fast, handles large SVGs); fall back to puppeteer.
async function tryRsvg(svgString, outPath, width, height) {
  const tmpSvg = join(tmpdir(), `render_${Date.now()}.svg`);
  writeFileSync(tmpSvg, svgString, 'utf8');
  try {
    await execFileAsync('/opt/homebrew/bin/rsvg-convert', [
      '-w', String(width), '-h', String(height), '-o', outPath, tmpSvg,
    ]);
    return true;
  } catch { return false; }
  finally { if (existsSync(tmpSvg)) unlinkSync(tmpSvg); }
}

export async function renderToPng(svgString, outPath, { width = 1200, height = 1200 } = {}) {
  if (await tryRsvg(svgString, outPath, width, height)) return;

  // Fallback: puppeteer
  const html = `<!DOCTYPE html>
<html><head><style>
  * { margin:0; padding:0; }
  body { background:#0d1117; width:${width}px; height:${height}px; overflow:hidden; }
  svg { display:block; }
</style></head>
<body>${svgString}</body></html>`;

  const browser = await puppeteer.launch({ headless: true, args: ['--no-sandbox'], protocolTimeout: 300_000 });
  const page = await browser.newPage();
  await page.setViewport({ width, height, deviceScaleFactor: 2 });
  await page.setContent(html, { waitUntil: 'networkidle0' });
  const buffer = await page.screenshot({ type: 'png', clip: { x: 0, y: 0, width, height } });
  await browser.close();
  writeFileSync(outPath, buffer);
}
