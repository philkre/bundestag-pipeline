// lib/export.js
import puppeteer from 'puppeteer';
import { writeFileSync } from 'node:fs';

export async function renderToPng(svgString, outPath, { width = 1200, height = 1200 } = {}) {
  const html = `<!DOCTYPE html>
<html>
<head><style>
  * { margin: 0; padding: 0; }
  body { background: #0d1117; width: ${width}px; height: ${height}px; overflow: hidden; }
  svg { display: block; }
</style></head>
<body>${svgString}</body>
</html>`;

  const browser = await puppeteer.launch({ headless: true, args: ['--no-sandbox'] });
  const page = await browser.newPage();
  await page.setViewport({ width, height, deviceScaleFactor: 2 });
  await page.setContent(html, { waitUntil: 'networkidle0' });
  const buffer = await page.screenshot({ type: 'png', clip: { x: 0, y: 0, width, height } });
  await browser.close();
  writeFileSync(outPath, buffer);
}
