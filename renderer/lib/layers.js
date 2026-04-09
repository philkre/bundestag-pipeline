// lib/layers.js
import { resolveColor } from './data.js';
import * as d3 from 'd3';

// Canonical display names for parties whose data names are all-caps
const DISPLAY_NAMES = {
  'DIE GRÜNEN':            'Die Grünen',
  'BÜNDNIS 90/DIE GRÜNEN': 'Die Grünen',
  'DIE LINKE':             'Die Linke',
  'Die Linke.':            'Die Linke',
  'DIE LINKE.':            'Die Linke',
};
function displayName(party) { return DISPLAY_NAMES[party] ?? party; }

// Helper: bounding box of a set of points
function bbox(points) {
  const xs = points.map(p => p.x), ys = points.map(p => p.y);
  return {
    minX: Math.min(...xs), maxX: Math.max(...xs),
    minY: Math.min(...ys), maxY: Math.max(...ys),
  };
}

// Helper: group nodes by party, return Map<party, node[]>
function byParty(nodes) {
  const m = new Map();
  for (const n of nodes) {
    if (!m.has(n.party)) m.set(n.party, []);
    m.get(n.party).push(n);
  }
  return m;
}

export function glowLayer(nodes, partyColors) {
  const groups = byParty(nodes);
  const filterId = 'glow-blur';
  let defs = `<defs>
    <filter id="${filterId}" x="-60%" y="-60%" width="220%" height="220%">
      <feGaussianBlur stdDeviation="38"/>
    </filter>`;

  let ellipses = '';
  let gi = 0;
  for (const [party, pnodes] of groups) {
    const color = resolveColor(party, partyColors);
    const b = bbox(pnodes);
    const cx = (b.minX + b.maxX) / 2, cy = (b.minY + b.maxY) / 2;
    const rx = (b.maxX - b.minX) / 2 + 32;
    const ry = (b.maxY - b.minY) / 2 + 32;
    const gradId = `glow-grad-${gi++}`;
    defs += `
    <radialGradient id="${gradId}" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="${color}" stop-opacity="0.82"/>
      <stop offset="100%" stop-color="${color}" stop-opacity="0"/>
    </radialGradient>`;
    ellipses += `<ellipse cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" rx="${rx.toFixed(1)}" ry="${ry.toFixed(1)}" fill="url(#${gradId})" filter="url(#${filterId})"/>\n`;
  }
  defs += '\n  </defs>';
  return `${defs}\n<g class="layer-glows">\n${ellipses}</g>`;
}

export function coalitionLayer(nodes, coalitionParties) {
  if (!coalitionParties || coalitionParties.size === 0) return '';

  const pts = nodes
    .filter(n => coalitionParties.has(n.party))
    .map(n => [n.x, n.y]);

  if (pts.length < 3) return '';

  const hull = d3.polygonHull(pts);
  if (!hull) return '';

  // Compute hull centroid
  const hcx = hull.reduce((s, p) => s + p[0], 0) / hull.length;
  const hcy = hull.reduce((s, p) => s + p[1], 0) / hull.length;

  // Expand each vertex outward from centroid by 40px
  const expanded = hull.map(([x, y]) => {
    const dx = x - hcx, dy = y - hcy;
    const len = Math.hypot(dx, dy) || 1;
    return [x + (dx / len) * 80, y + (dy / len) * 80];
  });

  // Smooth with Catmull-Rom closed spline
  const lineGen = d3.line().curve(d3.curveCatmullRomClosed);
  const d = lineGen(expanded);

  return `<g class="layer-coalition">
  <path d="${d}"
    fill="rgba(255,255,255,0.04)"
    stroke="rgba(255,255,255,0.50)"
    stroke-width="3"
    stroke-dasharray="8,5"/>
</g>`;
}
export function edgeLayer(nodes, edges, partyColors, { minWeight = 0.3, topEdgesPerPair = 15 } = {}) {
  const nodeById = new Map(nodes.map(n => [n.id, n]));

  // Group cross-party edges by sorted party pair key
  const pairMap = new Map();
  for (const e of edges) {
    if (e.weight <= 0) continue;          // always exclude negatives per spec
    if (e.weight <= minWeight) continue;  // exclude below threshold
    const src = nodeById.get(e.source), tgt = nodeById.get(e.target);
    if (!src || !tgt) continue;
    if (src.party === tgt.party) continue; // intra-party: skip
    const key = [src.party, tgt.party].sort().join('|||');
    if (!pairMap.has(key)) pairMap.set(key, { partyA: src.party, partyB: tgt.party, edges: [] });
    pairMap.get(key).edges.push({ e, src, tgt });
  }

  // Collect all weights for normalisation
  let maxW = minWeight;
  for (const e of edges) {
    if (e.weight > minWeight && e.weight > maxW) maxW = e.weight;
  }
  if (maxW <= minWeight) maxW = 1;

  const widthScale = (w) => 0.4 + ((w - minWeight) / (maxW - minWeight)) * (2.5 - 0.4);
  const opacityScale = (w) => 0.6 + ((w - minWeight) / (maxW - minWeight)) * (0.25);

  // Party size map for scaling k
  const partySizes = new Map();
  for (const n of nodes) partySizes.set(n.party, (partySizes.get(n.party) ?? 0) + 1);
  let maxSize = 1;
  for (const s of partySizes.values()) if (s > maxSize) maxSize = s;

  let defs = '<defs>';
  let lines = '';
  let edgeIdx = 0;

  for (const { partyA, partyB, edges: pairEdges } of pairMap.values()) {
    const sizeA = partySizes.get(partyA) ?? 1;
    const sizeB = partySizes.get(partyB) ?? 1;
    const k = Math.max(3, Math.ceil(topEdgesPerPair * Math.sqrt(Math.min(sizeA, sizeB) / maxSize)));

    // Diversify: take only the single best edge per source node, so every MP
    // with a strong cross-party connection gets one representative line rather
    // than all lines funnelling through the same high-kappa individual.
    const bySource = new Map();
    for (const item of pairEdges) {
      const sid = item.e.source;
      if (!bySource.has(sid) || item.e.weight > bySource.get(sid).e.weight)
        bySource.set(sid, item);
    }
    const top = [...bySource.values()]
      .sort((a, b) => b.e.weight - a.e.weight)
      .slice(0, k);

    // Per-edge gradient aligned along the actual edge geometry using
    // gradientUnits="userSpaceOnUse" — guarantees each node always receives
    // its own party colour regardless of its position on the canvas.
    for (const { e, src, tgt } of top) {
      const gradId = `eg-${edgeIdx++}`;
      const colorSrc = resolveColor(src.party, partyColors);
      const colorTgt = resolveColor(tgt.party, partyColors);
      defs += `
  <linearGradient id="${gradId}" gradientUnits="userSpaceOnUse" x1="${src.x.toFixed(1)}" y1="${src.y.toFixed(1)}" x2="${tgt.x.toFixed(1)}" y2="${tgt.y.toFixed(1)}">
    <stop offset="0%" stop-color="${colorSrc}" stop-opacity="0.8"/>
    <stop offset="100%" stop-color="${colorTgt}" stop-opacity="0.8"/>
  </linearGradient>`;

      const w = widthScale(e.weight).toFixed(2);
      const op = opacityScale(e.weight).toFixed(2);
      // Quadratic Bézier: control point offset perpendicular to the edge midpoint
      const mx = (src.x + tgt.x) / 2, my = (src.y + tgt.y) / 2;
      const dx = tgt.x - src.x, dy = tgt.y - src.y;
      const len = Math.hypot(dx, dy) || 1;
      const bend = Math.min(len * 0.18, 120); // max 120px arc height
      const qx = (mx - (dy / len) * bend).toFixed(1);
      const qy = (my + (dx / len) * bend).toFixed(1);
      lines += `<path d="M${src.x.toFixed(1)},${src.y.toFixed(1)} Q${qx},${qy} ${tgt.x.toFixed(1)},${tgt.y.toFixed(1)}" fill="none" stroke="url(#${gradId})" stroke-width="${w}" opacity="${op}" stroke-linecap="round"/>\n`;
    }
  }

  defs += '\n</defs>';
  return `${defs}\n<g class="layer-edges">\n${lines}</g>`;
}
export function nodeLayer(nodes, partyColors) {
  const circles = nodes.map(n => {
    const color = resolveColor(n.party, partyColors);
    return `<circle cx="${n.x.toFixed(1)}" cy="${n.y.toFixed(1)}" r="5" fill="${color}"/>`;
  }).join('\n');
  return `<g class="layer-nodes">\n${circles}\n</g>`;
}

export function labelLayer(nodes, partyColors, coalitionParties, title, { width = 2400, height = 2400 } = {}) {
  const cx = width / 2, cy = height / 2;

  // Party centroids and counts
  const groups = new Map();
  for (const n of nodes) {
    if (!groups.has(n.party)) groups.set(n.party, { xs: [], ys: [], count: 0 });
    const g = groups.get(n.party);
    g.xs.push(n.x); g.ys.push(n.y); g.count++;
  }

  let labels = '';
  for (const [party, { xs, ys, count }] of groups) {
    const pcx = xs.reduce((a, b) => a + b, 0) / xs.length;
    const pcy = ys.reduce((a, b) => a + b, 0) / ys.length;
    // Offset outward from canvas centre
    const dx = pcx - cx, dy = pcy - cy;
    const len = Math.hypot(dx, dy) || 1;
    const bboxW = Math.max(...xs) - Math.min(...xs);
    const bboxH = Math.max(...ys) - Math.min(...ys);
    const glowR = Math.max(bboxW / 2, bboxH / 2) * 1.3 + 20;
    const lx = pcx + (dx / len) * glowR * 1.3;
    const ly = pcy + (dy / len) * glowR * 1.3;
    const color = resolveColor(party, partyColors);
    labels += `<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" text-anchor="middle" fill="${color}" font-family="sans-serif" font-size="40" font-weight="bold" opacity="0.9">${displayName(party)}</text>\n`;
    labels += `<text x="${lx.toFixed(1)}" y="${(ly + 36).toFixed(1)}" text-anchor="middle" fill="${color}" font-family="sans-serif" font-size="30" opacity="0.6">(${count} MPs)</text>\n`;
  }

  // Legend (bottom-left)
  const coalitionList = [...groups.keys()].filter(p => coalitionParties.has(p));
  const oppositionList = [...groups.keys()].filter(p => !coalitionParties.has(p));
  const allParties = [...coalitionList, ...oppositionList];
  const legendX = 50, legendStartY = height - 60 - allParties.length * 42 - (coalitionList.length > 0 ? 60 : 0);
  let legend = `<g class="legend">`;
  let ly2 = legendStartY;

  if (coalitionList.length > 0) {
    legend += `<text x="${legendX}" y="${ly2}" fill="rgba(255,255,255,0.45)" font-family="sans-serif" font-size="28" font-style="italic">Koalition</text>\n`;
    ly2 += 42;
    coalitionList.forEach(party => {
      const color = resolveColor(party, partyColors);
      legend += `<rect x="${legendX}" y="${ly2 - 22}" width="26" height="26" rx="4" fill="${color}"/>\n`;
      legend += `<text x="${legendX + 38}" y="${ly2}" fill="rgba(255,255,255,0.85)" font-family="sans-serif" font-size="32">${displayName(party)}</text>\n`;
      ly2 += 42;
    });
    ly2 += 14;
  }

  if (oppositionList.length > 0) {
    legend += `<text x="${legendX}" y="${ly2}" fill="rgba(255,255,255,0.35)" font-family="sans-serif" font-size="28" font-style="italic">Opposition</text>\n`;
    ly2 += 42;
    oppositionList.forEach(party => {
      const color = resolveColor(party, partyColors);
      legend += `<rect x="${legendX}" y="${ly2 - 22}" width="26" height="26" rx="4" fill="${color}"/>\n`;
      legend += `<text x="${legendX + 38}" y="${ly2}" fill="rgba(255,255,255,0.85)" font-family="sans-serif" font-size="32">${displayName(party)}</text>\n`;
      ly2 += 42;
    });
  }

  legend += `</g>`;

  // Title (bottom-right)
  const titleSvg = title
    ? `<text x="${width - 60}" y="${height - 60}" text-anchor="end" fill="rgba(255,255,255,0.7)" font-family="sans-serif" font-size="52">${title}</text>`
    : '';

  return `<g class="layer-labels">\n${labels}${legend}\n${titleSvg}\n</g>`;
}
