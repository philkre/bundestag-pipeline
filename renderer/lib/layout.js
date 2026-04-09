// lib/layout.js
import * as d3 from 'd3';

export function partyGroups(nodes) {
  const groups = new Map();
  for (const n of nodes) {
    if (!groups.has(n.party)) groups.set(n.party, []);
    groups.get(n.party).push(n.id);
  }
  return groups;
}

// Simple seeded PRNG (mulberry32)
function makePrng(seed) {
  let s = seed >>> 0;
  return () => {
    s += 0x6d2b79f5;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function computeLayout(nodes, edges, { seed = 42, width = 1200, height = 1200, coalitionParties = new Set() } = {}) {
  const cx = width / 2, cy = height / 2;
  const rand = makePrng(seed);

  // Party centroids on a circle.
  // Coalition parties are grouped together (adjacent slots), then opposition parties,
  // each sub-group sorted by descending size so coalition members sit next to each other.
  const groups = partyGroups(nodes);
  const coalition = [...groups.entries()].filter(([p]) =>  coalitionParties.has(p)).sort((a, b) => b[1].length - a[1].length);
  const opposition = [...groups.entries()].filter(([p]) => !coalitionParties.has(p)).sort((a, b) => b[1].length - a[1].length);
  const sorted = [...coalition, ...opposition];
  const r = Math.min(width, height) * 0.28;
  const centroids = new Map();
  sorted.forEach(([party], i) => {
    const angle = (2 * Math.PI * i) / sorted.length;
    centroids.set(party, { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) });
  });

  // Initialise node positions
  const nodeData = nodes.map(n => {
    const { x: pcx, y: pcy } = centroids.get(n.party);
    const jitter = Math.min(width, height) * 0.10;
    return {
      ...n,
      x: pcx + (rand() * 2 - 1) * jitter,
      y: pcy + (rand() * 2 - 1) * jitter,
    };
  });

  const simulation = d3.forceSimulation(nodeData)
    .force('x', d3.forceX(d => centroids.get(d.party).x).strength(0.4))
    .force('y', d3.forceY(d => centroids.get(d.party).y).strength(0.4))
    .force('collide', d3.forceCollide(12))
    .force('charge', d3.forceManyBody().strength(-20))
    .stop();

  simulation.tick(300);

  // Clamp all nodes inside canvas with padding so blobs don't clip the edge
  const pad = 200;
  for (const n of nodeData) {
    n.x = Math.max(pad, Math.min(width - pad, n.x));
    n.y = Math.max(pad, Math.min(height - pad, n.y));
  }

  return nodeData;
}
