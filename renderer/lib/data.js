// lib/data.js
import { readFileSync } from 'node:fs';
import * as d3 from 'd3';

// Uses d3.csvParse to correctly handle quoted fields (e.g. names with commas).
// Column mapping: person_id → id, name → name, party → party
export function loadNodes(nodesPath) {
  const raw = readFileSync(nodesPath, 'utf8');
  return d3.csvParse(raw, row => ({
    id:    row.person_id.trim(),
    name:  row.name.trim(),
    party: row.party.trim(),
  }));
}

// Column mapping: source → source, target → target, weight → weight (float)
export function loadEdges(edgesPath) {
  const raw = readFileSync(edgesPath, 'utf8');
  return d3.csvParse(raw, row => ({
    source: row.source.trim(),
    target: row.target.trim(),
    weight: parseFloat(row.weight),
  }));
}

export function loadPartyColors(colorsPath) {
  return JSON.parse(readFileSync(colorsPath, 'utf8'));
}

export function loadCoalitions(coalitionsPath, parliamentLabel) {
  if (!parliamentLabel) return new Set();
  const data = JSON.parse(readFileSync(coalitionsPath, 'utf8'));
  if (!(parliamentLabel in data)) {
    process.stderr.write(`Warning: parliament "${parliamentLabel}" not found in ${coalitionsPath} — skipping coalition boundary.\n`);
    return new Set();
  }
  return new Set(data[parliamentLabel]);
}

// Resolve party → hex color. Lookup order:
//   1. Exact match in partyColors
//   2. Prefix match (e.g. "CDU/CSU" starts with key "CDU")
//   3. Fallback #888888
export function resolveColor(party, partyColors) {
  if (!partyColors) return '#888888';
  if (party in partyColors) return partyColors[party];
  const match = Object.keys(partyColors).find(k => party.startsWith(k));
  if (match) return partyColors[match];
  return '#888888';
}
