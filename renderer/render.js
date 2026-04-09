#!/usr/bin/env node
import { parseArgs } from 'node:util';
import { join } from 'node:path';
import { mkdirSync } from 'node:fs';
import { loadNodes, loadEdges, loadPartyColors, loadCoalitions } from './lib/data.js';
import { computeLayout } from './lib/layout.js';
import { glowLayer, coalitionLayer, edgeLayer, nodeLayer, labelLayer } from './lib/layers.js';
import { renderToPng } from './lib/export.js';

const { values: args } = parseArgs({
  options: {
    'out-dir':            { type: 'string', default: 'output/' },
    'party-colors':       { type: 'string', default: 'party_colours.json' },
    'coalitions':         { type: 'string', default: 'coalitions.json' },
    'parliament':         { type: 'string', default: '' },
    'title':              { type: 'string', default: '' },
    'min-weight':         { type: 'string', default: '0.15' },
    'top-edges-per-pair': { type: 'string', default: '40' },
    'seed':               { type: 'string', default: '42' },
  }
});

const outDir       = args['out-dir'];
const minWeight    = parseFloat(args['min-weight']);
const topEdges     = parseInt(args['top-edges-per-pair'], 10);
const seed         = parseInt(args['seed'], 10);
const parliament   = args['parliament'];
const title        = args['title'];
const W = 2400, H = 2400;

// Derive img filename: "Bundestag 2017 - 2021" → "bt_2017_2021.png"
const imgName = parliament
  ? 'bt_' + parliament.replace(/[^0-9]+/g, '_').replace(/^_+|_+$/g, '') + '.png'
  : 'network.png';
const imgDir  = join(outDir, '..', 'img');
mkdirSync(outDir, { recursive: true });
mkdirSync(imgDir, { recursive: true });

console.log(`Loading data from ${outDir}...`);
const nodes        = loadNodes(join(outDir, 'nodes.csv'));
const edges        = loadEdges(join(outDir, 'edges.csv'));
const partyColors  = loadPartyColors(args['party-colors']);
const coalition    = loadCoalitions(args['coalitions'], parliament);

console.log(`Nodes: ${nodes.length}, Edges: ${edges.length}`);
console.log(`Computing layout...`);
const laid = computeLayout(nodes, edges, { seed, width: W, height: H, coalitionParties: coalition });

console.log(`Generating SVG layers...`);
const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
  <rect width="${W}" height="${H}" fill="#0d1117"/>
  ${glowLayer(laid, partyColors)}
  ${coalitionLayer(laid, coalition)}
  ${edgeLayer(laid, edges, partyColors, { minWeight, topEdgesPerPair: topEdges })}
  ${nodeLayer(laid, partyColors)}
  ${labelLayer(laid, partyColors, coalition, title, { width: W, height: H })}
</svg>`;

const outPath = join(imgDir, imgName);
console.log(`Rendering to ${outPath}...`);
await renderToPng(svg, outPath, { width: W, height: H });
console.log(`Done. Wrote ${outPath}`);
