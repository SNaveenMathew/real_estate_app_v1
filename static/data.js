'use strict';
/* Data page: the catalog map (left) and the dataset workbench (right).
   Vanilla JS, no dependencies. Everything that comes from uploaded data is rendered with
   textContent / DOM nodes (never innerHTML): column names and cell values are untrusted. */

/* ---------------------------------------------------------------- tiny DOM helpers */
const $ = (sel, root = document) => root.querySelector(sel);
const SVGNS = 'http://www.w3.org/2000/svg';

function append(el, kids) {
  for (const k of kids.flat(Infinity)) {
    if (k == null || k === false) continue;
    el.append(k instanceof Node ? k : document.createTextNode(String(k)));
  }
}
function h(tag, props, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false || k === 'value') continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (k === 'checked' || k === 'disabled' || k === 'hidden' || k === 'selected') el[k] = !!v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  append(el, kids);
  if (props && props.value != null) el.value = props.value;      // after children so <select> options exist
  return el;
}
function s(tag, attrs, ...kids) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null) el.setAttribute(k, v);
  append(el, kids);
  return el;
}
const fmt = n => (n == null ? '–' : Number(n).toLocaleString('en-US'));
const pct = n => (n == null ? '–' : `${Number(n) % 1 ? Number(n).toFixed(1) : Number(n)}%`);
const str = v => (v == null || v === '' ? '—' : String(v));
const plural = (n, one, many) => `${fmt(n)} ${n === 1 ? one : (many || one + 's')}`;

/* ---------------------------------------------------------------- API */
async function api(path, opts = {}) {
  const res = await fetch('/api/onboarding' + path, opts);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { const j = await res.json(); if (j && j.detail) msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch (_) { /* keep default */ }
    throw new Error(msg);
  }
  return res.json();
}
const JSON_HDR = { 'Content-Type': 'application/json' };
const post = (p, body) => api(p, { method: 'POST', headers: JSON_HDR, body: JSON.stringify(body || {}) });
const put = (p, body) => api(p, { method: 'PUT', headers: JSON_HDR, body: JSON.stringify(body || {}) });

/* ---------------------------------------------------------------- state */
const LANES = ['housing', 'geography', 'risk', 'sales', 'safety', 'mobility', 'environment', 'demographics', 'other', 'system'];
const DOMAIN_COLOR = { housing: '#3d7be8', geography: '#0f9d8a', risk: '#e0742f', sales: '#b08a1e', safety: '#c2415c',
  mobility: '#5a67d8', environment: '#3f9a5b', demographics: '#8d6e63', other: '#7b8598', system: '#9aa5b8' };
const ROLE_LABEL = { key: 'Key (used to join)', label: 'Label (names things)', measure: 'Measure (number to average or rank)',
  dimension: 'Category', date: 'Date', geo: 'Coordinates or geometry', other: 'Other' };

const S = {
  catalog: null, meta: { formats: [], domains: [], roles: [], max_upload_mb: 250 }, datasets: [],
  ds: null, form: null, tab: 'describe', busy: null, sel: null, llm: null, selftest: null,
  colRole: 'all', colQuery: '', justAdded: new Set(), edits: {}, lastEnrich: null,
  filter: { q: '', hidden: new Set(), allCols: false, system: false }, upload: { sheet: '', skiprows: '' },
};

function toast(msg, kind = '') {
  const el = h('div', { class: 'toast ' + kind }, msg);
  $('#toasts').append(el);
  setTimeout(() => el.classList.add('out'), 4200);
  setTimeout(() => el.remove(), 4800);
}

async function withBusy(label, fn) {
  S.busy = label; renderPanel();
  try { return await fn(); }
  catch (e) { toast(e.message || String(e), 'err'); }
  finally { S.busy = null; renderPanel(); }
}

/* ---------------------------------------------------------------- loading */
async function loadDatasets() {
  const d = await api('/datasets');
  S.datasets = d.datasets; S.meta = d;
}
async function loadCatalog() {
  const q = S.ds ? `?dataset_id=${encodeURIComponent(S.ds.dataset_id)}` : '';
  S.catalog = await api('/catalog' + q);
  renderMap();
}

/* ================================================================ CATALOG MAP */
const NW = 236, RH = 19, HH = 46, LANE_GAP = 96, NODE_GAP = 26, TOP = 58, LEFT = 28;
const SQL_FUNCS = /^(LEFT|RIGHT|LPAD|UPPER|LOWER|TRIM|SUBSTR|SUBSTRING|CAST|AS|VARCHAR)$/i;
const identsOf = expr => (String(expr || '').match(/[A-Za-z_][A-Za-z0-9_]*/g) || []).filter(x => !SQL_FUNCS.test(x));
const shortType = t => { const u = String(t || '').toUpperCase(); return u.startsWith('VARCHAR') ? 'text' : /INT/.test(u) ? 'int' :
  /DOUBLE|FLOAT|DECIMAL|REAL/.test(u) ? 'num' : /TIMESTAMP|DATE/.test(u) ? 'date' : /BOOL/.test(u) ? 'bool' : u.toLowerCase().slice(0, 6); };
const clip = (t, n) => (String(t).length > n ? String(t).slice(0, n - 1) + '…' : String(t));

function graphModel() {
  const cat = S.catalog, draft = cat.draft, q = S.filter.q.trim().toLowerCase();
  let tables = cat.tables.filter(t => t.agent_visible || S.filter.system).map(t => ({ ...t, draft: false }));
  if (draft && draft.status === 'draft') {
    const t = draft.table;
    tables.push({ name: t.name, description: t.description, grain: t.grain, domain: t.domain || 'other', origin: 'upload', rows: t.rows,
      columns: t.columns.map(c => ({ name: c.name, type: c.type, note: '', role: c.role })), concepts: 0, draft: true, agent_visible: true });
  }
  const names = new Set(tables.map(t => t.name));
  let rels = cat.relationships.filter(r => names.has(r.left_table) && names.has(r.right_table)).map(r => ({ ...r, pending: false }));
  if (draft) for (const r of draft.relationships) {
    if (r.status === 'pending' && names.has(r.left_table) && names.has(r.right_table))
      rels.push({ key: 'proposal:' + r.proposal_id, proposal_id: r.proposal_id, left_table: r.left_table, left_expr: r.left_expr,
        right_table: r.right_table, right_expr: r.right_expr, cardinality: r.cardinality, confidence: r.confidence,
        origin: 'upload', preferred: true, pending: true });
  }
  tables = tables.filter(t => !S.filter.hidden.has(t.domain || 'other'));
  const vis = new Set(tables.map(t => t.name));
  rels = rels.filter(r => vis.has(r.left_table) && vis.has(r.right_table));
  for (const t of tables) {
    t.match = !q || t.name.toLowerCase().includes(q) || (t.description || '').toLowerCase().includes(q) ||
      t.columns.some(c => c.name.toLowerCase().includes(q));
  }
  return { tables, rels };
}

function visibleColumns(t, rels) {
  const join = new Set();
  for (const r of rels) {
    if (r.left_table === t.name) identsOf(r.left_expr).forEach(x => join.add(x));
    if (r.right_table === t.name) identsOf(r.right_expr).forEach(x => join.add(x));
  }
  const cols = t.columns;
  if (S.filter.allCols) return { shown: cols.slice(0, 60), more: Math.max(0, cols.length - 60), join };
  const shown = cols.filter(c => join.has(c.name));
  for (const c of cols) { if (shown.length >= 8) break; if (!shown.includes(c)) shown.push(c); }
  shown.sort((a, b) => cols.indexOf(a) - cols.indexOf(b));
  return { shown, more: cols.length - shown.length, join };
}

/* Order the area lanes so linked tables sit close together (links under review count double). The
   default order is kept unless a move strictly shortens the total link distance; "system" stays last. */
function orderLanes(tables, rels) {
  const present = LANES.filter(l => tables.some(t => (t.domain || 'other') === l));
  const movable = present.filter(l => l !== 'system'), tail = present.filter(l => l === 'system');
  const laneOf = {}; tables.forEach(t => { laneOf[t.name] = t.domain || 'other'; });
  const pairs = {};
  rels.forEach(r => {
    const a = laneOf[r.left_table], b = laneOf[r.right_table];
    if (a && b && a !== b) { const k = [a, b].sort().join('|'); pairs[k] = (pairs[k] || 0) + (r.pending || r.origin !== 'builtin' ? 2 : 1); }
  });
  const cost = order => Object.entries(pairs).reduce((sum, [k, n]) => { const [a, b] = k.split('|'); return sum + n * Math.abs(order.indexOf(a) - order.indexOf(b)); }, 0);
  let best = movable.slice(), bestCost = cost(best);
  for (let iter = 0; iter < 20; iter++) {
    let improved = false;
    for (const lane of movable) {
      for (let pos = 0; pos < movable.length; pos++) {
        const trial = best.filter(l => l !== lane); trial.splice(pos, 0, lane);
        const c = cost(trial);
        if (c < bestCost) { best = trial; bestCost = c; improved = true; }
      }
    }
    if (!improved) break;
  }
  return best.concat(tail);
}

function layout(tables, rels) {
  const lanes = orderLanes(tables, rels);
  const deg = {};
  rels.forEach(r => { deg[r.left_table] = (deg[r.left_table] || 0) + 1; deg[r.right_table] = (deg[r.right_table] || 0) + 1; });
  const nodes = new Map();
  lanes.forEach((lane, li) => {
    const inLane = tables.filter(t => (t.domain || 'other') === lane).sort((a, b) =>
      (b.draft - a.draft) || ((b.origin !== 'builtin') - (a.origin !== 'builtin')) || ((deg[b.name] || 0) - (deg[a.name] || 0)) || a.name.localeCompare(b.name));
    let y = TOP;
    for (const t of inLane) {
      const vc = visibleColumns(t, rels);
      const height = HH + (vc.shown.length + (vc.more ? 1 : 0)) * RH + 12;
      nodes.set(t.name, { t, lane, li, x: LEFT + li * (NW + LANE_GAP), y, h: height, vc });
      y += height + NODE_GAP;
    }
  });
  const width = LEFT * 2 + Math.max(lanes.length, 1) * NW + Math.max(0, lanes.length - 1) * LANE_GAP + 40;
  const height = Math.max(240, ...[...nodes.values()].map(n => n.y + n.h + 30));
  return { nodes, lanes, width, height };
}

const anchorY = (n, expr) => {
  for (const id of identsOf(expr)) {
    const i = n.vc.shown.findIndex(c => c.name === id);
    if (i >= 0) return n.y + HH + i * RH + RH / 2;
  }
  return n.y + HH / 2;
};
const ends = card => { const m = String(card || '').split('-to-'); return [m[0] === 'many' ? 'many' : m[0] === 'one' ? 'one' : '', m[1] === 'many' ? 'many' : m[1] === 'one' ? 'one' : '']; };

function glyph(x, y, d, kind, cls) {
  if (!kind) return null;
  const dd = kind === 'many'
    ? `M${x + d * 14},${y} L${x},${y - 6} M${x + d * 14},${y} L${x},${y} M${x + d * 14},${y} L${x},${y + 6}`
    : `M${x + d * 10},${y - 6} L${x + d * 10},${y + 6}`;
  return s('path', { class: 'edge-glyph ' + cls, d: dd, fill: 'none', stroke: '#7d889d', 'stroke-width': 1.4 });
}

function edgeClass(r) { return 'edge' + (r.pending ? ' is-pending' : r.origin !== 'builtin' ? ' is-mine' : '') + (r.preferred === false ? ' is-soft' : ''); }

function drawEdge(r, A, B, k) {
  const ya = anchorY(A, r.left_expr), yb = anchorY(B, r.right_expr);
  let x1, x2, d1, d2, c1, c2;
  if (A.li < B.li) { x1 = A.x + NW; x2 = B.x; d1 = 1; d2 = -1; const dx = Math.max(50, (x2 - x1) * 0.45); c1 = [x1 + dx, ya]; c2 = [x2 - dx, yb]; }
  else if (A.li > B.li) { x1 = A.x; x2 = B.x + NW; d1 = -1; d2 = 1; const dx = Math.max(50, (x1 - x2) * 0.45); c1 = [x1 - dx, ya]; c2 = [x2 + dx, yb]; }
  else { x1 = A.x + NW; x2 = B.x + NW; d1 = 1; d2 = 1; const b = 46 + 14 * (k % 4); c1 = [x1 + b, ya]; c2 = [x2 + b, yb]; }
  const far = Math.abs(A.li - B.li) > 1;      // passes behind other lanes: keep it quiet until selected
  const cls = edgeClass(r) + (far ? ' is-far' : ''), gc = r.pending ? 'is-pending' : r.origin !== 'builtin' ? 'is-mine' : '';
  const [ka, kb] = ends(r.cardinality);
  const stroke = r.pending ? '#d99a00' : r.origin !== 'builtin' ? '#4f8ef7' : '#9aa5b8';
  const active = S.sel && S.sel.type === 'rel' && S.sel.key === r.key;
  const g = s('g', { 'data-a': r.left_table, 'data-b': r.right_table });
  const path = s('path', { class: cls + (active ? ' is-active' : ''), d: `M${x1},${ya} C${c1[0]},${c1[1]} ${c2[0]},${c2[1]} ${x2},${yb}`, fill: 'none',
    stroke, 'stroke-width': active ? 3 : 1.5, 'stroke-opacity': far && !active ? 0.55 : 1, tabindex: 0, role: 'button',
    'aria-label': `Link ${r.left_table}.${r.left_expr} to ${r.right_table}.${r.right_expr}, ${r.cardinality || 'unspecified'}${r.pending ? ', awaiting your review' : ''}` },
    s('title', {}, `${r.left_table}.${r.left_expr} = ${r.right_table}.${r.right_expr} (${r.cardinality || '?'})`));
  const pick = () => select({ type: 'rel', key: r.key, rel: r });
  path.addEventListener('click', pick);
  path.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
  g.append(path, glyph(x1, ya, d1, ka, gc) || '', glyph(x2, yb, d2, kb, gc) || '');
  return g;
}

function drawNode(N) {
  const t = N.t, color = DOMAIN_COLOR[t.domain || 'other'] || '#7b8598';
  const selected = S.sel && S.sel.type === 'table' && S.sel.name === t.name;
  const mine = t.origin !== 'builtin';
  const g = s('g', { class: 'node' + (t.draft ? ' is-draft' : '') + (selected ? ' is-selected' : '') + (S.justAdded.has(t.name) ? ' just-added' : '') + (t.match ? '' : ' is-dim'),
    transform: `translate(${N.x},${N.y})`, tabindex: 0, role: 'button', 'data-table': t.name,
    'aria-label': `Table ${t.name}, ${t.columns.length} columns, ${t.draft ? 'under review' : mine ? 'added by you' : 'built in'}` });
  g.append(s('title', {}, `${t.name}\n${t.description || ''}${t.grain ? '\nGrain: ' + t.grain : ''}`));
  g.append(s('rect', { class: 'node-box', width: NW, height: N.h, rx: 10, fill: '#fff', stroke: t.draft ? '#d99a00' : '#ccd3e0', 'stroke-width': 1 }));
  g.append(s('rect', { width: 5, height: N.h, rx: 2.5, fill: color }));
  g.append(s('text', { class: 't-name', x: 16, y: 21, 'font-size': 12.5, 'font-weight': 600, fill: '#1a1a2e', 'font-family': 'ui-monospace,Menlo,Consolas,monospace' }, clip(t.name, 25)));
  g.append(s('text', { class: 't-meta', x: 16, y: 37, 'font-size': 11, fill: '#7b8598' },
    (t.draft || mine) ? `${fmt(t.rows)} rows` : `${fmt(t.rows)} rows · ${plural(t.columns.length, 'column')}`));
  // badges live on the meta line so a long table name never collides with them
  if (t.draft) g.append(s('text', { class: 't-badge review', x: NW - 10, y: 37, 'text-anchor': 'end', 'font-size': 10, 'font-weight': 700, fill: '#9a6a00' }, 'UNDER REVIEW'));
  else if (mine) g.append(s('text', { class: 't-badge', x: NW - 10, y: 37, 'text-anchor': 'end', 'font-size': 10, 'font-weight': 700, fill: '#6d4fc4' }, 'ADDED BY YOU'));
  g.append(s('line', { x1: 10, x2: NW - 10, y1: HH - 6, y2: HH - 6, stroke: '#edf0f5' }));
  N.vc.shown.forEach((c, i) => {
    const y = HH + i * RH + RH / 2 + 4, join = N.vc.join.has(c.name);
    if (join) g.append(s('circle', { cx: 10, cy: y - 4, r: 2.6, fill: color }));
    g.append(s('text', { class: 't-col' + (join ? ' is-join' : ''), x: 18, y, 'font-size': 11.5, 'font-weight': join ? 700 : 400, fill: '#3a4256', 'font-family': 'ui-monospace,Menlo,Consolas,monospace' }, clip(c.name, 27)));
    g.append(s('text', { class: 't-type', x: NW - 10, y, 'text-anchor': 'end', 'font-size': 10.5, fill: '#9aa3b5' }, shortType(c.type)));
  });
  if (N.vc.more) g.append(s('text', { class: 't-meta', x: 18, y: HH + N.vc.shown.length * RH + RH / 2 + 4, 'font-size': 11, fill: '#7b8598' }, `+ ${N.vc.more} more columns`));
  const pick = () => select({ type: 'table', name: t.name });
  g.addEventListener('click', pick);
  g.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
  return g;
}

function renderMap() {
  if (!S.catalog) return;
  const cat = S.catalog;
  const mine = cat.tables.filter(t => t.origin !== 'builtin').length;
  $('#map-summary').textContent = `${plural(cat.tables.filter(t => t.agent_visible).length, 'table')} · ${plural(cat.relationships.length, 'link')}` +
    (mine ? ` · ${mine} added by you` : '') + (cat.draft && cat.draft.status === 'draft' ? ' · 1 under review' : '');
  renderDomainFilters(cat);
  const { tables, rels } = graphModel();
  const L = layout(tables, rels);
  const svg = s('svg', { width: L.width, height: L.height, viewBox: `0 0 ${L.width} ${L.height}`, role: 'group', 'aria-label': 'Data model map' });
  L.lanes.forEach((lane, li) => {
    const label = (cat.domains.find(d => d.key === lane) || { label: lane }).label;
    svg.append(s('text', { class: 'lane-title', x: LEFT + li * (NW + LANE_GAP), y: 30, 'font-size': 12, 'font-weight': 600, fill: '#7b8598' }, label));
    svg.append(s('rect', { x: LEFT + li * (NW + LANE_GAP), y: 38, width: 28, height: 3, rx: 1.5, fill: DOMAIN_COLOR[lane] || '#7b8598' }));
  });
  const edges = s('g', { class: 'edges' });
  const sameLane = {};
  rels.forEach((r, k) => {
    const A = L.nodes.get(r.left_table), B = L.nodes.get(r.right_table);
    if (!A || !B) return;
    const key = A.li === B.li ? A.li : -1;
    sameLane[key] = (sameLane[key] || 0) + 1;
    edges.append(drawEdge(r, A, B, sameLane[key]));
  });
  svg.append(edges);
  for (const N of L.nodes.values()) svg.append(drawNode(N));
  $('#canvas').replaceChildren(L.nodes.size ? svg : h('p', { class: 'muted pad' }, 'Nothing to show with the current filters.'));
  applyEmphasis();
  renderInspector();
}

function applyEmphasis() {
  const svg = $('#canvas svg'); if (!svg || !S.sel) return;
  let keep = null;
  if (S.sel.type === 'table') {
    keep = new Set([S.sel.name]);
    svg.querySelectorAll('.edges g').forEach(g => { if (g.dataset.a === S.sel.name) keep.add(g.dataset.b); if (g.dataset.b === S.sel.name) keep.add(g.dataset.a); });
    svg.querySelectorAll('.node').forEach(n => { if (!keep.has(n.dataset.table)) n.classList.add('is-soft'); });
    svg.querySelectorAll('.edges g').forEach(g => { if (g.dataset.a !== S.sel.name && g.dataset.b !== S.sel.name) g.firstChild.classList.add('is-dim'); });
  } else if (S.sel.type === 'rel') {
    const r = S.sel.rel;
    svg.querySelectorAll('.node').forEach(n => { if (n.dataset.table !== r.left_table && n.dataset.table !== r.right_table) n.classList.add('is-soft'); });
    svg.querySelectorAll('.edges g').forEach(g => { if (!(g.dataset.a === r.left_table && g.dataset.b === r.right_table)) g.firstChild.classList.add('is-dim'); });
  }
}

function renderDomainFilters(cat) {
  const present = LANES.filter(l => cat.tables.some(t => (t.domain || 'other') === l && (t.agent_visible || S.filter.system)));
  $('#domain-filters').replaceChildren(...present.map(l => {
    const on = !S.filter.hidden.has(l);
    return h('button', { type: 'button', 'aria-pressed': String(on), onclick: () => { on ? S.filter.hidden.add(l) : S.filter.hidden.delete(l); renderMap(); } },
      h('span', { class: 'dot', style: { background: DOMAIN_COLOR[l] } }), (cat.domains.find(d => d.key === l) || { label: l }).label);
  }));
}

function renderLegend() {
  const line = (cls, stroke, dash) => { const v = s('svg', { width: 34, height: 12, viewBox: '0 0 34 12' });
    v.append(s('path', { d: 'M1,6 L33,6', stroke, 'stroke-width': 2, 'stroke-dasharray': dash || null, fill: 'none' })); return v; };
  const gl = (many) => { const v = s('svg', { width: 26, height: 14, viewBox: '0 0 26 14' });
    v.append(s('path', { d: many ? 'M2,7 L14,1 M2,7 L14,7 M2,7 L14,13' : 'M8,1 L8,13', stroke: '#7d889d', 'stroke-width': 1.4, fill: 'none' }));
    v.append(s('path', { d: 'M14,7 L26,7', stroke: '#9aa5b8', 'stroke-width': 1.5 })); return v; };
  $('#legend').replaceChildren(
    h('span', {}, line('', '#9aa5b8'), 'Built-in link'), h('span', {}, line('', '#4f8ef7'), 'Added by you'),
    h('span', {}, line('', '#d99a00', '6 4'), 'Awaiting your review'), h('span', {}, line('', '#9aa5b8', '2 4'), 'Not preferred by the agent'),
    h('span', {}, gl(true), 'many'), h('span', {}, gl(false), 'one'));
}

function select(sel) { S.sel = sel; renderMap(); }

/* ---------------------------------------------------------------- inspector */
function renderInspector() {
  const el = $('#inspector'), sel = S.sel;
  if (!sel || !S.catalog) { el.hidden = true; return; }
  const close = h('button', { class: 'btn small close', type: 'button', 'aria-label': 'Close details', onclick: () => { S.sel = null; renderMap(); } }, 'Close');
  if (sel.type === 'table') {
    const t = S.catalog.tables.find(x => x.name === sel.name) || graphModel().tables.find(x => x.name === sel.name);
    if (!t) { el.hidden = true; return; }
    const rels = [...S.catalog.relationships, ...(S.catalog.draft ? S.catalog.draft.relationships.filter(r => r.status === 'pending').map(r => ({ ...r, key: 'proposal:' + r.proposal_id, pending: true, origin: 'upload' })) : [])]
      .filter(r => r.left_table === t.name || r.right_table === t.name);
    const concepts = S.catalog.concepts.filter(c => c.tables.includes(t.name));
    el.hidden = false;
    el.replaceChildren(close,
      h('div', { class: 'row wrap' }, h('h3', {}, t.name), t.origin === 'builtin' ? h('span', { class: 'pill' }, 'built in') : h('span', { class: 'pill mine' }, 'added by you'),
        t.draft ? h('span', { class: 'pill warn' }, 'under review') : null,
        h('span', { class: 'pill', style: { background: DOMAIN_COLOR[t.domain || 'other'] + '22', color: '#333' } }, (S.catalog.domains.find(d => d.key === t.domain) || { label: t.domain }).label)),
      h('p', { style: { marginTop: '6px' } }, t.description || 'No description.'),
      h('dl', { class: 'kv' },
        h('dt', {}, 'Rows'), h('dd', {}, fmt(t.rows)), h('dt', {}, 'Grain'), h('dd', {}, str(t.grain)),
        h('dt', {}, 'Concepts'), h('dd', {}, concepts.length ? concepts.slice(0, 12).map(c => h('span', { class: 'chip alias' }, c.aliases[0] || c.key)) : 'None'),
        h('dt', {}, 'Links'), h('dd', {}, rels.length ? rels.map(r => h('div', {}, h('button', { class: 'link-btn mono', type: 'button', onclick: () => select({ type: 'rel', key: r.key, rel: r }) },
          `${r.left_table}.${r.left_expr} = ${r.right_table}.${r.right_expr}`), r.pending ? h('span', { class: 'pill warn', style: { marginLeft: '6px' } }, 'review') : null)) : 'None')),
      t.dataset_id ? h('div', { class: 'row', style: { marginTop: '8px' } }, h('button', { class: 'btn small', type: 'button', onclick: () => openDataset(t.dataset_id) }, 'Open this dataset')) : null,
      h('ul', { class: 'col-list' }, t.columns.map(c => h('li', {}, h('span', { class: 'mono' }, c.name), h('span', { class: 'muted' }, shortType(c.type)), h('span', { class: 'muted' }, clip(c.note || '', 110))))));
  } else {
    const r = sel.rel, ev = r.evidence && r.evidence.examples ? r.evidence : null;
    el.hidden = false;
    el.replaceChildren(close,
      h('div', { class: 'row wrap' }, h('h3', {}, `${r.left_table}.${r.left_expr} = ${r.right_table}.${r.right_expr}`),
        r.pending ? h('span', { class: 'pill warn' }, 'awaiting your review') : null),
      h('div', { class: 'row wrap', style: { marginTop: '6px' } }, r.cardinality ? h('span', { class: 'pill info' }, r.cardinality) : null,
        r.confidence ? confPill(r.confidence) : null, r.origin === 'builtin' ? h('span', { class: 'pill' }, 'built in') : h('span', { class: 'pill mine' }, 'added by you'),
        r.preferred === false ? h('span', { class: 'pill' }, 'not preferred') : null, r.bridge ? h('span', { class: 'pill' }, 'bridge') : null),
      h('dl', { class: 'kv' }, r.note ? [h('dt', {}, 'Why'), h('dd', {}, r.note)] : null, r.grain_effect ? [h('dt', {}, 'Row effect'), h('dd', {}, r.grain_effect)] : null,
        r.approved_at ? [h('dt', {}, 'Approved'), h('dd', {}, String(r.approved_at).slice(0, 16).replace('T', ' '))] : null),
      r.pending ? h('div', { class: 'actions' }, h('button', { class: 'btn primary small', type: 'button', onclick: () => { S.tab = 'review'; renderPanel(); } }, 'Review in the panel')) : null,
      ev ? h('div', { style: { marginTop: '10px' } }, h('h4', {}, 'How it maps when it was approved'), mappingTable(ev, `${r.right_table}`)) : null,
      r.origin !== 'builtin' && !r.pending ? h('div', { class: 'actions' }, h('button', { class: 'btn danger small', type: 'button', onclick: () => revoke(r.key) }, 'Revoke this link')) : null);
  }
}

/* ================================================================ WORKBENCH */
const confPill = c => h('span', { class: 'pill ' + (c === 'high' ? 'ok' : c === 'medium' ? 'warn' : 'bad') }, `${c} confidence`);
const statePill = p => ({ approved: h('span', { class: 'pill ok' }, 'Approved'), rejected: h('span', { class: 'pill' }, 'Rejected'),
  revoked: h('span', { class: 'pill' }, 'Revoked'), pending: h('span', { class: 'pill warn' }, 'Needs your review') }[p.status] || h('span', { class: 'pill' }, p.status));

function renderPanel() {
  const panel = $('#panel'), top = panel.scrollTop;
  panel.replaceChildren();
  append(panel, [S.busy ? [h('div', { class: 'progress', role: 'progressbar', 'aria-label': S.busy }), h('p', { class: 'muted', style: { marginBottom: '10px' } }, S.busy)] : null,
    S.ds ? datasetView() : homeView()]);      // append() flattens nested arrays and skips null
  panel.scrollTop = top;
}

/* ---------------------------------------------------------------- home */
function homeView() {
  return [
    h('div', { class: 'panel-head' }, h('h2', {}, 'Add a dataset'),
      h('p', { class: 'muted' }, 'Upload a file, describe it, then review the links the system proposes. The assistant only uses it after you approve.')),
    uploadCard(), datasetsCard(), modelCard(),
  ];
}

function uploadCard() {
  const input = h('input', { type: 'file', hidden: true, accept: (S.meta.formats || []).join(','), onchange: e => { const f = e.target.files[0]; e.target.value = ''; if (f) uploadFile(f); } });
  const drop = h('label', { class: 'drop', tabindex: 0 }, input, h('strong', {}, 'Choose a file'), ' or drop it here',
    h('small', {}, `CSV, TSV, Excel, JSON, Parquet, GeoJSON, GeoPackage, or a zipped shapefile · up to ${S.meta.max_upload_mb} MB`));
  drop.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('is-over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('is-over'));
  drop.addEventListener('drop', e => { e.preventDefault(); drop.classList.remove('is-over'); const f = e.dataTransfer && e.dataTransfer.files[0]; if (f) uploadFile(f); });
  return h('div', { class: 'card' }, drop,
    h('details', { style: { marginTop: '10px' } }, h('summary', { class: 'muted' }, 'Excel and CSV options'),
      h('div', { class: 'adjust' },
        h('label', { class: 'field' }, h('span', {}, 'Excel sheet'), h('input', { type: 'text', value: S.upload.sheet, placeholder: 'first sheet', oninput: e => { S.upload.sheet = e.target.value; } })),
        h('label', { class: 'field' }, h('span', {}, 'Header row (0 = first row)'), h('input', { type: 'number', min: 0, value: S.upload.skiprows, placeholder: 'detect', oninput: e => { S.upload.skiprows = e.target.value; } })))));
}

function datasetsCard() {
  const list = S.datasets;
  return h('div', { class: 'card' }, h('h3', {}, 'Datasets you added'),
    list.length ? h('ul', { class: 'ds-list' }, list.map(d => h('li', {},
      h('div', {}, h('button', { class: 'title', type: 'button', onclick: () => openDataset(d.dataset_id) }, d.title || d.table_name),
        h('div', { class: 'muted' }, h('span', { class: 'mono' }, d.table_name), ' · ', plural(d.row_count, 'row'))),
      h('span', { class: 'pill ' + (d.status === 'published' ? 'ok' : d.status === 'draft' ? 'info' : '') }, d.status === 'published' ? 'live' : d.status))))
      : h('p', { class: 'muted', style: { marginTop: '4px' } }, 'Nothing yet. Files you upload appear here while you review them.'));
}

function modelCard() {
  const m = S.llm, draft = m && m.tiers && m.tiers.draft && m.tiers.draft[0];
  const body = [];
  if (!m) body.push(h('p', { class: 'muted' }, 'Checking the drafting model…'));
  else {
    const anyOk = (m.tiers.draft || []).some(e => e.ok);
    body.push(h('div', { class: 'row wrap' }, h('span', { class: 'pill ' + (anyOk ? 'ok' : 'warn') }, anyOk ? 'reachable' : 'not reachable'),
      draft ? h('span', { class: 'muted mono' }, `${draft.model} @ ${draft.base_url}`) : null));
    if (!anyOk) body.push(h('p', { class: 'muted', style: { marginTop: '6px' } }, 'Descriptions will be filled by rules only. You can still write them yourself, and nothing on this page depends on the model.'));
    (m.notes || []).forEach(n => body.push(h('p', { class: 'muted', style: { marginTop: '6px' } }, n)));
  }
  if (S.selftest) {
    const t = S.selftest, sm = t.summary;
    body.push(h('div', { class: 'note ' + (t.verdict === 'PASS' ? 'ok' : 'warn'), style: { marginTop: '8px' } },
      h('b', {}, `Self-test: ${t.verdict}`), ` · valid ${pct(sm.valid_rate * 100)} · first try ${pct(sm.first_try_rate * 100)} · roles ${pct(sm.role_agreement * 100)} · synonyms ${pct(sm.synonym_rate * 100)}`,
      sm.median_latency_s != null ? ` · median ${sm.median_latency_s}s` : '', t.reasons.length ? h('ul', {}, t.reasons.map(r => h('li', {}, r))) : null));
  }
  return h('div', { class: 'card' }, h('div', { class: 'row between' }, h('div', {}, h('h3', {}, 'Drafting model'),
    h('p', { class: 'muted', style: { fontSize: '12.5px' } }, 'Optional. It only drafts wording; it never decides what joins.')),
    h('button', { class: 'btn small', type: 'button', disabled: !!S.busy, onclick: runSelftest }, 'Test it')), h('div', { style: { marginTop: '8px' } }, body));
}

async function runSelftest() {
  await withBusy('Scoring the drafting model on four sample datasets…', async () => { S.selftest = await post('/llm/selftest'); });
}

async function uploadFile(file) {
  await withBusy(`Reading and profiling ${file.name}…`, async () => {
    const fd = new FormData();
    fd.append('file', file);
    if (S.upload.sheet) fd.append('sheet', S.upload.sheet);
    if (S.upload.skiprows !== '') fd.append('skiprows', S.upload.skiprows);
    const ds = await api('/datasets', { method: 'POST', body: fd });
    await adopt(ds, 'describe');
    toast(`Read ${fmt(ds.row_count)} rows and ${ds.columns.length} columns. Describe it next.`, 'ok');
  });
}

async function adopt(ds, tab) {
  S.ds = ds; S.form = formFrom(ds); S.tab = tab || pickTab(ds); S.lastEnrich = null;
  await loadDatasets();
  await loadCatalog();
}
function pickTab(ds) {
  const pending = ds.proposals.some(p => p.status === 'pending');
  if (ds.status === 'published') return pending ? 'review' : 'result';
  return ds.proposals.length ? 'review' : 'describe';
}
async function openDataset(id) {
  await withBusy('Opening…', async () => { await adopt(await api('/datasets/' + id)); });
}
async function closeDataset() { S.ds = null; S.form = null; S.sel = null; await loadDatasets(); await loadCatalog(); renderPanel(); }

/* ---------------------------------------------------------------- dataset view */
function formFrom(ds) {
  const cols = {};
  ds.columns.forEach(c => { cols[c.name] = { role: c.role, description: c.description || '', unit: c.unit || '', synonyms: (c.synonyms || []).join(', '), include: c.include !== false }; });
  return { table_name: ds.table_name, title: ds.title, description: ds.description, grain: ds.grain, domain: ds.domain, cols };
}
function syncForm(ds) {
  ds.columns.forEach(c => { if (!S.form.cols[c.name]) S.form.cols[c.name] = { role: c.role, description: c.description || '', unit: c.unit || '', synonyms: (c.synonyms || []).join(', '), include: true }; });
}
async function saveForm() {
  const f = S.form;
  const body = { table_name: f.table_name, title: f.title, description: f.description, grain: f.grain, domain: f.domain,
    columns: Object.entries(f.cols).map(([name, c]) => ({ name, role: c.role, description: c.description, unit: c.unit, synonyms: c.synonyms, include: c.include })) };
  S.ds = await put(`/datasets/${S.ds.dataset_id}`, body);
}

function statusPill(ds) { return h('span', { class: 'pill ' + (ds.status === 'published' ? 'ok' : 'info') }, ds.status === 'published' ? 'live' : ds.status); }

function datasetView() {
  const ds = S.ds, pending = ds.proposals.filter(p => p.status === 'pending').length;
  const tab = (id, label, disabled) => h('button', { type: 'button', role: 'tab', 'aria-selected': String(S.tab === id), disabled, onclick: () => { S.tab = id; renderPanel(); } }, label);
  return [
    h('div', { class: 'row between' }, h('button', { class: 'link-btn', type: 'button', onclick: closeDataset }, '‹ All datasets'), statusPill(ds)),
    h('div', { class: 'panel-head' }, h('h2', {}, ds.title || ds.table_name),
      h('p', { class: 'muted' }, h('span', { class: 'mono' }, ds.table_name), ' · ', plural(ds.row_count, 'row'), ' · from ', ds.source_filename)),
    (ds.notes || []).length ? h('div', { class: 'note info', style: { marginBottom: '10px' } }, ds.notes.join(' ')) : null,
    h('div', { class: 'tabs', role: 'tablist' },
      tab('describe', '1  Describe', ds.status !== 'draft'),
      tab('review', `2  Review links${pending ? ` (${pending})` : ''}`, !ds.proposals.length),
      tab('result', '3  Live', ds.status !== 'published')),
    S.tab === 'describe' ? describeTab(ds) : S.tab === 'review' ? reviewTab(ds) : resultTab(ds),
  ];
}

/* ---------------------------------------------------------------- describe */
const TABLE_RE = /^[a-z][a-z0-9_]{2,47}$/;
function field(label, control, hint, extra) { return h('label', { class: 'field' }, h('span', {}, label), control, hint ? h('small', {}, hint) : null, extra || null); }

function describeTab(ds) {
  const f = S.form, err = h('small', { class: 'err' }, '');
  const checkName = () => { err.textContent = TABLE_RE.test(f.table_name) ? '' : 'Use lowercase letters, digits and underscores; start with a letter; 3-48 characters.'; };
  checkName();
  const dm = (S.ds.options || {}).draft_meta;
  const cols = h('div', {});
  const renderCols = () => cols.replaceChildren(...columnCards(ds));
  const roles = {};
  ds.columns.forEach(c => { const r = (S.form.cols[c.name] || {}).role || c.role; roles[r] = (roles[r] || 0) + 1; });
  const chips = h('div', { class: 'role-filter' });
  const renderChips = () => chips.replaceChildren(...['all', ...Object.keys(roles)].map(r => h('button', { type: 'button', 'aria-pressed': String(S.colRole === r),
    onclick: () => { S.colRole = r; renderChips(); renderCols(); } }, r === 'all' ? `All (${ds.columns.length})` : `${r} (${roles[r]})`)));
  renderChips(); renderCols();
  return [
    h('div', { class: 'card stack' },
      field('Dataset name', h('input', { type: 'text', value: f.title, oninput: e => { f.title = e.target.value; } })),
      field('Table name', h('input', { type: 'text', class: 'mono', value: f.table_name, oninput: e => { f.table_name = e.target.value.toLowerCase().trim(); checkName(); } }),
        'This is how the assistant and its SQL refer to the data.', err),
      field('What is in it?', h('textarea', { oninput: e => { f.description = e.target.value; } }, f.description)),
      h('div', { class: 'row', style: { alignItems: 'flex-start' } },
        h('div', { style: { flex: '1' } }, field('One row is…', h('input', { type: 'text', value: f.grain, placeholder: 'e.g. one row per census tract', oninput: e => { f.grain = e.target.value; } }))),
        h('div', { style: { width: '150px' } }, field('Area', h('select', { onchange: e => { f.domain = e.target.value; }, value: f.domain },
          (S.meta.domains || []).map(d => h('option', { value: d.key }, d.label))))))),
    h('div', { class: 'card' }, h('div', { class: 'row between' }, h('div', {}, h('h3', {}, 'Draft the wording with the local model'),
      h('p', { class: 'muted', style: { fontSize: '12.5px' } }, 'Suggests column descriptions, units, and the phrases you might use to ask about each column. Review everything; nothing is saved to the catalog yet.')),
      h('button', { class: 'btn', type: 'button', disabled: !!S.busy, onclick: draftFlow }, 'Draft')),
      dm ? h('div', { class: 'note ' + (dm.mode === 'model' ? 'ok' : 'warn'), style: { marginTop: '8px' } },
        dm.mode === 'model' ? `Drafted by ${dm.endpoint}. Check the wording before you continue.` : 'No model answered, so nothing was drafted. Write the descriptions yourself, or start a model and try again.',
        (dm.notes || []).length ? h('ul', {}, dm.notes.map(n => h('li', {}, n))) : null) : null),
    ...(ds.available_enrichments || []).map(e => h('div', { class: 'card' }, h('div', { class: 'row between' }, h('div', {}, h('h3', {}, e.label), h('p', { class: 'muted', style: { fontSize: '12.5px' } }, e.note)),
      h('button', { class: 'btn', type: 'button', disabled: !!S.busy, onclick: () => enrichFlow(e.kind) }, 'Run')))),
    S.lastEnrich ? h('div', { class: 'note ok', style: { marginBottom: '10px' } }, h('b', {}, S.lastEnrich.message), ' from ', S.lastEnrich.source, '.',
      h('ul', {}, (S.lastEnrich.examples || []).map(x => h('li', { class: 'mono' }, x.address ? `${x.address} → ` : `${x.lat}, ${x.lon} → `, str(x.tract_fips))))) : null,
    h('div', { class: 'card' }, h('h3', {}, 'Columns'),
      h('p', { class: 'muted', style: { fontSize: '12.5px' } }, 'Roles are guessed from the data. Synonyms are how you would ask about a column in chat, for example “walkability index”.'),
      h('input', { type: 'text', placeholder: 'Filter columns', 'aria-label': 'Filter columns', value: S.colQuery, style: { marginTop: '8px' }, oninput: e => { S.colQuery = e.target.value; renderCols(); } }),
      chips, cols),
    h('div', { class: 'sticky-actions' },
      h('button', { class: 'btn primary', type: 'button', disabled: !!S.busy, onclick: analyzeFlow }, 'Find links'),
      h('button', { class: 'btn', type: 'button', disabled: !!S.busy, onclick: () => withBusy('Saving…', async () => { await saveForm(); toast('Saved.', 'ok'); }) }, 'Save'),
      h('button', { class: 'btn danger', type: 'button', disabled: !!S.busy, onclick: discardFlow }, 'Discard')),
  ];
}

function columnCards(ds) {
  const q = S.colQuery.trim().toLowerCase();
  const list = ds.columns.filter(c => {
    const fc = S.form.cols[c.name]; if (!fc) return false;
    return (S.colRole === 'all' || fc.role === S.colRole) && (!q || c.name.includes(q) || (c.source_name || '').toLowerCase().includes(q));
  });
  if (!list.length) return [h('p', { class: 'muted' }, 'No columns match.')];
  return list.map(c => {
    const fc = S.form.cols[c.name];
    const card = h('div', { class: 'col-card' + (fc.include ? '' : ' is-off') });
    const box = h('input', { type: 'checkbox', checked: fc.include, 'aria-label': `Include ${c.name}`, onchange: e => { fc.include = e.target.checked; card.classList.toggle('is-off', !fc.include); } });
    card.append(
      h('div', { class: 'top' }, box, h('span', { class: 'mono', style: { fontWeight: 600 } }, c.name),
        c.source_name && c.source_name !== c.name ? h('span', { class: 'muted', title: 'Original header' }, `“${clip(c.source_name, 28)}”`) : null,
        h('span', { class: 'pill' }, c.dtype), c.derived ? h('span', { class: 'pill mine' }, 'derived') : null,
        c.drafted_by === 'model' ? h('span', { class: 'pill info', title: 'Drafted by the model; check it' }, 'model draft') : null,
        h('select', { 'aria-label': `Role of ${c.name}`, onchange: e => { fc.role = e.target.value; }, value: fc.role },
          (S.meta.roles || []).map(r => h('option', { value: r, title: ROLE_LABEL[r] }, r)))),
      h('div', { class: 'fields' },
        h('input', { type: 'text', class: 'wide', placeholder: 'What does this column hold?', value: fc.description, 'aria-label': `Description of ${c.name}`, oninput: e => { fc.description = e.target.value; } }),
        h('input', { type: 'text', placeholder: 'Words you would use to ask about it, comma-separated', value: fc.synonyms, 'aria-label': `Synonyms for ${c.name}`, oninput: e => { fc.synonyms = e.target.value; } }),
        h('input', { type: 'text', placeholder: 'unit', value: fc.unit, 'aria-label': `Unit of ${c.name}`, oninput: e => { fc.unit = e.target.value; } })),
      h('div', { class: 'samples', title: (c.samples || []).join(' | ') }, (c.samples || []).length ? c.samples.join('  ·  ') : 'no values'));
    return card;
  });
}

async function draftFlow() {
  await withBusy('Asking the drafting model… this can take a minute on a local model.', async () => {
    await saveForm();
    const out = await post(`/datasets/${S.ds.dataset_id}/draft`);
    S.ds = out; S.form = formFrom(out);
    toast(out.draft.ok ? 'Drafted. Check the wording before you continue.' : 'No model answered; nothing was drafted.', out.draft.ok ? 'ok' : '');
  });
}
async function enrichFlow(kind) {
  await withBusy('Running the enrichment…', async () => {
    await saveForm();
    const out = await post(`/datasets/${S.ds.dataset_id}/enrich`, { kind });
    S.ds = out; syncForm(out); S.lastEnrich = out.enrichment_result;
    await loadCatalog();
    toast(out.enrichment_result.message, 'ok');
  });
}
async function analyzeFlow() {
  await withBusy('Testing the columns against the existing data…', async () => {
    await saveForm();
    const out = await post(`/datasets/${S.ds.dataset_id}/analyze`);
    S.ds = out; S.tab = 'review'; await loadCatalog();
    const n = out.proposals.filter(p => p.kind === 'relationship').length;
    toast(n ? `${plural(n, 'link')} proposed. Check the examples before approving.` : 'No links found. You can still add the table on its own.', 'ok');
  });
}
async function discardFlow() {
  if (!window.confirm('Discard this draft? The uploaded copy is deleted; nothing has been added to the catalog.')) return;
  await withBusy('Discarding…', async () => { await post(`/datasets/${S.ds.dataset_id}/retire`); S.ds = null; S.form = null; await loadDatasets(); await loadCatalog(); toast('Draft discarded.'); });
}

/* ---------------------------------------------------------------- review */
function reviewTab(ds) {
  const props = ds.proposals, table = props.find(p => p.kind === 'table');
  const rels = props.filter(p => p.kind === 'relationship');
  const groups = new Map();
  rels.forEach(p => { if (!groups.has(p.group_key)) groups.set(p.group_key, []); groups.get(p.group_key).push(p); });
  const pending = props.filter(p => p.status === 'pending').length;
  const out = [h('div', { class: 'rev-head' }, h('h3', {}, pending ? `${plural(pending, 'change')} to review` : 'Everything has been decided'),
    h('button', { class: 'btn small', type: 'button', disabled: !!S.busy, onclick: analyzeFlow }, ds.status === 'published' ? 'Search for links again' : 'Re-run'))];
  if (!props.length) out.push(h('div', { class: 'note' }, 'No proposals yet. Go back to “Describe” and choose “Find links”.'));
  if (table) out.push(tableCard(table));
  if (rels.length) {
    const published = ds.status === 'published';
    out.push(h('div', { class: 'rev-head', style: { marginTop: '14px' } }, h('h3', {}, 'Links to existing data')));
    if (!published && table) out.push(h('div', { class: 'note info', style: { marginBottom: '10px' } }, 'Approve the table first. A link can only be added once the table exists.'));
    for (const [key, items] of groups) out.push(relGroup(key, items, published));
  } else if (table) out.push(h('div', { class: 'note', style: { marginTop: '10px' } }, 'No links to existing data were found. If your data has coordinates, run “Add census tract from coordinates” on the Describe tab; if it has street addresses, include the ZIP or city. You can still add the table on its own.'));
  return out;
}

function tableCard(p) {
  const pl = p.payload, ev = p.evidence, concepts = Object.values(pl.concepts || {});
  return h('div', { class: 'prop is-' + p.status },
    h('div', { class: 'row between' }, h('h4', {}, p.title), statePill(p)),
    h('p', { class: 'sum' }, p.summary),
    concepts.length ? h('div', { class: 'stack', style: { marginTop: '9px' } }, h('h4', {}, 'The assistant will recognize'),
      concepts.map(c => h('div', {}, h('span', { class: 'mono' }, (c.columns[0] || '').split('.')[1]), ' as ', c.aliases.slice(0, 5).map(a => h('span', { class: 'chip alias' }, a))))) : null,
    (ev.entity_domains || []).length ? h('p', { class: 'muted', style: { marginTop: '8px', fontSize: '12.5px' } }, 'Values of ', ev.entity_domains.map(d => h('span', { class: 'chip' }, d)), ' can be recognized inside questions.') : null,
    (ev.notices || []).length ? h('div', { class: 'note info', style: { marginTop: '9px' } }, h('b', {}, 'Wording that takes precedence'), h('ul', {}, ev.notices.map(w => h('li', {}, w)))) : null,
    (ev.warnings || []).length ? h('div', { class: 'note warn', style: { marginTop: '9px' } }, h('b', {}, 'Worth a look'), h('ul', {}, ev.warnings.map(w => h('li', {}, w)))) : null,
    (ev.suggestions || []).length ? h('div', { style: { marginTop: '9px' } }, h('h4', {}, 'You will be able to ask'), ev.suggestions.map(q => h('span', { class: 'chip ask' }, q))) : null,
    p.status === 'pending' ? h('div', { class: 'actions' },
      h('button', { class: 'btn primary', type: 'button', disabled: !!S.busy, onclick: () => decide(p, 'approve') }, 'Approve table'),
      h('button', { class: 'btn', type: 'button', disabled: !!S.busy, onclick: () => decide(p, 'reject') }, 'Reject')) : null);
}

function relGroup(key, items, published) {
  const first = items[0], ev = first.evidence, tr = ev.transform || {};
  const isAddr = tr.kind === 'address';
  const normalized = tr.kind && tr.kind !== 'identity' && !isAddr;
  const head = h('div', { class: 'ghead' },
    h('h3', {}, isAddr ? 'Street address matching' : first.payload.derive ? `Your column ${first.payload.derive.from} becomes ${first.payload.derive.column}` : `Your column ${ev.source_column || first.payload.left_expr}`),
    h('span', { class: 'rule' }, ev.rule_text || tr.label || ''));
  const pendingItems = items.filter(p => p.status === 'pending');
  const bulk = pendingItems.length > 1 && published ? h('button', { class: 'btn small', type: 'button', style: { marginTop: '6px' }, disabled: !!S.busy, onclick: () => bulkApprove(pendingItems) }, `Approve all ${pendingItems.length} in this group`) : null;
  return h('div', { class: 'group' }, head, normalized ? h('div', { class: 'note', style: { marginBottom: '8px' } }, 'A new column ', h('span', { class: 'mono' }, first.payload.derive.column), ' is added to your table when you approve; your original column is kept.') : null,
    items.map(p => relCard(p, published)), bulk);
}

function relCard(p, published) {
  const pl = p.payload, ev = p.evidence, st = ev.stats || {}, tgt = `${pl.right_table}.${pl.right_expr}`;
  const edit = S.edits[p.proposal_id] || (S.edits[p.proposal_id] = { cardinality: pl.cardinality, preferred: pl.preferred, note: pl.note || '' });
  const rightCov = st.right_covered_pct, leftPct = st.left_match_pct;
  const lead = rightCov >= leftPct
    ? h('span', {}, h('b', {}, pct(rightCov)), ` of ${pl.right_table} rows have a match in your data (${fmt(st.right_rows_matched)} of ${fmt(st.right_rows)})`)
    : h('span', {}, h('b', {}, pct(leftPct)), ` of your ${st.left_distinct ? 'distinct values' : 'rows'} exist in ${pl.right_table} (${fmt(st.matched_distinct)} of ${fmt(st.left_distinct || st.left_rows)})`);
  const canApprove = published;
  return h('div', { class: 'prop is-' + p.status },
    h('div', { class: 'row between' }, h('h4', {}, `${pl.left_table}.${pl.left_expr}  =  ${tgt}`), statePill(p)),
    h('div', { class: 'row wrap', style: { marginTop: '5px' } }, confPill(pl.confidence), h('span', { class: 'pill info' }, edit.cardinality), pl.preferred ? null : h('span', { class: 'pill' }, 'not preferred')),
    h('p', { class: 'sum' }, p.summary),
    h('div', { class: 'stat-line' }, lead,
      st.fanout_avg ? h('span', {}, 'Each key matches ', h('b', {}, st.fanout_avg), ` ${pl.right_table} rows on average (max ${fmt(st.fanout_max)})`) : null),
    (ev.warnings || []).length ? h('div', { class: 'note warn', style: { marginTop: '8px' } }, h('ul', { style: { margin: 0 } }, ev.warnings.map(w => h('li', {}, w)))) : null,
    ev.model_note ? h('div', { class: 'note info', style: { marginTop: '8px' } }, h('b', {}, `Model note (${ev.model_note.verdict}): `), ev.model_note.reason) : null,
    h('details', { class: 'how' }, h('summary', {}, 'Show how it maps'), mappingTable(ev, tgt)),
    p.status === 'pending' ? h('details', { class: 'how' }, h('summary', {}, 'Adjust before approving'),
      h('div', { class: 'adjust' },
        h('label', { class: 'field' }, h('span', {}, 'Cardinality'), h('select', { onchange: e => { edit.cardinality = e.target.value; }, value: edit.cardinality },
          ['many-to-one', 'one-to-many', 'one-to-one', 'many-to-many'].map(c => h('option', { value: c }, c)))),
        h('label', { class: 'check', style: { alignSelf: 'end' } }, h('input', { type: 'checkbox', checked: edit.preferred, onchange: e => { edit.preferred = e.target.checked; } }), 'Show this link to the assistant by default'),
        h('label', { class: 'field wide' }, h('span', {}, 'Note for the assistant'), h('input', { type: 'text', value: edit.note, oninput: e => { edit.note = e.target.value; } })))) : null,
    p.status === 'pending' ? h('div', { class: 'actions' },
      h('button', { class: 'btn primary', type: 'button', disabled: !canApprove || !!S.busy, title: canApprove ? '' : 'Approve the table first', onclick: () => decide(p, 'approve', { cardinality: edit.cardinality, preferred: edit.preferred, note: edit.note }) }, 'Approve link'),
      h('button', { class: 'btn', type: 'button', disabled: !!S.busy, onclick: () => decide(p, 'reject') }, 'Reject')) : null,
    p.status === 'approved' ? h('div', { class: 'actions' }, h('span', { class: 'muted' }, 'Live in General Chat and House Chat.'),
      h('button', { class: 'btn danger small', type: 'button', disabled: !!S.busy, onclick: () => revoke(`${pl.left_table}:${pl.left_expr}=${pl.right_table}:${pl.right_expr}`) }, 'Revoke')) : null);
}

function matchLabel(m) { return Object.values(m).filter(v => v != null && v !== '').slice(0, 3).join(' · '); }

/* The heart of verification: your value -> after the rule -> the rows it lands on. */
function mappingTable(ev, target) {
  const ex = ev.examples || [];
  const isAddr = ev.transform && ev.transform.kind === 'address';
  const rule = isAddr ? 'After normalizing' : (ev.transform && ev.transform.kind === 'identity') ? 'Compared as-is' : 'After the rule';
  const rows = ex.map(e => h('tr', {},
    h('td', {}, h('div', { class: 'mono' }, str(e.raw)),
      e.context && Object.keys(e.context).length ? h('div', { class: 'ctx' }, Object.entries(e.context).map(([k, v]) => `${k} ${str(v)}`).join(' · ')) : null,
      e.your_rows > 1 ? h('div', { class: 'ctx' }, `${fmt(e.your_rows)} of your rows`) : null),
    h('td', { class: 'arrow', 'aria-hidden': 'true' }, '→'),
    h('td', {}, e.normalized === e.raw ? h('span', { class: 'same' }, 'same') : h('span', { class: 'mono' }, str(e.normalized))),
    h('td', { class: 'arrow', 'aria-hidden': 'true' }, '→'),
    h('td', {}, (e.matches || []).map(m => h('span', { class: 'chip', title: JSON.stringify(m) }, matchLabel(m))),
      e.match_count > (e.matches || []).length ? h('span', { class: 'muted' }, ` +${fmt(e.match_count - e.matches.length)} more`) : null,
      e.tier ? h('span', { class: 'pill', style: { marginLeft: '4px' } }, `tier ${e.tier}`) : null)));
  const un = ev.unmatched || [];
  return h('div', {},
    ex.length ? h('table', { class: 'map-table' },
      h('thead', {}, h('tr', {}, h('th', {}, 'Your value'), h('th', {}), h('th', {}, rule), h('th', {}), h('th', {}, `Matches in ${target}`))), h('tbody', {}, rows))
      : h('p', { class: 'muted' }, 'No example rows.'),
    un.length ? h('p', { class: 'miss' }, h('b', {}, 'Values with no match: '), un.slice(0, 5).map(u => h('span', { class: 'chip' }, str(u.normalized != null ? u.normalized : u.raw))), (ev.stats && ev.stats.left_distinct > un.length) ? ' …' : '') : null);
}

/* ---------------------------------------------------------------- decisions */
async function decide(p, decision, edits) {
  await withBusy(decision === 'approve' ? 'Applying…' : 'Saving…', async () => {
    const out = await post(`/proposals/${p.proposal_id}/decision`, { decision, edits: edits || {} });
    S.ds = out;
    if (decision === 'approve' && p.kind === 'table') S.justAdded.add(out.table_name);
    await loadDatasets(); await loadCatalog();
    if (decision === 'approve') toast(p.kind === 'table' ? 'Table added. General Chat and House Chat can use it now.' : 'Link approved. It is live in General Chat and House Chat.', 'ok');
    else toast('Rejected.');
    if (S.ds.status === 'published' && !S.ds.proposals.some(x => x.status === 'pending')) S.tab = 'result';
  });
}
async function bulkApprove(items) {
  await withBusy(`Approving ${items.length} links…`, async () => {
    let out = null;
    for (const p of items) {
      const e = S.edits[p.proposal_id] || {};
      out = await post(`/proposals/${p.proposal_id}/decision`, { decision: 'approve', edits: { cardinality: e.cardinality, preferred: e.preferred, note: e.note } });
    }
    if (out) S.ds = out;
    await loadCatalog(); toast(`${items.length} links approved. They are live now.`, 'ok');
  });
}
async function revoke(relKey) {
  if (!window.confirm('Revoke this link? The assistant stops using it immediately. Your data is not changed.')) return;
  await withBusy('Revoking…', async () => {
    await post('/relationships/revoke', { rel_key: relKey });
    if (S.ds) S.ds = await api('/datasets/' + S.ds.dataset_id);
    S.sel = null; await loadCatalog(); toast('Link revoked.');
  });
}

/* ---------------------------------------------------------------- result */
function resultTab(ds) {
  const table = ds.proposals.find(p => p.kind === 'table' && p.status === 'approved');
  const rels = ds.proposals.filter(p => p.kind === 'relationship' && p.status === 'approved');
  const houseLinked = (S.catalog.house_linked || []).includes(ds.table_name);
  const suggestions = table ? table.evidence.suggestions || [] : [];
  return [
    h('div', { class: 'card' }, h('h3', {}, 'Live in the assistant'),
      h('p', { class: 'muted', style: { marginTop: '4px' } }, 'General Chat and House Chat read the catalog on every question, so this is already in effect.'),
      h('ul', { style: { margin: '10px 0 0', paddingLeft: '18px' } },
        h('li', {}, h('span', { class: 'mono' }, ds.table_name), ` is queryable (${plural(ds.row_count, 'row')}).`),
        rels.length ? rels.map(p => h('li', {}, h('span', { class: 'mono' }, `${p.payload.left_table}.${p.payload.left_expr} = ${p.payload.right_table}.${p.payload.right_expr}`), ` (${p.payload.cardinality})`)) : h('li', {}, 'No links yet, so it can be queried on its own but not joined to houses or tracts.'),
        houseLinked ? h('li', {}, 'House Chat can read the records linked to the house you have open.') : null)),
    suggestions.length ? h('div', { class: 'card' }, h('h3', {}, 'Try asking'), h('div', { style: { marginTop: '6px' } }, suggestions.map(q => h('span', { class: 'chip ask' }, q))),
      h('div', { class: 'actions' }, h('a', { class: 'btn primary', href: '/#general-chat' }, 'Open General Chat'))) : null,
    h('div', { class: 'card' }, h('h3', {}, 'Manage'),
      h('div', { class: 'actions' },
        h('button', { class: 'btn', type: 'button', disabled: !!S.busy, onclick: analyzeFlow }, 'Search for links again'),
        h('button', { class: 'btn danger', type: 'button', disabled: !!S.busy, onclick: retireFlow }, 'Retire this dataset')),
      h('p', { class: 'muted', style: { marginTop: '6px', fontSize: '12.5px' } }, 'Retiring hides the table and its links from the assistant. The data table is kept.')),
  ];
}
async function retireFlow() {
  if (!window.confirm('Retire this dataset? The assistant stops using its table and links immediately. The data table is kept.')) return;
  await withBusy('Retiring…', async () => { await post(`/datasets/${S.ds.dataset_id}/retire`); S.ds = null; S.form = null; await loadDatasets(); await loadCatalog(); toast('Dataset retired.'); });
}

/* ---------------------------------------------------------------- boot */
function bindToolbar() {
  let timer = null;
  $('#map-search').addEventListener('input', e => { clearTimeout(timer); timer = setTimeout(() => { S.filter.q = e.target.value; renderMap(); }, 150); });
  $('#opt-allcols').addEventListener('change', e => { S.filter.allCols = e.target.checked; renderMap(); });
  $('#opt-system').addEventListener('change', e => { S.filter.system = e.target.checked; renderMap(); });
}

async function init() {
  bindToolbar(); renderLegend();
  try { await Promise.all([loadDatasets(), loadCatalog()]); }
  catch (e) {
    $('#canvas').replaceChildren(h('div', { class: 'pad' }, h('div', { class: 'note bad' }, 'Could not load the catalog: ' + e.message),
      h('button', { class: 'btn', type: 'button', style: { marginTop: '8px' }, onclick: init }, 'Try again')));
  }
  renderPanel();
  api('/llm/status').then(x => { S.llm = x; if (!S.ds) renderPanel(); }).catch(() => { S.llm = { tiers: { draft: [] }, notes: [] }; if (!S.ds) renderPanel(); });
}
window.__dm = { S, uploadFile, openDataset, renderMap, renderPanel, saveForm, analyzeFlow, decide, adopt };
document.addEventListener('DOMContentLoaded', init);
