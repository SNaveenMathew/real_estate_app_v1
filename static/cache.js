'use strict';
/* Ontology page: the data model's classes, properties and relations, drawn as a graph.
   Vanilla JS, no dependencies. Data comes from GET /api/ontology/graph?detail=..., which is derived
   server-side from the live catalog, so an approved dataset appears here with no page change.
   Every string from the API is rendered with textContent / DOM nodes, never innerHTML. */

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
  if (props && props.value != null) el.value = props.value;
  return el;
}
function s(tag, attrs, ...kids) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null) el.setAttribute(k, v);
  append(el, kids);
  return el;
}
const fmt = n => (n == null || n === '' ? '–' : Number(n).toLocaleString('en-US'));
const clip = (t, n) => { const v = String(t == null ? '' : t); return v.length > n ? v.slice(0, n - 1) + '…' : v; };

/* ---------------------------------------------------------------- vocabulary */
const LANES = ['domain', 'dataset', 'concept', 'entity_type', 'transform', 'table', 'column'];
const LANE_LABEL = {
  domain: 'subject area', dataset: 'dataset', concept: 'concept',
  entity_type: 'entity type', transform: 'join-key rule', table: 'table', column: 'column (property)',
};
const LANE_WIDTH = {
  domain: 176, dataset: 208, concept: 286, entity_type: 178, transform: 214, table: 190, column: 236,
};
const KIND_COLOR = {
  domain: '#0f9d8a', dataset: '#6d4fc4', concept: '#3d7be8',
  entity_type: '#e0742f', transform: '#b5651d', table: '#7b8598', column: '#9aa5b8',
};
const DOMAIN_ORDER = ['housing', 'geography', 'risk', 'sales', 'safety', 'mobility', 'environment',
  'demographics', 'other', 'system'];
const DOMAIN_COLOR = {
  housing: '#3d7be8', geography: '#0f9d8a', risk: '#e0742f', sales: '#b08a1e', safety: '#c2415c',
  mobility: '#5a67d8', environment: '#3f9a5b', demographics: '#8d6e63', other: '#7b8598', system: '#9aa5b8',
};
const EDGE_COLOR = {
  in_domain: '#dfe5ef', in_dataset: '#e2daf6', in_table: '#e8edf5', uses_table: '#a8bad2',
  measures: '#c99a2e', about: '#e0a06a', requires_entity: '#c9770f', subclass_of: '#0f9d8a',
  overrides: '#5a67d8', joins: '#c2415c', transformed_by: '#b5651d', identified_by: '#8d6e63',
  derived_from: '#b08a1e',
};
const ROW = 32, HEAD = 48, LANE_GAP = 86;

const S = { graph: null, detail: 'standard', sel: null, zoom: 1, q: '' };

const rankOf = d => { const i = DOMAIN_ORDER.indexOf(d || 'other'); return i < 0 ? DOMAIN_ORDER.length : i; };

/* ---------------------------------------------------------------- stats strip */
function renderStats() {
  const box = $('#stats');
  box.replaceChildren();
  const g = S.graph;
  if (!g || !g.stats) { box.append(h('p', { class: 'muted pad' }, 'The ontology is unavailable.')); return; }
  const st = g.stats;
  const issues = g.issues || [];
  const errors = issues.filter(i => i.severity === 'error').length;
  const warnings = issues.filter(i => i.severity === 'warning').length;
  const card = (label, value, sub, title) => h('div', { class: 'stat', title: title || '' },
    h('div', { class: 'k' }, label), h('div', { class: 'v' }, value), h('div', { class: 'sub' }, sub));
  box.append(
    card('Subject areas', fmt(st.domains), 'top-level grouping of the model'),
    card('Concepts', fmt(st.concepts), 'what the assistant can reason about'),
    card('Tables', fmt(st.tables), 'physical tables the planner may read'),
    card('Columns', fmt(st.columns), 'properties, with role and unit'),
    card('Relations', fmt(st.relationships), 'approved join paths'),
    card('Key rules', fmt(st.transformations), `${fmt(st.relationships_with_transform)} joins normalized · ${fmt(st.opaque_relationships)} custom expressions`),
    card('Coverage',
      g.coverage && g.coverage.gaps.length
        ? `${g.coverage.gaps.length} setup gap${g.coverage.gaps.length === 1 ? '' : 's'}`
        : g.coverage && (g.coverage.data_gaps || []).length
          ? `${g.coverage.data_gaps.length} data gap${g.coverage.data_gaps.length === 1 ? '' : 's'}`
          : 'complete',
      g.coverage && (g.coverage.data_gaps || []).length
        ? 'metadata complete; physical or semantic join gaps remain'
        : 'visible tables with concepts and join paths'),
    card('Entity types', fmt(st.entity_types), 'named values a request can anchor to'),
    card('Datasets', fmt(st.datasets), 'built-in model plus uploads'),
    card('Model checks', `${errors} err · ${warnings} warn`,
      issues.length ? 'structural checks on the catalog' : 'no structural problems found',
      issues.slice(0, 6).map(i => `${i.severity}: ${i.subject} — ${i.message}`).join('\n')),
  );
  $('#ontology-summary').textContent =
    `${fmt(g.counts ? g.counts.nodes : g.nodes.length)} classes/properties and `
    + `${fmt(g.edges.length)} relations shown at “${g.detail}” granularity · catalog ${g.version}`;
}

/* ---------------------------------------------------------------- model + layout */
function matches(n, q) {
  if (!q) return true;
  const hay = [n.label, n.description, n.key, n.domain, n.table, n.column, n.iri, n.grain,
    ...(n.aliases || []), ...(n.tables || []), ...(n.entity_types || []), ...(n.domains || [])];
  return hay.some(v => String(v == null ? '' : v).toLowerCase().includes(q));
}

function buildModel() {
  const g = S.graph;
  const q = S.q.trim().toLowerCase();
  let nodes = g.nodes.filter(n => matches(n, q));
  let keep = new Set(nodes.map(n => n.id));
  if (q) {
    /* Keep one hop of context, so a matched table still shows the concepts that read it. */
    for (const e of g.edges) {
      if (keep.has(e.source)) keep.add(e.target);
      else if (keep.has(e.target)) keep.add(e.source);
    }
    nodes = g.nodes.filter(n => keep.has(n.id));
  }
  keep = new Set(nodes.map(n => n.id));
  const edges = g.edges.filter(e => keep.has(e.source) && keep.has(e.target));

  const lanes = LANES.filter(k => nodes.some(n => n.kind === k));
  const laneX = {};
  let x = 16;
  for (const k of lanes) { laneX[k] = x; x += LANE_WIDTH[k] + LANE_GAP; }

  const out = [];
  for (const kind of lanes) {
    const list = nodes.filter(n => n.kind === kind).sort((a, b) =>
      (rankOf(a.domain || a.key) - rankOf(b.domain || b.key)) || String(a.label).localeCompare(String(b.label)));
    let y = HEAD, lastDomain = null;
    for (const n of list) {
      const dom = n.domain || null;
      if (dom && lastDomain !== null && dom !== lastDomain) y += 14;
      if (dom) lastDomain = dom;
      out.push({ ...n, x: laneX[kind], y, w: LANE_WIDTH[kind], h: kind === 'column' ? 26 : 30 });
      y += ROW;
    }
  }
  return { nodes: out, edges, lanes, laneX };
}

function subLine(n) {
  if (n.kind === 'domain') return `${fmt(n.degree)} link${n.degree === 1 ? '' : 's'}`;
  if (n.kind === 'dataset') {
    const tables = (n.tables || []).length;
    return `${tables} table${tables === 1 ? '' : 's'}${n.synthetic ? ' · built in' : ''}`;
  }
  if (n.kind === 'concept') {
    if (n.gap) return 'topic not loaded in this model';
    const reads = (n.tables || []).join(', ');
    return reads ? `reads ${reads}` : `${(n.operations || []).length} operation(s)`;
  }
  if (n.kind === 'entity_type') {
    const cols = (n.columns || []).length;
    return n.individuals == null ? `${cols} source column${cols === 1 ? '' : 's'}`
      : `${fmt(n.individuals)} values · ${cols} source column${cols === 1 ? '' : 's'}`;
  }
  if (n.kind === 'transform') {
    return `${n.needs_width ? 'needs a width' : 'no width'} · ${n.normalizes_target ? 'normalizes both sides' : 'source side only'}`;
  }
  if (n.kind === 'table') return `${fmt(n.rows)} rows · ${n.domain || 'other'}`;
  const bits = [n.role || 'column'];
  if (n.unit) bits.push(n.unit);
  if (n.hidden) bits.push('hidden');
  return bits.join(' · ');
}

/* ---------------------------------------------------------------- rendering */
function fillFor(node) {
  if (node.kind === 'concept' && node.domain) return DOMAIN_COLOR[node.domain] || KIND_COLOR.concept;
  return KIND_COLOR[node.kind] || '#8fa6c9';
}

function renderMap() {
  const canvas = $('#canvas');
  canvas.replaceChildren();
  if (!S.graph) return;
  const model = buildModel();
  const nodes = model.nodes;
  if (!nodes.length) {
    canvas.append(h('p', { class: 'muted pad' }, S.q
      ? 'Nothing in the ontology matches that search.'
      : 'The catalog has no classes yet.'));
    renderLegend(model);
    return;
  }

  const width = Math.max(...nodes.map(n => n.x + n.w)) + 24;
  const height = Math.max(...nodes.map(n => n.y + n.h), HEAD + 120) + 30;
  const svg = s('svg', { width: width * S.zoom, height: height * S.zoom, viewBox: `0 0 ${width} ${height}` });
  const root = s('g', { transform: `scale(${S.zoom})` });
  svg.append(root);

  for (const kind of model.lanes) {
    root.append(s('text', { class: 'band', x: model.laneX[kind], y: 26 }, LANE_LABEL[kind] || kind));
  }

  /* edges first, so nodes sit on top */
  const byId = new Map(nodes.map(n => [n.id, n]));
  for (const e of model.edges) {
    const a = byId.get(e.source), b = byId.get(e.target);
    if (!a || !b) continue;
    const hot = S.sel && (e.source === S.sel || e.target === S.sel);
    let d;
    if (b.x >= a.x + a.w) {
      const x1 = a.x + a.w, y1 = a.y + a.h / 2, x2 = b.x, y2 = b.y + b.h / 2;
      const dx = Math.max(26, (x2 - x1) * 0.42);
      d = `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
    } else {
      /* same lane (a join or a concept hierarchy): loop out to the right */
      const x1 = a.x + a.w, y1 = a.y + a.h / 2, x2 = b.x + b.w, y2 = b.y + b.h / 2;
      const dx = 30 + Math.abs(y2 - y1) * 0.22;
      d = `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 + dx} ${y2}, ${x2} ${y2}`;
    }
    root.append(s('path', {
      class: 'gedge' + (hot ? ' hot' : ''),
      d,
      stroke: hot ? null : (EDGE_COLOR[e.predicate] || '#c8d1e0'),
    }));
  }

  for (const node of nodes) {
    const group = s('g', {
      class: 'gnode' + (S.sel === node.id ? ' sel' : ''),
      tabindex: '0', role: 'button',
      'aria-label': `${node.kind}: ${node.label}`,
    });
    group.append(s('rect', {
      x: node.x, y: node.y, width: node.w, height: node.h, rx: 7,
      fill: '#fff', stroke: KIND_COLOR[node.kind] || '#8fa6c9',
    }));
    group.append(s('rect', {
      x: node.x, y: node.y, width: 4, height: node.h, rx: 2, fill: fillFor(node), stroke: 'none',
    }));
    group.append(s('text', { class: 'nlabel', x: node.x + 12, y: node.y + 13 },
      clip(node.label || node.id, node.kind === 'column' ? 34 : 40)));
    group.append(s('text', { class: 'nsub', x: node.x + 12, y: node.y + 23 }, clip(subLine(node), 62)));
    const activate = () => showNode(node);
    group.addEventListener('click', activate);
    group.addEventListener('keydown', ev => {
      if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); activate(); }
    });
    root.append(group);
  }

  canvas.append(svg);
  renderLegend(model);
}

function renderLegend(model) {
  const box = $('#legend');
  box.replaceChildren();
  for (const kind of model.lanes) {
    box.append(h('span', null, h('i', { style: { background: KIND_COLOR[kind] } }), LANE_LABEL[kind] || kind));
  }
  box.append(
    h('span', null, h('i', { class: 'edge', style: { background: EDGE_COLOR.joins } }), 'approved join'),
    h('span', null, h('i', { class: 'edge', style: { background: EDGE_COLOR.uses_table } }), 'reads / measures'),
    h('span', null, h('i', { class: 'edge', style: { background: EDGE_COLOR.subclass_of } }), 'narrower concept'),
  );
}

/* ---------------------------------------------------------------- inspector */
function kv(pairs) {
  const dl = h('dl', { class: 'kv' });
  for (const [k, v] of pairs) dl.append(h('dt', null, k), h('dd', null, v == null || v === '' ? '—' : v));
  return dl;
}

function connections(node) {
  const byId = new Map(S.graph.nodes.map(n => [n.id, n]));
  const rows = S.graph.edges.filter(e => e.source === node.id || e.target === node.id).slice(0, 40);
  if (!rows.length) return h('p', { class: 'muted' }, 'No relations at this granularity.');
  const list = h('ul', { class: 'links' });
  for (const e of rows) {
    const outgoing = e.source === node.id;
    const other = byId.get(outgoing ? e.target : e.source);
    const label = (S.graph.predicates || {})[e.predicate] || e.predicate;
    const attrs = [e.cardinality, e.confidence ? `${e.confidence} confidence` : null, e.match_mode,
      e.predicate === 'transformed_by' ? (e.column || e.expr) : null,
      e.predicate === 'transformed_by' ? e.rule : null]
      .filter(Boolean).join(' · ');
    list.append(h('li', null,
      h('span', { class: 'mono' }, outgoing ? '→' : '←'), ' ', label, ' ',
      h('span', { class: 'mono' }, other ? (other.label || other.id) : '?'),
      attrs ? h('span', { class: 'muted' }, ` (${attrs})`) : null));
  }
  return list;
}

function showNode(node) {
  S.sel = node.id;
  $('#panel').classList.add('open');
  $('#btn-toggle-panel').classList.add('active');
  $('#btn-toggle-panel').setAttribute('aria-expanded', 'true');
  $('#ins-title').textContent = node.label || node.id;
  const body = $('#ins-body');
  const pairs = [['Kind', node.kind], ['IRI', h('span', { class: 'mono' }, node.iri || '—')]];
  if (node.description) pairs.push(['About', node.description]);

  if (node.kind === 'concept') {
    pairs.push(
      ['Area', node.domain || '—'],
      ['Catalog key', h('span', { class: 'mono' }, node.key)],
      ['Reads', (node.tables || []).join(', ') || '—'],
      ['Measures', (node.columns || []).join(', ') || '—'],
      ['Entity types', (node.entity_types || []).join(', ') || '—'],
      ['Grain', node.grain || '—'],
      ['Null policy', node.null_policy || '—'],
      ['Required terms', (node.required_terms || []).join(', ') || '—'],
      ['Excluded terms', (node.excluded_terms || []).join(', ') || '—'],
      ['Default operation', node.default_operation || '—'],
      ['Filters', (node.filters || []).length ? `${(node.filters || []).length} declared` : '—'],
      ['Narrower than', (node.broader || []).join(', ') || '—'],
      ['Takes precedence over', (node.overrides || []).join(', ') || '—'],
      ['Origin', node.origin || 'builtin'],
      ['Dataset', node.dataset_id || '—'],
    );
    if ((node.aliases || []).length) {
      pairs.push(['Synonyms', (node.aliases || []).slice(0, 18).join(', ')]);
    }
    const ops = (node.operations || []).map(o => `${o.op}(${o.expr})${o.group_by ? ` by ${o.group_by}` : ''}`);
    body.replaceChildren(
      kv(pairs),
      ops.length ? h('div', { class: 'card' },
        h('h3', null, 'Operations the planner can select'),
        h('ul', { class: 'links' }, ops.slice(0, 20).map(o => h('li', { class: 'mono' }, o)))) : null,
      h('div', { class: 'card' }, h('h3', null, 'Relations'), connections(node)),
    );
    renderMap();
    return;
  }
  if (node.kind === 'table') {
    pairs.push(['Area', node.domain || '—'], ['Grain', node.grain || '—'],
      ['Rows', fmt(node.rows)], ['Visible to the agents', node.agent_visible ? 'yes' : 'no'],
      ['Default filter', h('span', { class: 'mono' }, node.default_filter || '—')],
      ['Setup hint', node.setup_hint || '—'], ['Filter hint', node.filter_hint || '—'],
      ['Origin', node.origin || 'builtin'], ['Dataset', node.dataset_id || '—']);
    body.replaceChildren(kv(pairs), h('div', { class: 'card' }, h('h3', null, 'Relations'), connections(node)));
    renderMap();
    return;
  }
  if (node.kind === 'column') {
    pairs.push(['Table', node.table], ['Type', node.type || '—'], ['Role', node.role || '—'],
      ['Unit', node.unit || '—'], ['Derived from', node.derived_from || '—'],
      ['Hidden from prompts', node.hidden ? 'yes' : 'no'], ['Primary key', node.primary_key ? 'yes' : 'no'],
      ['Not null', node.not_null ? 'yes' : 'no'], ['Origin', node.origin || 'builtin']);
    body.replaceChildren(kv(pairs), h('div', { class: 'card' }, h('h3', null, 'Relations'), connections(node)));
    renderMap();
    return;
  }
  if (node.kind === 'entity_type') {
    const cols = node.columns || [];
    pairs.push(['Live values', node.individuals == null ? 'not counted' : fmt(node.individuals)],
      ['Domains', (node.domains || []).join(', ') || '—']);
    body.replaceChildren(kv(pairs),
      h('div', { class: 'card' }, h('h3', null, 'Identification sources'),
        cols.length ? kv(cols.map(c => [`${c.table}.${c.column}`, `${c.match_mode}${c.domain ? ' · ' + c.domain : ''}`]))
          : h('p', { class: 'muted' }, 'None.')),
      h('div', { class: 'card' }, h('h3', null, 'Relations'), connections(node)));
    renderMap();
    return;
  }
  if (node.kind === 'transform') {
    pairs.push(['Rule key', h('span', { class: 'mono' }, node.key)],
      ['Needs width', node.needs_width ? 'yes' : 'no'],
      ['Accepted widths', (node.widths || []).join(', ') || '—'],
      ['SQL pattern', h('span', { class: 'mono' }, node.sql_pattern || '—')],
      ['Target side', node.normalizes_target ? 'normalized too' : 'unchanged']);
    body.replaceChildren(kv(pairs), h('div', { class: 'card' }, h('h3', null, 'Relations'), connections(node)));
    renderMap();
    return;
  }
  if (node.kind === 'dataset') {
    pairs.push(['Status', node.status || '—'], ['Area', node.domain || '—'], ['Grain', node.grain || '—'],
      ['Tables', (node.tables || []).join(', ') || '—'], ['Rows', fmt(node.rows)],
      ['Source file', node.source || '—']);
    body.replaceChildren(kv(pairs),
      node.synthetic ? h('p', { class: 'muted' },
        'A grouping for the tables the application ships with, not a catalogue row of its own.') : null,
      h('div', { class: 'card' }, h('h3', null, 'Relations'), connections(node)));
    renderMap();
    return;
  }
  /* subject area */
  const members = S.graph.nodes.filter(n => n.domain === node.key && n.kind !== 'domain');
  pairs.push(['Members at this granularity', fmt(members.length)]);
  body.replaceChildren(kv(pairs),
    h('div', { class: 'card' }, h('h3', null, 'Members'),
      members.length
        ? h('ul', { class: 'links' }, members.slice(0, 60).map(n => h('li', null,
          h('span', { class: 'mono' }, n.kind), ' ', n.label || n.id)))
        : h('p', { class: 'muted' }, 'Nothing in this area at this granularity.')),
    h('div', { class: 'card' }, h('h3', null, 'Relations'), connections(node)));
  renderMap();
}

/* ---------------------------------------------------------------- actions */
function setZoom(z) {
  S.zoom = Math.max(0.25, Math.min(2.5, z));
  $('#zoom-level').textContent = `${Math.round(S.zoom * 100)}%`;
  renderMap();
}

function fitZoom() {
  const canvas = $('#canvas');
  if (!S.graph) return;
  const model = buildModel();
  if (!model.nodes.length) return;
  const width = Math.max(...model.nodes.map(n => n.x + n.w)) + 24;
  const available = Math.max(240, canvas.clientWidth - 34);
  setZoom(available / width);
}

function setDetail(detail) {
  if (S.detail === detail) return;
  S.detail = detail;
  for (const [id, key] of [['#view-overview', 'overview'], ['#view-standard', 'standard'], ['#view-full', 'full']]) {
    const el = $(id);
    const on = key === detail;
    el.classList.toggle('on', on);
    el.setAttribute('aria-pressed', String(on));
  }
  load();
}

async function load() {
  try {
    const res = await fetch(`/api/ontology/graph?detail=${encodeURIComponent(S.detail)}`);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    S.graph = await res.json();
    S.sel = null;
    renderStats();
    renderMap();
  } catch (e) {
    $('#canvas').replaceChildren(h('p', { class: 'muted pad' }, `Could not load the ontology: ${e.message}`));
  }
}

/* ---------------------------------------------------------------- wire up */
document.addEventListener('DOMContentLoaded', () => {
  const panel = $('#panel');
  const toggle = () => {
    const open = panel.classList.toggle('open');
    $('#btn-toggle-panel').classList.toggle('active', open);
    $('#btn-toggle-panel').setAttribute('aria-expanded', String(open));
  };
  $('#btn-toggle-panel').addEventListener('click', toggle);
  $('#btn-close-panel').addEventListener('click', toggle);

  $('#view-overview').addEventListener('click', () => setDetail('overview'));
  $('#view-standard').addEventListener('click', () => setDetail('standard'));
  $('#view-full').addEventListener('click', () => setDetail('full'));

  let searchTimer = null;
  $('#ontology-search').addEventListener('input', ev => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => { S.q = ev.target.value; renderMap(); }, 140);
  });
  $('#btn-zoom-in').addEventListener('click', () => setZoom(S.zoom + 0.1));
  $('#btn-zoom-out').addEventListener('click', () => setZoom(S.zoom - 0.1));
  $('#btn-zoom-reset').addEventListener('click', () => setZoom(1));
  $('#btn-zoom-fit').addEventListener('click', fitZoom);

  load();
});
