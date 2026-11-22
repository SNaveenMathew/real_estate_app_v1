'use strict';
/* Cache page: what the assistant has reused, and how it connects to the data model.
   Vanilla JS, no dependencies. Everything that comes from the cache or the catalog is rendered with
   textContent / DOM nodes (never innerHTML): cached answers and user questions are untrusted text. */

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
const timeAgo = iso => {
  if (!iso) return 'never';
  const ms = Date.now() - new Date(iso).getTime();
  const mins = Math.round(ms / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins} min ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `${hrs} hr${hrs === 1 ? '' : 's'} ago`;
  const days = Math.round(hrs / 24);
  return `${days} day${days === 1 ? '' : 's'} ago`;
};

/* ---------------------------------------------------------------- API */
async function api(path, opts = {}) {
  const res = await fetch('/api/cache' + path, opts);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { const j = await res.json(); if (j && j.detail) msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch (_) { /* keep default */ }
    throw new Error(msg);
  }
  return res.json();
}
const JSON_HDR = { 'Content-Type': 'application/json' };

/* ---------------------------------------------------------------- state */
const DOMAIN_ORDER = ['housing', 'geography', 'risk', 'sales', 'safety', 'mobility', 'environment',
  'demographics', 'other', 'system'];
const DOMAIN_COLOR = { housing: '#3d7be8', geography: '#0f9d8a', risk: '#e0742f', sales: '#b08a1e',
  safety: '#c2415c', mobility: '#5a67d8', environment: '#3f9a5b', demographics: '#8d6e63',
  other: '#7b8598', system: '#9aa5b8' };

const S = {
  graph: null, view: 'questions', sel: null, zoom: 1, showEmpty: true, q: '',
  entry: null, entryKey: null, loadingEntry: false,
};

function toast(msg, kind = '') {
  const el = h('div', { class: 'toast ' + kind }, msg);
  $('#toasts').append(el);
  setTimeout(() => el.classList.add('out'), 4200);
  setTimeout(() => el.remove(), 4800);
}

const rankOf = d => { const i = DOMAIN_ORDER.indexOf(d || 'other'); return i < 0 ? DOMAIN_ORDER.length : i; };

/* ---------------------------------------------------------------- stats strip */
function renderStats() {
  const box = $('#stats');
  box.replaceChildren();
  const g = S.graph;
  if (!g || !g.stats) { box.append(h('p', { class: 'muted pad' }, 'The cache is unavailable.')); return; }
  const st = g.stats;
  const card = (label, data, on, note) => {
    const entries = data ? data.entries : 0;
    const hits = data ? data.hits : 0;
    const reused = entries ? Math.round((hits / (entries + hits || 1)) * 100) : 0;
    return h('div', { class: 'stat' + (on ? '' : ' off'), title: note || '' },
      h('div', { class: 'k' }, label),
      h('div', { class: 'v' }, fmt(entries)),
      h('div', { class: 'sub' }, on ? `${fmt(hits)} reuse${hits === 1 ? '' : 's'} · ${reused}% of lookups` : 'turned off'),
      h('div', { class: 'bar' }, h('i', { style: { width: `${Math.max(0, Math.min(100, reused))}%` } })));
  };
  box.append(
    card('Prompt / context', st.prompt, st.prompt && st.prompt.enabled,
      'Temperature-0 model calls replayed verbatim for an unchanged prompt.'),
    card('Semantic answers', st.semantic, st.semantic && st.semantic.enabled,
      'Questions the planner routed to the same concepts, tables, operation and scope.'),
    card('SQL / CTE results', st.sql, st.sql && st.sql.enabled,
      'Identical statements, including CTEs, shared by both chats.'),
    h('div', { class: 'stat' },
      h('div', { class: 'k' }, 'Data fingerprint'),
      h('div', { class: 'v mono', style: { fontSize: '13px' } }, st.data_version || 'unavailable'),
      h('div', { class: 'sub' }, 'A reloaded dataset changes this, and every entry stops matching.')),
  );
  const total = st.total_entries || 0;
  $('#cache-summary').textContent = total
    ? `${fmt(total)} stored ${total === 1 ? 'entry' : 'entries'} · semantic match needs ${Math.round((st.thresholds && st.thresholds.semantic || 0.85) * 100)}% token overlap after the same plan`
    : 'Nothing cached yet — ask a question in either chat, then come back.';
}

/* ---------------------------------------------------------------- graph model */
const COLS = {
  concept: { x: 16, w: 252 }, question: { x: 412, w: 336 }, table: { x: 880, w: 224 },
  role: { x: 16, w: 252 }, prompt: { x: 412, w: 336 },
};
const ROW = 34, HEAD = 46;

function buildQuestions() {
  const g = S.graph;
  const q = S.q.trim().toLowerCase();
  const has = (node, extra) => !q || [node.label, node.request_text, node.description, extra, ...(node.tables || [])]
    .some(v => String(v || '').toLowerCase().includes(q));

  let cons = g.abstractions.filter(c => S.showEmpty || c.questions > 0);
  let qs = g.questions.slice();
  let tabs = g.tables.slice();

  let conIds = new Set(cons.map(c => c.id));
  if (q) {
    qs = qs.filter(n => has(n, n.operation));
    const usedCon = new Set();
    qs.forEach(n => n.concepts.forEach(k => usedCon.add('c:' + k)));
    cons = cons.filter(c => usedCon.has(c.id) || String(c.label || '').toLowerCase().includes(q)
      || String(c.description || '').toLowerCase().includes(q));
    conIds = new Set(cons.map(c => c.id));
    qs = qs.filter(n => n.concepts.length === 0 || n.concepts.some(k => conIds.has('c:' + k)));
    const usedTab = new Set();
    qs.forEach(n => n.tables.forEach(t => usedTab.add('t:' + t)));
    tabs = tabs.filter(t => usedTab.has(t.id) || String(t.label || '').toLowerCase().includes(q));
  } else {
    qs = qs.filter(n => n.concepts.some(k => conIds.has('c:' + k)) || n.concepts.length === 0);
  }

  /* order: concepts by area lane, then by label; questions follow the order of their first
     concept so links between the two columns stay short. */
  const conOrder = cons.slice().sort((a, b) =>
    (rankOf(a.domain) - rankOf(b.domain)) || String(a.label).localeCompare(String(b.label)));
  const conIndex = new Map(conOrder.map((c, i) => [c.id, i]));
  const qOrder = qs.slice().sort((a, b) => {
    const ai = a.concepts.length ? Math.min(...a.concepts.map(k => conIndex.has('c:' + k) ? conIndex.get('c:' + k) : 1e6)) : 1e6;
    const bi = b.concepts.length ? Math.min(...b.concepts.map(k => conIndex.has('c:' + k) ? conIndex.get('c:' + k) : 1e6)) : 1e6;
    return (ai - bi) || String(a.label).localeCompare(String(b.label));
  });
  const tabOrder = tabs.slice().sort((a, b) =>
    (rankOf(a.domain) - rankOf(b.domain)) || String(a.label).localeCompare(String(b.label)));

  /* positions */
  const nodes = [];
  const conY = new Map();
  let y = HEAD, lastDomain = null;
  for (const c of conOrder) {
    if (c.domain !== lastDomain) { if (lastDomain !== null) y += 22; lastDomain = c.domain; }
    conY.set(c.id, y);
    nodes.push({ ...c, x: COLS.concept.x, y, w: COLS.concept.w, h: 30 });
    y += ROW;
  }
  /* domain bands drawn once, at the first node of each lane */
  const bands = [];
  const seen = new Set();
  for (const c of conOrder) {
    if (seen.has(c.domain)) continue;
    seen.add(c.domain);
    bands.push({ domain: c.domain, y: conY.get(c.id) - 3 });
  }

  qOrder.forEach((n, i) => {
    nodes.push({ ...n, x: COLS.question.x, y: HEAD + i * ROW, w: COLS.question.w, h: 28 });
  });
  tabOrder.forEach((n, i) => {
    nodes.push({ ...n, x: COLS.table.x, y: HEAD + i * ROW, w: COLS.table.w, h: 26 });
  });

  const byId = new Map(nodes.map(n => [n.id, n]));
  const edges = [];
  for (const n of qOrder.concat(tabOrder)) {
    for (const e of g.edges) {
      if (e.target === n.id && byId.has(e.source)) edges.push(e);
    }
  }
  return { nodes, edges, bands, kind: 'questions' };
}

function buildPrompts() {
  const g = S.graph;
  const q = S.q.trim().toLowerCase();
  const match = n => !q || String(n.label || '').toLowerCase().includes(q);
  const roles = (g.roles || []).slice();
  const calls = (g.prompt_calls || []).filter(match);
  const roleIds = new Set(roles.map(r => r.id));
  const nodes = [];
  roles.forEach((r, i) => nodes.push({ ...r, x: COLS.role.x, y: HEAD + i * ROW, w: COLS.role.w, h: 28 }));
  calls.forEach((c, i) => nodes.push({ ...c, x: COLS.prompt.x, y: HEAD + i * ROW, w: COLS.prompt.w, h: 26 }));
  const byId = new Map(nodes.map(n => [n.id, n]));
  const edges = calls.filter(c => roleIds.has('r:' + c.role)).map(c => ({ source: 'r:' + c.role, target: c.id }));
  return { nodes, edges: edges.filter(e => byId.has(e.source) && byId.has(e.target)), bands: [], kind: 'prompts' };
}

/* ---------------------------------------------------------------- rendering */
function fillFor(node) {
  if (node.kind === 'concept') return DOMAIN_COLOR[node.domain] || DOMAIN_COLOR.other;
  if (node.kind === 'question') return node.hits > 0 ? '#2f6fe0' : '#8fa6c9';
  if (node.kind === 'table') return '#7b8598';
  if (node.kind === 'role') return '#6d4fc4';
  return '#8fa6c9';
}

function subLine(node) {
  if (node.kind === 'concept') return node.questions ? `${fmt(node.questions)} cached · ${fmt(node.hits)} reuses` : 'no reuse yet';
  if (node.kind === 'question') return [node.operation || 'answer', node.scope && node.scope !== 'query_database' ? node.scope : null,
    `${fmt(node.hits)} reuse${node.hits === 1 ? '' : 's'}`].filter(Boolean).join(' · ');
  if (node.kind === 'table') return `${fmt(node.questions)} cached queries · ${fmt(node.hits)} reuses`;
  if (node.kind === 'role') return `${fmt(node.calls)} cached call${node.calls === 1 ? '' : 's'} · ${fmt(node.hits)} reuses`;
  return `${fmt(node.hits)} reuse${node.hits === 1 ? '' : 's'} · ${timeAgo(node.created_at)}`;
}

function renderMap() {
  const canvas = $('#canvas');
  canvas.replaceChildren();
  if (!S.graph) return;
  const model = S.view === 'prompts' ? buildPrompts() : buildQuestions();
  const nodes = model.nodes;
  if (!nodes.length) {
    canvas.append(h('p', { class: 'muted pad' }, S.q
      ? 'Nothing cached matches that search.'
      : 'Nothing cached yet. Ask a question in either chat and it will appear here.'));
    renderLegend(model);
    return;
  }

  const width = Math.max(...nodes.map(n => n.x + n.w)) + 24;
  const height = Math.max(...nodes.map(n => n.y + n.h), HEAD + 120) + 30;
  const svg = s('svg', { width: width * S.zoom, height: height * S.zoom, viewBox: `0 0 ${width} ${height}` });
  const root = s('g', { transform: `scale(${S.zoom})` });
  svg.append(root);

  /* edges first, so nodes sit on top */
  const byId = new Map(nodes.map(n => [n.id, n]));
  for (const e of model.edges) {
    const a = byId.get(e.source), b = byId.get(e.target);
    if (!a || !b) continue;
    const x1 = a.x + a.w, y1 = a.y + a.h / 2, x2 = b.x, y2 = b.y + b.h / 2;
    const dx = Math.max(28, (x2 - x1) * 0.45);
    const hot = S.sel && (e.source === S.sel || e.target === S.sel);
    root.append(s('path', {
      class: 'gedge' + (hot ? ' hot' : ''),
      d: `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`,
    }));
  }

  /* area lane labels above the abstraction column */
  for (const band of model.bands) {
    const label = (S.graph.domains.find(d => d.key === band.domain) || {}).label || band.domain;
    root.append(s('text', { class: 'band', x: COLS.concept.x, y: band.y }, label));
  }

  for (const node of nodes) {
    const group = s('g', {
      class: 'gnode' + (S.sel === node.id ? ' sel' : ''),
      tabindex: '0', role: 'button',
      'aria-label': `${node.kind}: ${node.label}`,
    });
    group.append(s('rect', {
      x: node.x, y: node.y, width: node.w, height: node.h, rx: 7,
      fill: '#fff', stroke: fillFor(node),
    }));
    group.append(s('rect', {
      x: node.x, y: node.y, width: 4, height: node.h, rx: 2, fill: fillFor(node), stroke: 'none',
    }));
    group.append(s('text', { class: 'nlabel', x: node.x + 12, y: node.y + 13 }, clip(node.label || node.id, 42)));
    group.append(s('text', { class: 'nsub', x: node.x + 12, y: node.y + 23 }, clip(subLine(node), 62)));
    const activate = () => select(node);
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
  if (model.kind === 'prompts') {
    box.append(h('span', null, h('i', { style: { background: '#6d4fc4' } }), 'model role'),
      h('span', null, h('i', { style: { background: '#8fa6c9' } }), 'cached call'));
    return;
  }
  box.append(
    h('span', null, h('i', { style: { background: '#3d7be8' } }), 'abstraction (a catalog concept)'),
    h('span', null, h('i', { style: { background: '#2f6fe0' } }), 'cached question, already reused'),
    h('span', null, h('i', { style: { background: '#8fa6c9' } }), 'cached question, not reused yet'),
    h('span', null, h('i', { style: { background: '#7b8598' } }), 'table or CTE it reads'));
}

/* ---------------------------------------------------------------- inspector */
function kv(pairs) {
  const dl = h('dl', { class: 'kv' });
  for (const [k, v] of pairs) { dl.append(h('dt', null, k), h('dd', null, v == null || v === '' ? '—' : v)); }
  return dl;
}

async function select(node) {
  S.sel = node.id;
  $('#panel').classList.add('open');
  $('#btn-toggle-panel').classList.add('active');
  $('#btn-toggle-panel').setAttribute('aria-expanded', 'true');
  $('#ins-title').textContent = node.label || node.id;
  const body = $('#ins-body');
  body.replaceChildren(h('p', { class: 'muted' }, 'Loading…'));
  renderMap();

  const pairs = [['Kind', node.kind]];
  if (node.kind === 'concept') {
    pairs.push(['Area', node.domain], ['Catalog key', h('span', { class: 'mono' }, node.key)],
      ['Reads', (node.tables || []).join(', ') || '—'], ['Origin', node.origin],
      ['Cached questions', fmt(node.questions)], ['Reuses', fmt(node.hits)]);
    body.replaceChildren(
      h('p', { class: 'muted', style: { marginBottom: '12px' } },
        node.description || 'A semantic concept from the live catalog.'),
      kv(pairs),
      h('div', { class: 'card' },
        h('h3', null, 'What this shows'),
        h('p', { class: 'muted' },
          'The catalog concepts are the higher-level abstractions the assistant reasons in — walkability, '
          + 'transit connectivity, risk, price and anything else the catalog or an uploaded dataset declares. '
          + 'A concept with no cached questions is still listed: it is something the assistant can answer, '
          + 'it just has not been asked twice yet.')));
    return;
  }
  if (node.kind === 'table') {
    pairs.push(['Area', node.domain], ['Rows', fmt(node.rows)],
      ['Cached questions reading it', fmt(node.questions)], ['Reuses', fmt(node.hits)]);
    body.replaceChildren(kv(pairs));
    return;
  }
  if (node.kind === 'question') {
    pairs.push(['Asked', node.request_text], ['Operation', node.operation || '—'],
      ['Scope', node.scope || '—'], ['Concepts', (node.concepts || []).join(', ') || '—'],
      ['Tables', (node.tables || []).join(', ') || '—'],
      ['Reuses', fmt(node.hits)], ['Cached', timeAgo(node.created_at)]);
    body.replaceChildren(kv(pairs));
    const key = node.cache_key;
    await loadEntry(key, body);
    return;
  }
  if (node.kind === 'role') {
    pairs.push(['Cached calls', fmt(node.calls)], ['Reuses', fmt(node.hits)]);
    body.replaceChildren(kv(pairs), h('p', { class: 'muted' },
      'Every cached model call for this role. The chart on the right shows them; click one to read the stored reply.'));
    return;
  }
  /* a prompt call */
  pairs.push(['Role', node.role], ['Reuses', fmt(node.hits)], ['Cached', timeAgo(node.created_at)],
    ['Reply length', `${fmt(node.response_length)} characters`]);
  body.replaceChildren(kv(pairs));
  await loadEntry(node.cache_key, body);
}

async function loadEntry(cacheKey, body) {
  if (!cacheKey) return;
  S.entryKey = cacheKey;
  S.loadingEntry = true;
  try {
    const data = await api('/entry/' + encodeURIComponent(cacheKey));
    if (S.entryKey !== cacheKey) return;
    const card = h('div', { class: 'card' });
    if (data.sql) {
      card.append(h('h3', null, 'Cached statement'), h('pre', { class: 'blob' }, data.sql));
      if (data.meta && data.meta.tables && data.meta.tables.length) {
        card.append(h('p', { class: 'muted', style: { marginTop: '8px' } },
          `Tables: ${data.meta.tables.join(', ')} · ${fmt(data.meta.row_count)} rows`
          + (data.meta.truncated ? ' (truncated)' : '')));
      }
    }
    if (data.response) {
      card.append(h('h3', { style: { marginTop: data.sql ? '14px' : 0 } }, 'Cached answer'),
        h('pre', { class: 'blob' }, data.response));
    }
    if (!card.childNodes.length) card.append(h('p', { class: 'muted' }, 'Nothing stored to preview.'));
    body.append(card);
  } catch (e) {
    body.append(h('p', { class: 'muted' }, `Could not read that entry: ${e.message}`));
  } finally {
    S.loadingEntry = false;
  }
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
  const model = S.view === 'prompts' ? buildPrompts() : buildQuestions();
  if (!model.nodes.length) return;
  const width = Math.max(...model.nodes.map(n => n.x + n.w)) + 24;
  const available = Math.max(240, canvas.clientWidth - 34);
  setZoom(available / width);
}

async function load() {
  try {
    S.graph = await api('/graph?include_empty=true');
    renderStats();
    renderMap();
  } catch (e) {
    $('#canvas').replaceChildren(h('p', { class: 'muted pad' }, `Could not load the cache: ${e.message}`));
  }
}

async function clearCache() {
  try {
    const out = await api('/clear', { method: 'POST', headers: JSON_HDR, body: JSON.stringify({}) });
    if (!out.ok) throw new Error(out.error || 'Clear failed.');
    toast('Cache cleared. The next questions will be computed again.', 'ok');
    S.sel = null;
    await load();
  } catch (e) {
    toast(e.message || String(e), 'err');
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

  $('#view-questions').addEventListener('click', () => {
    S.view = 'questions';
    $('#view-questions').classList.add('on'); $('#view-questions').setAttribute('aria-pressed', 'true');
    $('#view-prompts').classList.remove('on'); $('#view-prompts').setAttribute('aria-pressed', 'false');
    renderMap();
  });
  $('#view-prompts').addEventListener('click', () => {
    S.view = 'prompts';
    $('#view-prompts').classList.add('on'); $('#view-prompts').setAttribute('aria-pressed', 'true');
    $('#view-questions').classList.remove('on'); $('#view-questions').setAttribute('aria-pressed', 'false');
    renderMap();
  });

  let searchTimer = null;
  $('#cache-search').addEventListener('input', ev => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => { S.q = ev.target.value; renderMap(); }, 140);
  });
  $('#opt-empty').addEventListener('change', ev => { S.showEmpty = ev.target.checked; renderMap(); });
  $('#btn-zoom-in').addEventListener('click', () => setZoom(S.zoom + 0.1));
  $('#btn-zoom-out').addEventListener('click', () => setZoom(S.zoom - 0.1));
  $('#btn-zoom-reset').addEventListener('click', () => setZoom(1));
  $('#btn-zoom-fit').addEventListener('click', fitZoom);
  $('#btn-clear').addEventListener('click', clearCache);

  load();
});
