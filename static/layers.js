'use strict';
/* Map layers: the panel and the Leaflet rendering for every layer /api/layers reports - Houses (built
   into app.js), Commute (built into commute.js), and everything generic (Crime, Bike Routes, Risk,
   Population, Sold Homes, and any dataset approved on the Data page). Loaded after app.js and
   commute.js; it reaches into their top-level `map`, `clusterGroup`, `CommuteUI` the same way commute.js
   already reaches into app.js's `map` and `state`.

   Toggle rules: at most one "fill" layer at a time (radio behaviour - two area fills stacked mostly
   just hide one under the other), any number of "overlay" layers together (points/heat/lines draw as
   distinct glyphs, not solid coverage, so they never hide each other or the active fill). Houses is an
   overlay and starts checked; everything else starts unchecked, every page load. */
const MapLayers = (() => {
  const S = {
    specs: [], byName: {}, on: new Set(), fill: null, measure: {}, weight: {},
    leaflet: {}, loading: new Set(), error: '', control: null, fetchSeq: {},
  };

  const BIKE_COLORS = { protected_bike_lanes: '#0d9488', bike_lanes: '#16a34a', trails: '#65a30d',
    bikeable_sidewalks: '#0891b2', sharrows: '#ca8a04', cautionary_bike_route: '#ea580c', on_street_bike_route: '#6b7280' };
  const BIKE_LABELS = { protected_bike_lanes: 'Protected bike lane', bike_lanes: 'Bike lane', trails: 'Trail',
    bikeable_sidewalks: 'Bikeable sidewalk', sharrows: 'Sharrows', cautionary_bike_route: 'Cautionary route',
    on_street_bike_route: 'On-street route' };
  const HEAT_GRADIENT = { 0.2: '#3b82f6', 0.4: '#eab308', 0.65: '#f97316', 1.0: '#ef4444' };
  const RAMP = ['#3b82f6', '#eab308', '#f97316', '#ef4444'];   // the same 4-stop feel as the heat gradient, for choropleth/polygon fills

  function el(tag, props, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v == null || v === false) continue;
      if (k === 'class') n.className = v;
      else if (k === 'html') n.innerHTML = v;      // only ever used with our own fixed strings below, never server/user data
      else if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2), v);
      else if (k === 'checked' || k === 'disabled') n[k] = !!v;
      else n.setAttribute(k, v === true ? '' : v);
    }
    for (const kid of kids.flat(Infinity)) {
      if (kid == null || kid === false) continue;
      n.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return n;
  }

  async function api(path, params) {
    const url = new URL('/api/layers' + path, location.origin);
    for (const [k, v] of Object.entries(params || {})) if (v != null) url.searchParams.set(k, v);
    const r = await fetch(url);
    if (!r.ok) {
      let msg = `${r.status} ${r.statusText}`;
      try { const j = await r.json(); if (j && j.detail) msg = j.detail; } catch (_) { /* keep default */ }
      throw new Error(msg);
    }
    return r.json();
  }

  function bbox() {
    const b = map.getBounds();
    return { west: b.getWest(), south: b.getSouth(), east: b.getEast(), north: b.getNorth() };
  }

  // Same formula the old crime-only heat layer used: a fixed on-screen cell size as you zoom, clamped
  // to sane degree bounds at either extreme.
  function gridDegForZoom() {
    return Math.max(0.0005, Math.min(0.5, 0.003 * Math.pow(2, 14 - map.getZoom())));
  }

  function numericColor(value, lo, hi) {
    if (value == null || lo == null || hi == null || hi <= lo) return '#9aa5b8';
    const t = Math.max(0, Math.min(1, (value - lo) / (hi - lo))) * (RAMP.length - 1);
    const i = Math.min(RAMP.length - 2, Math.floor(t));
    return RAMP[t - i < 0.5 ? i : i + 1];    // 4 flat bands read more clearly on a small legend than a true blend
  }

  function hashColor(name) {
    let h = 0;
    for (const c of name) h = (h * 31 + c.charCodeAt(0)) >>> 0;
    return `hsl(${h % 360}, 62%, 45%)`;
  }

  /* ------------------------------------------------------------------ load + panel */
  async function loadSpecs() {
    const out = await api('');
    S.specs = out.layers;
    S.byName = Object.fromEntries(out.layers.map(l => [l.name, l]));
    S.exclusiveFill = new Set(out.exclusive_groups.fill || []);
    for (const l of out.layers) {
      if (l.default_on) S.on.add(l.name);
      if (l.group === 'fill' && l.default_on) S.fill = l.name;
      if (l.kind === 'heat' && l.weight) S.weight[l.name] = l.weight;
      if ((l.kind === 'choropleth' || l.kind === 'polygons') && l.default_measure) S.measure[l.name] = l.default_measure;
    }
  }

  function statusLine() {
    const busy = S.loading.size;
    if (busy) return `Loading\u2026`;
    if (S.error) return 'Error';
    return '';
  }

  function row(spec) {
    const checked = S.on.has(spec.name);
    const isFill = spec.group === 'fill';
    const disabled = !spec.available;
    const input = el('input', {
      type: isFill ? 'radio' : 'checkbox', name: isFill ? 'ml-fill' : undefined,
      checked: isFill ? S.fill === spec.name : checked, disabled,
      onchange: () => toggle(spec.name, isFill),
    });
    const swatchColor = spec.kind === 'heat' ? '#ef4444' : spec.kind === 'lines' ? (spec.name === 'bike_routes' ? '#16a34a' : hashColor(spec.name))
      : (spec.kind === 'choropleth' || spec.kind === 'polygons') ? '#f97316' : '#4f8ef7';
    const kids = [input, el('span', { class: 'ml-swatch', style: `background:${swatchColor}` }),
      el('span', {}, spec.title), spec.row_count != null ? el('span', { class: 'ml-count' }, fmtCount(spec.row_count)) : null];
    const wrap = el('label', { class: 'ml-row' + (disabled ? ' disabled' : '') }, kids);
    const extra = [];
    if (disabled && spec.reason) extra.push(el('p', { class: 'ml-reason' }, spec.reason));
    if ((checked || (isFill && S.fill === spec.name)) && spec.available) {
      if (spec.kind === 'heat' && spec.weight_options && spec.weight_options.length > 1) {
        extra.push(weightPicker(spec));
      }
      if ((spec.kind === 'choropleth' || spec.kind === 'polygons') && spec.measures && spec.measures.length > 1) {
        extra.push(measurePicker(spec));
      }
    }
    return [wrap, ...extra];
  }

  function fmtCount(n) {
    if (n >= 1000) return `${Math.round(n / 1000)}k`;
    return String(n);
  }

  function weightPicker(spec) {
    const sel = el('select', { class: 'ml-select', 'aria-label': `${spec.title} weight`,
      onchange: (e) => { S.weight[spec.name] = e.target.value || null; refresh(spec.name); } },
      el('option', { value: '' }, 'Weight: count only'),
      spec.weight_options.map(w => el('option', { value: w.column, selected: S.weight[spec.name] === w.column }, w.label)));
    return sel;
  }

  function measurePicker(spec) {
    const current = S.measure[spec.name] || spec.default_measure;
    const sel = el('select', { class: 'ml-select', 'aria-label': `${spec.title} measure`,
      onchange: (e) => { S.measure[spec.name] = e.target.value; refresh(spec.name); } },
      spec.measures.map(m => el('option', { value: m.column, selected: m.column === current }, m.label)));
    return sel;
  }

  function panelBody() {
    const houses = S.byName.houses;
    const fills = S.specs.filter(l => l.group === 'fill');
    // Exclude BOTH special entries (houses, commute) from the generic list: each gets its own
    // dedicated row/rendering below, so including them here would show them twice.
    const overlays = S.specs.filter(l => l.group === 'overlay' && !l.special);
    const commute = S.byName.commute;
    const groups = [];
    if (houses) groups.push(el('div', { class: 'ml-group' }, row(houses)));
    if (fills.length) {
      groups.push(el('div', { class: 'ml-group' },
        el('div', { class: 'ml-group-label' }, 'Area (pick one)'),
        fillNoneRow(), fills.map(row)));
    }
    if (overlays.length || commute) {
      groups.push(el('div', { class: 'ml-group' },
        el('div', { class: 'ml-group-label' }, 'Overlays'),
        overlays.map(row), commute ? commuteRow(commute) : null));
    }
    return [el('div', { class: 'ml-title' }, 'Map layers', el('span', { class: 'ml-status' }, statusLine())), groups, legendBlock()];
  }

  function fillNoneRow() {
    return el('label', { class: 'ml-row' },
      el('input', { type: 'radio', name: 'ml-fill', checked: S.fill === null, onchange: () => setFill(null) }),
      el('span', { class: 'ml-swatch', style: 'background:#e5e7eb' }), el('span', {}, 'None'));
  }

  function commuteRow(spec) {
    const on = typeof CommuteUI !== 'undefined' && CommuteUI._S.layerOn;
    const disabled = !spec.available;
    const input = el('input', { type: 'checkbox', id: 'layer-commute', checked: on, disabled,
      onchange: (e) => { if (typeof CommuteUI !== 'undefined') CommuteUI.setLayer(e.target.checked); } });
    return [el('label', { class: 'ml-row' + (disabled ? ' disabled' : '') },
      input, el('span', { class: 'ml-swatch', style: 'background:#1a1a2e' }), el('span', {}, spec.title)),
      el('div', { id: 'commute-legend', class: 'commute-legend', style: 'display:none' })];
  }

  function legendBlock() {
    const active = [];
    if (S.fill && S.byName[S.fill] && S.byName[S.fill].available) active.push(fillLegend(S.byName[S.fill]));
    for (const name of S.on) {
      const spec = S.byName[name];
      if (spec && spec.kind === 'heat') active.push(heatLegend(spec));
      if (spec && spec.name === 'bike_routes') active.push(bikeLegend());
    }
    if (!active.length) return null;
    return el('div', { class: 'ml-legend' }, active);
  }

  function fillLegend(spec) {
    const cached = S.leaflet[spec.name] && S.leaflet[spec.name]._mlRange;
    const label = (spec.measures.find(m => m.column === (S.measure[spec.name] || spec.default_measure)) || {}).label || spec.title;
    const gradient = `linear-gradient(90deg, ${RAMP.join(',')})`;
    return el('div', {},
      el('div', { class: 'ml-legend-title' }, label),
      el('div', { class: 'ml-legend-gradient', style: `background:${gradient}` }),
      el('div', { class: 'ml-legend-scale' },
        el('span', {}, cached ? fmtNum(cached[0]) : 'low'), el('span', {}, cached ? fmtNum(cached[1]) : 'high')));
  }

  function heatLegend(spec) {
    const gradient = `linear-gradient(90deg, ${Object.entries(HEAT_GRADIENT).map(([p, c]) => `${c} ${p * 100}%`).join(',')})`;
    return el('div', {}, el('div', { class: 'ml-legend-title' }, `${spec.title} density`),
      el('div', { class: 'ml-legend-gradient', style: `background:${gradient}` }),
      el('div', { class: 'ml-legend-scale' }, el('span', {}, 'fewer'), el('span', {}, 'more')));
  }

  function bikeLegend() {
    return el('div', {}, el('div', { class: 'ml-legend-title' }, 'Bike Routes'),
      Object.entries(BIKE_LABELS).map(([k, label]) =>
        el('div', { class: 'ml-legend-row' }, el('span', { class: 'ml-legend-line', style: `background:${BIKE_COLORS[k]}` }), label)));
  }

  function fmtNum(v) {
    if (v == null) return '\u2013';
    return Math.abs(v) >= 100 ? Math.round(v).toLocaleString() : (Math.round(v * 10) / 10).toString();
  }

  function render() {
    if (!S.control) return;
    // panelBody() returns nested arrays (groups of rows); replaceChildren does not flatten those on its
    // own the way el()'s own children-handling does, so flatten explicitly before spreading.
    const flat = panelBody().flat(Infinity).filter((x) => x != null && x !== false);
    S.control.getContainer().replaceChildren(...flat);
  }

  /* ------------------------------------------------------------------ toggling + fetching */
  function toggle(name, isFill) {
    if (name === 'houses') {
      if (S.on.has('houses')) { S.on.delete('houses'); map.removeLayer(clusterGroup); }
      else { S.on.add('houses'); map.addLayer(clusterGroup); }
      render();
      return;
    }
    if (isFill) { setFill(S.fill === name ? null : name); return; }
    if (S.on.has(name)) { S.on.delete(name); removeLeaflet(name); }
    else { S.on.add(name); refresh(name); }
    render();
  }

  function setFill(name) {
    if (S.fill && S.fill !== name) removeLeaflet(S.fill);
    S.fill = name;
    if (name) refresh(name);
    render();
  }

  function removeLeaflet(name) {
    // Bump the fetch sequence so an in-flight request for this layer (started before it was turned off,
    // e.g. by a moveend refresh) is discarded as stale when it resolves, instead of silently re-adding a
    // layer the user just removed.
    S.fetchSeq[name] = (S.fetchSeq[name] || 0) + 1;
    if (S.leaflet[name]) { map.removeLayer(S.leaflet[name]); delete S.leaflet[name]; }
  }

  async function refresh(name) {
    const spec = S.byName[name];
    if (!spec || !spec.available) return;
    if (!S.on.has(name) && S.fill !== name) return;
    const seq = (S.fetchSeq[name] = (S.fetchSeq[name] || 0) + 1);
    S.loading.add(name); S.error = ''; render();
    try {
      const params = { ...bbox() };
      if (spec.kind === 'heat') { params.weight = S.weight[name] || undefined; params.grid_deg = gridDegForZoom(); }
      if (spec.kind === 'choropleth' || spec.kind === 'polygons') params.measure = S.measure[name] || spec.default_measure;
      const data = await api('/' + encodeURIComponent(name), params);
      if (seq !== S.fetchSeq[name]) return;    // a newer request for this layer has already landed
      mount(name, spec, data);
    } catch (e) {
      if (seq === S.fetchSeq[name]) S.error = e.message;
    } finally {
      S.loading.delete(name);
      if (seq === S.fetchSeq[name]) render();
    }
  }

  async function refreshVisible() {
    const names = [...S.on].filter(n => n !== 'houses' && S.byName[n] && S.byName[n].available);
    if (S.fill) names.push(S.fill);
    await Promise.all(names.map(refresh));
  }

  /* ------------------------------------------------------------------ mounting each kind onto Leaflet */
  function mount(name, spec, data) {
    removeLeaflet(name);
    let layer;
    if (spec.kind === 'heat') layer = mountHeat(data);
    else if (spec.kind === 'points') layer = mountPoints(spec, data);
    else if (spec.kind === 'lines') layer = mountLines(name, data);
    else if (spec.kind === 'polygons') layer = mountPolygons(spec, name, data);
    else if (spec.kind === 'choropleth') layer = mountChoropleth(spec, name, data);
    if (!layer) return;
    layer.addTo(map);
    S.leaflet[name] = layer;
  }

  function mountHeat(data) {
    return L.heatLayer(data.points.map(p => [p[0], p[1], p[2]]), {
      radius: 18, blur: 22, maxZoom: map.getZoom(), max: Math.max(data.max_weight, 1), gradient: HEAT_GRADIENT,
    });
  }

  function mountPoints(spec, data) {
    const group = L.featureGroup();
    for (const f of data.features) {
      const [lon, lat] = f.geometry.coordinates;
      const m = L.circleMarker([lat, lon], { radius: 6, color: '#fff', weight: 1.5, fillColor: '#4f8ef7', fillOpacity: 0.9 });
      m.bindPopup(popupHtml(spec.title, f.properties));
      group.addLayer(m);
    }
    return group;
  }

  function mountLines(name, data) {
    const isBike = name === 'bike_routes';
    return L.geoJSON(data, {
      style: (f) => ({
        color: (f.properties && f.properties.color) || (isBike && BIKE_COLORS[f.properties.layer_type]) || '#4f8ef7',
        weight: isBike ? 5 : 3, opacity: isBike ? 0.85 : 0.75,
      }),
      onEachFeature: (f, layer) => {
        const label = isBike ? `${BIKE_LABELS[f.properties.layer_type] || f.properties.layer_type || ''} \u2014 ${f.properties.city || ''}`
          : (f.properties && (f.properties.name || f.properties.route_id)) || '';
        if (label) layer.bindTooltip(String(label), { sticky: true });
      },
    });
  }

  function mountPolygons(spec, name, data) {
    const measure = S.measure[name] || spec.default_measure;
    if (S.leaflet[name]) S.leaflet[name]._mlRange = [data.min, data.max];
    const range = [data.min, data.max];
    const layer = L.geoJSON(data, {
      style: (f) => ({ color: '#fff', weight: 1, fillOpacity: 0.6,
        fillColor: measure ? numericColor(f.properties[measure], range[0], range[1]) : '#4f8ef7' }),
      onEachFeature: (f, l) => l.bindPopup(popupHtml(spec.title, f.properties)),
    });
    layer._mlRange = range;
    return layer;
  }

  function mountChoropleth(spec, name, data) {
    const range = [data.min, data.max];
    if (data.warning) S.error = data.warning;
    const layer = L.geoJSON(data, {
      style: (f) => ({ color: '#fff', weight: 0.6, fillOpacity: 0.55, fillColor: numericColor(f.properties.value, range[0], range[1]) }),
      onEachFeature: (f, l) => l.bindTooltip(`${data.measure_label || data.measure}: ${fmtNum(f.properties.value)}`),
    });
    layer._mlRange = range;
    return layer;
  }

  function popupHtml(title, props) {
    const rows = Object.entries(props).filter(([, v]) => v != null && v !== '').slice(0, 8)
      .map(([k, v]) => `<div><b>${escapeHtml(prettify(k))}:</b> ${escapeHtml(String(v))}</div>`).join('');
    return `<div class="ml-popup"><div style="font-weight:700;margin-bottom:4px">${escapeHtml(title)}</div>${rows}</div>`;
  }
  function prettify(name) { return name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()); }

  /* ------------------------------------------------------------------ the Leaflet control + boot */
  function mount_control() {
    const Control = L.Control.extend({
      options: { position: 'topright' },
      onAdd() {
        const div = L.DomUtil.create('div', 'ml-control');
        L.DomEvent.disableClickPropagation(div);
        L.DomEvent.disableScrollPropagation(div);
        return div;
      },
    });
    S.control = new Control();
    map.addControl(S.control);
    render();
  }

  let moveTimer = null;
  function onMoveEnd() {
    clearTimeout(moveTimer);
    moveTimer = setTimeout(refreshVisible, 250);
  }

  async function init() {
    if (S.booted) return;
    S.booted = true;
    mount_control();
    try {
      await loadSpecs();
      map.addLayer(clusterGroup);           // Houses starts on, every page load, regardless of any earlier state
      render();
      await refreshVisible();
    } catch (e) {
      S.error = e.message; render();
    }
    map.on('moveend', onMoveEnd);
  }

  document.addEventListener('DOMContentLoaded', () => { if (document.readyState !== 'loading') init(); });
  if (document.readyState !== 'loading') init();

  return { refreshVisible, _S: S };
})();
window.MapLayers = MapLayers;
