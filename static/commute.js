'use strict';
/* Commute: the sidebar's Commute tab and the map layer that puts commute minutes on each house.
   Loaded after app.js (it uses app.js's `state`, `map`, `makeIcon`, `openSidebar` and `switchTab`).
   Anything that comes from a server (addresses, labels, warnings) is rendered with textContent, never innerHTML. */
const CommuteUI = (() => {
  const MODE_LABEL = { drive: 'Drive', bike: 'Bike', walk: 'Walk', transit: 'Transit' };
  const MODE_ORDER = ['drive', 'bike', 'walk', 'transit'];
  const S = {
    cfg: null, rows: {}, work: null,                 // server state
    layerOn: false, mode: 'drive', maxMin: 30,       // map layer settings
    house: null, houseRow: undefined,                // selected house; its estimate (undefined = still loading, null = none)
    editing: false, picking: false, busy: '', error: '', poll: null, workMarker: null, slider: null,
  };
  try {
    const saved = JSON.parse(localStorage.getItem('commute.prefs') || '{}');
    if (MODE_ORDER.includes(saved.mode)) S.mode = saved.mode;
    if (saved.maxMin >= 10 && saved.maxMin <= 120) S.maxMin = saved.maxMin;
  } catch (_) { /* preferences are a convenience only */ }
  const savePrefs = () => { try { localStorage.setItem('commute.prefs', JSON.stringify({ mode: S.mode, maxMin: S.maxMin })); } catch (_) { /* ignore */ } };

  /* ------------------------------------------------------------------ helpers */
  function el(tag, props, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v == null || v === false) continue;
      if (k === 'class') n.className = v;
      else if (k === 'style') n.style.cssText = v;
      else if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2), v);
      else if (k === 'value') n.value = v;
      else if (k === 'checked' || k === 'disabled') n[k] = !!v;
      else n.setAttribute(k, v === true ? '' : v);
    }
    for (const kid of kids.flat(Infinity)) {
      if (kid == null || kid === false) continue;
      n.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return n;
  }
  async function api(path, opts) {
    const r = await fetch('/api/commute' + path, opts);
    let body = null;
    try { body = await r.json(); } catch (_) { /* no body */ }
    if (!r.ok) throw new Error((body && body.detail) || `${r.status} ${r.statusText}`);
    return body;
  }
  const send = (method, path, body) => api(path, { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
  const fmtMin = m => (m == null ? '–' : m < 60 ? `${Math.round(m)} min` : `${Math.floor(m / 60)} h ${String(Math.round(m % 60)).padStart(2, '0')} min`);
  const fmtMi = m => (m == null ? '' : `${m.toFixed(m < 10 ? 1 : 0)} mi`);
  const fmtWhen = iso => {
    const d = new Date(String(iso || '').replace(' ', 'T'));
    return isNaN(d) ? '' : d.toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
  };
  const hostOf = url => { try { return new URL(url).host; } catch (_) { return url; } };
  const enabledModes = () => (S.cfg && S.cfg.modes) || ['drive', 'bike', 'walk'];
  // Bucket on the minutes the person actually sees (rounded), so a chip that reads "30" is never red next to a "≤ 30" legend.
  const bucket = m => { const r = Math.round(m); return r <= Math.round(S.maxMin * 2 / 3) ? 'good' : r <= S.maxMin ? 'ok' : 'far'; };
  /* The Census geocoder answers in ALL CAPS ("500 GRANT ST, PITTSBURGH, PA, 15219"); show it as an address. */
  const KEEP_UPPER = new Set(['NE', 'NW', 'SE', 'SW', 'US', 'PO']);
  function prettyAddress(label) {
    if (!label || label !== label.toUpperCase() || !/[A-Z]/.test(label)) return label || '';
    return label.split(',').map(part => {
      part = part.trim();
      if (/^[A-Z]{2}$/.test(part) || /^\d{5}(-\d{4})?$/.test(part)) return part;          // state, ZIP
      return part.split(/\s+/).map(w => KEEP_UPPER.has(w) ? w : w.charAt(0) + w.slice(1).toLowerCase()).join(' ');
    }).join(', ');
  }

  /* ------------------------------------------------------------------ server state */
  async function refreshConfig() { S.cfg = await api('/config'); }
  async function refreshSummary() {
    const s = await api('/summary');
    S.rows = s.rows; S.work = s.work;
    syncWorkMarker(); refreshMarkers();
  }
  async function loadHouseRow() {
    if (!S.house) return;
    const id = S.house;
    try {
      const r = await api('/house/' + encodeURIComponent(id));
      if (S.house === id) { S.houseRow = r.commute; if (r.work) S.work = r.work; }
    } catch (_) { if (S.house === id) S.houseRow = null; }
  }
  async function onHouseSelected(id) {
    S.house = id; S.houseRow = undefined; render();
    await loadHouseRow(); render();
  }

  function startPolling() {
    if (S.poll) return;
    S.poll = setInterval(async () => {
      try {
        const st = await api('/status');
        S.cfg = Object.assign({}, S.cfg, { job: st.job, counts: st.counts });
        if (st.job.state === 'running') { render(); return; }
        stopPolling();
        await refreshConfig(); await refreshSummary(); await loadHouseRow();
        render();
      } catch (_) { stopPolling(); }
    }, 1200);
  }
  function stopPolling() { if (S.poll) { clearInterval(S.poll); S.poll = null; } }

  async function saveWork(payload) {
    S.busy = 'Looking up the address…'; S.error = ''; render();
    try {
      const r = await send('PUT', '/work', payload);
      S.work = r.work; S.editing = false; S.rows = {};
      await refreshConfig(); syncWorkMarker(); refreshMarkers();
      startPolling();
    } catch (e) { S.error = e.message; }
    finally { S.busy = ''; render(); }
  }
  async function compute(scope) {
    S.error = '';
    try { const r = await send('POST', '/refresh', { scope }); S.cfg = Object.assign({}, S.cfg, { job: r.job }); startPolling(); }
    catch (e) { S.error = e.message; }
    render();
  }
  async function removeWork() {
    if (!window.confirm('Forget your work location and delete the commute estimates? The houses are not changed.')) return;
    try { S.cfg = await send('DELETE', '/work'); S.work = null; S.rows = {}; S.houseRow = null; setLayer(false); }
    catch (e) { S.error = e.message; }
    render();
  }

  /* pick the work location by clicking the map */
  function pickOnMap() {
    S.picking = true; render();
    map.getContainer().classList.add('picking-commute');
    const done = () => { S.picking = false; map.getContainer().classList.remove('picking-commute'); document.removeEventListener('keydown', onKey); };
    const onKey = e => { if (e.key === 'Escape') { map.off('click', onClick); done(); render(); } };
    const onClick = e => {
      done();
      const { lat, lng } = e.latlng;
      saveWork({ lat, lon: lng, label: `Dropped pin (${lat.toFixed(4)}, ${lng.toFixed(4)})` });
    };
    map.once('click', onClick);
    document.addEventListener('keydown', onKey);
  }

  /* ------------------------------------------------------------------ the map layer */
  function badgeFor(id) {
    if (!S.layerOn) return null;
    const r = S.rows[id], m = r && r[S.mode + '_min'];
    if (m == null) return null;
    return { min: m, cls: bucket(m) };
  }
  const badgeHtml = id => { const b = badgeFor(id); return b ? `<span class="commute-chip ${b.cls}">${Math.round(b.min)}</span>` : ''; };
  const dimClass = id => {
    if (typeof state !== 'undefined' && id === state.selectedHouseId) return '';       // the house you opened never fades
    const b = badgeFor(id);
    return b && b.cls === 'far' ? ' marker-dim' : '';
  };

  function refreshMarkers() {
    if (typeof state === 'undefined' || !state.markers) return;
    for (const [id, entry] of Object.entries(state.markers)) {
      entry.marker.setIcon(makeIcon(entry.props.status, id === state.selectedHouseId, entry.props.is_favorite, id));
    }
  }
  function syncWorkMarker() {
    if (S.workMarker) { map.removeLayer(S.workMarker); S.workMarker = null; }
    if (!S.layerOn || !S.work) return;
    const tip = document.createElement('span');
    tip.textContent = 'Work: ' + (S.work.label || 'saved location');
    S.workMarker = L.marker([S.work.lat, S.work.lon], {
      icon: L.divIcon({ className: '', html: '<div class="work-pin">🏢</div>', iconSize: [34, 34], iconAnchor: [17, 17] }),
      zIndexOffset: 2000, keyboard: false,
    }).bindTooltip(tip, { direction: 'top', offset: [0, -14] }).addTo(map);
  }
  function renderLegend() {
    const box = document.getElementById('commute-legend');
    if (!box) return;
    box.style.display = S.layerOn ? '' : 'none';
    box.replaceChildren(
      el('span', {}, el('i', { class: 'lg good' }), `≤ ${Math.round(S.maxMin * 2 / 3)}`),
      el('span', {}, el('i', { class: 'lg ok' }), `≤ ${S.maxMin}`),
      el('span', {}, el('i', { class: 'lg far' }), `over ${S.maxMin} min`),
      el('span', { class: 'lg-mode' }, MODE_LABEL[S.mode].toLowerCase()));
  }
  function setLayer(on) {
    if (on && !(S.cfg && S.cfg.configured)) {          // nothing to show yet: take the person to the setup form
      openSidebar(); switchTab('commute');
      S.error = ''; render();
      on = false;
    }
    S.layerOn = on;
    for (const cb of document.querySelectorAll('#layer-commute, #cm-layer')) cb.checked = on;
    syncWorkMarker(); refreshMarkers(); renderLegend();
    if (document.getElementById('cm-mode-box')) render();
  }

  /* ------------------------------------------------------------------ the tab */
  function setupCard() {
    const cfg = S.cfg || {};
    const input = el('input', {
      type: 'text', id: 'cm-address', class: 'cm-input', 'aria-label': 'Work address',
      placeholder: 'Street address, city, state (or 40.4406, -79.9959)',
      value: (S.editing && S.work && S.work.source !== 'map' ? S.work.label : '') || cfg.env_address || '',
    });
    const go = () => saveWork({ address: input.value });
    input.addEventListener('keydown', e => { if (e.key === 'Enter') go(); });
    const publicHosts = Object.values(cfg.providers || {}).filter(p => p.public).map(p => hostOf(p.url));
    return el('div', { class: 'cm-card' },
      el('h4', {}, S.editing ? 'Change your work location' : 'Where do you work?'),
      el('p', { class: 'cm-muted' }, 'See how long each house is from your workplace by car, bike and on foot. These are free-flow estimates from OpenStreetMap roads.'),
      input,
      S.error ? el('p', { class: 'cm-error', role: 'alert' }, S.error) : null,
      el('div', { class: 'cm-actions' },
        el('button', { class: 'btn-small', type: 'button', disabled: !!S.busy, onclick: go }, 'Find and compute'),
        el('button', { class: 'btn-small cm-secondary', type: 'button', disabled: !!S.busy, onclick: pickOnMap }, 'Click the map instead'),
        S.editing ? el('button', { class: 'btn-small cm-secondary', type: 'button', onclick: () => { S.editing = false; S.error = ''; render(); } }, 'Cancel') : null),
      S.picking ? el('p', { class: 'cm-hint' }, 'Click your workplace on the map. Press Esc to cancel.') : null,
      publicHosts.length
        ? el('p', { class: 'cm-muted cm-privacy' }, `Your work location and each house's coordinates are sent to ${[...new Set(publicHosts)].join(', ')}. Run your own routing server to keep them private (see the README).`)
        : el('p', { class: 'cm-muted cm-privacy' }, 'Routing runs on your own server, so nothing leaves your network except the address lookup.'));
  }

  function progressCard() {
    const j = (S.cfg && S.cfg.job) || {};
    if (j.state !== 'running') return null;
    const pct = j.total ? Math.round(100 * j.done / j.total) : 5;
    return el('div', { class: 'cm-card' },
      el('h4', {}, 'Computing commutes'),
      el('div', { class: 'cm-progress', role: 'progressbar', 'aria-label': 'Computing commutes', 'aria-valuemin': 0, 'aria-valuemax': j.total || 0, 'aria-valuenow': j.done || 0 },
        el('span', { style: `width:${pct}%` })),
      el('p', { class: 'cm-muted' }, j.total ? `${j.done} of ${j.total} houses` : 'Starting…'));
  }

  function workCard() {
    const w = S.work || (S.cfg && S.cfg.work), c = (S.cfg && S.cfg.counts) || {}, j = (S.cfg && S.cfg.job) || {};
    const need = c.to_compute || 0;
    return el('div', { class: 'cm-card' },
      el('div', { class: 'cm-head' },
        el('div', { class: 'cm-where' }, el('p', { class: 'cm-kicker' }, 'Commuting to'),
          el('p', { class: 'cm-work' }, w ? prettyAddress(w.label) || `${w.lat.toFixed(4)}, ${w.lon.toFixed(4)}` : '')),
        el('div', { class: 'cm-links' },
          el('button', { class: 'cm-link', type: 'button', onclick: () => { S.editing = true; S.error = ''; render(); } }, 'Change'),
          el('button', { class: 'cm-link', type: 'button', onclick: removeWork }, 'Remove'))),
      need > 0 && j.state !== 'running'
        ? el('div', { class: 'cm-banner' }, el('span', {}, `${need} house${need === 1 ? '' : 's'} need${need === 1 ? 's' : ''} a commute estimate.`),
            el('button', { class: 'btn-small', type: 'button', onclick: () => compute('missing') }, 'Compute'))
        : null,
      S.error ? el('p', { class: 'cm-error', role: 'alert' }, S.error) : null,
      j.state === 'error' ? el('p', { class: 'cm-error', role: 'alert' }, j.message) : null,
      (j.warnings || []).length && j.state !== 'running'
        ? el('div', { class: 'cm-warn', role: 'status' }, el('b', {}, 'Some estimates are missing'), el('ul', {}, j.warnings.map(x => el('li', {}, x))))
        : null);
  }

  function modeRow(mode, minutes, miles) {
    const axis = Math.max(60, S.maxMin * 2);
    const b = bucket(minutes);
    return el('div', { class: 'cm-mode' },
      el('div', { class: 'cm-mode-name' }, MODE_LABEL[mode]),
      minutes > 180
        ? el('div', { class: 'cm-mode-long' }, 'a long trip')
        : el('div', { class: 'cm-bar', role: 'img', 'aria-label': `${MODE_LABEL[mode]}: ${fmtMin(minutes)}` },
            el('span', { class: `cm-fill ${b}`, style: `width:${Math.min(100, minutes / axis * 100)}%` })),
      el('div', { class: 'cm-mode-val' }, el('strong', {}, fmtMin(minutes)), miles != null ? el('span', { class: 'cm-muted' }, ` · ${fmtMi(miles)}`) : null));
  }
  function unavailableRow(mode, r) {
    const far = r.straight_line_miles;
    const why = mode === 'walk' && far > 6 ? `too far to walk (${fmtMi(far)} straight-line)`
      : mode === 'bike' && far > 30 ? `too far to bike (${fmtMi(far)} straight-line)`
      : mode === 'transit' ? 'no itinerary found'
      : 'no route found';
    return el('div', { class: 'cm-mode cm-mode-na' }, el('div', { class: 'cm-mode-name' }, MODE_LABEL[mode]), el('div', { class: 'cm-muted cm-na' }, why));
  }

  function houseCard() {
    if (!S.house) {
      return el('div', { class: 'cm-card' }, el('h4', {}, 'This house'), el('p', { class: 'cm-muted' }, 'Select a house on the map to see how long its commute is.'));
    }
    const r = S.houseRow, j = (S.cfg && S.cfg.job) || {};
    if (r === undefined) return el('div', { class: 'cm-card' }, el('h4', {}, 'This house'), el('p', { class: 'cm-muted' }, 'Loading…'));
    if (r === null) {
      return el('div', { class: 'cm-card' }, el('h4', {}, 'This house'),
        el('p', { class: 'cm-muted' }, j.state === 'running' ? 'Its estimate is being computed.' : 'No estimate yet.'),
        j.state === 'running' ? null : el('div', { class: 'cm-actions' }, el('button', { class: 'btn-small', type: 'button', onclick: () => compute('missing') }, 'Compute')));
    }
    const rows = enabledModes().map(mode => {
      const m = r[mode + '_min'];
      return m == null ? unavailableRow(mode, r) : modeRow(mode, m, mode === 'transit' ? null : r[mode + '_miles']);
    });
    const factor = S.cfg && S.cfg.drive_factor;
    return el('div', { class: 'cm-card' },
      el('h4', {}, 'This house'),
      el('div', { class: 'cm-modes' }, rows),
      !enabledModes().includes('transit') ? el('p', { class: 'cm-muted cm-transit-hint' }, 'Transit times need a self-hosted OpenTripPlanner (see the README).') : null,
      el('p', { class: 'cm-muted cm-foot' },
        'Free-flow estimate along OpenStreetMap roads, house to work, with no traffic or signal timing',
        factor && factor !== 1 ? ` (drive times ×${factor})` : '', r.computed_at ? `. Computed ${fmtWhen(r.computed_at)}.` : '.'),
      r.fresh === false
        ? el('div', { class: 'cm-banner' }, el('span', {}, 'Out of date: the work location or routing settings changed.'),
            el('button', { class: 'btn-small', type: 'button', onclick: () => compute('missing') }, 'Recompute'))
        : null);
  }

  function mapCard() {
    const has = Object.keys(S.rows).length > 0;
    const slider = el('input', { type: 'range', id: 'cm-max', min: 10, max: 120, step: 5, value: S.maxMin, 'aria-label': 'Maximum commute in minutes' });
    const out = el('output', { for: 'cm-max' }, `${S.maxMin} min`);
    let t = null;
    slider.addEventListener('input', () => {
      S.maxMin = Number(slider.value); out.textContent = `${S.maxMin} min`; savePrefs();
      clearTimeout(t); t = setTimeout(() => { refreshMarkers(); renderLegend(); render(); }, 150);
    });
    const modes = enabledModes().filter(m => Object.values(S.rows).some(r => r[m + '_min'] != null));
    return el('div', { class: 'cm-card', id: 'cm-mode-box' },
      el('h4', {}, 'On the map'),
      el('label', { class: 'cm-check' }, el('input', { type: 'checkbox', id: 'cm-layer', checked: S.layerOn, disabled: !has, onchange: e => setLayer(e.target.checked) }), 'Show commute minutes on each house'),
      !has ? el('p', { class: 'cm-muted' }, 'Available once the estimates are computed.') : null,
      modes.length > 1
        ? el('div', { class: 'cm-seg', role: 'radiogroup', 'aria-label': 'Travel mode shown on the map' },
            modes.map(m => el('button', { type: 'button', role: 'radio', 'aria-checked': String(S.mode === m), class: S.mode === m ? 'on' : '',
              onclick: () => { S.mode = m; savePrefs(); refreshMarkers(); renderLegend(); render(); } }, MODE_LABEL[m])))
        : null,
      el('div', { class: 'cm-slider' }, el('label', { for: 'cm-max' }, el('span', {}, 'My maximum commute'), out), slider),
      el('p', { class: 'cm-muted' }, 'Houses over this limit turn red and fade, so the ones worth a look stand out.'));
  }

  function dataCard() {
    const cfg = S.cfg || {}, entries = Object.entries(cfg.providers || {});
    return el('div', { class: 'cm-card' },
      el('h4', {}, 'Where the numbers come from'),
      el('ul', { class: 'cm-providers' }, entries.map(([mode, p]) =>
        el('li', {}, el('span', {}, MODE_LABEL[mode]), el('span', { class: 'cm-muted' }, hostOf(p.url)), el('span', { class: p.public ? 'cm-tag pub' : 'cm-tag loc' }, p.public ? 'public server' : 'your server')))),
      el('div', { class: 'cm-actions' },
        el('button', { class: 'btn-small cm-secondary', type: 'button', disabled: !!(cfg.job && cfg.job.state === 'running'), onclick: () => compute('all') }, 'Recompute every house')));
  }

  function render() {
    const root = document.getElementById('commute-content');
    if (!root) return;
    if (!S.cfg) { root.replaceChildren(el('p', { class: 'cm-muted' }, 'Loading…')); return; }
    const configured = S.cfg.configured;
    root.replaceChildren(el('div', { class: 'cm-root' }, (!configured || S.editing)
      ? [setupCard(), progressCard()]
      : [workCard(), progressCard(), houseCard(), mapCard(), dataCard()]));
    renderLegend();
  }

  async function init() {
    if (S.booted) return;
    S.booted = true;
    render();
    try {
      await refreshConfig(); await refreshSummary();
      if (S.cfg.job && S.cfg.job.state === 'running') startPolling();
    } catch (e) { S.error = e.message; }
    render();
  }
  document.addEventListener('DOMContentLoaded', init);
  if (document.readyState !== 'loading') init();

  return { onHouseSelected, badgeHtml, dimClass, setLayer, refreshMarkers, _S: S };
})();
window.CommuteUI = CommuteUI;
