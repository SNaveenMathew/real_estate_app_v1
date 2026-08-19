/* ── App state ───────────────────────────────────────────────────────── */
const state = {
  selectedHouseId: null,
  houseChatHistory: {},   // {house_id: [{role, content}]}
  generalChatHistory: [],
  generalChatSessionId: (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : `general-${Date.now()}-${Math.random().toString(16).slice(2)}`,
  markers: {},            // {house_id: L.Marker}
};

/* ── Map init ────────────────────────────────────────────────────────── */
const map = L.map('map', { zoomControl: true }).setView([37.09, -95.71], 5);

L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
  attribution: '© OpenStreetMap contributors © CARTO',
  subdomains: 'abcd', maxZoom: 19,
}).addTo(map);

/* ── Cluster group — created once, added to map ──────────────────────── */
const clusterGroup = L.markerClusterGroup({
  // Show individual markers once zoomed in enough to see streets
  disableClusteringAtZoom: 15,

  // Cluster appearance: match the app's dark navy palette
  iconCreateFunction(cluster) {
    const count = cluster.getChildCount();
    // Pick size bucket
    let size = 'small';
    if (count >= 100) size = 'large';
    else if (count >= 10) size = 'medium';
    return L.divIcon({
      html: `<div class="cluster-icon cluster-${size}"><span>${count}</span></div>`,
      className: '',
      iconSize: L.point(40, 40),
    });
  },

  // Smooth spider-out animation when two markers overlap exactly
  spiderfyOnMaxZoom: true,
  showCoverageOnHover: false,
  zoomToBoundsOnClick: true,

  // Performance: chunk-render large datasets without blocking the UI
  chunkedLoading: true,
});
map.addLayer(clusterGroup);

/* ── Marker helpers ──────────────────────────────────────────────────── */
function statusClass(status) {
  if (!status) return 'default';
  const s = status.toLowerCase();
  if (s.includes('active'))     return 'active';
  if (s.includes('pending'))    return 'pending';
  if (s.includes('contingent')) return 'contingent';
  if (s.includes('pre'))        return 'pre';
  return 'default';
}

function makeIcon(status, selected = false, isFavorite = false) {
  const sc = statusClass(status);
  if (isFavorite) {
    const selCls = selected ? ' selected' : '';
    return L.divIcon({
      html: `<div class="marker-heart ${sc}${selCls}">♥</div>`,
      className: '', iconSize: [30, 28], iconAnchor: [15, 28],
    });
  }
  const cls = `marker-pin ${sc}${selected ? ' marker-selected' : ''}`;
  return L.divIcon({
    html: `<div class="${cls}"></div>`,
    className: '', iconSize: [30, 30], iconAnchor: [15, 30],
  });
}

/* ── Load houses ─────────────────────────────────────────────────────── */
async function loadHouses() {
  const resp = await fetch('/api/houses');
  const geojson = await resp.json();

  if (!geojson.features || geojson.features.length === 0) {
    document.getElementById('stats-badge').textContent = '0 houses loaded';
    return;
  }

  const bounds = [];
  const layersToAdd = [];

  geojson.features.forEach(f => {
    const p = f.properties;
    const [lon, lat] = f.geometry.coordinates;
    const marker = L.marker([lat, lon], { icon: makeIcon(p.status, false, p.is_favorite) });

    marker.on('click', () => selectHouse(p.house_id));
    marker.bindTooltip(p.address || p.house_id, { direction: 'top', offset: [0, -20] });

    // Store for later (selectHouse needs to update the icon)
    state.markers[p.house_id] = { marker, props: p };
    layersToAdd.push(marker);
    bounds.push([lat, lon]);
  });

  // Add all markers to the cluster group in one call (much faster than one-by-one)
  clusterGroup.addLayers(layersToAdd);

  if (bounds.length > 0) map.fitBounds(bounds, { padding: [40, 40] });

  updateStatsBadge();
}

async function updateStatsBadge() {
  try {
    const resp = await fetch('/api/stats');
    const s = await resp.json();
    document.getElementById('stats-badge').textContent =
      `${s.houses} houses · ${s.nri_tracts.toLocaleString()} NRI tracts · ${s.vector_documents} docs`;
  } catch {
    document.getElementById('stats-badge').textContent = 'Stats unavailable';
  }
}

/* ── Select house ─────────────────────────────────────────────────────── */
async function selectHouse(houseId) {
  // Deselect old
  if (state.selectedHouseId && state.markers[state.selectedHouseId]) {
    const { marker, props } = state.markers[state.selectedHouseId];
    marker.setIcon(makeIcon(props.status, false, props.is_favorite));
  }

  state.selectedHouseId = houseId;
  const { marker, props } = state.markers[houseId];
  marker.setIcon(makeIcon(props.status, true, props.is_favorite));

  openSidebar();
  document.getElementById('house-title').textContent = props.address || houseId;

  // Switch to details tab
  switchTab('details');

  // Load full details
  try {
    const resp = await fetch(`/api/house/${houseId}`);
    const data = await resp.json();
    renderDetails(data.house, data.nri);
    renderRisk(data.nri);
    renderPhotos(data.documents || []);
    renderHistory(data.history || []);

    // Update favorite button
    const favBtn = document.getElementById('btn-favorite');
    const isFav = data.house.is_favorite;
    favBtn.textContent = isFav ? '♥' : '♡';
    favBtn.className = `btn-favorite ${isFav ? 'active' : 'inactive'}`;
  } catch (e) {
    document.getElementById('house-details-content').innerHTML =
      `<p style="color:red">Error loading house: ${e.message}</p>`;
  }

  // Load description separately
  try {
    const descResp = await fetch(`/api/house/${houseId}/description`);
    const descData = await descResp.json();
    renderDescription(descData.description);
  } catch (e) {
    renderDescription(null);
  }

  // Init chat if needed
  if (!state.houseChatHistory[houseId]) {
    state.houseChatHistory[houseId] = [];
    renderHouseChat([]);
    appendMsg('house', 'assistant', 'How can I help you with this property? I can estimate a price, explain the risk profile, or answer questions about what you\'ve shared.');
  } else {
    renderHouseChat(state.houseChatHistory[houseId]);
  }
}

/* ── Render details ──────────────────────────────────────────────────── */
function fmt$(n) { return n != null ? '$' + Number(n).toLocaleString() : '—'; }
function fmtN(n, suffix = '') { return n != null ? Number(n).toLocaleString() + suffix : '—'; }

function statusBadge(status) {
  const colors = {
    active: '#4f8ef7', pending: '#f59e0b', contingent: '#8b5cf6',
    pre: '#10b981', default: '#6b7280',
  };
  const cls = statusClass(status);
  return `<span class="status-badge" style="background:${colors[cls]}20;color:${colors[cls]}">${status || 'Unknown'}</span>`;
}

function scoreColor(score) {
  if (score == null) return '#e0e0e0';
  if (score >= 70)   return '#10b981';
  if (score >= 50)   return '#f59e0b';
  return '#ef4444';
}

function renderDetails(h, nri) {
  const el = document.getElementById('house-details-content');
  if (!h) { el.innerHTML = '<p>No details available.</p>'; return; }

  const riskColor = nri ? ratingColor(nri.risk_ratng) : '#e0e0e0';

  el.innerHTML = `
    <div class="detail-grid">
      <div class="detail-card full">
        <div class="label">Status</div>
        <div>${statusBadge(h.status)}</div>
        ${h.address ? `<div style="font-size:12px;color:#666;margin-top:4px">${h.address}, ${h.city || ''}, ${h.state || ''} ${h.zip || ''}</div>` : ''}
      </div>
      <div class="detail-card">
        <div class="label">List Price</div>
        <div class="value">${fmt$(h.price)}</div>
      </div>
      <div class="detail-card">
        <div class="label">Price/sqft</div>
        <div class="value">${h.price && h.sqft ? '$' + (h.price / h.sqft).toFixed(0) : '—'}</div>
      </div>
      <div class="detail-card">
        <div class="label">Beds / Baths</div>
        <div class="value">${fmtN(h.beds)} / ${fmtN(h.baths)}</div>
      </div>
      <div class="detail-card">
        <div class="label">Sq Ft</div>
        <div class="value">${fmtN(h.sqft)}</div>
      </div>
      <div class="detail-card">
        <div class="label">Year Built</div>
        <div class="value">${h.year_built || '—'}</div>
      </div>
      <div class="detail-card">
        <div class="label">HOA/mo</div>
        <div class="value">${h.hoa_fee ? fmt$(h.hoa_fee) : 'None'}</div>
      </div>
      ${nri ? `
      <div class="detail-card">
        <div class="label">NRI Risk</div>
        <div class="value" style="color:${riskColor}">${nri.risk_ratng || '—'}</div>
      </div>
      <div class="detail-card">
        <div class="label">Exp. Annual Loss</div>
        <div class="value">${nri.eal_valt ? fmt$(nri.eal_valt) : '—'}</div>
      </div>
      ` : ''}
      <div class="detail-card full">
        <div class="label">Livability Scores</div>
        <div class="score-row">
          ${scoreChip(h.walk_score, 'Walk')}
          ${scoreChip(h.bike_score, 'Bike')}
          ${scoreChip(h.transit_score, 'Transit')}
        </div>
      </div>
      ${h.tract_fips ? `
      <div class="detail-card full">
        <div class="label">Census Tract</div>
        <div style="font-size:12px;color:#666">${h.tract_fips}</div>
      </div>` : ''}
    </div>
  `;
}

function scoreChip(score, label) {
  const bg = scoreColor(score);
  return `<div class="score-pill" style="background:${bg}20;color:${bg}">
    ${score != null ? score : '—'}
    <span class="score-label">${label}</span>
  </div>`;
}

/* ── Render history table ────────────────────────────────────────────── */
function renderHistory(history) {
  // Find or create history section inside tab-details
  let section = document.getElementById('history-section');
  if (!section) {
    section = document.createElement('div');
    section.id = 'history-section';
    section.style.cssText = 'margin-top:16px';
    document.getElementById('house-details-content').after(section);
  }

  if (!history || history.length === 0) {
    section.innerHTML = '';
    return;
  }

  const rows = history.map(h => {
    const date  = h.snapshot_date || '—';
    const type  = h.source_type === 'sold' ? '🏷 Sold' : '📋 Listed';
    const stat  = h.status || '—';
    const price = h.price ? `$${Number(h.price).toLocaleString()}` : '—';
    const beds  = h.beds  ? `${h.beds}bd` : '';
    const baths = h.baths ? `${h.baths}ba` : '';
    const sqft  = h.sqft  ? `${Number(h.sqft).toLocaleString()} sf` : '';
    const details = [beds, baths, sqft].filter(Boolean).join(' / ') || '—';
    const src   = h.source_file || '—';

    // Color the type badge
    const badgeColor = h.source_type === 'sold' ? '#10b981' : '#4f8ef7';
    return `
      <tr>
        <td>${date}</td>
        <td><span style="background:${badgeColor}20;color:${badgeColor};
                         padding:2px 6px;border-radius:4px;font-size:11px;
                         font-weight:700">${type}</span></td>
        <td>${stat}</td>
        <td style="font-weight:600">${price}</td>
        <td style="color:#666">${details}</td>
      </tr>`;
  }).join('');

  section.innerHTML = `
    <div style="font-size:11px;font-weight:700;text-transform:uppercase;
                letter-spacing:.5px;color:#888;margin-bottom:8px">
      Price &amp; Status History
    </div>
    <div style="overflow-x:auto">
      <table style="width:100%;border-collapse:collapse;font-size:12px">
        <thead>
          <tr style="background:#f8f9fc;color:#666">
            <th style="padding:6px 8px;text-align:left;border-bottom:1px solid #eee">Date</th>
            <th style="padding:6px 8px;text-align:left;border-bottom:1px solid #eee">Type</th>
            <th style="padding:6px 8px;text-align:left;border-bottom:1px solid #eee">Status</th>
            <th style="padding:6px 8px;text-align:left;border-bottom:1px solid #eee">Price</th>
            <th style="padding:6px 8px;text-align:left;border-bottom:1px solid #eee">Details</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

/* ── Render risk ─────────────────────────────────────────────────────── */
const HAZARDS = [
  ['rfld_risks', 'Riverine Flooding'], ['hrcn_risks', 'Hurricane'],
  ['trnd_risks', 'Tornado'],           ['wfir_risks', 'Wildfire'],
  ['erqk_risks', 'Earthquake'],        ['cfld_risks', 'Coastal Flooding'],
  ['swnd_risks', 'Strong Wind'],       ['hail_risks', 'Hail'],
  ['hwav_risks', 'Heat Wave'],         ['drgt_risks', 'Drought'],
  ['ltng_risks', 'Lightning'],         ['wntw_risks', 'Winter Weather'],
  ['istm_risks', 'Ice Storm'],         ['lnds_risks', 'Landslide'],
  ['cwav_risks', 'Cold Wave'],         ['tsun_risks', 'Tsunami'],
  ['vlcn_risks', 'Volcanic Activity'], ['avln_risks', 'Avalanche'],
];

function ratingColor(rating) {
  if (!rating) return '#9ca3af';
  const r = rating.toLowerCase();
  if (r.includes('very high')) return '#ef4444';
  if (r.includes('relatively high')) return '#f97316';
  if (r.includes('moderate') || r.includes('medium')) return '#f59e0b';
  if (r.includes('relatively low')) return '#84cc16';
  if (r.includes('very low'))  return '#10b981';
  return '#9ca3af';
}

function renderRisk(nri) {
  const el = document.getElementById('risk-content');
  if (!nri) {
    el.innerHTML = '<p style="color:#888;font-size:13px;padding:20px 0">No NRI data for this census tract.<br>Make sure NRI data is loaded and tract FIPS is resolved.</p>';
    return;
  }

  const rc = ratingColor(nri.risk_ratng);
  const hazardBars = HAZARDS
    .filter(([k]) => nri[k] != null && nri[k] > 0)
    .sort((a, b) => (nri[b[0]] || 0) - (nri[a[0]] || 0))
    .map(([k, label]) => {
      const val = nri[k];
      const pct = Math.min(100, val);
      const color = val > 60 ? '#ef4444' : val > 30 ? '#f59e0b' : '#10b981';
      return `<div class="hazard-bar">
        <div class="hazard-name">${label} <span style="float:right;font-weight:700;color:${color}">${val.toFixed(1)}</span></div>
        <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div>
      </div>`;
    }).join('');

  el.innerHTML = `
    <div class="risk-summary">
      <div style="font-size:11px;opacity:.6;text-transform:uppercase;letter-spacing:.5px">Composite Risk Score</div>
      <div class="score-big" style="color:${rc}">${nri.risk_score != null ? nri.risk_score.toFixed(1) : '—'}</div>
      <div class="rating">${nri.risk_ratng || '—'} · ${nri.risk_npctl != null ? nri.risk_npctl.toFixed(0) + 'th percentile' : ''}</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px;font-size:12px">
        <div><div style="opacity:.6">Social Vulnerability</div><div style="font-weight:700">${nri.sovi_ratng || '—'}</div></div>
        <div><div style="opacity:.6">Community Resilience</div><div style="font-weight:700">${nri.resl_ratng || '—'}</div></div>
        <div><div style="opacity:.6">Exp. Annual Loss</div><div style="font-weight:700">${nri.eal_valt != null ? '$' + Number(nri.eal_valt).toLocaleString() : '—'}</div></div>
        <div><div style="opacity:.6">County</div><div style="font-weight:700">${nri.county_name || '—'}, ${nri.state_name || ''}</div></div>
      </div>
    </div>
    <div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:#888;margin-bottom:10px">
      Hazard Risk Scores (0–100)
    </div>
    ${hazardBars || '<p style="color:#aaa;font-size:13px">No hazard scores available.</p>'}
  `;
}

/* ── Description editor (single description per house) ───────────────── */
function renderDescription(desc) {
  const viewMode  = document.getElementById('desc-view-mode');
  const editMode  = document.getElementById('desc-edit-mode');
  const descText  = document.getElementById('desc-text');
  const descEmpty = document.getElementById('desc-empty');
  const addBtn    = document.getElementById('btn-add-desc');
  const descContent = document.getElementById('desc-text-content');

  editMode.style.display = 'none';

  if (desc) {
    descText.style.display  = 'block';
    descEmpty.style.display = 'none';
    addBtn.style.display    = 'none';
    descContent.textContent = desc.text;
    descContent._docId = desc.id;
  } else {
    descText.style.display  = 'none';
    descEmpty.style.display = 'block';
    addBtn.style.display    = 'inline-block';
  }
}

function showDescEditMode(currentText = '') {
  document.getElementById('desc-view-mode').style.display = 'none';
  const editMode = document.getElementById('desc-edit-mode');
  editMode.style.display = 'flex';
  document.getElementById('desc-edit-input').value = currentText;
  document.getElementById('desc-edit-input').focus();
}

function hideDescEditMode() {
  document.getElementById('desc-view-mode').style.display = 'block';
  document.getElementById('desc-edit-mode').style.display = 'none';
}

// Edit button
document.getElementById('btn-edit-desc').addEventListener('click', () => {
  const current = document.getElementById('desc-text-content').textContent;
  showDescEditMode(current);
});

// Add button (when no description)
document.getElementById('btn-add-desc').addEventListener('click', () => {
  document.getElementById('desc-empty').style.display = 'none';
  document.getElementById('btn-add-desc').style.display = 'none';
  showDescEditMode('');
});

// Cancel edit
document.getElementById('btn-cancel-desc').addEventListener('click', hideDescEditMode);

// Save description
document.getElementById('btn-save-desc').addEventListener('click', async () => {
  if (!state.selectedHouseId) return;
  const text = document.getElementById('desc-edit-input').value.trim();
  if (!text) return;
  try {
    await fetch(`/api/house/${state.selectedHouseId}/description`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, doc_type: 'description' }),
    });
    const resp = await fetch(`/api/house/${state.selectedHouseId}/description`);
    const data = await resp.json();
    renderDescription(data.description);
    hideDescEditMode();
    updateStatsBadge();
  } catch (e) { alert('Save failed: ' + e.message); }
});

// Delete description
document.getElementById('btn-delete-desc').addEventListener('click', async () => {
  if (!state.selectedHouseId) return;
  if (!confirm('Delete the saved description?')) return;
  try {
    await fetch(`/api/house/${state.selectedHouseId}/description`, { method: 'DELETE' });
    renderDescription(null);
    updateStatsBadge();
  } catch (e) { alert('Delete failed: ' + e.message); }
});

/* ── Photos rendering ────────────────────────────────────────────────── */
function renderPhotos(docs) {
  const el = document.getElementById('photos-content');
  if (!el) return;
  const photos = (docs || []).filter(d => d.metadata?.doc_type === 'photo');
  if (!photos.length) {
    el.innerHTML = '<p class="no-desc">No photos uploaded yet.</p>';
    return;
  }
  el.innerHTML = photos.map(d => `
    <div class="doc-item">
      <div class="doc-type">photo</div>
      <div class="doc-text">${d.text.substring(0, 120)}${d.text.length > 120 ? '...' : ''}</div>
    </div>
  `).join('');
}

/* ── Tabs ────────────────────────────────────────────────────────────── */
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.toggle('active', p.id === `tab-${name}`));
}

document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => switchTab(tab.dataset.tab));
});

/* ── Favorite toggle ─────────────────────────────────────────────────── */
document.getElementById('btn-favorite').addEventListener('click', async () => {
  if (!state.selectedHouseId) return;
  try {
    const resp = await fetch(`/api/house/${state.selectedHouseId}/favorite`, { method: 'POST' });
    const data = await resp.json();
    const isFav = data.is_favorite;

    // Update button
    const favBtn = document.getElementById('btn-favorite');
    favBtn.textContent = isFav ? '♥' : '♡';
    favBtn.className = `btn-favorite ${isFav ? 'active' : 'inactive'}`;

    // Update stored props and re-render marker
    const entry = state.markers[state.selectedHouseId];
    if (entry) {
      entry.props.is_favorite = isFav;
      entry.marker.setIcon(makeIcon(entry.props.status, true, isFav));
    }
  } catch (e) { alert('Could not update favorite: ' + e.message); }
});

/* ── Sidebar ─────────────────────────────────────────────────────────── */
function openSidebar()  { document.getElementById('sidebar').classList.add('open'); }
function closeSidebar() { document.getElementById('sidebar').classList.remove('open'); }
document.getElementById('btn-close-sidebar').addEventListener('click', closeSidebar);

/* ── House chat ──────────────────────────────────────────────────────── */
function renderHouseChat(history) {
  const el = document.getElementById('house-chat-messages');
  el.innerHTML = '';
  history.forEach(h => appendMsg('house', h.role, h.content));
}

function openTrace(traceUrl, button) {
  if (!traceUrl) return;
  const win = window.open(traceUrl, '_blank');
  if (!win) {
    window.location.assign(traceUrl);
    return;
  }
  try { win.opener = null; } catch (_) {}
  if (button) button.blur();
}

function addTraceButton(messageEl, traceUrl) {
  if (!traceUrl || messageEl.querySelector('.btn-trace')) return;

  const traceButton = document.createElement('button');
  traceButton.type = 'button';
  traceButton.className = 'btn-trace';
  traceButton.title = 'Open the exact Phoenix trace for this answer';
  traceButton.setAttribute('aria-label', 'View the exact Phoenix trace for this answer');
  traceButton.textContent = '⌁ Trace';
  traceButton.addEventListener('click', (event) => {
    event.preventDefault();
    event.stopPropagation();
    openTrace(traceUrl, traceButton);
  });
  messageEl.appendChild(traceButton);
}

function appendMsg(ctx, role, content, traceUrl = null) {
  const containerId = ctx === 'house' ? 'house-chat-messages' : 'general-chat-messages';
  const el = document.getElementById(containerId);
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  const html = typeof marked !== 'undefined' ? marked.parse(content) : content;
  div.innerHTML = `<div class="bubble">${html}</div>`;

  if (ctx === 'general' && role === 'assistant') {
    div.dataset.traceUrl = traceUrl || '';
    addTraceButton(div, traceUrl);
  }

  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
  return div;
}

function appendTyping(ctx) {
  const containerId = ctx === 'house' ? 'house-chat-messages' : 'general-chat-messages';
  const el = document.getElementById(containerId);
  const div = document.createElement('div');
  div.className = 'msg assistant typing';
  div.innerHTML = '<div class="bubble">Thinking…</div>';
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
  return div;
}

async function sendHouseMessage() {
  const input = document.getElementById('house-chat-input');
  const msg = input.value.trim();
  if (!msg || !state.selectedHouseId) return;

  input.value = '';
  const sendBtn = document.getElementById('house-chat-send');
  sendBtn.disabled = true;

  // Switch to chat tab
  switchTab('chat');
  appendMsg('house', 'user', msg);
  const typing = appendTyping('house');

  try {
    const history = state.houseChatHistory[state.selectedHouseId] || [];
    const resp = await fetch(`/api/house/${state.selectedHouseId}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg, history }),
    });
    const data = await resp.json();
    typing.remove();
    appendMsg('house', 'assistant', data.reply);
    state.houseChatHistory[state.selectedHouseId] = data.history;

    // Refresh description display if auto-saved
    if (data.auto_saved) {
      const descResp = await fetch(`/api/house/${state.selectedHouseId}/description`);
      const descData = await descResp.json();
      renderDescription(descData.description);
      updateStatsBadge();
    }
  } catch (e) {
    typing.remove();
    appendMsg('house', 'assistant', `Error: ${e.message}`);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

document.getElementById('house-chat-send').addEventListener('click', sendHouseMessage);
document.getElementById('house-chat-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendHouseMessage(); }
});

/* ── Photo upload ────────────────────────────────────────────────────── */
document.getElementById('btn-upload-photo').addEventListener('click', async () => {
  const fileInput = document.getElementById('photo-input');
  const caption = document.getElementById('photo-caption').value;
  if (!fileInput.files[0] || !state.selectedHouseId) return;

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  formData.append('caption', caption);

  try {
    await fetch(`/api/house/${state.selectedHouseId}/photo`, {
      method: 'POST', body: formData,
    });
    fileInput.value = '';
    document.getElementById('photo-caption').value = '';
    // Refresh photos
    const detailResp = await fetch(`/api/house/${state.selectedHouseId}`);
    const detailData = await detailResp.json();
    renderPhotos(detailData.documents || []);
  } catch (e) {
    alert('Upload error: ' + e.message);
  }
});

/* ── General chat ────────────────────────────────────────────────────── */
document.getElementById('btn-general-chat').addEventListener('click', () => {
  document.getElementById('general-modal').classList.add('open');
  document.getElementById('general-chat-input').focus();
});
document.getElementById('btn-close-general').addEventListener('click', () => {
  document.getElementById('general-modal').classList.remove('open');
});
// Close on backdrop click
document.getElementById('general-modal').addEventListener('click', e => {
  if (e.target === e.currentTarget) e.currentTarget.classList.remove('open');
});


function bikeRouteBearing(a, b) {
  const [lat1, lon1] = a.map(Number);
  const [lat2, lon2] = b.map(Number);
  const p1 = lat1 * Math.PI / 180, p2 = lat2 * Math.PI / 180;
  const dl = (lon2 - lon1) * Math.PI / 180;
  const y = Math.sin(dl) * Math.cos(p2);
  const x = Math.cos(p1) * Math.sin(p2) - Math.sin(p1) * Math.cos(p2) * Math.cos(dl);
  return (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
}

function makeRouteArrow(latlng, bearing) {
  return L.marker(latlng, {
    icon: L.divIcon({
      className: 'bike-route-arrow',
      html: `<div style="transform:rotate(${bearing}deg)">➤</div>`,
      iconSize: [24,24],
      iconAnchor: [12,12],
    }),
    interactive: false,
  });
}

async function renderBikeRouteInChat(messageEl, vis) {
  const wrapper = document.createElement('div');
  wrapper.className = 'bike-route-result';
  wrapper.innerHTML = `
    <div class="bike-route-result-header">
      <strong>BikePGH route</strong>
      <span>${Number(vis.distance_miles || 0).toFixed(1)} mi · ${Number(vis.duration_minutes || 0).toFixed(0)} min estimate</span>
    </div>
    <div class="bike-route-chat-map"></div>
    <div class="bike-route-legend"></div>
  `;
  messageEl.querySelector('.bubble')?.appendChild(wrapper);

  const mapEl = wrapper.querySelector('.bike-route-chat-map');
  const routeMap = L.map(mapEl, {
    zoomControl: true,
    attributionControl: true,
    scrollWheelZoom: false,
  }).setView([40.4406, -80.0018], 13);

  L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '© OpenStreetMap contributors © CARTO',
    subdomains: 'abcd',
    maxZoom: 19,
  }).addTo(routeMap);

  const asLatLon = (p) => {
    if (!Array.isArray(p) || p.length < 2) return null;
    const lat = Number(p[0]);
    const lon = Number(p[1]);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return null;
    return [lat, lon];
  };

  const routeCoords = (vis.route_shape || []).map(asLatLon).filter(Boolean);
  if (routeCoords.length < 2) {
    wrapper.querySelector('.bike-route-legend').textContent =
      'Route found, but no valid route geometry was returned.';
    return;
  }

  // Draw the route first as a subtle guide. BikePGH infrastructure is
  // deliberately rendered AFTER the route so the original BikePGH colors
  // are the visible top layer.
  L.polyline(routeCoords, {
    color: '#ffffff',
    weight: 9,
    opacity: 0.95,
    lineCap: 'round',
    lineJoin: 'round',
  }).addTo(routeMap);
  L.polyline(routeCoords, {
    color: '#1d4ed8',
    weight: 5,
    opacity: 0.9,
    lineCap: 'round',
    lineJoin: 'round',
  }).addTo(routeMap);

  // Infrastructure is route-edge-specific: only BikePGH segments that
  // Dijkstra actually selected are drawn, using the original layer colors.
  const used = vis.used_infrastructure || {features: []};
  const features = Array.isArray(used.features) ? used.features : [];
  const routeLayer = L.featureGroup().addTo(routeMap);

  for (const feature of features) {
    const color = feature?.properties?.color || '#666';
    const geom = feature?.geometry;
    if (!geom || geom.type !== 'LineString' || !Array.isArray(geom.coordinates)) continue;

    const pts = geom.coordinates
      .filter(p => Array.isArray(p) && p.length >= 2)
      .map(([lon, lat]) => [Number(lat), Number(lon)])
      .filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));

    if (pts.length < 2) continue;

    L.polyline(pts, {
      color,
      weight: 6,
      opacity: 0.98,
      lineCap: 'round',
      lineJoin: 'round',
    }).addTo(routeLayer);
  }

  const start = vis.start
    ? asLatLon([vis.start.lat, vis.start.lon])
    : routeCoords[0];
  const end = vis.end
    ? asLatLon([vis.end.lat, vis.end.lon])
    : routeCoords[routeCoords.length - 1];

  if (start) {
    L.marker(start, {
      icon: L.divIcon({
        className: 'bike-route-endpoint',
        html: '<div class="bike-route-start-dot"></div>',
        iconSize: [16, 16],
        iconAnchor: [8, 8],
      }),
    }).addTo(routeMap).bindTooltip('Start', {permanent: false});
  }
  if (end) {
    L.marker(end, {
      icon: L.divIcon({
        className: 'bike-route-endpoint',
        html: '<div class="bike-route-end-dot"></div>',
        iconSize: [16, 16],
        iconAnchor: [8, 8],
      }),
    }).addTo(routeMap).bindTooltip('Destination', {permanent: false});
  }

  // Replace the old dense arrow markers with a few small, subtle chevrons
  // spaced along the route. They indicate travel direction without looking
  // like turn symbols on every noded graph edge.
  function addDirectionChevron(position, bearing) {
    return L.marker(position, {
      icon: L.divIcon({
        className: 'bike-route-direction-chevron',
        html: '<span></span>',
        iconSize: [18, 18],
        iconAnchor: [9, 9],
      }),
      interactive: false,
      zIndexOffset: 700,
    }).addTo(routeMap);
  }

  // Inject a tiny one-time CSS rule for the chevrons/endpoints.
  if (!document.getElementById('bike-route-chat-style')) {
    const style = document.createElement('style');
    style.id = 'bike-route-chat-style';
    style.textContent = `
      .bike-route-direction-chevron span {
        display:block;
        width:0;
        height:0;
        margin:2px;
        border-top:7px solid transparent;
        border-bottom:7px solid transparent;
        border-left:11px solid #1d4ed8;
        filter: drop-shadow(0 0 1px #fff);
      }
      .bike-route-start-dot,
      .bike-route-end-dot {
        width:14px;height:14px;border-radius:50%;
        border:2px solid #fff;box-shadow:0 1px 3px rgba(0,0,0,.35);
      }
      .bike-route-start-dot { background:#16a34a; }
      .bike-route-end-dot { background:#dc2626; }
    `;
    document.head.appendChild(style);
  }

  // Rotate a small number of chevrons using the route geometry. Use
  // approximately evenly spaced positions and a local bearing so the
  // direction is clear without clutter.
  const arrowCount = Math.min(5, Math.max(2, Math.floor(routeCoords.length / 40)));
  for (let k = 1; k <= arrowCount; k++) {
    const idx = Math.max(1, Math.min(
      routeCoords.length - 2,
      Math.round(k * (routeCoords.length - 1) / (arrowCount + 1))
    ));
    const a = routeCoords[Math.max(0, idx - 2)];
    const b = routeCoords[Math.min(routeCoords.length - 1, idx + 2)];
    const bearing = bikeRouteBearing(a, b);
    const marker = addDirectionChevron(routeCoords[idx], bearing);
    const el = marker.getElement()?.querySelector('span');
    if (el) el.style.transform = `rotate(${bearing}deg)`;
  }

  const bounds = L.latLngBounds(routeCoords);
  if (start) bounds.extend(start);
  if (end) bounds.extend(end);
  if (bounds.isValid()) {
    routeMap.fitBounds(bounds, { padding: [20, 20], maxZoom: 16 });
  }

  // Legend now contains only layer families actually used by the route.
  const usedLabels = new Map();
  for (const feature of features) {
    const p = feature?.properties || {};
    const label = p.label || p.layer_type;
    if (label) usedLabels.set(label, p.color || '#666');
  }
  const legendEl = wrapper.querySelector('.bike-route-legend');
  if (usedLabels.size) {
    legendEl.innerHTML =
      '<div class="bike-route-legend-title">BikePGH infrastructure used</div>' +
      [...usedLabels.entries()]
        .map(([label, color]) => `<span><i style="background:${escapeHtml(color)}"></i>${escapeHtml(label)}</span>`)
        .join('');
  } else {
    legendEl.textContent = 'BikePGH route infrastructure';
  }

  setTimeout(() => {
    routeMap.invalidateSize();
    if (bounds.isValid()) {
      routeMap.fitBounds(bounds, {padding: [20, 20], maxZoom: 16});
    }
  }, 100);
}

async function sendGeneralMessage() {
  const input = document.getElementById('general-chat-input');
  const msg = input.value.trim();
  if (!msg) return;

  input.value = '';
  const sendBtn = document.getElementById('general-chat-send');
  sendBtn.disabled = true;

  appendMsg('general', 'user', msg);
  const typing = appendTyping('general');

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: msg,
        history: state.generalChatHistory,
        session_id: state.generalChatSessionId,
      }),
    });
    const data = await resp.json();
    typing.remove();
    if (!resp.ok) {
      throw new Error(data.detail || data.error || `Request failed (${resp.status})`);
    }

    const traceId = data.observability?.trace_id || null;
    const traceUrl = data.observability?.trace_url ||
      (traceId ? `/redirects/traces/${encodeURIComponent(traceId)}` : null);
    const assistantMsg = appendMsg(
      'general',
      'assistant',
      data.reply,
      traceUrl,
    );

    // Keep the exact trace metadata on the answer in local state too, so a
    // future re-render can recreate the same button instead of losing it.
    if (Array.isArray(data.history)) {
      state.generalChatHistory = data.history;
    }
    if (data.visualization && data.visualization.type === 'bike_route') {
      renderBikeRouteInChat(assistantMsg, data.visualization);
    }
  } catch (e) {
    typing.remove();
    appendMsg('general', 'assistant', `Error: ${e.message}`);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

document.getElementById('general-chat-send').addEventListener('click', sendGeneralMessage);
document.getElementById('general-chat-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendGeneralMessage(); }
});

/* ── Map layers (Crime / NRI) ────────────────────────────────────────────
   Exactly one optional overlay can be active at a time. Both layers are
   viewport-scoped: we request only what's within the current map bounds,
   and re-request (debounced) as the user pans/zooms, so this stays fast
   regardless of how much crime/NRI data is loaded server-side. */

const layerState = {
  active: null,       // null | 'crime' | 'nri' | 'bike'
  layerObj: null,      // the Leaflet layer currently on the map, if any
  fetchToken: 0,        // guards against a slow/stale request overwriting a newer one
};

function currentBboxQuery() {
  const b = map.getBounds();
  return `west=${b.getWest()}&south=${b.getSouth()}&east=${b.getEast()}&north=${b.getNorth()}`;
}

// Coarser grid cells when zoomed out, finer when zoomed in — keeps the
// server-side aggregation (see services/layers.py::get_crime_heatmap)
// resolved appropriately for whatever's actually on screen, and keeps a
// wide, zoomed-out view from needing so many grid cells that the
// MAX_GRID_CELLS safety cap starts silently dropping real areas. Reference:
// ~300m cells (0.003°) at zoom 14, halving/doubling per zoom level.
function gridDegForZoom(zoom) {
  const deg = 0.003 * Math.pow(2, 14 - zoom);
  return Math.min(0.5, Math.max(0.0005, deg));
}

function setLayerLoading(isLoading) {
  const el = document.getElementById('layer-status');
  if (el) el.textContent = isLoading ? 'Loading…' : '';
}

function clearActiveLayer() {
  if (layerState.layerObj) {
    map.removeLayer(layerState.layerObj);
    layerState.layerObj = null;
  }
  const legend = document.getElementById('layer-legend');
  if (legend) legend.style.display = 'none';
}

function renderLegend(kind, data) {
  const el = document.getElementById('layer-legend');
  if (!el) return;
  if (kind === 'crime') {
    el.innerHTML = `
      <div class="legend-title">Crime severity (weighted density)</div>
      <div class="legend-gradient"></div>
      <div class="legend-scale"><span>0</span><span>${data.max_weight.toFixed(1)}</span></div>
      <div class="legend-note">${data.incident_count.toLocaleString()} incident(s) in view — scale is relative to this view, not the whole map</div>
      ${data.truncated ? '<div class="legend-note">Zoom in for full detail</div>' : ''}
    `;
  } else if (kind === 'nri') {
    const ratings = ['Very Low', 'Relatively Low', 'Relatively Moderate', 'Relatively High', 'Very High'];
    el.innerHTML = `
      <div class="legend-title">FEMA Risk Rating</div>
      ${ratings.map(r => `<div class="legend-row"><span class="legend-swatch" style="background:${ratingColor(r)}"></span>${r}</div>`).join('')}
      ${data.warning ? `<div class="legend-note">${data.warning}</div>` : ''}
      ${data.truncated ? '<div class="legend-note">Zoom in to see all tracts</div>' : ''}
    `;
  } else if (kind === 'bike') {
    const order = ['bike_lanes','bikeable_sidewalks','cautionary_bike_route','on_street_bike_route','protected_bike_lanes','sharrows','trails'];
    el.innerHTML = `
      <div class="legend-title">Bike Lanes</div>
      ${order.map(k => `<div class="legend-row"><span class="legend-swatch bike-legend-line" style="background:${BIKE_LAYER_COLORS[k]}"></span>${BIKE_LAYER_LABELS[k]}</div>`).join('')}
      <div class="legend-note">${(data.feature_count || 0).toLocaleString()} feature(s) in view${data.truncated ? ' — zoom in for full detail' : ''}</div>
    `;
  }
  el.style.display = 'block';
}

const BIKE_LAYER_COLORS = {
  bike_lanes: '#4682b4',
  bikeable_sidewalks: '#add8e6',
  cautionary_bike_route: '#ff0000',
  on_street_bike_route: '#90ee90',
  protected_bike_lanes: '#006400',
  sharrows: '#ffa500',
  trails: '#ffc0cb',
};
const BIKE_LAYER_LABELS = {
  bike_lanes: 'Bike Lanes',
  bikeable_sidewalks: 'Bikeable Sidewalks',
  cautionary_bike_route: 'Cautionary Bike Route',
  on_street_bike_route: 'On Street Bike Route',
  protected_bike_lanes: 'Protected Bike Lanes',
  sharrows: 'Sharrows',
  trails: 'Trails',
};

async function refreshCrimeLayer() {
  if (typeof L.heatLayer !== 'function') {
    console.error('leaflet.heat did not load — check network access to unpkg.com');
    return;
  }
  const token = ++layerState.fetchToken;
  setLayerLoading(true);
  try {
    const zoom = map.getZoom();
    const gridDeg = gridDegForZoom(zoom);
    const resp = await fetch(`/api/layers/crime?${currentBboxQuery()}&grid_deg=${gridDeg}`);
    if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
    const data = await resp.json();
    if (token !== layerState.fetchToken) return; // a newer request has since superseded this one

    const heatPoints = data.points; // already [lat, lon, weight]
    const maxVal = Math.max(data.max_weight, 1);
    // maxZoom here is intentionally set to the CURRENT zoom, not a fixed
    // value. Leaflet.heat scales each point's intensity by
    // 1 / 2^(maxZoom - currentZoom) internally — with a fixed maxZoom that
    // means the exact same underlying data renders dimmer and dimmer the
    // further you zoom out below it (e.g. a fixed maxZoom of 17 leaves you
    // at a quarter intensity by zoom 15, 1/16 by zoom 13 — this is *why*
    // areas can look like "low crime" purely from the current zoom level).
    // Pinning maxZoom to the current zoom makes that factor always 1, so
    // the ONLY thing driving color intensity is `max` — our own
    // server-computed severity ceiling for whatever's actually in view
    // right now. The gradient (the color mapping itself) is untouched.
    if (layerState.active === 'crime' && layerState.layerObj) {
      layerState.layerObj.setLatLngs(heatPoints);
      layerState.layerObj.setOptions({ max: maxVal, maxZoom: zoom });
    } else {
      clearActiveLayer();
      layerState.layerObj = L.heatLayer(heatPoints, {
        radius: 18, blur: 22, maxZoom: zoom, max: maxVal,
        gradient: { 0.2: '#3b82f6', 0.4: '#eab308', 0.65: '#f97316', 1.0: '#ef4444' },
      }).addTo(map);
    }
    layerState.active = 'crime';
    renderLegend('crime', data);
  } catch (e) {
    console.error('Crime layer error:', e);
  } finally {
    if (token === layerState.fetchToken) setLayerLoading(false);
  }
}

async function refreshNriLayer() {
  const token = ++layerState.fetchToken;
  setLayerLoading(true);
  try {
    const resp = await fetch(`/api/layers/nri?${currentBboxQuery()}`);
    if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
    const data = await resp.json();
    if (token !== layerState.fetchToken) return;

    clearActiveLayer();
    layerState.layerObj = L.geoJSON(data, {
      style: f => ({
        color: '#fff', weight: 1, fillOpacity: 0.55,
        fillColor: ratingColor(f.properties.risk_ratng),
      }),
      onEachFeature: (f, lyr) => {
        const p = f.properties;
        const scoreTxt = (p.risk_score != null) ? Number(p.risk_score).toFixed(1) : '—';
        lyr.bindTooltip(
          `<b>${p.county_name || 'Unknown county'}, ${p.state_name || ''}</b><br>` +
          `Risk: ${p.risk_ratng || 'No data'} (${scoreTxt})`,
          { sticky: true }
        );
      },
    }).addTo(map);
    layerState.active = 'nri';
    renderLegend('nri', data);
  } catch (e) {
    console.error('NRI layer error:', e);
  } finally {
    if (token === layerState.fetchToken) setLayerLoading(false);
  }
}

async function refreshBikeLayer() {
  const token = ++layerState.fetchToken;
  setLayerLoading(true);
  try {
    const resp = await fetch(`/api/layers/bike?${currentBboxQuery()}&exclusive=true`);
    if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
    const data = await resp.json();
    if (token !== layerState.fetchToken) return;

    clearActiveLayer();
    layerState.layerObj = L.geoJSON(data, {
      style: f => ({
        color: f.properties?.color || BIKE_LAYER_COLORS[f.properties?.layer_type] || '#4f8ef7',
        weight: 5,
        opacity: 0.85,
      }),
      onEachFeature: (f, lyr) => {
        const p = f.properties || {};
        lyr.bindTooltip(
          `<b>${p.layer_label || 'Bike route'}</b>${p.city ? `<br>${p.city}` : ''}`,
          { sticky: true }
        );
      },
    }).addTo(map);
    layerState.active = 'bike';
    renderLegend('bike', data);
  } catch (e) {
    console.error('Bike Lanes layer error:', e);
  } finally {
    if (token === layerState.fetchToken) setLayerLoading(false);
  }
}

function setActiveLayer(kind) {
  document.querySelectorAll('.layer-option').forEach(b => {
    b.classList.toggle('active', b.dataset.layer === kind);
  });
  if (kind === 'none') {
    layerState.active = null;
    layerState.fetchToken++; // invalidate any in-flight request
    clearActiveLayer();
    setLayerLoading(false);
    return;
  }
  if (kind === 'crime') refreshCrimeLayer();
  else if (kind === 'nri') refreshNriLayer();
  else if (kind === 'bike') refreshBikeLayer();
}

// Re-fetch the active layer as the map pans/zooms (debounced).
let _layerMoveTimer = null;
map.on('moveend', () => {
  if (!layerState.active) return;
  clearTimeout(_layerMoveTimer);
  _layerMoveTimer = setTimeout(() => {
    if (layerState.active === 'crime') refreshCrimeLayer();
    else if (layerState.active === 'nri') refreshNriLayer();
    else if (layerState.active === 'bike') refreshBikeLayer();
  }, 400);
});

// Keep the crime layer's color scale in sync with zoom immediately, not
// just after the debounced re-fetch above completes — see the maxZoom note
// in refreshCrimeLayer(). This is a cheap client-side option update (no
// network call), so there's no reason to wait for the full data refresh.
map.on('zoomend', () => {
  if (layerState.active === 'crime' && layerState.layerObj) {
    layerState.layerObj.setOptions({ maxZoom: map.getZoom() });
  }
});


/* ── Bike route planner ───────────────────────────────────────────────── */
let bikeRouteLayer = null;

function clearBikeRoute() {
  if (bikeRouteLayer) {
    map.removeLayer(bikeRouteLayer);
    bikeRouteLayer = null;
  }
  const panel = document.getElementById('bike-route-result');
  if (panel) panel.innerHTML = '';
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, c => ({
    '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'
  }[c]));
}

function BikeRouteControl() {
  const Control = L.Control.extend({
    options: { position: 'topleft' },
    onAdd() {
      const div = L.DomUtil.create('div', 'bike-route-control');
      div.innerHTML = `
        <div class="bike-route-title">🚲 Find a bikeable route</div>
        <input id="bike-route-start" placeholder="Start: Mount Washington" />
        <input id="bike-route-end" placeholder="End: Point State Park" />
        <button id="bike-route-submit" type="button">Find Route</button>
        <div id="bike-route-status" class="route-status"></div>
        <div id="bike-route-result" class="route-status"></div>
        <div class="route-attribution">Uses OpenStreetMap, Nominatim, and Valhalla. Endpoints can be place names or addresses.</div>
      `;
      L.DomEvent.disableClickPropagation(div);
      L.DomEvent.disableScrollPropagation(div);
      div.querySelector('#bike-route-submit').addEventListener('click', findBikeRoute);
      return div;
    }
  });
  return Control;
}

async function findBikeRoute() {
  const start = document.getElementById('bike-route-start')?.value.trim();
  const end = document.getElementById('bike-route-end')?.value.trim();
  const status = document.getElementById('bike-route-status');
  const result = document.getElementById('bike-route-result');
  if (!start || !end) {
    status.textContent = 'Enter both a start and destination.';
    return;
  }

  clearBikeRoute();
  status.textContent = 'Finding places and routing…';
  try {
    const resp = await fetch('/api/bike/route', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({start, end, city: 'Pittsburgh, PA'})
    });
    const raw = await resp.text();
    let data = null;
    try { data = raw ? JSON.parse(raw) : {}; } catch (_) {}
    if (!resp.ok) throw new Error(data?.detail || raw || `HTTP ${resp.status}`);
    const coords = data.route.shape || [];
    if (coords.length < 2) throw new Error('No route geometry was returned.');

    bikeRouteLayer = L.polyline(coords, {
      color: '#2563eb',
      weight: 7,
      opacity: 0.9
    }).addTo(map);

    const bounds = bikeRouteLayer.getBounds();
    if (bounds.isValid()) map.fitBounds(bounds.pad(0.12));

    const summary = data.route.summary || {};
    const facilities = data.route.local_bike_facilities || {};
    const minutes = summary.time != null ? Math.round(Number(summary.time) / 60) : null;
    result.innerHTML =
      `<b>${escapeHtml(data.start.display_name)}</b> → <b>${escapeHtml(data.end.display_name)}</b><br>` +
      `${summary.length != null ? Number(summary.length).toFixed(1) + ' mi' : ''}` +
      `${minutes != null ? ` · ~${minutes} min` : ''}<br>` +
      `Mapped BikePGH facility overlap: ${Number(facilities.facility_overlap_pct || 0).toFixed(0)}%` +
      `${facilities.facility_segments?.length ? `<br>${facilities.facility_segments.map(x => escapeHtml(x.label)).join(', ')}` : ''}` +
      `<br><small>${escapeHtml(data.note || '')}</small>`;
    status.textContent = `Route found (${data.alternatives_considered || 1} route option(s) considered).`;
  } catch (e) {
    console.error(e);
    status.textContent = `Route error: ${e.message || e}`;
  }
}

const LayerControl = L.Control.extend({
  options: { position: 'topright' },
  onAdd() {
    const div = L.DomUtil.create('div', 'layer-toggle-control');
    div.innerHTML = `
      <div class="layer-toggle-title">Map Layer <span id="layer-status"></span></div>
      <div class="layer-toggle-buttons">
        <button class="layer-option active" data-layer="none" type="button">None</button>
        <button class="layer-option" data-layer="crime" type="button">Crime</button>
        <button class="layer-option" data-layer="nri" type="button">NRI</button>
        <button class="layer-option" data-layer="bike" type="button">Bike Lanes</button>
      </div>
      <div class="layer-toggle-legend" id="layer-legend" style="display:none"></div>
    `;
    L.DomEvent.disableClickPropagation(div);
    L.DomEvent.disableScrollPropagation(div);
    div.querySelectorAll('.layer-option').forEach(btn => {
      btn.addEventListener('click', () => setActiveLayer(btn.dataset.layer));
    });
    return div;
  },
});

/* ── Boot ────────────────────────────────────────────────────────────── */
loadHouses();
map.addControl(new LayerControl());
map.addControl(new BikeRouteControl());
