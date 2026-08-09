/* ── App state ───────────────────────────────────────────────────────── */
const state = {
  selectedHouseId: null,
  houseChatHistory: {},   // {house_id: [{role, content}]}
  generalChatHistory: [],
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
  if (r.includes('medium'))    return '#f59e0b';
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

function appendMsg(ctx, role, content) {
  const containerId = ctx === 'house' ? 'house-chat-messages' : 'general-chat-messages';
  const el = document.getElementById(containerId);
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  const html = typeof marked !== 'undefined' ? marked.parse(content) : content;
  div.innerHTML = `<div class="bubble">${html}</div>`;
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
      body: JSON.stringify({ message: msg, history: state.generalChatHistory }),
    });
    const data = await resp.json();
    typing.remove();
    appendMsg('general', 'assistant', data.reply);
    state.generalChatHistory = data.history;
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

/* ── Boot ────────────────────────────────────────────────────────────── */
loadHouses();
