/* ============================================================
   AuthentiCheck — Dashboard Logic
   ============================================================ */

/* ── Sample Data ── */

const reviews = [
  {
    id: 1,
    text: 'Ergonomic sa kamay, smooth ang scroll wheel at maganda ang clicks. Worth it talaga para sa presyo.',
    stars: 5, platform: 'Shopee', label: 'auth', clip: 0.91,
    probs: [0.85, 0.05, 0.05, 0.05], sentiment: 'Positive',
    note: 'High p_auth, consistent image, matching 5★ rating'
  },
  {
    id: 2,
    text: 'Panget ng mouse, hindi gumagana yung bluetooth agad nung pagbukas ko! Wag kayo umorder dito sayang pera.',
    stars: 5, platform: 'Lazada', label: 'dec', clip: 0.88,
    probs: [0.82, 0.08, 0.05, 0.05], sentiment: '—',
    note: 'XGBoost detected extreme rating mismatch (5★ rating vs highly negative text)'
  },
  {
    id: 3,
    text: 'Ang ganda ng pagkain doon sa restaurant malapit sa mall.',
    stars: 5, platform: 'Shopee', label: 'irr', clip: 0.09,
    probs: [0.05, 0.02, 0.92, 0.01], sentiment: '—',
    note: 'High p_irr & low S_clip (0.09) — off-topic food review for a mouse'
  },
  {
    id: 4,
    text: 'Okay. Good.',
    stars: 4, platform: 'Lazada', label: 'liv', clip: 0.77,
    probs: [0.05, 0.04, 0.10, 0.81], sentiment: '—',
    note: 'High p_LIV probability — vague text, lacks informational value'
  },
  {
    id: 5,
    text: 'Solid ang build quality, medyo mahal pero sulit. Battery life okay lang, sana type-C na.',
    stars: 4, platform: 'Shopee', label: 'auth', clip: 0.86,
    probs: [0.74, 0.11, 0.05, 0.10], sentiment: 'Mixed',
    note: 'Valid mixed sentiment review with consistent rating and image'
  },
  {
    id: 6,
    text: 'Best mouse ever!!! 10/10 would recommend to everyone!!! I love this so much buy it now!!!!!',
    stars: 5, platform: 'Lazada', label: 'dec', clip: 0.12,
    probs: [0.20, 0.75, 0.03, 0.02], sentiment: '—',
    note: 'High p_dec (artificially positive text) paired with unaligned image (S_clip 0.12)'
  },
  {
    id: 7,
    text: 'Di ko pa nagagamit pero maganda packaging.',
    stars: 5, platform: 'Shopee', label: 'liv', clip: 0.79,
    probs: [0.15, 0.05, 0.12, 0.68], sentiment: '—',
    note: 'High p_LIV — review only discusses packaging, no product experience'
  },
  {
    id: 8,
    text: 'Sira ang box pagdating. Nagrereklamo na ko sa courier. Ref no. 28812.',
    stars: 1, platform: 'Lazada', label: 'irr', clip: 0.13,
    probs: [0.10, 0.03, 0.77, 0.10], sentiment: '—',
    note: 'High p_irr — logistics/courier complaint, not a product review'
  },
];

const absa = [
  {
    aspect: 'Ergonomics', pos: 312, neu: 48, neg: 21,
    samples: [
      'Grabe ang ganda sa kamay, hindi nakakangawit kahit matagal gamitin.',
      'Medyo malaki para sa maliit na kamay.',
      'Sumasakit wrist ko after an hour, not recommended.'
    ]
  },
  {
    aspect: 'Connectivity / Bluetooth', pos: 198, neu: 61, neg: 55,
    samples: [
      'Mabilis mag-connect at stable ang connection!',
      'Minsan nade-disconnect pero okay lang.',
      'Laggy ang bluetooth, hindi magamit pang-games.'
    ]
  },
  {
    aspect: 'Battery life', pos: 410, neu: 90, neg: 38,
    samples: [
      'Isang charge lang umaabot ng ilang weeks!',
      'Okay lang battery life, standard lang.',
      'Mabilis maubos battery, kailangan lagi i-charge.'
    ]
  },
  {
    aspect: 'Price / value', pos: 502, neu: 72, neg: 14,
    samples: [
      'Worth it talaga sa presyo, premium feel no regrets!',
      'Medyo mahal pero okay na rin.',
      'Mas mahal kaysa sa expected, hindi sulit.'
    ]
  },
  {
    aspect: 'Build quality', pos: 244, neu: 55, neg: 43,
    samples: [
      'Solid ang build, hindi parang mumurahing plastic.',
      'Plastic pero okay naman ang weight.',
      'Ang nipis ng plastic material, parang madaling mabasag.'
    ]
  },
  {
    aspect: 'Scroll wheel', pos: 287, neu: 44, neg: 29,
    samples: [
      'Ang smooth ng infinite scroll! Super satisfying.',
      'Okay naman ang scroll wheel.',
      'Medyo maingay at magaspang yung scroll wheel.'
    ]
  },
  {
    aspect: 'Click feel / Buttons', pos: 198, neu: 82, neg: 12,
    samples: [
      'Silent clicks and very responsive side buttons!',
      'Normal clicks, hindi silent.',
      'Matigas i-click yung left button, nakakangawit.'
    ]
  },
  {
    aspect: 'Shipping / delivery', pos: 155, neu: 60, neg: 24,
    samples: [
      'Mabilis dumating, 2 days lang nandito na!',
      'Standard shipping time.',
      'Matagal dumating, umabot ng 2 weeks.'
    ]
  },
];

/* ── State ── */
let dashFilter       = 'all';
let selectedDash     = null;
let classifyFilter   = 'all';
let selectedClassify = null;
let uploadedImages   = [];

/* ── Helpers ── */

function lbl(l) {
  return { auth: 'auth', liv: 'liv', irr: 'irr', dec: 'dec' }[l] || 'auth';
}

function lblTxt(l) {
  return { auth: 'Authentic', liv: 'Vague / LIV', irr: 'Irrelevant', dec: 'Deceptive' }[l] || l;
}

function stars(n) {
  return '★'.repeat(n) + '☆'.repeat(5 - n);
}

/* ── Toast ── */
function showToast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 2200);
}

/* ── Page Navigation ── */
function goPage(id) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById('page-' + id).classList.add('active');

  document.querySelectorAll('.sidebar-nav').forEach(n => n.classList.remove('active'));
  document.getElementById('nav-' + id).classList.add('active');

  if (id === 'classify') renderClassify();
  if (id === 'sentiment') renderABSA();
}

/* ── Dashboard ── */

function renderDash() {
  const filtered = dashFilter === 'all'
    ? reviews
    : reviews.filter(r => r.label === dashFilter);

  document.getElementById('dash-review-list').innerHTML = filtered.map(r => `
    <div class="rev ${selectedDash === r.id ? 'selected' : ''}" onclick="selectDash(${r.id})">
      <div class="rev-meta">
        <div style="display:flex;align-items:center;gap:6px;">
          <span class="stars" style="font-size:12px;">${stars(r.stars)}</span>
          <span style="font-size:11px;color:var(--text3);">${r.platform}</span>
        </div>
        <span class="badge ${lbl(r.label)}">${lblTxt(r.label)}</span>
      </div>
      <div class="rev-text">${r.text}</div>
      ${r.note ? `<div class="rev-note">${r.note}</div>` : ''}
    </div>
  `).join('');
}

function selectDash(id) {
  selectedDash = id;
  renderDash();

  const r = reviews.find(x => x.id === id);
  document.getElementById('dash-detail').innerHTML = `
    <div class="card-title">Review #${r.id}</div>
    <div class="detail-row">
      <span class="detail-label">Classification</span>
      <span class="badge ${lbl(r.label)}">${lblTxt(r.label)}</span>
    </div>
    <div class="detail-row" style="flex-direction:column;align-items:flex-start;gap:6px;">
      <span class="detail-label">DOST-RoBERTa Probabilities</span>
      <div style="display:flex;gap:4px;flex-wrap:wrap;">
        <span class="chip">p_auth: ${r.probs[0]}</span>
        <span class="chip">p_dec: ${r.probs[1]}</span>
        <span class="chip">p_irr: ${r.probs[2]}</span>
        <span class="chip">p_liv: ${r.probs[3]}</span>
      </div>
    </div>
    <div class="detail-row">
      <span class="detail-label">M-CLIP score</span>
      <span class="chip">${r.clip} — ${r.clip > 0.5 ? 'consistent' : 'mismatch'}</span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Star rating</span>
      <span class="stars" style="font-size:13px;">${stars(r.stars)}</span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Platform</span>
      <span class="chip">${r.platform}</span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Sentiment</span>
      <span class="chip">${r.sentiment}</span>
    </div>
  `;
}

function setDashFilter(f, el) {
  dashFilter = f;
  selectedDash = null;
  document.querySelectorAll('#dash-ftabs .ftab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('dash-detail').innerHTML =
    '<div class="card-title">Review detail</div>' +
    '<div style="font-size:12px;color:var(--text3);">Click a review to see its analysis.</div>';
  renderDash();
}

function filterDash(f, el) {
  document.querySelectorAll('#dash-metrics .metric').forEach(m => m.classList.remove('selected'));
  el.classList.add('selected');
  dashFilter = f;
  selectedDash = null;

  const tabMap = { all: 0, auth: 1, dec: 2, liv: 3, irr: 4 };
  document.querySelectorAll('#dash-ftabs .ftab').forEach((t, i) =>
    t.classList.toggle('active', i === tabMap[f] || (f === 'all' && i === 0))
  );

  document.getElementById('dash-detail').innerHTML =
    '<div class="card-title">Review detail</div>' +
    '<div style="font-size:12px;color:var(--text3);">Click a review to see its analysis.</div>';
  renderDash();
}

/* ── Classification ── */

function renderClassify() {
  const filtered = classifyFilter === 'all'
    ? reviews
    : reviews.filter(r => r.label === classifyFilter);

  document.getElementById('classify-list').innerHTML = filtered.map(r => `
    <div class="rev ${selectedClassify === r.id ? 'selected' : ''}" onclick="selectClassify(${r.id})">
      <div class="rev-meta">
        <div style="display:flex;align-items:center;gap:6px;">
          <span class="stars" style="font-size:12px;">${stars(r.stars)}</span>
          <span style="font-size:11px;color:var(--text3);">${r.platform}</span>
          <span style="font-size:11px;color:var(--text3);">CLIP: ${r.clip}</span>
        </div>
        <span class="badge ${lbl(r.label)}">${lblTxt(r.label)}</span>
      </div>
      <div class="rev-text">${r.text}</div>
      ${r.note ? `<div class="rev-note">${r.note}</div>` : ''}
    </div>
  `).join('');
}

function selectClassify(id) {
  selectedClassify = id;
  renderClassify();

  const r = reviews.find(x => x.id === id);
  document.getElementById('classify-detail').innerHTML = `
    <div class="card-title">Review #${r.id} — full analysis</div>
    <div class="detail-row">
      <span class="detail-label">Final classification</span>
      <span class="badge ${lbl(r.label)}">${lblTxt(r.label)}</span>
    </div>
    <div class="detail-row" style="flex-direction:column;align-items:flex-start;gap:6px;">
      <span class="detail-label">DOST-RoBERTa Probabilities</span>
      <div style="display:flex;gap:4px;flex-wrap:wrap;">
        <span class="chip">p_auth: ${r.probs[0]}</span>
        <span class="chip">p_dec: ${r.probs[1]}</span>
        <span class="chip">p_irr: ${r.probs[2]}</span>
        <span class="chip">p_liv: ${r.probs[3]}</span>
      </div>
    </div>
    <div class="detail-row">
      <span class="detail-label">M-CLIP score</span>
      <span class="chip">${r.clip}</span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Star rating</span>
      <span class="stars" style="font-size:13px;">${stars(r.stars)}</span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Platform</span>
      <span class="chip">${r.platform}</span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Forwarded to ABSA</span>
      <span style="font-size:12px;">
        ${r.label === 'auth'
          ? '<span style="color:var(--green)">Yes</span>'
          : '<span style="color:var(--red)">No — filtered out</span>'}
      </span>
    </div>
  `;
}

function setClassifyFilter(f, el) {
  classifyFilter = f;
  selectedClassify = null;
  document.querySelectorAll('#classify-tabs .ftab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('classify-detail').innerHTML =
    '<div class="card-title">Review detail</div>' +
    '<div style="font-size:12px;color:var(--text3);">Select a review to inspect its scores.</div>';
  renderClassify();
}

/* ── Sentiment (ABSA) ── */

function renderABSA() {
  document.getElementById('absa-list').innerHTML = absa.map((a, i) => `
    <div class="absa-row" onclick="showABSADetail(${i})">
      <span>${a.aspect}</span>
      <div class="pills">
        <span class="pill pos">+${a.pos}</span>
        <span class="pill neu">${a.neu}</span>
        <span class="pill neg">-${a.neg}</span>
      </div>
    </div>
  `).join('');
}

function showABSADetail(i) {
  const a = absa[i];
  const total = a.pos + a.neu + a.neg;

  document.getElementById('absa-detail').innerHTML = `
    <div class="card-title">${a.aspect}</div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px;">
      <div style="background:#EAF3DE;border-radius:var(--radius);padding:10px;text-align:center;">
        <div style="font-size:20px;font-weight:600;color:#27500A;">${Math.round(a.pos / total * 100)}%</div>
        <div style="font-size:11px;color:#3B6D11;">Positive (${a.pos})</div>
      </div>
      <div style="background:#F1EFE8;border-radius:var(--radius);padding:10px;text-align:center;">
        <div style="font-size:20px;font-weight:600;color:#444441;">${Math.round(a.neu / total * 100)}%</div>
        <div style="font-size:11px;color:#5F5E5A;">Neutral (${a.neu})</div>
      </div>
      <div style="background:var(--red-light);border-radius:var(--radius);padding:10px;text-align:center;">
        <div style="font-size:20px;font-weight:600;color:#791F1F;">${Math.round(a.neg / total * 100)}%</div>
        <div style="font-size:11px;color:var(--red);">Negative (${a.neg})</div>
      </div>
    </div>
    <div style="font-size:11px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px;">
      Sample excerpts
    </div>
    ${a.samples.map(s => `
      <div style="font-size:12px;padding:8px 10px;background:var(--surface2);border-radius:var(--radius);margin-bottom:6px;color:var(--text2);line-height:1.5;">
        "${s}"
      </div>
    `).join('')}
  `;
}

/* ── Image Handling (Upload page) ── */

function handleImages(input) {
  Array.from(input.files).forEach(f => {
    const reader = new FileReader();
    reader.onload = e => {
      uploadedImages.push({ name: f.name, url: e.target.result });
      renderImgGrid();
    };
    reader.readAsDataURL(f);
  });
  input.value = '';
}

function renderImgGrid() {
  const grid = document.getElementById('img-preview-grid');
  const thumbs = uploadedImages.map((img, i) => `
    <div class="img-thumb-wrap">
      <img class="img-thumb" src="${img.url}" alt="${img.name}">
      <button class="img-remove" onclick="removeImg(${i})" title="Remove">&times;</button>
    </div>
  `).join('');

  const addMore = `
    <div class="img-placeholder" onclick="document.getElementById('img-input').click()">
      <i class="ti ti-plus"></i>
      <span>Add more</span>
    </div>
  `;

  grid.innerHTML = thumbs + addMore;

  document.getElementById('img-count').textContent = uploadedImages.length
    ? `${uploadedImages.length} image${uploadedImages.length > 1 ? 's' : ''} uploaded — these will be matched against review text using M-CLIP`
    : '';
}

function removeImg(i) {
  uploadedImages.splice(i, 1);
  if (uploadedImages.length === 0) {
    document.getElementById('img-preview-grid').innerHTML = '';
    document.getElementById('img-count').textContent = '';
  } else {
    renderImgGrid();
  }
}

function handleSingleImg(input) {
  const f = input.files[0];
  if (!f) return;

  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById('single-img-preview').innerHTML = `
      <div class="img-thumb-wrap" style="width:80px;height:80px;">
        <img class="img-thumb" src="${e.target.result}" style="width:80px;height:80px;">
        <button class="img-remove" onclick="clearSingleImg()">&times;</button>
      </div>
    `;
  };
  reader.readAsDataURL(f);
}

function clearSingleImg() {
  document.getElementById('single-img-input').value = '';
  document.getElementById('single-img-preview').innerHTML = `
    <div class="img-placeholder" onclick="document.getElementById('single-img-input').click()">
      <i class="ti ti-photo-plus"></i>
      <span>Add photo</span>
    </div>
  `;
}

/* ── CSV Handling ── */

function handleCSV(input) {
  const f = input.files[0];
  if (!f) return;

  const statusEl = document.getElementById('csv-status');
  statusEl.style.color = 'var(--primary)';
  statusEl.textContent = 'Reading file…';

  const reader = new FileReader();
  reader.onload = () => {
    const lineCount = reader.result.split('\n').filter(l => l.trim()).length;
    statusEl.style.color = 'var(--green)';
    statusEl.textContent = `✓ ${f.name} loaded — ${lineCount - 1} reviews found. Go to Classification to view.`;
    setTimeout(() => goPage('classify'), 1800);
  };
  reader.readAsText(f);
  input.value = '';
}

function dropCSV(e) {
  e.preventDefault();
  dragLeave(e);
  const f = e.dataTransfer.files[0];
  if (f && f.name.endsWith('.csv')) {
    handleCSV({ files: [f] });
  }
}

function dropImages(e) {
  e.preventDefault();
  dragLeave(e);
  Array.from(e.dataTransfer.files)
    .filter(f => f.type.startsWith('image/'))
    .forEach(f => {
      const reader = new FileReader();
      reader.onload = ev => {
        uploadedImages.push({ name: f.name, url: ev.target.result });
        renderImgGrid();
      };
      reader.readAsDataURL(f);
    });
}

function dragOver(e)  { e.preventDefault(); e.currentTarget.classList.add('dragover'); }
function dragLeave(e) { e.currentTarget.classList.remove('dragover'); }

function fakeLoad() {
  showToast('Sample dataset loaded!');
  setTimeout(() => goPage('dashboard'), 800);
}

/* ── Single Review Classifier ── */

function classifyTest() {
  const text    = document.getElementById('test-text').value.trim();
  const starsN  = parseInt(document.getElementById('test-stars').value);
  const resultEl = document.getElementById('test-result');

  if (!text) {
    resultEl.innerHTML = '<span style="color:var(--red);font-size:12px;">Please enter review text first.</span>';
    return;
  }

  resultEl.innerHTML = '<span style="font-size:12px;color:var(--text3);">Analyzing…</span>';

  setTimeout(() => {
    const words  = text.split(/\s+/);
    const hasImg = document.getElementById('single-img-input').files.length > 0;

    // Simulate M-CLIP image-text similarity
    const clip = hasImg
      ? parseFloat((0.75 + Math.random() * 0.20).toFixed(2))
      : parseFloat((0.20 + Math.random() * 0.75).toFixed(2));

    const hasNeg = /(sira|hindi|disappointing|broken|bad|pangit|ayaw|wala|mahal|late|mabagal)/i.test(text);
    const hasPos = /(ganda|sulit|worth|okay|solid|maganda|maayos|mabilis|magaling|perfect|goods|best|love)/i.test(text);

    // Simulate DOST-RoBERTa text classification probabilities
    const probs = { auth: 0.10, dec: 0.10, irr: 0.10, liv: 0.10 };
    let textPred = 'auth';

    if (words.length <= 4)                                             textPred = 'liv';
    else if (/(delivery|seller|shipping|courier|ref no)/i.test(text)) textPred = 'irr';
    else if (/(best product ever|10\/10|buy it now|scam)/i.test(text)) textPred = 'dec';
    else                                                               textPred = 'auth';

    probs[textPred] = 0.65 + Math.random() * 0.15;
    const remainder = 1 - probs[textPred];
    const others = Object.keys(probs).filter(k => k !== textPred);
    probs[others[0]] = parseFloat((remainder * 0.50).toFixed(2));
    probs[others[1]] = parseFloat((remainder * 0.30).toFixed(2));
    probs[others[2]] = parseFloat((1 - probs[textPred] - probs[others[0]] - probs[others[1]]).toFixed(2));
    probs[textPred]  = parseFloat(probs[textPred].toFixed(2));

    // Simulate XGBoost late-fusion decision
    let finalLabel     = textPred;
    let conflictReason = '';

    const isNegativeText = hasNeg && !hasPos;
    const isPositiveText = hasPos && !hasNeg;

    if ((isNegativeText && starsN >= 4) || (isPositiveText && starsN <= 2)) {
      finalLabel     = 'dec';
      conflictReason = 'XGBoost resolved conflict: extreme rating mismatch detected.';
    } else if (textPred === 'auth' && clip < 0.3) {
      finalLabel     = 'irr';
      conflictReason = 'XGBoost resolved conflict: unaligned image similarity score.';
    }

    const explanations = {
      auth: 'This review appears genuine — it has sufficient informational value, consistent rating, and no multimodal mismatch.',
      liv:  'This review has high p_LIV probability. Very short, lacks product-specific detail, or lacks informational value.',
      irr:  'This review appears off-topic. High p_irr or low CLIP score suggests image-text mismatch or unrelated content.',
      dec:  'Possible deceptive review detected. High p_dec or extreme rating mismatch detected by XGBoost.'
    };

    resultEl.innerHTML = `
      <div style="border:0.5px solid var(--border);border-radius:var(--radius);padding:12px;">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
          <span style="font-weight:600;font-size:13px;">Result</span>
          <span class="badge ${finalLabel}">${lblTxt(finalLabel)}</span>
        </div>
        <div class="detail-row" style="flex-direction:column;align-items:flex-start;gap:6px;">
          <span class="detail-label">DOST-RoBERTa Probabilities (Text only)</span>
          <div style="display:flex;gap:4px;flex-wrap:wrap;">
            <span class="chip">p_auth: ${probs.auth.toFixed(2)}</span>
            <span class="chip">p_dec: ${probs.dec.toFixed(2)}</span>
            <span class="chip">p_irr: ${probs.irr.toFixed(2)}</span>
            <span class="chip">p_liv: ${probs.liv.toFixed(2)}</span>
          </div>
        </div>
        <div class="detail-row">
          <span class="detail-label">M-CLIP score (simulated)</span>
          <span class="chip">${clip.toFixed(2)} — ${clip > 0.5 ? 'image consistent' : 'possible mismatch'}</span>
        </div>
        <div class="detail-row">
          <span class="detail-label">Image uploaded</span>
          <span class="chip">${hasImg ? 'Yes' : 'No'}</span>
        </div>
        <div class="detail-row">
          <span class="detail-label">Stars</span>
          <span style="color:#BA7517;">${'★'.repeat(starsN) + '☆'.repeat(5 - starsN)}</span>
        </div>
        ${conflictReason
          ? `<div style="margin-top:10px;font-size:11px;color:var(--red);font-weight:500;">${conflictReason}</div>`
          : ''}
        <div style="margin-top:10px;font-size:12px;padding:8px 10px;border-radius:var(--radius);background:var(--surface2);color:var(--text2);">
          ${explanations[finalLabel]}
        </div>
      </div>
    `;
  }, 900);
}

/* ── Init ── */
renderDash();

if (window.location.hash === '#upload') {
  goPage('upload');
}