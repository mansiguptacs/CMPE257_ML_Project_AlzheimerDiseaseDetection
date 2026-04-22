/* ═══════════════════════════════════════════════════════
   NeuroScan AI — Main Application Logic
═══════════════════════════════════════════════════════ */

const API = "http://localhost:8000";

// ── Feature category definitions ──────────────────────
const FEATURE_GROUPS = {
  clinical: [
    "M/F","Age","Educ","SES","MMSE","eTIV","nWBV","ASF","Delay"
  ],
  tissue: [
    "csf_vol_mm3","gm_vol_mm3","wm_vol_mm3",
    "csf_voxels","gm_voxels","wm_voxels",
    "brain_parenchyma_vol_mm3","total_segmented_vol_mm3",
    "csf_frac","gm_frac","wm_frac","brain_parenchyma_frac",
    "csf_to_brain_ratio","gm_wm_ratio","reconstructed_nwbv",
    "brain_parenchyma_to_etiv","gm_to_etiv","wm_to_etiv","csf_to_etiv",
    "nwbv_abs_error"
  ],
  regional: null   // everything else = regional
};

// ── State ─────────────────────────────────────────────
const state = {
  patientInfo: null,
  modes: [],
  selectedMode: null,
  lastResult: null,
  fiChart: null,
  activeFeatureGroup: "clinical",
};

// ── Cached DOM refs ───────────────────────────────────
const $ = id => document.getElementById(id);

// ═════════════════════════════════════════════════════
//  INIT
// ═════════════════════════════════════════════════════
async function init() {
  await Promise.all([loadPatient(), loadModes()]);
  setupFeatureTabs();
}

// ═════════════════════════════════════════════════════
//  PATIENT INFO
// ═════════════════════════════════════════════════════
async function loadPatient() {
  try {
    const res  = await fetch(`${API}/patient-info`);
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    state.patientInfo = data;
    renderPatientCard(data);
  } catch (e) {
    console.error("Failed to load patient info:", e);
    $("patient-grid").innerHTML = `<div style="color:var(--accent-red);font-size:.8rem;grid-column:1/-1;">
      ⚠ Could not reach backend at ${API}. Is the server running?</div>`;
  }
}

function renderPatientCard(d) {
  $("session-id-badge").textContent = d.session_id;

  const fields = [
    { label: "Age",       value: d.age ? `${d.age} yrs` : "—"       },
    { label: "Sex",       value: d.sex || "—"                        },
    { label: "Education", value: d.education ? `${d.education} yrs` : "—" },
    { label: "SES",       value: d.ses ?? "—"                        },
    { label: "MMSE",      value: d.mmse ?? "—"                       },
    { label: "eTIV (mm³)",value: d.etiv ? fmtNum(d.etiv, 0) : "—"   },
    { label: "nWBV",      value: d.nwbv ? d.nwbv.toFixed(4) : "—"   },
    { label: "ASF",       value: d.asf  ? d.asf.toFixed(4) : "—"    },
  ];

  $("patient-grid").innerHTML = fields.map(f => `
    <div class="patient-stat">
      <div class="ps-label">${f.label}</div>
      <div class="ps-value">${f.value}</div>
    </div>`).join("");

  // Ground truth
  const gtRow = $("ground-truth-row");
  $("gt-value").textContent = `CDR ${d.cdr ?? "—"} · ${d.ground_truth_label}`;
  gtRow.style.display = "flex";
}

// ═════════════════════════════════════════════════════
//  ABLATION MODES
// ═════════════════════════════════════════════════════
async function loadModes() {
  try {
    const res   = await fetch(`${API}/modes`);
    const modes = await res.json();
    state.modes = modes;
    renderModes(modes);
    // Select first mode by default
    selectMode(modes[0].key);
  } catch (e) {
    console.error("Failed to load modes:", e);
  }
}

function renderModes(modes) {
  const container = $("mode-options");
  const icons = { full: "🔬", no_mmse: "🧪", imaging_only: "🗺" };
  container.innerHTML = modes.map(m => `
    <div class="mode-option" id="mode-opt-${m.key}" onclick="selectMode('${m.key}')" role="radio" tabindex="0">
      <div class="mode-radio"></div>
      <div class="mode-text">${icons[m.key] || "▸"} ${m.label}</div>
    </div>`).join("");
}

function selectMode(key) {
  state.selectedMode = key;
  document.querySelectorAll(".mode-option").forEach(el => el.classList.remove("selected"));
  const opt = $(`mode-opt-${key}`);
  if (opt) opt.classList.add("selected");
  $("predict-btn").disabled = false;
}

// ═════════════════════════════════════════════════════
//  PREDICT
// ═════════════════════════════════════════════════════
$("predict-btn").addEventListener("click", runPrediction);

async function runPrediction() {
  const mode = state.selectedMode;
  if (!mode) return;

  const btn = $("predict-btn");
  btn.classList.add("loading");
  btn.disabled = true;

  try {
    const res = await fetch(`${API}/predict/${mode}`, { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    const result = await res.json();
    state.lastResult = result;
    renderResult(result);
    renderFeatureExplorer(result.features, state.activeFeatureGroup);
  } catch (e) {
    console.error("Prediction failed:", e);
    alert(`Prediction error: ${e.message}`);
  } finally {
    btn.classList.remove("loading");
    btn.disabled = false;
  }
}

// ═════════════════════════════════════════════════════
//  RENDER RESULT
// ═════════════════════════════════════════════════════
function renderResult(r) {
  $("result-placeholder").style.display = "none";
  const rc = $("result-content");
  rc.style.display = "block";

  // Verdict badge
  const isDemented = r.prediction === 1;
  const badge      = $("verdict-badge");
  badge.className  = `verdict-badge ${isDemented ? "demented" : "non-demented"}`;
  $("verdict-icon").textContent = isDemented ? "🔴" : "🟢";
  $("verdict-text").textContent = r.prediction_label;
  $("mode-label-result").textContent = r.mode_label;

  // Confidence gauge
  const conf = r.prob_demented;
  animateGauge(conf);
  $("gauge-val").textContent = `${(conf * 100).toFixed(1)}%`;

  // Probability bars
  const pctND = (r.prob_non_demented * 100).toFixed(1);
  const pctD  = (r.prob_demented     * 100).toFixed(1);
  $("prob-nd-bar").style.width = `${pctND}%`;
  $("prob-d-bar").style.width  = `${pctD}%`;
  $("prob-nd-val").textContent = `${pctND}%`;
  $("prob-d-val").textContent  = `${pctD}%`;
  $("features-used-note").textContent = `${r.num_features_used} features used in this mode`;

  // Feature importance chart
  renderFIChart(r.feature_importance.slice(0, 15));
}

// ── Gauge animation ────────────────────────────────────
function animateGauge(prob) {
  // Arc length of the semicircle (π * r = π * 80 ≈ 251.2)
  const totalArc = 251.2;
  const offset   = totalArc * (1 - prob);
  $("gauge-arc").style.strokeDashoffset = offset;

  // Needle: -90° → 0° = 0 prob, +90° = 1.0 prob
  const deg = -90 + prob * 180;
  $("gauge-needle").style.transform = `rotate(${deg}deg)`;
}

// ── Feature importance chart ───────────────────────────
function renderFIChart(fiData) {
  const labels = fiData.map(f => f.feature);
  const values = fiData.map(f => parseFloat(f.importance.toFixed(4)));

  // Colour bars by feature source
  const colors = labels.map(l => {
    if (/hippocampus|ventricle|entorhinal|temporal/.test(l)) return "rgba(124,58,237,0.75)";
    if (/csf_|gm_|wm_|brain_parenchyma|reconstructed|_frac|_ratio|_to_etiv/.test(l)) return "rgba(14,165,233,0.75)";
    return "rgba(59,130,246,0.75)";
  });

  if (state.fiChart) state.fiChart.destroy();

  state.fiChart = new Chart($("fi-chart"), {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Importance",
        data: values,
        backgroundColor: colors,
        borderColor: colors.map(c => c.replace("0.75", "1")),
        borderWidth: 1,
        borderRadius: 4,
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 700, easing: "easeOutQuart" },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#ffffff",
          borderColor: "#e2e8f0",
          borderWidth: 1,
          bodyColor: "#0f172a",
          titleColor: "#475569",
          boxShadowOffsetX: 2,
          boxShadowOffsetY: 2,
          boxShadowBlur: 8,
          callbacks: {
            label: ctx => ` ${(ctx.parsed.x * 100).toFixed(2)}%`
          }
        }
      },
      scales: {
        x: {
          ticks: { color: "#94a3b8", font: { size: 10 } },
          grid:  { color: "#f1f5f9" },
        },
        y: {
          ticks: {
            color: "#475569",
            font: { family: "'JetBrains Mono', monospace", size: 10 },
            maxTicksLimit: 15,
          },
          grid: { display: false },
        }
      }
    }
  });

  // Chart legend
  const legendEl = document.createElement("div");
  legendEl.style.cssText = "display:flex;gap:12px;flex-wrap:wrap;margin-top:6px;font-size:0.66rem;color:#94a3b8;";
  legendEl.innerHTML = [
    ['rgba(59,130,246,0.75)', 'Original Clinical'],
    ['rgba(14,165,233,0.75)', 'Tissue Features'],
    ['rgba(124,58,237,0.75)', 'Regional ROI'],
  ].map(([c,l]) => `<span style="display:flex;align-items:center;gap:5px;">
    <span style="width:10px;height:10px;border-radius:2px;background:${c};display:inline-block;"></span>${l}</span>`).join("");

  const section = document.querySelector(".chart-section");
  const existing = section.querySelector(".chart-legend");
  if (existing) existing.remove();
  legendEl.className = "chart-legend";
  section.appendChild(legendEl);
}

// ═════════════════════════════════════════════════════
//  FEATURE EXPLORER
// ═════════════════════════════════════════════════════
function setupFeatureTabs() {
  document.querySelectorAll(".feat-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".feat-tab").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      state.activeFeatureGroup = tab.dataset.group;
      if (state.lastResult) {
        renderFeatureExplorer(state.lastResult.features, state.activeFeatureGroup);
      }
    });
  });
}

function renderFeatureExplorer(features, group) {
  if (!features) return;

  const allKeys  = Object.keys(features);
  let filteredKeys;

  if (group === "clinical") {
    filteredKeys = FEATURE_GROUPS.clinical.filter(k => k in features);
  } else if (group === "tissue") {
    filteredKeys = FEATURE_GROUPS.tissue.filter(k => k in features);
  } else {
    // regional = everything not in clinical or tissue
    const exclude = new Set([...FEATURE_GROUPS.clinical, ...FEATURE_GROUPS.tissue]);
    filteredKeys  = allKeys.filter(k => !exclude.has(k));
  }

  if (filteredKeys.length === 0) {
    $("feat-list").innerHTML = `<div style="color:var(--text-muted);font-size:.8rem;padding:12px;">
      No features available in this mode for the selected category.</div>`;
    return;
  }

  $("feat-list").innerHTML = filteredKeys.map(k => {
    const val = features[k];
    const displayVal = val === null || val === undefined
      ? '<span class="feat-val feat-null">—</span>'
      : `<span class="feat-val">${fmtVal(k, val)}</span>`;
    return `<div class="feat-item">
      <span class="feat-name" title="${k}">${k}</span>
      ${displayVal}
    </div>`;
  }).join("");
}

// ═════════════════════════════════════════════════════
//  UTILITIES
// ═════════════════════════════════════════════════════
function fmtNum(n, decimals = 2) {
  if (n === null || n === undefined) return "—";
  return Number(n).toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function fmtVal(key, val) {
  if (val === null || val === undefined) return "—";
  if (typeof val === "string") return val;
  // Volume columns → comma-sep integers
  if (/_mm3$|_voxels$/.test(key)) return fmtNum(val, 0);
  // Fraction / ratio columns
  if (/_frac$|_ratio$|_fraction$|nwbv|_to_etiv$|asym|lateral/.test(key))
    return Number(val).toFixed(4);
  // General float
  if (Number.isFinite(val)) return Number(val).toLocaleString("en-US", { maximumFractionDigits: 2 });
  return String(val);
}

// ── Start ─────────────────────────────────────────────
window.selectMode = selectMode;   // expose for inline onclick
init();
