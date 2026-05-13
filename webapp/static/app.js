// Constants & Configuration
const MODEL_COLORS = {
  "XGBoost":           "#f59e0b",
  "LightGBM":          "#10b981",
  "CatBoost":          "#6366f1",
  "TabPFN":            "#3b82f6",
  "SAP RPT-1 OSS":     "#ec4899",
  "Voting Ensemble":   "#fbbf24",
  "Stacking Ensemble": "#a78bfa",
};

const MODEL_EMOJIS = {
  "XGBoost":           "🟡",
  "LightGBM":          "🟢",
  "CatBoost":          "🟣",
  "TabPFN":            "🟦",
  "SAP RPT-1 OSS":     "🩷",
  "Voting Ensemble":   "🏆",
  "Stacking Ensemble": "✨",
};

const ENSEMBLE_NAMES = ["Voting Ensemble", "Stacking Ensemble"];

// DOM Elements
const dropZone       = document.getElementById("drop-zone");
const fileInput      = document.getElementById("file-input");
const uploadError    = document.getElementById("upload-error");
const uploadSection  = document.getElementById("upload-section");
const previewSection = document.getElementById("preview-section");
const previewMeta    = document.getElementById("preview-meta");
const targetSelect   = document.getElementById("target-select");
const previewTable   = document.getElementById("preview-table");
const changeFileBtn  = document.getElementById("change-file-btn");
const runBtn         = document.getElementById("run-btn");
const loadingSection = document.getElementById("loading-section");
const resultsSection = document.getElementById("results-section");
const resetBtn       = document.getElementById("reset-btn");
const exportCsvBtn   = document.getElementById("export-csv-btn");
const exportJsonBtn  = document.getElementById("export-json-btn");
const downloadPlotsBtn = document.getElementById("download-plots-btn");

const resumeSection  = document.getElementById("resume-section");
const resumeFilename = document.getElementById("resume-filename");
const resumeClearBtn = document.getElementById("resume-clear-btn");
const resumeGoBtn    = document.getElementById("resume-go-btn");

let currentFile = null;
let chartInstances = [];

// Drag & Drop Handling
if (dropZone) {
  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") fileInput.click(); });

  dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
  dropZone.addEventListener("dragleave",  () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  });
}

if (fileInput) {
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });
}

if (changeFileBtn) changeFileBtn.addEventListener("click", resetToUpload);
if (resetBtn) resetBtn.addEventListener("click", resetToUpload);

if (exportCsvBtn) exportCsvBtn.addEventListener("click", () => {
  const data = JSON.parse(sessionStorage.getItem("lastResults"));
  if (data) exportToCSV(data);
});

if (exportJsonBtn) exportJsonBtn.addEventListener("click", () => {
  const data = JSON.parse(sessionStorage.getItem("lastResults"));
  if (data) exportToJSON(data);
});

if (downloadPlotsBtn) downloadPlotsBtn.addEventListener("click", async () => {
  const data = JSON.parse(sessionStorage.getItem("lastResults"));
  if (!data) {
    alert("Please run a benchmark first to generate results before downloading plots.");
    return;
  }

  const originalText = downloadPlotsBtn.innerHTML;
  downloadPlotsBtn.innerHTML = "⏳ Generating...";
  downloadPlotsBtn.disabled = true;

  try {
    const res = await fetch("/download-plots", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });
    if (!res.ok) throw new Error("Failed to generate plots");
    
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "modelmatrix_plots.pdf";
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  } catch (e) {
    alert("Error generating plots: " + e.message);
  } finally {
    downloadPlotsBtn.innerHTML = originalText;
    downloadPlotsBtn.disabled = false;
  }
});

if (resumeClearBtn) resumeClearBtn.addEventListener("click", () => {
  sessionStorage.removeItem("lastResults");
  sessionStorage.removeItem("lastFileName");
  window.location.reload();
});

if (resumeGoBtn) resumeGoBtn.addEventListener("click", () => {
  window.location.href = "/static/arena.html";
});

// File selection and preview initialization
async function handleFile(file) {
  uploadError.hidden = true;

  if (!file.name.endsWith(".csv")) {
    showError("Please upload a .csv file.");
    return;
  }

  const MAX_MB = 5;
  if (file.size > MAX_MB * 1024 * 1024) {
    showError(`File is too large (${(file.size / 1048576).toFixed(1)} MB). Maximum is ${MAX_MB} MB.`);
    return;
  }

  currentFile = file;

  const fd = new FormData();
  fd.append("file", file);

  try {
    const res = await fetch("/preview", { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json();
      showError(err.detail || "Failed to read CSV.");
      return;
    }
    const data = await res.json();
    renderPreview(data, file);
  } catch (e) {
    showError("Network error: " + e.message);
  }
}

function renderPreview(data, file) {
  // Meta badges
  previewMeta.innerHTML = `
    <span class="meta-badge">📄 ${file.name}</span>
    <span class="meta-badge">${data.n_rows.toLocaleString()} rows</span>
    <span class="meta-badge">${data.n_cols} columns</span>
  `;

  // Target column selector
  targetSelect.innerHTML = "";
  data.columns.forEach(col => {
    const opt = document.createElement("option");
    opt.value = col;
    opt.textContent = col;
    if (col === data.default_target) opt.selected = true;
    targetSelect.appendChild(opt);
  });

  // Preview table
  const cols = data.columns;
  let thead = "<thead><tr>" + cols.map(c => `<th class="${c === data.default_target ? 'target-col' : ''}">${esc(c)}</th>`).join("") + "</tr></thead>";
  let tbody = "<tbody>" + data.preview.map(row =>
    "<tr>" + cols.map(c => `<td class="${c === data.default_target ? 'target-col' : ''}">${esc(String(row[c] ?? ""))}</td>`).join("") + "</tr>"
  ).join("") + "</tbody>";
  previewTable.innerHTML = thead + tbody;

  // Highlight target column on select change
  targetSelect.addEventListener("change", () => highlightTarget(targetSelect.value, cols));

  uploadSection.hidden = true;
  previewSection.hidden = false;
}

function highlightTarget(targetCol, cols) {
  previewTable.querySelectorAll("th, td").forEach(el => el.classList.remove("target-col"));
  const idx = cols.indexOf(targetCol);
  if (idx < 0) return;
  previewTable.querySelectorAll("tr").forEach(row => {
    const cells = row.querySelectorAll("th, td");
    if (cells[idx]) cells[idx].classList.add("target-col");
  });
}

// Execute benchmarking suite
if (runBtn) {
  runBtn.addEventListener("click", async () => {
  if (!currentFile) return;

  previewSection.hidden = true;
  loadingSection.hidden = false;

  // Animate loader steps
  const steps = ["step-xgb", "step-lgb", "step-cat", "step-tabpfn", "step-sap", "step-vote", "step-stack"];
  const delays = [0, 150, 300, 450, 600, 750, 900];
  let stepIdx = 0;
  const stepTimer = setInterval(() => {
    if (stepIdx > 0) {
      document.getElementById(steps[stepIdx - 1])?.classList.remove("active");
      document.getElementById(steps[stepIdx - 1])?.classList.add("done");
    }
    if (stepIdx < steps.length) {
      document.getElementById(steps[stepIdx])?.classList.add("active");
      stepIdx++;
    } else {
      clearInterval(stepTimer);
    }
  }, 1400);

  const fd = new FormData();
  fd.append("file", currentFile);
  fd.append("target_col", targetSelect.value);

  try {
    const res = await fetch("/benchmark", { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json();
      clearInterval(stepTimer);
      loadingSection.hidden = true;
      previewSection.hidden = false;
      showError(err.detail || "Benchmarking failed.");
      return;
    }
    const data = await res.json();
    clearInterval(stepTimer);
    loadingSection.hidden = true;
    sessionStorage.setItem("lastResults", JSON.stringify(data));
    sessionStorage.setItem("lastFileName", currentFile.name);
    window.location.href = "/static/arena.html";
  } catch (e) {
    clearInterval(stepTimer);
    loadingSection.hidden = true;
    previewSection.hidden = false;
    showError("Network error: " + e.message);
  }
});
}

// Visualization of benchmarking results
function renderResults(data) {
  const { dataset_info, task, results, recommendation, n_folds } = data;
  const isCLF = task === "classification";
  const primaryKey = isCLF ? "roc_auc" : "r2";
  const primaryLabel = isCLF ? "ROC-AUC" : "R²";

  const fileName = sessionStorage.getItem("lastFileName") || "Dataset";

  // ── Info bar
  const taskBadge = isCLF
    ? `<span class="info-tag">🏷 Classification</span>`
    : `<span class="info-tag green">📈 Regression</span>`;
  document.getElementById("info-bar").innerHTML = `
    <span class="info-tag">📄 ${esc(fileName)}</span>
    ${taskBadge}
    <span class="info-tag">${dataset_info.n_samples.toLocaleString()} samples</span>
    <span class="info-tag">${dataset_info.n_features} features</span>
    <span class="info-tag">Target: <strong>${esc(dataset_info.target_col)}</strong></span>
    ${isCLF ? `<span class="info-tag pink">${dataset_info.n_classes} classes</span>` : ""}
    <span class="info-tag">${n_folds}-Fold CV</span>
  `;

  // ── KPI cards
  const kpiGrid = document.getElementById("kpi-grid");
  kpiGrid.innerHTML = "";

  const validModels = Object.entries(results).filter(([, v]) => !v.error);
  const bestEntry   = validModels.reduce((best, [name, v]) =>
    (v.mean[primaryKey] || 0) > (best[1].mean[primaryKey] || 0) ? [name, v] : best
  , validModels[0]);

  const kpis = [
    {
      label: "Best Model",
      value: bestEntry[0],
      sub:   `${primaryLabel}: ${fmt(bestEntry[1].mean[primaryKey])}`,
      color: MODEL_COLORS[bestEntry[0]],
    },
    {
      label: `Best ${primaryLabel}`,
      value: fmt(bestEntry[1].mean[primaryKey]),
      sub:   `± ${fmt(bestEntry[1].std[primaryKey])} std`,
      color: "#818cf8",
    },
    {
      label: "Models Evaluated",
      value: validModels.length,
      sub:   `${n_folds}-fold cross-validation`,
      color: "#10b981",
    },
    {
      label: "Dataset Size",
      value: dataset_info.n_samples.toLocaleString(),
      sub:   `${dataset_info.n_features} features · ${isCLF ? dataset_info.n_classes + " classes" : "regression"}`,
      color: "#f59e0b",
    },
  ];

  kpis.forEach(k => {
    const card = document.createElement("div");
    card.className = "kpi-card";
    card.style.setProperty("--accent-bar", k.color);
    card.innerHTML = `
      <div class="kpi-label">${k.label}</div>
      <div class="kpi-value" style="color:${k.color}">${esc(String(k.value))}</div>
      <div class="kpi-sub">${k.sub}</div>
    `;
    kpiGrid.appendChild(card);
  });

  // ── Legend
  const legendEl = document.getElementById("legend");
  legendEl.innerHTML = Object.entries(MODEL_COLORS).map(([name, color]) =>
    `<div class="legend-item">
      <div class="legend-dot" style="background:${color}"></div>
      <span>${name}</span>
    </div>`
  ).join("");

  // ── Charts
  chartInstances.forEach(c => c.destroy());
  chartInstances = [];
  const chartsGrid = document.getElementById("charts-grid");
  chartsGrid.innerHTML = "";

  const metricsToChart = isCLF
    ? [["roc_auc", "ROC-AUC"], ["accuracy", "Accuracy"], ["f1_macro", "F1-Macro"]]
    : [["r2", "R²"], ["mae", "MAE"], ["rmse", "RMSE"]];

  metricsToChart.forEach(([key, label]) => {
    const modelNames = Object.keys(results).filter(n => !results[n].error && results[n].mean[key] != null);
    if (!modelNames.length) return;

    const vals  = modelNames.map(n => roundN(results[n].mean[key], 4));
    const errs  = modelNames.map(n => roundN(results[n].std[key] || 0, 4));
    const bgs   = modelNames.map(n => (MODEL_COLORS[n] || "#888") + "cc");
    const bords = modelNames.map(n => MODEL_COLORS[n] || "#888");

    const isErrorMetric = ["mae", "rmse", "log_loss"].includes(key.toLowerCase());
    const highQual = isErrorMetric ? "poor" : "excellent";
    const lowQual  = isErrorMetric ? "excellent" : "poor";

    const card = document.createElement("div");
    card.className = "chart-card";
    const canvasId = `chart-${key}`;
    card.innerHTML = `
      <h4>${label}</h4>
      <div class="chart-sub">${label} (mean ± std over ${n_folds} folds)</div>
      <canvas id="${canvasId}"></canvas>
      <div class="chart-interpretation">
        <div class="interp-item"><span>High ${label} = </span> <span class="badge ${highQual}">${highQual}</span></div>
        <div class="interp-item"><span>Low ${label} = </span> <span class="badge ${lowQual}">${lowQual}</span></div>
      </div>
    `;
    chartsGrid.appendChild(card);

    const minVal = Math.min(...vals);
    const maxVal = Math.max(...vals);
    const pad    = Math.max((maxVal - minVal) * 0.15, 0.02);

    const inst = new Chart(document.getElementById(canvasId), {
      type: "bar",
      data: {
        labels: modelNames,
        datasets: [{
          label,
          data: vals,
          backgroundColor: bgs,
          borderColor:     bords,
          borderWidth: 2,
          borderRadius: 8,
        }],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => `${label}: ${ctx.parsed.y.toFixed(4)} ± ${errs[ctx.dataIndex].toFixed(4)}`,
            },
          },
        },
        scales: {
          y: {
            min: Math.max(key === "roc_auc" || key === "accuracy" ? 0 : -Infinity, minVal - pad),
            max: key === "roc_auc" || key === "accuracy" ? Math.min(1, maxVal + pad) : maxVal + pad,
            grid: { color: "rgba(100, 116, 139, 0.1)" },
            ticks: { color: "rgba(100, 116, 139, 0.8)", font: { size: 11 } },
          },
          x: {
            grid: { display: false },
            ticks: { color: "rgba(100, 116, 139, 0.8)", font: { size: 12 } },
          },
        },
      },
    });
    chartInstances.push(inst);
  });

  // ── Full table
  const thead = document.getElementById("results-thead");
  const tbody = document.getElementById("results-tbody");

  const allMetrics = isCLF
    ? ["accuracy", "f1_macro", "roc_auc", "log_loss", "fit_time"]
    : ["r2", "mae", "rmse", "fit_time"];
  const metricLabels = isCLF
    ? ["Accuracy", "F1-Macro", "ROC-AUC", "Log Loss", "Fit Time"]
    : ["R²", "MAE", "RMSE", "Fit Time"];

  thead.innerHTML = "<tr><th>Model</th>" + metricLabels.map(l => `<th>${l}</th>`).join("") + "</tr>";
  tbody.innerHTML = Object.entries(results).map(([name, d]) => {
    if (d.error) {
      const errText = d.error.startsWith("Error:") ? d.error : `Error: ${d.error}`;
      return `<tr><td><span class="model-dot" style="background:${MODEL_COLORS[name] || '#888'}"></span>${name}</td><td colspan="${allMetrics.length}" style="color:#f87171">${esc(errText)}</td></tr>`;
    }
    const cells = allMetrics.map(k => {
      const v = d.mean[k];
      if (v == null) return `<td class="mono" style="color:#374151">—</td>`;
      const isTime = k === "fit_time";
      if (isTime) return `<td class="mono" style="color:#94a3b8">${v.toFixed(3)}s</td>`;
      const cls = scoreClass(v, k, task);
      return `<td class="mono ${cls}">${v.toFixed(4)}<span style="color:#374151;font-size:.7em"> ±${(d.std[k]||0).toFixed(3)}</span></td>`;
    }).join("");
    return `<tr><td><span class="model-dot" style="background:${MODEL_COLORS[name] || '#888'}"></span><strong>${name}</strong></td>${cells}</tr>`;
  }).join("");

  // ── Recommendations
  const recGrid = document.getElementById("recommendation-grid");
  recGrid.innerHTML = "";
  const recs = recommendation.recommendations || {};
  const recDefs = [
    { key: "best_overall",      label: "🏆 Best Overall",      winner: true  },
    { key: "production",        label: "🚀 Production Ready",  winner: false },
    { key: "best_accuracy",     label: "🎯 Highest Accuracy",  winner: false },
    { key: "best_speed",        label: "⚡ Fastest Training",  winner: false },
    { key: "best_consistency",  label: "🛡 Most Consistent",   winner: false },
  ];

  recDefs.forEach(({ key, label, winner }) => {
    const rec = recs[key];
    if (!rec) return;
    const color = MODEL_COLORS[rec.model] || "#888";
    const score = rec.score != null
      ? `<div class="rec-score">${recommendation.primary_metric}: ${typeof rec.score === "number" ? rec.score.toFixed(4) : rec.score}</div>`
      : "";
    const card = document.createElement("div");
    card.className = `rec-card ${key}${winner ? " winner" : ""}`;
    card.innerHTML = `
      <div class="rec-type">${label}</div>
      <div class="rec-model-name">
        ${winner ? '<span class="rec-trophy">🏆</span>' : ""}
        <span style="color:${color}">${rec.model}</span>
      </div>
      ${score}
      <p class="rec-reason">${esc(rec.reason)}</p>
    `;
    recGrid.appendChild(card);
  });

  // ── Ensemble Analysis section
  renderEnsembleSection(data.ensemble_info || {}, results, recommendation, task);

  // ── Interactive Playground
  renderPlayground(data.dataset_info, recommendation.recommendations?.best_overall, task);

  // ── Statistical Rigor
  renderStatisticalSection(data.stats || {});

  resultsSection.hidden = false;
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderStatisticalSection(stats) {
  const tbody = document.getElementById("rigor-tbody");
  const badge = document.getElementById("friedman-badge");
  if (!tbody || !stats.ranking) return;

  const isSig = stats.significant;
  badge.className = `p-value-badge ${isSig ? 'significant' : 'not-significant'}`;
  badge.textContent = isSig 
    ? `Significant (p=${stats.friedman_p})` 
    : `Not Significant (p=${stats.friedman_p})`;

  tbody.innerHTML = stats.ranking.map(r => {
    const stability = r.win_rate;
    return `
      <tr>
        <td>
          <span class="rank-pill ${r.avg_rank <= 1.5 ? 'rank-1' : ''}" style="${r.avg_rank > 1.5 ? 'background: transparent; box-shadow: none;' : ''}">${r.avg_rank <= 1.5 ? '🏆' : ''}</span>
          <strong>${r.model}</strong>
        </td>
        <td class="mono">${r.avg_rank}</td>
        <td>
          <div class="stability-bar">
            <div class="stability-fill" style="width: ${stability}%"></div>
          </div>
          <span class="mono">${stability}%</span>
        </td>
        <td>
          <span class="badge ${stability > 50 ? 'excellent' : (stability > 20 ? 'neutral' : 'poor')}">
            ${stability > 50 ? 'Dominant' : (stability > 20 ? 'Competitive' : 'Volatile')}
          </span>
        </td>
      </tr>
    `;
  }).join("");
}

// ── Playground Logic ──────────────────────────────────────────────────────────
function renderPlayground(datasetInfo, bestOverall, task) {
  const form = document.getElementById("playground-form");
  const valueEl = document.getElementById("prediction-value");
  const subEl   = document.getElementById("prediction-sub");
  const probEl  = document.getElementById("probability-bars");

  if (!form || !bestOverall) return;
  form.innerHTML = "";
  
  const features = datasetInfo.columns || [];
  const preview = datasetInfo.preview ? datasetInfo.preview[0] : {};

  features.forEach(f => {
    const div = document.createElement("div");
    div.className = "playground-field";
    
    const types = datasetInfo.feature_types || {};
    const isNumeric = types[f] === "numeric";
    
    const sampleVal = preview[f];
    const val = sampleVal != null ? sampleVal : (isNumeric ? 0 : "");
    const placeholder = isNumeric ? "Enter value..." : "Enter text...";
    
    div.innerHTML = `
      <label>${f.replace(/_/g, " ")}</label>
      <input type="text" 
             data-feature="${f}" 
             value="${val}" 
             placeholder="${placeholder}"
             onclick="this.select()">
    `;
    form.appendChild(div);
  });

  const updatePrediction = async () => {
    const inputs = form.querySelectorAll("input");
    const data = {};
    inputs.forEach(i => {
      const v = i.value;
      data[i.dataset.feature] = isNaN(parseFloat(v)) ? v : parseFloat(v);
    });

    valueEl.style.opacity = "0.5";
    try {
      const resp = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      });
      const res = await resp.json();
      
      valueEl.style.opacity = "1";
      if (res.error) {
        valueEl.textContent = "Error";
        subEl.textContent = res.error;
        return;
      }

      if (task === "classification") {
        valueEl.textContent = res.prediction || "—";
        subEl.textContent = `Most likely class (via ${bestOverall.model})`;
        
        if (res.probabilities && res.labels) {
          probEl.innerHTML = res.probabilities.map((p, i) => `
            <div class="prob-row">
              <div class="prob-meta"><span>${res.labels[i] || 'Class '+i}</span><span>${(p*100).toFixed(1)}%</span></div>
              <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:${p*100}%"></div></div>
            </div>
          `).join("");
        }
      } else {
        const val = Number(res.prediction);
        valueEl.textContent = isNaN(val) ? "—" : val.toFixed(4);
        subEl.textContent = `Regression output (via ${bestOverall.model})`;
        probEl.innerHTML = "";
      }
    } catch (e) {
      valueEl.style.opacity = "1";
      valueEl.textContent = "Error";
      subEl.textContent = "Service unavailable";
    }
  };

  form.addEventListener("input", debounce(updatePrediction, 300));
  updatePrediction(); // Initial prediction
}

function debounce(fn, ms) {
  let timeout;
  return (...args) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => fn.apply(this, args), ms);
  };
}

// ── Ensemble Analysis renderer ────────────────────────────────────────────────
function renderEnsembleSection(ensembleInfo, results, recommendation, task) {
  const grid  = document.getElementById("ensemble-grid");
  const title = document.getElementById("ensemble-section-title");
  grid.innerHTML = "";

  const entries = Object.entries(ensembleInfo).filter(([name]) => results[name] && !results[name].error);
  if (!entries.length) {
    title.hidden = true;
    grid.hidden  = true;
    return;
  }
  title.hidden = false;
  grid.hidden  = false;

  const primaryKey   = task === "classification" ? "roc_auc" : "r2";
  const primaryLabel = task === "classification" ? "ROC-AUC" : "R²";

  // Find the best individual model score (excluding ensembles) for gain %
  const indivScores = Object.entries(results)
    .filter(([n, v]) => !ENSEMBLE_NAMES.includes(n) && !v.error && v.mean[primaryKey] != null)
    .map(([, v]) => v.mean[primaryKey]);
  const bestIndivScore = indivScores.length ? Math.max(...indivScores) : 0;

  entries.forEach(([name, info]) => {
    const cv    = results[name];
    const score = cv.mean[primaryKey] ?? 0;
    const std   = cv.std[primaryKey]  ?? 0;
    const ft    = cv.mean.fit_time    ?? 0;
    const color = MODEL_COLORS[name] || "#888";
    const gain  = bestIndivScore > 0 ? ((score - bestIndivScore) / bestIndivScore * 100) : 0;
    const gainStr = gain >= 0
      ? `<span class="gain-pos">▲ +${gain.toFixed(2)}% vs best individual</span>`
      : `<span class="gain-neg">▼ ${gain.toFixed(2)}% vs best individual</span>`;

    const componentPills = (info.components || []).map(c =>
      `<span class="comp-pill" style="border-color:${MODEL_COLORS[c] || '#888'};color:${MODEL_COLORS[c] || '#888'}">${c}</span>`
    ).join("");

    const metaTag = info.meta_learner
      ? `<div class="ens-meta">Meta-learner: <strong>${esc(info.meta_learner)}</strong></div>` : "";

    const card = document.createElement("div");
    card.className = "ens-card";
    card.style.setProperty("--ens-color", color);
    card.innerHTML = `
      <div class="ens-header">
        <span class="ens-emoji">${MODEL_EMOJIS[name] || "🧩"}</span>
        <span class="ens-name" style="color:${color}">${name}</span>
        <span class="ens-type-badge">${info.type === "voting" ? "Soft Voting" : "Stacking"}</span>
      </div>
      <div class="ens-score">
        <span class="ens-score-val">${score.toFixed(4)}</span>
        <span class="ens-score-label"> ${primaryLabel} ± ${std.toFixed(3)}</span>
      </div>
      <div class="ens-gain">${gainStr}</div>
      ${metaTag}
      <div class="ens-desc">${esc(info.description || "")}</div>
      <div class="ens-components-label">Component Models</div>
      <div class="ens-components">${componentPills}</div>
      <div class="ens-footer">Avg fit time: ${ft.toFixed(3)}s per fold</div>
    `;
    grid.appendChild(card);
  });
}



// ── Helpers ───────────────────────────────────────────────────────────────────
function resetToUpload() {
  currentFile = null;
  if (fileInput) fileInput.value = "";
  if (uploadError) uploadError.hidden = true;
  if (previewSection) previewSection.hidden = true;
  if (loadingSection) loadingSection.hidden = true;
  if (resultsSection) resultsSection.hidden = true;
  if (uploadSection) uploadSection.hidden = false;
  chartInstances.forEach(c => c.destroy());
  chartInstances = [];
  sessionStorage.removeItem("lastResults");
  sessionStorage.removeItem("lastFileName");
  if (window.location.pathname.includes("arena.html")) {
    window.location.href = "/static/uploader.html";
  } else {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
}

function showError(msg) {
  if (!uploadError) return;
  uploadError.textContent = msg;
  uploadError.hidden = false;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function exportToCSV(data) {
  const results = data.results;
  const models = Object.keys(results);
  if (models.length === 0) return;
  
  const metricKeys = new Set();
  models.forEach(m => {
    if (results[m].mean) {
      Object.keys(results[m].mean).forEach(k => metricKeys.add(k));
    }
  });
  const metrics = Array.from(metricKeys).sort();
  
  let csv = "Model," + metrics.map(m => m + " (mean)").join(",") + "\n";
  models.forEach(m => {
    if (results[m].error) {
      const errText = results[m].error.startsWith("Error:") ? results[m].error : `Error: ${results[m].error}`;
      csv += `${m.replace(/,/g, "")},${errText.replace(/,/g, " ")}\n`;
      return;
    }
    let row = [m.replace(/,/g, "")];
    metrics.forEach(met => {
      let val = results[m].mean ? results[m].mean[met] : "";
      row.push(val !== undefined && val !== null ? val : "");
    });
    csv += row.join(",") + "\n";
  });
  
  downloadFile(csv, "benchmark_results.csv", "text/csv");
}

function exportToJSON(data) {
  const json = JSON.stringify(data, null, 2);
  downloadFile(json, "benchmark_results.json", "application/json");
}

function downloadFile(content, fileName, contentType) {
  const blob = new Blob([content], { type: contentType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = fileName;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 100);
}

function fmt(v) {
  if (v == null || isNaN(v)) return "—";
  return Number(v).toFixed(4);
}

function roundN(v, n) {
  return Math.round(v * Math.pow(10, n)) / Math.pow(10, n);
}

function esc(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function scoreClass(v, metric, task) {
  if (metric === "fit_time") return "";
  const higherBetter = !["mae", "rmse", "mse", "log_loss"].includes(metric);
  if (!higherBetter) {
    if (v < 0.1)  return "col-excellent";
    if (v < 0.3)  return "col-good";
    if (v < 0.5)  return "col-fair";
    return "col-poor";
  }
  if (metric === "roc_auc" || metric === "accuracy") {
    if (v >= 0.95) return "col-excellent";
    if (v >= 0.88) return "col-good";
    if (v >= 0.75) return "col-fair";
    return "col-poor";
  }
  if (metric === "r2") {
    if (v >= 0.75) return "col-excellent";
    if (v >= 0.5)  return "col-good";
    if (v >= 0.25) return "col-fair";
    return "col-poor";
  }
  if (v >= 0.85) return "col-excellent";
  if (v >= 0.70) return "col-good";
  if (v >= 0.55) return "col-fair";
  return "col-poor";
}

// ── Restore state on load ────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  checkResumeState();
});

// ── Handle Back Button (BFCache) ──────────────────────────────────────────────
window.addEventListener("pageshow", function(e) {
  checkResumeState();
});

// Theme Toggle Logic
const themeToggle = document.getElementById("theme-toggle");
const themeIconDark = document.getElementById("theme-icon-dark");
const themeIconLight = document.getElementById("theme-icon-light");

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("theme", theme);
  if (theme === "light") {
    if (themeIconDark) themeIconDark.style.display = "block";
    if (themeIconLight) themeIconLight.style.display = "none";
  } else {
    if (themeIconDark) themeIconDark.style.display = "none";
    if (themeIconLight) themeIconLight.style.display = "block";
  }
}

if (themeToggle) {
  themeToggle.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    setTheme(current === "dark" ? "light" : "dark");
  });
}

// Initial theme load
const savedTheme = localStorage.getItem("theme") || "dark";
setTheme(savedTheme);

function checkResumeState() {
  const savedResults = sessionStorage.getItem("lastResults");
  const savedFile = sessionStorage.getItem("lastFileName");
  const isUploader = window.location.pathname.includes("uploader.html") || window.location.pathname === "/";
  const isArena = window.location.pathname.includes("arena.html");

  // Handle MBench logo link privilege
  const navLogo = document.getElementById("nav-logo");
  if (navLogo) {
    // Privilege: Only uploader page in fresh mode (no results) can go to landing
    if (isUploader && !savedResults) {
      navLogo.classList.add("active-link");
      navLogo.style.pointerEvents = "auto";
    } else {
      navLogo.classList.remove("active-link");
      navLogo.style.pointerEvents = "none";
    }
  }

  if (savedResults && savedFile) {
    if (isUploader) {
      // Always show resume card if data exists, until cleared
      if (uploadSection) uploadSection.hidden = true;
      if (previewSection) previewSection.hidden = true;
      if (loadingSection) loadingSection.hidden = true;
      if (resumeSection) {
        resumeSection.hidden = false;
        resumeFilename.textContent = savedFile;
      }
    } else if (isArena) {
      // Auto-render on results page if data exists
      try {
        const data = JSON.parse(savedResults);
        renderResults(data);
      } catch (e) {
        window.location.href = "/static/uploader.html";
      }
    }
  } else {
    // No saved data: reset to default
    if (isUploader) {
      if (resumeSection) resumeSection.hidden = true;
      if (uploadSection) uploadSection.hidden = false;
    } else if (isArena) {
      window.location.href = "/static/uploader.html";
    }
  }
}
