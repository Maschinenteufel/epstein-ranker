const DATA_URL = "../data/epstein_ranked.jsonl";

const elements = {
  scoreFilter: document.getElementById("scoreFilter"),
  scoreValue: document.getElementById("scoreValue"),
  leadFilter: document.getElementById("leadTypeFilter"),
  powerFilter: document.getElementById("powerFilter"),
  searchInput: document.getElementById("searchInput"),
  limitInput: document.getElementById("limitInput"),
  resetFilters: document.getElementById("resetFilters"),
  refreshButton: document.getElementById("refreshButton"),
  countStat: document.getElementById("countStat"),
  avgStat: document.getElementById("avgStat"),
  leadStat: document.getElementById("leadStat"),
  updatedStat: document.getElementById("updatedStat"),
  detailDrawer: document.getElementById("detailDrawer"),
  detailTitle: document.getElementById("detailTitle"),
  detailReason: document.getElementById("detailReason"),
  detailInsights: document.getElementById("detailInsights"),
  detailLeadTypes: document.getElementById("detailLeadTypes"),
  detailPower: document.getElementById("detailPower"),
  detailAgencies: document.getElementById("detailAgencies"),
  detailText: document.getElementById("detailText"),
  detailClose: document.getElementById("detailClose"),
};

const state = {
  raw: [],
  filtered: [],
  lastUpdated: null,
  gridOptions: null,
  leadChart: null,
  scoreChart: null,
  powerChart: null,
  agencyChart: null,
  leadChoices: null,
  powerChoices: null,
};

const gridColumnDefs = [
  {
    headerName: "Score",
    field: "importance_score",
    width: 110,
    filter: "agNumberColumnFilter",
    cellClass: "score-cell",
    tooltipValueGetter: (params) => `Score: ${params.value ?? 0}`,
  },
  {
    headerName: "Headline / File",
    field: "headline",
    flex: 2,
    minWidth: 280,
    cellRenderer: (params) => {
      const headline = params.value || "Untitled lead";
      const file = params.data?.filename ?? "";
      return `
        <div class="cell-headline">
          <strong class="cell-text">${headline}</strong>
          <span>${file}</span>
        </div>
      `;
    },
    tooltipValueGetter: (params) =>
      `${params.data?.headline || "Untitled lead"}\n${params.data?.filename || ""}`,
  },
  {
    headerName: "Reason",
    field: "reason",
    flex: 2,
    minWidth: 260,
    cellRenderer: (params) => `<span class="cell-text">${params.value || ""}</span>`,
    tooltipValueGetter: (params) => params.data?.reason || "",
  },
  {
    headerName: "Power Mentions",
    field: "power_mentions",
    minWidth: 200,
    cellRenderer: (params) => `<span class="cell-text">${(params.value || []).join(", ")}</span>`,
    tooltipValueGetter: (params) => (params.data?.power_mentions || []).join(", "),
  },
  {
    headerName: "Lead Types",
    field: "lead_types",
    minWidth: 200,
    cellRenderer: (params) => `<span class="cell-text">${(params.value || []).join(", ")}</span>`,
    tooltipValueGetter: (params) => (params.data?.lead_types || []).join(", "),
  },
  {
    headerName: "Agencies",
    field: "agency_involvement",
    minWidth: 200,
    cellRenderer: (params) => `<span class="cell-text">${(params.value || []).join(", ")}</span>`,
    tooltipValueGetter: (params) => (params.data?.agency_involvement || []).join(", "),
  },
  {
    headerName: "Key Insights",
    field: "key_insights",
    flex: 2,
    minWidth: 260,
    cellRenderer: (params) =>
      `<span class="cell-text">${(params.value || []).join(" • ")}</span>`,
    tooltipValueGetter: (params) => (params.data?.key_insights || []).join(" • "),
  },
  {
    headerName: "Tags",
    field: "tags",
    minWidth: 180,
    cellRenderer: (params) => `<span class="cell-text">${(params.value || []).join(", ")}</span>`,
    tooltipValueGetter: (params) => (params.data?.tags || []).join(", "),
  },
];

function initGrid() {
  const gridElement = document.getElementById("grid");
  state.gridOptions = {
    columnDefs: gridColumnDefs,
    defaultColDef: {
      resizable: true,
      sortable: true,
      filter: true,
      flex: 1,
      minWidth: 130,
      wrapText: false,
      autoHeight: false,
      tooltipComponentParams: { color: "#fff" },
    },
    animateRows: true,
    pagination: true,
    paginationPageSize: 25,
    rowHeight: 70,
    onGridReady: (params) => {
      params.api.setRowData(state.filtered);
      params.columnApi.applyColumnState({
        state: [{ colId: "importance_score", sort: "desc" }],
        defaultState: { sort: null },
      });
      const topRow = params.api.getDisplayedRowAtIndex(0)?.data;
      if (topRow) {
        params.api.ensureIndexVisible(0);
        renderDetail(topRow);
      } else if (state.filtered.length > 0) {
        renderDetail(state.filtered[0]);
      }
    },
    onRowClicked: (event) => renderDetail(event.data),
    getRowId: (params) => params.data.filename,
  };

  new agGrid.Grid(gridElement, state.gridOptions);
}

function parseJsonl(text) {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      try {
        return JSON.parse(line);
      } catch (err) {
        console.warn("Skipping malformed JSONL line", err);
        return null;
      }
    })
    .filter(Boolean);
}

function normalizeRow(row) {
  const importance = Number(row.importance_score ?? 0);
  const arrays = (value) => (Array.isArray(value) ? value : []);
  const normalized = {
    filename: row.filename,
    source_row_index: row.metadata?.source_row_index ?? null,
    headline: row.headline || row.metadata?.original_row?.filename || "Untitled lead",
    importance_score: Number.isFinite(importance) ? importance : 0,
    reason: row.reason || "",
    key_insights: arrays(row.key_insights),
    tags: arrays(row.tags),
    power_mentions: arrays(row.power_mentions),
    agency_involvement: arrays(row.agency_involvement),
    lead_types: arrays(row.lead_types),
    metadata: row.metadata || {},
    original_text: row.metadata?.original_row?.text || "",
  };
  normalized.search_blob = [
    normalized.headline,
    normalized.reason,
    normalized.key_insights.join(" "),
    normalized.tags.join(" "),
    normalized.power_mentions.join(" "),
    normalized.lead_types.join(" "),
    normalized.original_text,
  ]
    .join(" ")
    .toLowerCase();
  return normalized;
}

function populateFilters(data) {
  const leadTypeSet = new Set();
  const powerSet = new Set();
  data.forEach((row) => {
    row.lead_types.forEach((t) => leadTypeSet.add(t));
    row.power_mentions.forEach((p) => powerSet.add(p));
  });
  setChoiceOptions(state.leadChoices, Array.from(leadTypeSet));
  setChoiceOptions(state.powerChoices, Array.from(powerSet));
}

function setChoiceOptions(choiceInstance, values) {
  if (!choiceInstance) {
    return;
  }
  const options = values.map((value) => ({
    value,
    label: value,
  }));
  choiceInstance.clearChoices();
  choiceInstance.setChoices(options, "value", "label", true);
}

function getSelectedValues(choiceInstance) {
  if (!choiceInstance) return [];
  const value = choiceInstance.getValue(true);
  if (Array.isArray(value)) {
    return value;
  }
  if (value) {
    return [value];
  }
  return [];
}

function applyFilters() {
  const minScore = Number(elements.scoreFilter.value) || 0;
  elements.scoreValue.textContent = minScore.toString();
  const leadSelected = new Set(getSelectedValues(state.leadChoices));
  const powerSelected = new Set(getSelectedValues(state.powerChoices));
  const limit = Number(elements.limitInput.value) || null;
  const term = elements.searchInput.value.trim().toLowerCase();

  let filtered = state.raw.filter((row) => row.importance_score >= minScore);
  if (leadSelected.size > 0) {
    filtered = filtered.filter((row) => row.lead_types.some((lead) => leadSelected.has(lead)));
  }
  if (powerSelected.size > 0) {
    filtered = filtered.filter((row) => row.power_mentions.some((name) => powerSelected.has(name)));
  }
  if (term) {
    filtered = filtered.filter((row) => row.search_blob.includes(term));
  }
  if (limit && limit > 0) {
    filtered = filtered.slice(0, limit);
  }
  filtered.sort((a, b) => b.importance_score - a.importance_score);

  state.filtered = filtered;
  if (state.gridOptions?.api) {
    state.gridOptions.api.setRowData(filtered);
    state.gridOptions.columnApi.applyColumnState({
      state: [{ colId: "importance_score", sort: "desc" }],
      defaultState: { sort: null },
    });
    state.gridOptions.api.ensureIndexVisible(0);
    const firstRow = state.gridOptions.api.getDisplayedRowAtIndex(0)?.data;
    if (firstRow) {
      renderDetail(firstRow);
    } else {
      clearDetail();
    }
  } else {
    if (filtered.length > 0) {
      renderDetail(filtered[0]);
    } else {
      clearDetail();
    }
  }
  updateSummary();
  updateCharts();
}

function updateSummary() {
  const count = state.filtered.length;
  const average =
    count === 0
      ? 0
      : state.filtered.reduce((sum, row) => sum + row.importance_score, 0) / count;
  const leadCounts = aggregateCounts(state.filtered, "lead_types");
  const topLead = leadCounts.length ? leadCounts[0].label : "None";
  elements.countStat.textContent = count.toLocaleString();
  elements.avgStat.textContent = average.toFixed(1);
  elements.leadStat.textContent = topLead;
  elements.updatedStat.textContent = state.lastUpdated
    ? state.lastUpdated.toLocaleTimeString()
    : "–";
}

function aggregateCounts(rows, field) {
  const counter = new Map();
  rows.forEach((row) => {
    row[field].forEach((value) => {
      counter.set(value, (counter.get(value) || 0) + 1);
    });
  });
  return Array.from(counter.entries())
    .map(([label, value]) => ({ label, value }))
    .sort((a, b) => b.value - a.value);
}

function updateCharts() {
  updateLeadChart();
  updateScoreChart();
  updatePowerChart();
  updateAgencyChart();
}

function updateLeadChart() {
  const ctx = document.getElementById("leadChart").getContext("2d");
  const topLeadTypes = aggregateCounts(state.filtered, "lead_types").slice(0, 10);
  const labels = topLeadTypes.map((item) => item.label);
  const values = topLeadTypes.map((item) => item.value);

  if (state.leadChart) {
    state.leadChart.destroy();
  }

  state.leadChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Lead count",
          data: values,
          backgroundColor: "#5ad0ff",
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: { ticks: { color: "#c3d2e8" } },
        y: { ticks: { color: "#c3d2e8" } },
      },
    },
  });
}

function updateScoreChart() {
  const ctx = document.getElementById("scoreChart").getContext("2d");
  const buckets = Array(10).fill(0);
  state.filtered.forEach((row) => {
    const index = Math.min(9, Math.floor(row.importance_score / 10));
    buckets[index] += 1;
  });
  const labels = buckets.map((_, idx) => `${idx * 10}-${idx * 10 + 9}`);
  labels[9] = "90-100";

  if (state.scoreChart) {
    state.scoreChart.destroy();
  }

  state.scoreChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Rows",
          data: buckets,
          borderColor: "#ffb347",
          backgroundColor: "rgba(255, 179, 71, 0.25)",
          fill: true,
          tension: 0.3,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: { ticks: { color: "#c3d2e8" } },
        y: { ticks: { color: "#c3d2e8" } },
      },
    },
  });
}

function updatePowerChart() {
  const ctx = document.getElementById("powerChart").getContext("2d");
  const topPower = aggregateCounts(state.filtered, "power_mentions").slice(0, 8);
  const labels = topPower.map((item) => item.label);
  const values = topPower.map((item) => item.value);

  if (state.powerChart) {
    state.powerChart.destroy();
  }

  state.powerChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Mentions",
          data: values,
          backgroundColor: "rgba(255, 99, 132, 0.65)",
        },
      ],
    },
    options: {
      responsive: true,
      indexAxis: "y",
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: { ticks: { color: "#c3d2e8" } },
        y: { ticks: { color: "#c3d2e8" } },
      },
    },
  });
}

function updateAgencyChart() {
  const ctx = document.getElementById("agencyChart").getContext("2d");
  const topAgencies = aggregateCounts(state.filtered, "agency_involvement").slice(0, 8);
  const labels = topAgencies.map((item) => item.label);
  const values = topAgencies.map((item) => item.value);

  if (state.agencyChart) {
    state.agencyChart.destroy();
  }

  state.agencyChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Mentions",
          data: values,
          backgroundColor: "rgba(90, 208, 255, 0.7)",
        },
      ],
    },
    options: {
      responsive: true,
      indexAxis: "y",
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: { ticks: { color: "#c3d2e8" } },
        y: { ticks: { color: "#c3d2e8" } },
      },
    },
  });
}

async function loadData() {
  try {
    const response = await fetch(`${DATA_URL}?t=${Date.now()}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch data: ${response.status}`);
    }
    const text = await response.text();
    const parsed = parseJsonl(text);
    state.raw = parsed.map(normalizeRow);
    state.lastUpdated = new Date();
    populateFilters(state.raw);
    applyFilters();
  } catch (error) {
    console.error("Failed to load data", error);
    alert("Unable to load epstein_ranked.jsonl. Ensure the ranking script has produced output.");
  }
}

function resetFilters() {
  elements.scoreFilter.value = 0;
  elements.scoreValue.textContent = "0";
  state.leadChoices?.removeActiveItems();
  state.powerChoices?.removeActiveItems();
  elements.searchInput.value = "";
  elements.limitInput.value = "";
  applyFilters();
}

function wireEvents() {
  ["change", "input"].forEach((eventName) => {
    elements.scoreFilter.addEventListener(eventName, applyFilters);
    elements.searchInput.addEventListener(eventName, debounce(applyFilters, 200));
    elements.limitInput.addEventListener(eventName, debounce(applyFilters, 200));
  });
  elements.leadFilter.addEventListener("change", applyFilters);
  elements.powerFilter.addEventListener("change", applyFilters);
  elements.resetFilters.addEventListener("click", resetFilters);
  elements.refreshButton.addEventListener("click", () => loadData());
  elements.detailClose.addEventListener("click", () => clearDetail());
}

function debounce(fn, delay) {
  let timer;
  return (...args) => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => fn(...args), delay);
  };
}

document.addEventListener("DOMContentLoaded", () => {
  initGrid();
  initChoices();
  wireEvents();
  loadData();
});

function initChoices() {
  state.leadChoices = new Choices(elements.leadFilter, {
    removeItemButton: true,
    placeholder: true,
    placeholderValue: "Select lead types",
    searchPlaceholderValue: "Search lead types",
  });
  state.powerChoices = new Choices(elements.powerFilter, {
    removeItemButton: true,
    placeholder: true,
    placeholderValue: "Select power mentions",
    searchPlaceholderValue: "Search people/agencies",
  });
}

function detailCellRenderer(params) {
  const data = params.data;
  const container = document.createElement("div");
  container.className = "detail-panel";
  container.innerHTML = `
    <section>
      <h3>Reason</h3>
      <p>${escapeHtml(data.reason || "—")}</p>
      <h3>Lead Types</h3>
      <p>${data.lead_types.join(", ") || "—"}</p>
      <h3>Power Mentions</h3>
      <p>${data.power_mentions.join(", ") || "—"}</p>
      <h3>Agencies</h3>
      <p>${data.agency_involvement.join(", ") || "—"}</p>
    </section>
    <section>
      <h3>Key Insights</h3>
      <ul>${data.key_insights.map((insight) => `<li>${escapeHtml(insight)}</li>`).join("") || "<li>—</li>"}</ul>
    </section>
    <section>
      <h3>Original Text</h3>
      <pre>${escapeHtml(data.original_text || "No source text captured.")}</pre>
    </section>
  `;
  return container;
}

function escapeHtml(str) {
  return (str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function renderDetail(row) {
  if (!row) {
    clearDetail();
    return;
  }
  elements.detailDrawer.classList.remove("hidden");
  elements.detailTitle.textContent = `${row.headline || row.filename} (${row.filename})`;
  elements.detailReason.textContent = row.reason || "—";
  elements.detailLeadTypes.textContent = row.lead_types.join(", ") || "—";
  elements.detailPower.textContent = row.power_mentions.join(", ") || "—";
  elements.detailAgencies.textContent = row.agency_involvement.join(", ") || "—";
  elements.detailText.textContent = row.original_text || "No source text captured.";
  elements.detailInsights.innerHTML =
    row.key_insights.length > 0
      ? row.key_insights.map((item) => `<li>${escapeHtml(item)}</li>`).join("")
      : "<li>—</li>";
}

function clearDetail() {
  elements.detailDrawer.classList.add("hidden");
  elements.detailTitle.textContent = "Select a row to inspect full context";
  elements.detailReason.textContent = "—";
  elements.detailLeadTypes.textContent = "—";
  elements.detailPower.textContent = "—";
  elements.detailAgencies.textContent = "—";
  elements.detailText.textContent = "—";
  elements.detailInsights.innerHTML = "";
}
