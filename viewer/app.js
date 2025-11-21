import { POWER_ALIASES } from "./power_aliases.js";

const DATA_URL = "data/epstein_ranked.jsonl";
const CHUNK_MANIFEST_URL = "data/chunks.json";
const DEFAULT_CHUNK_SIZE = 1000;
const POWER_ALIAS_MAP = {
  "barack obama": ["barack obama", "obama", "barack", "president obama"],
  "donald trump": ["donald trump", "trump", "donald j trump", "president trump"],
  "joe biden": ["joe biden", "joseph biden", "president biden"],
  "bill clinton": ["bill clinton", "william clinton", "william j clinton", "president clinton"],
  "bill gates": ["bill gates", "william gates", "william h gates", "gates"],
  "hillary clinton": ["hillary clinton", "hillary rodham clinton"],
  "jeffrey epstein": ["jeffrey epstein", "jeff epstein", "epstein"],
  "elon musk": ["elon musk", "elon r musk", "musk"],
};

const elements = {
  scoreFilter: document.getElementById("scoreFilter"),
  scoreValue: document.getElementById("scoreValue"),
  leadFilter: document.getElementById("leadTypeFilter"),
  powerFilter: document.getElementById("powerFilter"),
  searchInput: document.getElementById("searchInput"),
  limitInput: document.getElementById("limitInput"),
  resetFilters: document.getElementById("resetFilters"),
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
  detailTags: document.getElementById("detailTags"),
  detailModel: document.getElementById("detailModel"),
  detailText: document.getElementById("detailText"),
  detailTextPreview: document.getElementById("detailTextPreview"),
  detailTextToggle: document.getElementById("detailTextToggle"),
  detailClose: document.getElementById("detailClose"),
  loadingOverlay: document.getElementById("loadingOverlay"),
  loadingTitle: document.getElementById("loadingTitle"),
  loadingSubtitle: document.getElementById("loadingSubtitle"),
  loadingProgress: document.getElementById("loadingProgress"),
  inlineLoader: document.getElementById("inlineLoader"),
  inlineLoaderText: document.getElementById("inlineLoaderText"),
};

const state = {
  raw: [],
  filtered: [],
  lastUpdated: null,
  manifestMetadata: null,
  gridOptions: null,
  leadChart: null,
  scoreChart: null,
  powerChart: null,
  agencyChart: null,
  leadChoices: null,
  powerChoices: null,
  loading: {
    totalChunks: 0,
    loadedChunks: 0,
  },
  currentLoadId: 0,
  activeRowId: null,
  powerDisplayNames: {},
};

const powerAliasLookup = buildPowerAliasLookup(POWER_ALIASES);
const canonicalPowerList = buildCanonicalPowerList(powerAliasLookup);
const powerKeywordMap = buildPowerKeywordMap(POWER_ALIASES, powerAliasLookup);

function resetLoadingState(title = "Loading Epstein Files…", subtitle = "Preparing data") {
  state.powerDisplayNames = {};
  state.loading.loadedChunks = 0;
  state.loading.totalChunks = 0;
  if (elements.loadingOverlay) {
    elements.loadingOverlay.classList.remove("hidden");
    elements.loadingTitle.textContent = title;
    elements.loadingSubtitle.textContent = subtitle;
  }
  if (elements.loadingProgress) {
    elements.loadingProgress.style.width = "0%";
  }
  hideInlineLoader();
}

function updateLoadingProgress(loaded, total, subtitle) {
  state.loading.loadedChunks = loaded;
  state.loading.totalChunks = total;
  const percent = total > 0 ? Math.min(100, Math.round((loaded / total) * 100)) : 0;

  if (elements.loadingProgress) {
    elements.loadingProgress.style.width = `${percent}%`;
  }

  if (elements.loadingSubtitle) {
    elements.loadingSubtitle.textContent =
      subtitle || (total > 0 ? `Loading ${loaded}/${total} files (${percent}%)` : "Loading data…");
  }

  if (elements.inlineLoader && !elements.inlineLoader.classList.contains("hidden")) {
    elements.inlineLoaderText.textContent =
      subtitle || (total > 0 ? `Loading remaining files (${percent}%)` : "Loading files…");
  }
}

function finishInitialLoadingUI() {
  if (elements.loadingOverlay) {
    elements.loadingOverlay.classList.add("hidden");
  }
}

function showInlineLoader(message) {
  if (!elements.inlineLoader) return;
  elements.inlineLoaderText.textContent = message;
  elements.inlineLoader.classList.remove("hidden");
}

function hideInlineLoader() {
  if (!elements.inlineLoader) return;
  elements.inlineLoader.classList.add("hidden");
  elements.inlineLoaderText.textContent = "";
}

const gridColumnDefs = [
  {
    headerName: "Score",
    field: "importance_score",
    width: 80,
    filter: "agNumberColumnFilter",
    cellClass: "score-cell",
    tooltipValueGetter: (params) => `Score: ${params.value ?? 0}`,
  },
  {
    headerName: "Headline",
    field: "headline",
    flex: 6,
    minWidth: 400,
    cellRenderer: (params) => {
      const headline = params.value || "Untitled lead";
      return `<strong class="cell-text">${headline}</strong>`;
    },
    tooltipValueGetter: (params) => params.data?.headline || "Untitled lead",
  },
  {
    headerName: "File",
    field: "filename",
    flex: 1,
    minWidth: 120,
    cellRenderer: (params) => `<span class="cell-text">${params.value || ""}</span>`,
    tooltipValueGetter: (params) => params.data?.filename || "",
  },
  {
    headerName: "Power Mentions",
    field: "power_mentions",
    flex: 2,
    minWidth: 180,
    cellRenderer: (params) => `<span class="cell-text">${(params.value || []).join(", ")}</span>`,
    tooltipValueGetter: (params) => (params.data?.power_mentions || []).join(", "),
  },
  {
    headerName: "Lead Types",
    field: "lead_types",
    flex: 1.2,
    minWidth: 150,
    cellRenderer: (params) => `<span class="cell-text">${(params.value || []).join(", ")}</span>`,
    tooltipValueGetter: (params) => (params.data?.lead_types || []).join(", "),
  },
  {
    headerName: "Agencies",
    field: "agency_involvement",
    flex: 1.2,
    minWidth: 140,
    cellRenderer: (params) => `<span class="cell-text">${(params.value || []).join(", ")}</span>`,
    tooltipValueGetter: (params) => (params.data?.agency_involvement || []).join(", "),
  },
  {
    headerName: "Tags",
    field: "tags",
    flex: 1.2,
    minWidth: 140,
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
    rowHeight: 58,
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

function buildPowerAliasLookup(aliasMap) {
  const lookup = new Map();
  Object.entries(aliasMap || {}).forEach(([canonical, aliases]) => {
    const canonicalKey = cleanPowerAlias(canonical);
    if (canonicalKey && !lookup.has(canonicalKey)) {
      lookup.set(canonicalKey, canonical);
    }
    (aliases || []).forEach((alias) => {
      const key = cleanPowerAlias(alias);
      if (!key) return;
      if (!lookup.has(key)) {
        lookup.set(key, canonical);
      }
    });
  });
  return lookup;
}

function buildCanonicalPowerList(lookup) {
  const list = [];
  const seen = new Set();
  lookup.forEach((canonical) => {
    const clean = cleanPowerAlias(canonical);
    if (!clean || seen.has(clean)) return;
    seen.add(clean);
    list.push({ canonical, clean });
  });
  return list;
}

function buildPowerKeywordMap(aliasMap) {
  const keywordMap = new Map();
  Object.entries(aliasMap || {}).forEach(([canonical, aliases]) => {
    const cleanCanonical = cleanPowerAlias(canonical);
    if (!cleanCanonical) return;
    const keywords = new Set();
    keywords.add(cleanCanonical);
    cleanCanonical.split(" ").forEach((token) => keywords.add(token));
    (aliases || []).forEach((alias) => {
      const cleanAlias = cleanPowerAlias(alias);
      if (!cleanAlias) return;
      keywords.add(cleanAlias);
      cleanAlias.split(" ").forEach((token) => keywords.add(token));
    });
    keywordMap.set(canonical, Array.from(keywords));
  });
  return keywordMap;
}

function cleanPowerAlias(name) {
  if (!name) return "";
  return String(name)
    .toLowerCase()
    .replace(/[–—]/g, "-")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizePowerMentions(values) {
  const normalized = [];
  const seen = new Set();
  (values || []).forEach((originalName) => {
    const candidates = generatePowerAliasCandidates(originalName);
    let canonical = null;
    for (const key of candidates.keys) {
      canonical = powerAliasLookup.get(key);
      if (canonical) break;
    }
    if (!canonical) {
      // Fallback: try token match on the best key
      const tokenFallback = findCanonicalByToken(candidates.bestKey);
      if (tokenFallback) {
        canonical = tokenFallback;
      }
    }
    if (!canonical) {
      canonical = originalName;
    }
    const display = typeof canonical === "string" ? canonical : originalName;
    const displayKey = cleanPowerAlias(display);
    if (displayKey && !seen.has(displayKey)) {
      seen.add(displayKey);
      normalized.push(display);
    }
  });
  return normalized;
}

function generatePowerAliasCandidates(name) {
  const result = { keys: [], bestKey: "" };
  if (!name) return result;
  const candidates = [];
  const trimmed = String(name).trim();
  candidates.push(trimmed);
  const flipped = flipCommaName(trimmed);
  if (flipped) candidates.push(flipped);
  for (const candidate of candidates) {
    const key = cleanPowerAlias(candidate);
    if (key) {
      result.keys.push(key);
      if (!result.bestKey) {
        result.bestKey = key;
      }
    }
  }
  return result;
}

function flipCommaName(name) {
  if (!name || !name.includes(",")) return null;
  const [last, rest] = name.split(",", 2).map((part) => part.trim());
  if (!last || !rest) return null;
  // Only flip simple personal-name patterns to avoid mangling organizations.
  const lastIsSingleWord = /^[A-Za-z'.-]+$/.test(last);
  const restWordCount = rest.split(/\s+/).filter(Boolean).length;
  if (!lastIsSingleWord || restWordCount === 0 || restWordCount > 3) {
    return null;
  }
  return `${rest} ${last}`.trim();
}

function findCanonicalByToken(key) {
  if (!key || key.length < 3) return null;
  const matches = canonicalPowerList.filter((item) => {
    const words = item.clean.split(" ");
    return words.includes(key);
  });
  if (matches.length === 1) {
    return matches[0].canonical;
  }
  return null;
}

function normalizeRow(row) {
  const importance = Number(row.importance_score ?? 0);
  const arrays = (value) => (Array.isArray(value) ? value : []);
  const rawPowers = arrays(row.power_mentions)
    .map((p) => (typeof p === "string" ? p.trim() : String(p ?? "").trim()))
    .filter(Boolean);
  const normalizedPowers = normalizePowerMentions(rawPowers);
  const normalized = {
    filename: row.filename,
    source_row_index: row.metadata?.source_row_index ?? null,
    headline: row.headline || row.metadata?.original_row?.filename || "Untitled lead",
    importance_score: Number.isFinite(importance) ? importance : 0,
    reason: row.reason || "",
    key_insights: arrays(row.key_insights),
    tags: arrays(row.tags),
    power_mentions_raw: rawPowers,
    power_mentions: normalizedPowers.length > 0 ? normalizedPowers : rawPowers,
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

function populateFilters(data, preserveSelection = false) {
  const prevLead = preserveSelection ? getSelectedValues(state.leadChoices) : [];
  const prevPower = preserveSelection ? getSelectedValues(state.powerChoices) : [];
  const leadTypeSet = new Set();
  const powerSet = new Set();
  data.forEach((row) => {
    row.lead_types.forEach((t) => leadTypeSet.add(t));
    row.power_mentions.forEach((p) => powerSet.add(p));
  });
  setChoiceOptions(state.leadChoices, Array.from(leadTypeSet).sort(), prevLead);
  setChoiceOptions(
    state.powerChoices,
    Array.from(powerSet).sort(),
    prevPower,
    powerKeywordMap
  );
}

function setChoiceOptions(choiceInstance, values, previouslySelected = [], keywordMap = null) {
  if (!choiceInstance) {
    return;
  }
  const selectedSet = new Set(
    previouslySelected.length > 0 ? previouslySelected : getSelectedValues(choiceInstance)
  );
  const options = values.map((value) => ({
    value,
    label: value,
    customProperties: keywordMap ? { keywords: keywordMap.get(value) || [] } : undefined,
    selected: selectedSet.has(value),
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
    const api = state.gridOptions.api;
    const columnApi = state.gridOptions.columnApi;
    api.setRowData(filtered);
    columnApi.applyColumnState({
      state: [{ colId: "importance_score", sort: "desc" }],
      defaultState: { sort: null },
    });
    let targetRow = null;
    if (state.activeRowId) {
      const rowNode = api.getRowNode(state.activeRowId);
      if (rowNode?.data) {
        targetRow = rowNode.data;
        api.ensureNodeVisible(rowNode);
      }
    }
    if (!targetRow && filtered.length > 0) {
      targetRow = filtered[0];
      api.ensureIndexVisible(0);
    }
    targetRow ? renderDetail(targetRow) : clearDetail();
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

  // Display count with total info if available
  const totalLoaded = state.raw.length;
  if (state.manifestMetadata && state.manifestMetadata.total_dataset_rows) {
    const totalDataset = state.manifestMetadata.total_dataset_rows;
    if (typeof totalDataset === 'number') {
      if (count === totalLoaded) {
        // No filters applied
        elements.countStat.textContent = `${count.toLocaleString()} of ${totalDataset.toLocaleString()} loaded`;
      } else {
        // Filters applied
        elements.countStat.textContent = `${count.toLocaleString()} of ${totalLoaded.toLocaleString()} loaded (${totalDataset.toLocaleString()} total)`;
      }
    } else {
      elements.countStat.textContent = `${count.toLocaleString()} (${totalLoaded.toLocaleString()} loaded)`;
    }
  } else {
    elements.countStat.textContent = `${count.toLocaleString()} (${totalLoaded.toLocaleString()} loaded)`;
  }

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
  const loadId = Date.now();
  state.currentLoadId = loadId;
  resetLoadingState("Loading Epstein Files…", "Fetching chunk manifest");
  try {
    const manifest = await fetchManifest();
    if (manifest && manifest.chunks && manifest.chunks.length > 0) {
      await loadChunks(manifest, loadId);
      finishInitialLoadingUI();
      return;
    }
  } catch (err) {
    console.warn("Chunk manifest unavailable, falling back to epstein_ranked.jsonl.", err);
  }

  try {
    await loadSequentialChunks(DEFAULT_CHUNK_SIZE, 500, loadId);
    finishInitialLoadingUI();
    return;
  } catch (seqErr) {
    console.warn("Sequential chunk scan failed, falling back to single file.", seqErr);
  }

  try {
    await loadSingleFile(loadId);
    finishInitialLoadingUI();
  } catch (error) {
    console.error("Failed to load data", error);
    finishInitialLoadingUI();
    hideInlineLoader();
    alert(
      "Unable to load ranked outputs. Ensure contrib/epstein_ranked_*.jsonl files or data/epstein_ranked.jsonl exist."
    );
  }
}

async function fetchManifest() {
  try {
    const response = await fetch(`${CHUNK_MANIFEST_URL}?t=${Date.now()}`);
    if (!response.ok) {
      return null;
    }
    const data = await response.json();

    // Handle both old format (array) and new format (object with metadata)
    if (Array.isArray(data)) {
      // Old format: just an array of chunks
      return { chunks: data, metadata: null };
    } else if (data.chunks && Array.isArray(data.chunks)) {
      // New format: object with metadata and chunks
      return { chunks: data.chunks, metadata: data.metadata || null };
    }

    return null;
  } catch (error) {
    return null;
  }
}

async function loadChunks(manifestData, loadId, initialChunkCount = 2) {
  const chunks = manifestData.chunks || manifestData;
  const metadata = manifestData.metadata || null;

  if (!Array.isArray(chunks) || chunks.length === 0) {
    throw new Error("Chunk manifest contained no readable data.");
  }

  state.manifestMetadata = metadata;
  state.loading.totalChunks = chunks.length;
  state.loading.loadedChunks = 0;
  updateLoadingProgress(0, chunks.length, "Fetching initial files…");

  const initialRows = [];
  const initialBatch = chunks.slice(0, Math.min(initialChunkCount, chunks.length));

  for (const entry of initialBatch) {
    const rows = await fetchChunkEntry(entry, loadId);
    if (loadId !== state.currentLoadId) return;
    if (rows.length > 0) {
      initialRows.push(...rows);
    }
    state.loading.loadedChunks += 1;
    updateLoadingProgress(state.loading.loadedChunks, state.loading.totalChunks, "Loading top-ranked files…");
  }

  if (initialRows.length === 0) {
    throw new Error("Chunk manifest contained no readable data.");
  }

  await hydrateRows(initialRows, { append: false });

  if (chunks.length <= initialBatch.length) {
    hideInlineLoader();
    return;
  }

  showInlineLoader("Loading remaining files…");
  const remainingChunks = chunks.slice(initialBatch.length);
  backgroundLoadChunks(remainingChunks, loadId).catch((error) => {
    console.warn("Background chunk load failed", error);
    hideInlineLoader();
  });
}

async function backgroundLoadChunks(chunks, loadId) {
  for (const entry of chunks) {
    if (loadId !== state.currentLoadId) return;
    const rows = await fetchChunkEntry(entry, loadId);
    if (rows.length > 0 && loadId === state.currentLoadId) {
      await hydrateRows(rows, { append: true, preserveFilters: true });
    }
    state.loading.loadedChunks += 1;
    updateLoadingProgress(state.loading.loadedChunks, state.loading.totalChunks, "Loading remaining files…");
  }
  if (loadId === state.currentLoadId) {
    hideInlineLoader();
  }
}

async function loadSequentialChunks(chunkSize = DEFAULT_CHUNK_SIZE, maxChunks = 500, loadId = state.currentLoadId) {
  const rows = [];
  let start = 1;
  let attempts = 0;
  let misses = 0;
  state.manifestMetadata = null;
  state.loading.totalChunks = maxChunks;
  state.loading.loadedChunks = 0;
  while (attempts < maxChunks && misses < 5) {
    if (loadId !== state.currentLoadId) {
      return;
    }
    const end = start + chunkSize - 1;
    const path = `contrib/epstein_ranked_${String(start).padStart(5, "0")}_${String(
      end
    ).padStart(5, "0")}.jsonl`;
    attempts += 1;
    try {
      const response = await fetch(`${path}?t=${Date.now()}`);
      if (!response.ok) {
        misses += 1;
        start += chunkSize;
        state.loading.loadedChunks += 1;
        updateLoadingProgress(state.loading.loadedChunks, state.loading.totalChunks, "Scanning local chunks…");
        continue;
      }
      const text = await response.text();
      rows.push(...parseJsonl(text));
      misses = 0;
      start += chunkSize;
      state.loading.loadedChunks += 1;
      updateLoadingProgress(state.loading.loadedChunks, state.loading.totalChunks, "Scanning local chunks…");
    } catch (error) {
      console.warn("Chunk scan error", path, error);
      misses += 1;
      start += chunkSize;
      state.loading.loadedChunks += 1;
      updateLoadingProgress(state.loading.loadedChunks, state.loading.totalChunks, "Scanning local chunks…");
    }
  }
  if (rows.length === 0) {
    throw new Error("No sequential chunks readable");
  }
  updateLoadingProgress(state.loading.loadedChunks, state.loading.loadedChunks, "Loaded local chunks");
  await hydrateRows(rows, { append: false });
}

async function loadSingleFile(loadId = state.currentLoadId) {
  const response = await fetch(`${DATA_URL}?t=${Date.now()}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch data: ${response.status}`);
  }
  const text = await response.text();
  const parsed = parseJsonl(text);
  if (loadId !== state.currentLoadId) return;
  updateLoadingProgress(1, 1, "Loaded ranked file");
  await hydrateRows(parsed, { append: false });
}

async function fetchChunkEntry(entry, loadId) {
  const jsonPath = resolveChunkPath(entry?.json);
  if (!jsonPath) return [];
  try {
    const response = await fetch(`${jsonPath}?t=${Date.now()}`);
    if (!response.ok) {
      console.warn("Failed to fetch chunk", jsonPath);
      return [];
    }
    const text = await response.text();
    if (loadId !== state.currentLoadId) {
      return [];
    }
    return parseJsonl(text);
  } catch (error) {
    console.warn("Chunk load error", jsonPath, error);
    return [];
  }
}

async function hydrateRows(rows, { append = false, preserveFilters = false } = {}) {
  const normalized = rows.map(normalizeRow);
  state.raw = append ? state.raw.concat(normalized) : normalized;
  state.lastUpdated = new Date();
  populateFilters(state.raw, preserveFilters);
  applyFilters();
}

function resolveChunkPath(path) {
  if (!path) return null;
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  // Remove leading ../ if present (we serve from viewer/ with symlinks)
  if (path.startsWith("../")) {
    return path.substring(3);
  }
  // Remove leading ./ if present
  if (path.startsWith("./")) {
    return path.substring(2);
  }
  // Return path as-is (should be relative to viewer/)
  return path;
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
  elements.detailClose.addEventListener("click", () => clearDetail());
  elements.detailTextToggle.addEventListener("click", toggleDetailText);
}

function toggleDetailText() {
  const isExpanded = !elements.detailText.classList.contains("hidden");
  if (isExpanded) {
    // Collapse
    elements.detailText.classList.add("hidden");
    elements.detailTextPreview.classList.remove("hidden");
    elements.detailTextToggle.textContent = "Expand";
  } else {
    // Expand
    elements.detailText.classList.remove("hidden");
    elements.detailTextPreview.classList.add("hidden");
    elements.detailTextToggle.textContent = "Collapse";
  }
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
    searchResultLimit: 500,
    renderChoiceLimit: 500,
    fuseOptions: {
      keys: ["label", "value", "customProperties.keywords"],
      threshold: 0.3,
      ignoreLocation: true,
    },
  });
  state.powerChoices = new Choices(elements.powerFilter, {
    removeItemButton: true,
    placeholder: true,
    placeholderValue: "Select power mentions",
    searchPlaceholderValue: "Search people/agencies",
    searchResultLimit: 500,
    renderChoiceLimit: 500,
    fuseOptions: {
      keys: ["label", "value", "customProperties.keywords"],
      threshold: 0.3,
      ignoreLocation: true,
    },
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
  state.activeRowId = row.filename || null;
  elements.detailDrawer.classList.remove("hidden");
  elements.detailTitle.textContent = `${row.headline || row.filename} (${row.filename})`;
  elements.detailReason.textContent = row.reason || "—";
  elements.detailLeadTypes.textContent = row.lead_types.join(", ") || "—";
  elements.detailPower.textContent = row.power_mentions.join(", ") || "—";
  elements.detailAgencies.textContent = row.agency_involvement.join(", ") || "—";
  elements.detailTags.textContent = row.tags.join(", ") || "—";

  // Display model if available in metadata
  const model = row.metadata?.config?.model || "—";
  elements.detailModel.textContent = model;

  // Handle original text with collapse/expand
  const originalText = row.original_text || "No source text captured.";
  const wordCount = originalText.split(/\s+/).filter(Boolean).length;
  const snippet = originalText.split(/\s+/).slice(0, 30).join(" ");

  elements.detailText.textContent = originalText;
  elements.detailTextPreview.textContent = `${snippet}... (${wordCount.toLocaleString()} words)`;

  // Reset to collapsed state
  elements.detailText.classList.add("hidden");
  elements.detailTextPreview.classList.remove("hidden");
  elements.detailTextToggle.textContent = "Expand";

  elements.detailInsights.innerHTML =
    row.key_insights.length > 0
      ? row.key_insights.map((item) => `<li>${escapeHtml(item)}</li>`).join("")
      : "<li>—</li>";
}

function clearDetail() {
  elements.detailDrawer.classList.add("hidden");
  state.activeRowId = null;
  elements.detailTitle.textContent = "Select a row to inspect full context";
  elements.detailReason.textContent = "—";
  elements.detailLeadTypes.textContent = "—";
  elements.detailPower.textContent = "—";
  elements.detailAgencies.textContent = "—";
  elements.detailTags.textContent = "—";
  elements.detailModel.textContent = "—";
  elements.detailText.textContent = "—";
  elements.detailTextPreview.textContent = "—";
  elements.detailText.classList.add("hidden");
  elements.detailTextPreview.classList.remove("hidden");
  elements.detailTextToggle.textContent = "Expand";
  elements.detailInsights.innerHTML = "";
}
