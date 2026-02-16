import { POWER_ALIASES } from "./power_aliases.js";

const DEFAULT_CHUNK_SIZE = 1000;
const DATASET_STORAGE_KEY = "epstein_viewer_dataset";
const ALL_VOLUMES_VALUE = "__all__";
const DATASETS = {
  oversight: {
    key: "oversight",
    label: "Oversight Committee",
    title: "U.S. House Oversight Epstein Estate Documents",
    subtitle:
      "AI-ranked analysis of 20,000+ pages released by the House Oversight Committee. These are the estate documents already made public—not the unreleased files still pending disclosure.",
    estimateLabel: "of ~25,800 documents.",
    manifestUrl: "/data/chunks.json",
    dataUrl: "/data/epstein_ranked.jsonl",
    fallbackChunkPrefix: "/contrib",
    fallbackChunkSize: 1000,
  },
  doj_fta: {
    key: "doj_fta",
    label: "DOJ File Transparency Act",
    title: "DOJ Epstein Files (File Transparency Act)",
    subtitle:
      "AI-ranked analysis of the DOJ File Transparency Act corpus. This dataset is processed independently from the House Oversight corpus.",
    estimateLabel: "of ~954,704+ documents.",
    manifestUrl: "/contrib/fta/chunks.json",
    dataUrl: "/data/workspaces/standardworks_epstein_files/results/epstein_ranked.jsonl",
    fallbackChunkPrefix: "/contrib/fta",
    fallbackChunkSize: 1000,
    volumeManifestPrefix: "/contrib/fta",
    volumeCount: 12,
  },
};
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
  datasetSelector: document.getElementById("datasetSelector"),
  volumeFilterGroup: document.getElementById("volumeFilterGroup"),
  volumeFilter: document.getElementById("volumeFilter"),
  datasetTitle: document.getElementById("datasetTitle"),
  datasetSubtitle: document.getElementById("datasetSubtitle"),
  processingLabel: document.getElementById("processingLabel"),
  processingIntro: document.getElementById("processingIntro"),
  datasetEstimate: document.getElementById("datasetEstimate"),
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
  detailSourceUrl: document.getElementById("detailSourceUrl"),
  detailTextPanel: document.getElementById("detailTextPanel"),
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
  processedCount: document.getElementById("processedCount"),
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
  filtersEnabled: false,
  activeDatasetKey: "oversight",
};

const powerAliasLookup = buildPowerAliasLookup(POWER_ALIASES);
const canonicalPowerList = buildCanonicalPowerList(powerAliasLookup);
const powerKeywordMap = buildPowerKeywordMap(POWER_ALIASES, powerAliasLookup);
const canonicalAliasKeyMap = buildCanonicalAliasKeyMap(POWER_ALIASES);

function getActiveDataset() {
  return DATASETS[state.activeDatasetKey] || DATASETS.oversight;
}

function updateDatasetCopy() {
  const dataset = getActiveDataset();
  document.title = `${dataset.title} - Intelligence Briefing`;
  if (elements.datasetTitle) {
    elements.datasetTitle.textContent = dataset.title;
  }
  if (elements.datasetSubtitle) {
    elements.datasetSubtitle.textContent = dataset.subtitle;
  }
  if (elements.processingLabel) {
    elements.processingLabel.textContent = "Note:";
  }
  if (elements.processingIntro) {
    elements.processingIntro.textContent =
      "We are still processing the remaining files. Currently showing";
  }
  if (elements.datasetEstimate) {
    elements.datasetEstimate.textContent = dataset.estimateLabel;
  }
  if (elements.datasetSelector) {
    elements.datasetSelector.value = dataset.key;
  }
  updateVolumeFilterVisibility();
}

function resetDatasetState() {
  state.raw = [];
  state.filtered = [];
  state.lastUpdated = null;
  state.manifestMetadata = null;
  state.activeRowId = null;
  if (state.gridOptions?.api) {
    state.gridOptions.api.setRowData([]);
  }
  updateSummary();
  clearDetail();
}

function resetLoadingState(title = "Loading Epstein Files…", subtitle = "Preparing data") {
  state.powerDisplayNames = {};
  state.loading.loadedChunks = 0;
  state.loading.totalChunks = 0;
  setFiltersEnabled(false);
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

function setFiltersEnabled(enabled, { triggerApply = false } = {}) {
  state.filtersEnabled = !!enabled;
  const controls = [
    elements.scoreFilter,
    elements.volumeFilter,
    elements.leadFilter,
    elements.powerFilter,
    elements.searchInput,
    elements.limitInput,
    elements.resetFilters,
  ];
  controls.forEach((el) => {
    if (el) el.disabled = !enabled;
  });
  if (state.leadChoices) {
    enabled ? state.leadChoices.enable() : state.leadChoices.disable();
  }
  if (state.powerChoices) {
    enabled ? state.powerChoices.enable() : state.powerChoices.disable();
  }
  const waitNote = document.getElementById("filterWaitNote");
  if (waitNote) {
    waitNote.classList.toggle("hidden", enabled);
  }
  if (enabled && triggerApply) {
    applyFilters({ force: true });
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

function isMobileView() {
  return window.innerWidth <= 768;
}

function getGridColumnDefs() {
  const isMobile = isMobileView();
  return [
    {
      headerName: "Score",
      field: "importance_score",
      width: isMobile ? 65 : 80,
      minWidth: isMobile ? 65 : 80,
      maxWidth: isMobile ? 65 : 100,
      filter: "agNumberColumnFilter",
      cellClass: "score-cell",
      pinned: isMobile ? "left" : null,
      tooltipValueGetter: (params) => `Score: ${params.value ?? 0}`,
    },
    {
      headerName: "Volume",
      field: "volume_label",
      width: 115,
      minWidth: 95,
      maxWidth: 135,
      hide: isMobile || state.activeDatasetKey !== "doj_fta",
      cellRenderer: (params) => `<span class="cell-text">${params.value || "—"}</span>`,
      tooltipValueGetter: (params) => params.data?.volume_label || "—",
    },
    {
      headerName: "Headline",
      field: "headline",
      flex: isMobile ? 1 : 6,
      minWidth: isMobile ? 200 : 400,
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
      hide: isMobile,
      cellRenderer: (params) => `<span class="cell-text">${params.value || ""}</span>`,
      tooltipValueGetter: (params) => params.data?.filename || "",
    },
    {
      headerName: "Power Mentions",
      field: "power_mentions",
      flex: 2,
      minWidth: 180,
      hide: isMobile,
      cellRenderer: (params) => `<span class="cell-text">${(params.value || []).join(", ")}</span>`,
      tooltipValueGetter: (params) => (params.data?.power_mentions || []).join(", "),
    },
    {
      headerName: "Lead Types",
      field: "lead_types",
      flex: 1.2,
      minWidth: 150,
      hide: isMobile,
      cellRenderer: (params) => `<span class="cell-text">${(params.value || []).join(", ")}</span>`,
      tooltipValueGetter: (params) => (params.data?.lead_types || []).join(", "),
    },
    {
      headerName: "Agencies",
      field: "agency_involvement",
      flex: 1.2,
      minWidth: 140,
      hide: isMobile,
      cellRenderer: (params) => `<span class="cell-text">${(params.value || []).join(", ")}</span>`,
      tooltipValueGetter: (params) => (params.data?.agency_involvement || []).join(", "),
    },
    {
      headerName: "Tags",
      field: "tags",
      flex: 1.2,
      minWidth: 140,
      hide: isMobile,
      cellRenderer: (params) => `<span class="cell-text">${(params.value || []).join(", ")}</span>`,
      tooltipValueGetter: (params) => (params.data?.tags || []).join(", "),
    },
  ];
}

function initGrid() {
  const gridElement = document.getElementById("grid");
  const isMobile = isMobileView();
  state.gridOptions = {
    columnDefs: getGridColumnDefs(),
    defaultColDef: {
      resizable: true,
      sortable: true,
      filter: true,
      flex: 1,
      minWidth: isMobile ? 80 : 130,
      wrapText: false,
      autoHeight: false,
      tooltipComponentParams: { color: "#fff" },
    },
    suppressMovableColumns: true,
    animateRows: true,
    pagination: true,
    paginationPageSize: isMobile ? 15 : 25,
    rowHeight: isMobile ? 50 : 58,
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
    onRowClicked: (event) => renderDetail(event.data, { scrollToDetail: true }),
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

function buildCanonicalAliasKeyMap(aliasMap) {
  const map = new Map();
  Object.entries(aliasMap || {}).forEach(([canonical, aliases]) => {
    const canonicalKey = cleanPowerAlias(canonical);
    if (!canonicalKey) return;
    const keys = new Set();
    keys.add(canonicalKey);
    (aliases || []).forEach((alias) => {
      const key = cleanPowerAlias(alias);
      if (key) {
        keys.add(key);
      }
    });
    map.set(canonical, keys);
  });
  return map;
}

function canonicalizePowerSelection(value) {
  const candidates = generatePowerAliasCandidates(value);
  for (const key of candidates.keys) {
    const canonical = powerAliasLookup.get(key);
    if (canonical) return canonical;
  }
  const fallback = findCanonicalByToken(candidates.bestKey);
  return fallback || value;
}

function aliasKeysForCanonical(canonical) {
  const canonicalKey = cleanPowerAlias(canonical);
  if (!canonicalKey) return new Set();

  if (canonicalAliasKeyMap.has(canonical)) {
    return new Set(canonicalAliasKeyMap.get(canonical));
  }
  return new Set([canonicalKey]);
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

function expandPowerSelection(values) {
  const canonicalKeys = new Set();
  const aliasKeys = new Set();
  (values || []).forEach((value) => {
    const selectedCanonical = canonicalizePowerSelection(value);
    const canonicalKey = cleanPowerAlias(selectedCanonical);
    if (!canonicalKey) return;
    canonicalKeys.add(canonicalKey);
    aliasKeysForCanonical(selectedCanonical).forEach((key) => aliasKeys.add(key));
  });
  return { canonicalKeys, aliasKeys };
}

function matchesPowerSelection(name, selection) {
  const canonical = canonicalizePowerSelection(name);
  const canonicalKey = cleanPowerAlias(canonical);
  if (!canonicalKey || !selection || selection.aliasKeys.size === 0) return false;
  if (selection.aliasKeys.has(canonicalKey)) return true;
  const rowAliasKeys = aliasKeysForCanonical(canonical);
  return Array.from(rowAliasKeys).some((key) => selection.aliasKeys.has(key));
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

function deriveJusticePdfUrl(filename) {
  if (!filename) return null;
  const datasetMatch = String(filename).match(/DataSet\\s*0*(\\d+)/i);
  const eftaMatch = String(filename).match(/(EFTA\\d{8})/i);
  if (!datasetMatch || !eftaMatch) return null;
  const datasetNumber = Number.parseInt(datasetMatch[1], 10);
  if (!Number.isFinite(datasetNumber) || datasetNumber < 1) return null;
  const eftaId = eftaMatch[1].toUpperCase();
  return `https://www.justice.gov/epstein/files/DataSet%20${datasetNumber}/${eftaId}.pdf`;
}

function deriveVolumeLabel(row) {
  const formatVolume = (num) => `VOL${String(num).padStart(5, "0")}`;
  const readVolume = (value) => {
    if (!value) return null;
    const text = String(value);
    const volMatch = text.match(/(?:^|[_/\\-])vol(?:ume)?[_-]?0*(\d{1,5})(?:$|[_/\\-])/i);
    if (volMatch) {
      const n = Number.parseInt(volMatch[1], 10);
      if (Number.isFinite(n) && n > 0) {
        return formatVolume(n);
      }
    }
    const datasetMatch = text.match(/DataSet(?:%20|\s)*0*(\d{1,5})/i);
    if (datasetMatch) {
      const n = Number.parseInt(datasetMatch[1], 10);
      if (Number.isFinite(n) && n > 0) {
        return formatVolume(n);
      }
    }
    return null;
  };

  return (
    readVolume(row?.metadata?.config?.dataset_tag) ||
    readVolume(row?.filename) ||
    readVolume(row?.source_pdf_url) ||
    readVolume(row?.metadata?.source_pdf_url) ||
    ""
  );
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
    volume_label: deriveVolumeLabel(row),
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
    source_pdf_url:
      row.source_pdf_url ||
      row.metadata?.source_pdf_url ||
      deriveJusticePdfUrl(row.filename || row.metadata?.original_row?.filename || ""),
    original_text:
      typeof row.metadata?.original_row?.text === "string"
        ? row.metadata.original_row.text
        : "",
  };
  normalized.search_blob = [
    normalized.volume_label,
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
  const leadCounts = buildCountMap(data, "lead_types");
  const powerCounts = buildCountMap(data, "power_mentions");

  const sortedLeads = sortValuesByCount(Array.from(leadCounts.keys()), leadCounts);
  const sortedPowers = sortValuesByCount(Array.from(powerCounts.keys()), powerCounts);
  populateVolumeFilter(data, preserveSelection);

  setChoiceOptions(state.leadChoices, sortedLeads, prevLead, null, leadCounts, leadCounts);
  setChoiceOptions(
    state.powerChoices,
    sortedPowers,
    prevPower,
    powerKeywordMap,
    powerCounts,
    powerCounts
  );
}

function sortVolumeLabels(values) {
  return Array.from(values).sort((a, b) => {
    const aNum = Number.parseInt(String(a).replace(/\D+/g, ""), 10);
    const bNum = Number.parseInt(String(b).replace(/\D+/g, ""), 10);
    if (Number.isFinite(aNum) && Number.isFinite(bNum) && aNum !== bNum) {
      return aNum - bNum;
    }
    return String(a).localeCompare(String(b));
  });
}

function populateVolumeFilter(data, preserveSelection = false) {
  if (!elements.volumeFilter) return;
  const previous = preserveSelection ? elements.volumeFilter.value : ALL_VOLUMES_VALUE;
  const volumeLabels = sortVolumeLabels(
    new Set(
      data
        .map((row) => row.volume_label)
        .filter((value) => typeof value === "string" && value.trim().length > 0)
    )
  );
  const options = [
    `<option value="${ALL_VOLUMES_VALUE}">All volumes</option>`,
    ...volumeLabels.map((value) => `<option value="${value}">${value}</option>`),
  ];
  elements.volumeFilter.innerHTML = options.join("");
  const nextValue = volumeLabels.includes(previous) ? previous : ALL_VOLUMES_VALUE;
  elements.volumeFilter.value = nextValue;
}

function updateVolumeFilterVisibility() {
  const isDoj = state.activeDatasetKey === "doj_fta";
  if (elements.volumeFilterGroup) {
    elements.volumeFilterGroup.classList.toggle("hidden", !isDoj);
  }
  if (!isDoj && elements.volumeFilter) {
    elements.volumeFilter.value = ALL_VOLUMES_VALUE;
  }
  if (state.gridOptions?.columnApi) {
    state.gridOptions.columnApi.setColumnsVisible(["volume_label"], isDoj && !isMobileView());
  }
}

function setChoiceOptions(
  choiceInstance,
  values,
  previouslySelected = [],
  keywordMap = null,
  countMap = null,
  baseCountMap = null
) {
  if (!choiceInstance) {
    return;
  }
  const selectedSet = new Set(
    previouslySelected.length > 0 ? previouslySelected : getSelectedValues(choiceInstance)
  );
  const options = values.map((value) => {
    const customProps = {};
    if (keywordMap) {
      customProps.keywords = keywordMap.get(value) || [];
    }
    if (countMap) {
      customProps.count = countMap.get ? countMap.get(value) || 0 : countMap[value] || 0;
    }
    if (baseCountMap) {
      customProps.baseCount = baseCountMap.get
        ? baseCountMap.get(value) || 0
        : baseCountMap[value] || 0;
    }
    return {
      value,
      label: value,
      customProperties: Object.keys(customProps).length > 0 ? customProps : undefined,
      selected: selectedSet.has(value),
    };
  });
  refreshChoices(choiceInstance, options, Array.from(selectedSet));
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

function applyFilters(options = {}) {
  const force = options.force || false;
  if (!state.filtersEnabled && !force) return;
  const minScore = Number(elements.scoreFilter.value) || 0;
  elements.scoreValue.textContent = minScore.toString();
  const leadSelected = new Set(getSelectedValues(state.leadChoices));
  const powerSelectedRaw = getSelectedValues(state.powerChoices);
  const powerSelection = expandPowerSelection(powerSelectedRaw);
  const selectedVolume = elements.volumeFilter?.value || ALL_VOLUMES_VALUE;
  const limit = Number(elements.limitInput.value) || null;
  const term = elements.searchInput.value.trim().toLowerCase();

  let filtered = state.raw.filter((row) => row.importance_score >= minScore);
  if (leadSelected.size > 0) {
    filtered = filtered.filter((row) => row.lead_types.some((lead) => leadSelected.has(lead)));
  }
  if (powerSelection.aliasKeys.size > 0) {
    filtered = filtered.filter((row) =>
      row.power_mentions.some((name) => matchesPowerSelection(name, powerSelection))
    );
  }
  if (selectedVolume !== ALL_VOLUMES_VALUE) {
    filtered = filtered.filter((row) => row.volume_label === selectedVolume);
  }
  if (term) {
    filtered = filtered.filter((row) => row.search_blob.includes(term));
  }
  if (limit && limit > 0) {
    filtered = filtered.slice(0, limit);
  }
  filtered.sort((a, b) => b.importance_score - a.importance_score);

  state.filtered = filtered;
  updateChoiceOrdering(filtered, leadSelected, powerSelectedRaw);
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

  // Update processed count in the notice
  if (elements.processedCount) {
    elements.processedCount.textContent = totalLoaded.toLocaleString();
  }
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

function buildCountMap(rows, field) {
  const map = new Map();
  rows.forEach((row) => {
    (row[field] || []).forEach((value) => {
      map.set(value, (map.get(value) || 0) + 1);
    });
  });
  return map;
}

function sortValuesByCount(values, primaryCounts, secondaryCounts = null) {
  return Array.from(values).sort((a, b) => {
    const aPrimary = primaryCounts?.get ? primaryCounts.get(a) || 0 : 0;
    const bPrimary = primaryCounts?.get ? primaryCounts.get(b) || 0 : 0;
    if (aPrimary !== bPrimary) return bPrimary - aPrimary;
    if (secondaryCounts) {
      const aSecondary = secondaryCounts.get ? secondaryCounts.get(a) || 0 : 0;
      const bSecondary = secondaryCounts.get ? secondaryCounts.get(b) || 0 : 0;
      if (aSecondary !== bSecondary) return bSecondary - aSecondary;
    }
    return a.localeCompare(b);
  });
}

function refreshChoices(choiceInstance, options, selectedValues = []) {
  if (!choiceInstance) return;
  const uniqueSelected = Array.from(new Set(selectedValues));
  choiceInstance.clearStore();
  choiceInstance.setChoices(options, "value", "label", true);
  if (uniqueSelected.length > 0) {
    choiceInstance.setChoiceByValue(uniqueSelected);
  }
}

function updateChoiceOrdering(filteredRows, leadSelectedSet, powerSelectedRaw) {
  const leadCountsAll = buildCountMap(state.raw, "lead_types");
  const powerCountsAll = buildCountMap(state.raw, "power_mentions");
  const leadCountsFiltered = buildCountMap(filteredRows, "lead_types");
  const powerCountsFiltered = buildCountMap(filteredRows, "power_mentions");

  const sortedLeads = sortValuesByCount(
    Array.from(leadCountsAll.keys()),
    leadCountsFiltered,
    leadCountsAll
  );
  const sortedPowers = sortValuesByCount(
    Array.from(powerCountsAll.keys()),
    powerCountsFiltered,
    powerCountsAll
  );

  setChoiceOptions(
    state.leadChoices,
    sortedLeads,
    Array.from(leadSelectedSet || []),
    null,
    leadCountsFiltered,
    leadCountsAll
  );
  setChoiceOptions(
    state.powerChoices,
    sortedPowers,
    powerSelectedRaw || [],
    powerKeywordMap,
    powerCountsFiltered,
    powerCountsAll
  );
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
      maintainAspectRatio: false,
      indexAxis: "y",
      layout: {
        padding: { top: 10, bottom: 20, left: 10, right: 10 },
      },
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
      maintainAspectRatio: false,
      layout: {
        padding: { top: 10, bottom: 20, left: 10, right: 10 },
      },
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
      maintainAspectRatio: false,
      indexAxis: "y",
      layout: {
        padding: { top: 10, bottom: 20, left: 10, right: 10 },
      },
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
          backgroundColor: "rgba(153, 102, 255, 0.7)",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: "y",
      layout: {
        padding: { top: 10, bottom: 20, left: 10, right: 10 },
      },
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
  const dataset = getActiveDataset();
  const loadId = Date.now();
  state.currentLoadId = loadId;
  resetDatasetState();
  resetLoadingState(`Loading ${dataset.label}…`, "Fetching chunk manifest");
  try {
    const manifest = await fetchDatasetManifest(dataset);
    if (manifest && manifest.chunks && manifest.chunks.length > 0) {
      await loadChunks(manifest, loadId);
      finishInitialLoadingUI();
      return;
    }
  } catch (err) {
    console.warn(`Chunk manifest unavailable for ${dataset.label}, falling back to JSONL.`, err);
  }

  try {
    await loadSequentialChunks(
      dataset.fallbackChunkSize || DEFAULT_CHUNK_SIZE,
      500,
      loadId,
      dataset.fallbackChunkPrefix
    );
    finishInitialLoadingUI();
    setFiltersEnabled(true, { triggerApply: true });
    return;
  } catch (seqErr) {
    console.warn("Sequential chunk scan failed, falling back to single file.", seqErr);
  }

  if (dataset.volumeManifestPrefix && dataset.volumeCount) {
    try {
      await loadSequentialVolumeChunks(
        dataset.volumeManifestPrefix,
        dataset.volumeCount,
        dataset.fallbackChunkSize || DEFAULT_CHUNK_SIZE,
        500,
        loadId
      );
      finishInitialLoadingUI();
      setFiltersEnabled(true, { triggerApply: true });
      return;
    } catch (volErr) {
      console.warn("Per-volume chunk scan failed, falling back to single file.", volErr);
    }
  }

  try {
    await loadSingleFile(loadId, dataset.dataUrl);
    finishInitialLoadingUI();
    setFiltersEnabled(true, { triggerApply: true });
  } catch (error) {
    console.error("Failed to load data", error);
    finishInitialLoadingUI();
    hideInlineLoader();
    const extraHint =
      dataset.key === "doj_fta"
        ? ` Ensure files like ${dataset.fallbackChunkPrefix}/VOL00001/epstein_ranked_00001_01000.jsonl exist, or provide ${dataset.manifestUrl}.`
        : "";
    alert(
      `Unable to load ranked outputs for ${dataset.label}. Ensure chunk files under ${dataset.fallbackChunkPrefix}/epstein_ranked_*.jsonl or ${dataset.dataUrl} exist.${extraHint}`
    );
  }
}

async function fetchDatasetManifest(dataset) {
  const primary = await fetchManifest(dataset.manifestUrl);
  if (primary && primary.chunks && primary.chunks.length > 0) {
    return primary;
  }
  if (dataset.volumeManifestPrefix && dataset.volumeCount) {
    const merged = await fetchVolumeManifests(dataset.volumeManifestPrefix, dataset.volumeCount);
    if (merged && merged.chunks && merged.chunks.length > 0) {
      return merged;
    }
  }
  return primary;
}

async function fetchVolumeManifests(prefix, volumeCount) {
  const manifests = [];
  for (let i = 1; i <= volumeCount; i += 1) {
    const vol = String(i).padStart(5, "0");
    const manifestUrl = `${prefix}/VOL${vol}/chunks.json`;
    const manifest = await fetchManifest(manifestUrl);
    if (!manifest || !Array.isArray(manifest.chunks) || manifest.chunks.length === 0) {
      continue;
    }
    manifests.push(manifest);
  }

  if (manifests.length === 0) {
    return null;
  }

  const deduped = new Map();
  let rowsProcessed = 0;
  let lastUpdated = null;

  for (const manifest of manifests) {
    for (const chunk of manifest.chunks) {
      if (!chunk || !chunk.json) continue;
      const key = `${chunk.start_row}:${chunk.end_row}:${chunk.json}`;
      if (!deduped.has(key)) {
        deduped.set(key, chunk);
      }
    }
    const m = manifest.metadata || {};
    if (typeof m.rows_processed === "number") {
      rowsProcessed += m.rows_processed;
    }
    if (m.last_updated && (!lastUpdated || String(m.last_updated) > String(lastUpdated))) {
      lastUpdated = m.last_updated;
    }
  }

  const chunks = Array.from(deduped.values()).sort(
    (a, b) => (a.start_row || 0) - (b.start_row || 0)
  );

  return {
    metadata: {
      total_dataset_rows: "unknown",
      rows_processed: rowsProcessed,
      last_updated: lastUpdated,
    },
    chunks,
  };
}

async function fetchManifest(manifestUrl) {
  try {
    const response = await fetch(`${manifestUrl}?t=${Date.now()}`);
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
    setFiltersEnabled(true, { triggerApply: true });
    return;
  }

  showInlineLoader("Loading remaining files…");
  const remainingChunks = chunks.slice(initialBatch.length);
  backgroundLoadChunks(remainingChunks, loadId).catch((error) => {
    console.warn("Background chunk load failed", error);
    hideInlineLoader();
  });
}

async function backgroundLoadChunks(chunks, loadId, concurrency = 8) {
  if (!Array.isArray(chunks) || chunks.length === 0) {
    setFiltersEnabled(true, { triggerApply: true });
    return;
  }

  let index = 0;
  const worker = async () => {
    while (index < chunks.length) {
      const current = index;
      index += 1;
      const entry = chunks[current];
      if (loadId !== state.currentLoadId) return;
      try {
        const rows = await fetchChunkEntry(entry, loadId);
        if (rows.length > 0 && loadId === state.currentLoadId) {
          await hydrateRows(rows, { append: true, preserveFilters: true });
        }
      } finally {
        state.loading.loadedChunks += 1;
        updateLoadingProgress(
          state.loading.loadedChunks,
          state.loading.totalChunks,
          "Loading remaining files…"
        );
      }
    }
  };

  const workers = Array.from({ length: Math.min(concurrency, chunks.length) }, () => worker());
  await Promise.all(workers);

  if (loadId === state.currentLoadId) {
    hideInlineLoader();
    setFiltersEnabled(true, { triggerApply: true });
  }
}

async function loadSequentialChunks(
  chunkSize = DEFAULT_CHUNK_SIZE,
  maxChunks = 500,
  loadId = state.currentLoadId,
  chunkPrefix = "/contrib"
) {
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
    const path = `${chunkPrefix}/epstein_ranked_${String(start).padStart(5, "0")}_${String(
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

async function loadSequentialVolumeChunks(
  prefix,
  volumeCount,
  chunkSize = DEFAULT_CHUNK_SIZE,
  maxChunksPerVolume = 500,
  loadId = state.currentLoadId
) {
  const rows = [];
  let totalAttempts = 0;
  state.manifestMetadata = null;
  state.loading.totalChunks = volumeCount * maxChunksPerVolume;
  state.loading.loadedChunks = 0;

  for (let i = 1; i <= volumeCount; i += 1) {
    if (loadId !== state.currentLoadId) {
      return;
    }
    const vol = String(i).padStart(5, "0");
    let start = 1;
    let attempts = 0;
    let misses = 0;

    while (attempts < maxChunksPerVolume && misses < 5) {
      if (loadId !== state.currentLoadId) {
        return;
      }
      const end = start + chunkSize - 1;
      const path = `${prefix}/VOL${vol}/epstein_ranked_${String(start).padStart(5, "0")}_${String(
        end
      ).padStart(5, "0")}.jsonl`;
      attempts += 1;
      totalAttempts += 1;
      try {
        const response = await fetch(`${path}?t=${Date.now()}`);
        if (!response.ok) {
          misses += 1;
          start += chunkSize;
          state.loading.loadedChunks += 1;
          updateLoadingProgress(
            state.loading.loadedChunks,
            state.loading.totalChunks,
            `Scanning volume VOL${vol}…`
          );
          continue;
        }
        const text = await response.text();
        rows.push(...parseJsonl(text));
        misses = 0;
        start += chunkSize;
        state.loading.loadedChunks += 1;
        updateLoadingProgress(
          state.loading.loadedChunks,
          state.loading.totalChunks,
          `Scanning volume VOL${vol}…`
        );
      } catch (error) {
        misses += 1;
        start += chunkSize;
        state.loading.loadedChunks += 1;
        updateLoadingProgress(
          state.loading.loadedChunks,
          state.loading.totalChunks,
          `Scanning volume VOL${vol}…`
        );
      }
    }
  }

  if (rows.length === 0) {
    throw new Error("No per-volume chunks readable");
  }

  if (totalAttempts > 0) {
    updateLoadingProgress(state.loading.loadedChunks, state.loading.loadedChunks, "Loaded local chunks");
  }
  await hydrateRows(rows, { append: false });
}

async function loadSingleFile(loadId = state.currentLoadId, dataUrl = "/data/epstein_ranked.jsonl") {
  const response = await fetch(`${dataUrl}?t=${Date.now()}`);
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
  applyFilters({ force: true });
}

function resolveChunkPath(path) {
  if (!path) return null;
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  // Remove leading ../ if present
  if (path.startsWith("../")) {
    path = path.substring(3);
  }
  // Remove leading ./ if present
  if (path.startsWith("./")) {
    path = path.substring(2);
  }
  // Make absolute if not already
  if (!path.startsWith("/")) {
    return "/" + path;
  }
  return path;
}

function resetFilters() {
  elements.scoreFilter.value = 0;
  elements.scoreValue.textContent = "0";
  if (elements.volumeFilter) {
    elements.volumeFilter.value = ALL_VOLUMES_VALUE;
  }
  state.leadChoices?.removeActiveItems();
  state.powerChoices?.removeActiveItems();
  elements.searchInput.value = "";
  elements.limitInput.value = "";
  applyFilters({ force: true });
}

async function switchDataset(nextKey) {
  if (!DATASETS[nextKey] || state.activeDatasetKey === nextKey) {
    return;
  }
  state.activeDatasetKey = nextKey;
  window.localStorage.setItem(DATASET_STORAGE_KEY, nextKey);
  updateDatasetCopy();
  resetFilters();
  await loadData();
}

function initDatasetSelector() {
  if (!elements.datasetSelector) {
    return;
  }
  const options = Object.values(DATASETS).map(
    (dataset) => `<option value="${dataset.key}">${dataset.label}</option>`
  );
  elements.datasetSelector.innerHTML = options.join("");

  const stored = window.localStorage.getItem(DATASET_STORAGE_KEY);
  const preferred = DATASETS[stored] ? stored : "oversight";
  state.activeDatasetKey = preferred;
  updateDatasetCopy();

  elements.datasetSelector.addEventListener("change", (event) => {
    const nextKey = event.target.value;
    switchDataset(nextKey).catch((err) => {
      console.error("Failed to switch dataset", err);
    });
  });
}

function wireEvents() {
  ["change", "input"].forEach((eventName) => {
    elements.scoreFilter.addEventListener(eventName, applyFilters);
    elements.searchInput.addEventListener(eventName, debounce(applyFilters, 200));
    elements.limitInput.addEventListener(eventName, debounce(applyFilters, 200));
  });
  elements.leadFilter.addEventListener("change", applyFilters);
  elements.powerFilter.addEventListener("change", applyFilters);
  elements.volumeFilter?.addEventListener("change", applyFilters);
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
  initDatasetSelector();
  wireEvents();
  loadData();
});

function initChoices() {
  const frequencySorter = (a, b) => {
    const countA = a.customProperties?.count ?? 0;
    const countB = b.customProperties?.count ?? 0;
    if (countA !== countB) return countB - countA;
    const baseA = a.customProperties?.baseCount ?? 0;
    const baseB = b.customProperties?.baseCount ?? 0;
    if (baseA !== baseB) return baseB - baseA;
    const indexA = a.customProperties?.originalIndex ?? 9999;
    const indexB = b.customProperties?.originalIndex ?? 9999;
    return indexA - indexB;
  };

  state.leadChoices = new Choices(elements.leadFilter, {
    removeItemButton: true,
    placeholder: true,
    placeholderValue: "Select leads…",
    searchPlaceholderValue: "Search…",
    shouldSort: true,
    sorter: frequencySorter,
    searchResultLimit: 500,
    renderChoiceLimit: 500,
    fuseOptions: {
      keys: ["label", "value", "customProperties.keywords"],
      threshold: 0.3,
      ignoreLocation: true,
      shouldSort: false,
    },
  });
  state.powerChoices = new Choices(elements.powerFilter, {
    removeItemButton: true,
    placeholder: true,
    placeholderValue: "Select names…",
    searchPlaceholderValue: "Search…",
    shouldSort: true,
    sorter: frequencySorter,
    searchResultLimit: 500,
    renderChoiceLimit: 500,
    fuseOptions: {
      keys: ["label", "value", "customProperties.keywords"],
      threshold: 0.3,
      ignoreLocation: true,
      shouldSort: false,
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

function getHighlightTerms() {
  const terms = [];

  // Add search input term
  const searchTerm = elements.searchInput?.value?.trim();
  if (searchTerm) {
    terms.push(searchTerm);
  }

  // Add selected power mentions
  const selectedPowers = getSelectedValues(state.powerChoices);
  selectedPowers.forEach(power => {
    if (power) {
      terms.push(power);
      // Also add individual words from multi-word names
      const words = power.split(/\s+/).filter(w => w.length > 2);
      terms.push(...words);
    }
  });

  // Add selected lead types
  const selectedLeads = getSelectedValues(state.leadChoices);
  selectedLeads.forEach(lead => {
    if (lead) {
      terms.push(lead);
    }
  });

  return terms.filter(Boolean);
}

function highlightText(text, terms) {
  if (!text || !terms || terms.length === 0) {
    return escapeHtml(text);
  }

  // Escape the text first
  let result = escapeHtml(text);

  // Sort terms by length (longest first) to avoid partial replacements
  const sortedTerms = [...new Set(terms)].sort((a, b) => b.length - a.length);

  // Create a map to store positions and terms
  const matches = [];

  sortedTerms.forEach(term => {
    const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const regex = new RegExp(escaped, 'gi');
    let match;

    const lowerText = text.toLowerCase();
    const lowerTerm = term.toLowerCase();
    let searchPos = 0;

    while ((match = lowerText.indexOf(lowerTerm, searchPos)) !== -1) {
      matches.push({
        start: match,
        end: match + term.length,
        term: text.substring(match, match + term.length)
      });
      searchPos = match + 1;
    }
  });

  // Sort matches by position
  matches.sort((a, b) => a.start - b.start);

  // Remove overlapping matches
  const filteredMatches = [];
  let lastEnd = -1;
  matches.forEach(match => {
    if (match.start >= lastEnd) {
      filteredMatches.push(match);
      lastEnd = match.end;
    }
  });

  // Build the highlighted text
  if (filteredMatches.length === 0) {
    return result;
  }

  let highlighted = '';
  let lastIndex = 0;

  filteredMatches.forEach(match => {
    // Add text before the match
    highlighted += escapeHtml(text.substring(lastIndex, match.start));
    // Add highlighted match
    highlighted += '<mark class="highlight">' + escapeHtml(match.term) + '</mark>';
    lastIndex = match.end;
  });

  // Add remaining text
  highlighted += escapeHtml(text.substring(lastIndex));

  return highlighted;
}

function renderDetail(row, options = {}) {
  if (!row) {
    clearDetail();
    return;
  }
  state.activeRowId = row.filename || null;
  elements.detailDrawer.classList.remove("hidden");

  // Get terms to highlight
  const highlightTerms = getHighlightTerms();

  const headlineText = `${row.headline || row.filename} (${row.filename})`;
  elements.detailTitle.innerHTML = highlightText(headlineText, highlightTerms);
  elements.detailReason.innerHTML = highlightText(row.reason || "—", highlightTerms);
  elements.detailLeadTypes.innerHTML = highlightText(row.lead_types.join(", ") || "—", highlightTerms);
  elements.detailPower.innerHTML = highlightText(row.power_mentions.join(", ") || "—", highlightTerms);
  elements.detailAgencies.innerHTML = highlightText(row.agency_involvement.join(", ") || "—", highlightTerms);
  elements.detailTags.innerHTML = highlightText(row.tags.join(", ") || "—", highlightTerms);

  // Display model if available in metadata
  const model = row.metadata?.config?.model || "—";
  elements.detailModel.textContent = model;
  if (row.source_pdf_url) {
    elements.detailSourceUrl.textContent = "Open source file";
    elements.detailSourceUrl.href = row.source_pdf_url;
  } else {
    elements.detailSourceUrl.textContent = "—";
    elements.detailSourceUrl.removeAttribute("href");
  }

  // Hide the text panel entirely when no source text is available (image-first runs).
  const originalText = (row.original_text || "").trim();
  if (!originalText) {
    if (elements.detailTextPanel) {
      elements.detailTextPanel.classList.add("hidden");
    }
  } else {
    if (elements.detailTextPanel) {
      elements.detailTextPanel.classList.remove("hidden");
    }
    const wordCount = originalText.split(/\s+/).filter(Boolean).length;
    const snippet = originalText.split(/\s+/).slice(0, 30).join(" ");
    const highlightedText = highlightText(originalText, highlightTerms);
    const highlightedSnippet = highlightText(snippet, highlightTerms);
    elements.detailText.innerHTML = highlightedText;
    elements.detailTextPreview.innerHTML = `${highlightedSnippet}... (${wordCount.toLocaleString()} words)`;
    elements.detailText.classList.add("hidden");
    elements.detailTextPreview.classList.remove("hidden");
    elements.detailTextToggle.textContent = "Expand";
  }

  elements.detailInsights.innerHTML =
    row.key_insights.length > 0
      ? row.key_insights.map((item) => `<li>${highlightText(item, highlightTerms)}</li>`).join("")
      : "<li>—</li>";

  // Scroll to detail drawer when user clicks a row
  if (options.scrollToDetail) {
    setTimeout(() => {
      elements.detailDrawer.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 100);
  }
}

function clearDetail() {
  elements.detailDrawer.classList.add("hidden");
  state.activeRowId = null;
  elements.detailTitle.innerHTML = "Select a row to inspect full context";
  elements.detailReason.innerHTML = "—";
  elements.detailLeadTypes.innerHTML = "—";
  elements.detailPower.innerHTML = "—";
  elements.detailAgencies.innerHTML = "—";
  elements.detailTags.innerHTML = "—";
  elements.detailModel.textContent = "—";
  elements.detailSourceUrl.textContent = "—";
  elements.detailSourceUrl.removeAttribute("href");
  if (elements.detailTextPanel) {
    elements.detailTextPanel.classList.remove("hidden");
  }
  elements.detailText.innerHTML = "—";
  elements.detailTextPreview.innerHTML = "—";
  elements.detailText.classList.add("hidden");
  elements.detailTextPreview.classList.remove("hidden");
  elements.detailTextToggle.textContent = "Expand";
  elements.detailInsights.innerHTML = "";
}
