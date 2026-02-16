from __future__ import annotations

LEAD_TYPE_CANONICAL_MAP = {
    "financial flow": {
        "financial flow",
        "financial flows",
        "financial transaction",
        "money trail",
        "money movement",
        "money laundering",
        "overpriced sale",
        "payment chain",
    },
    "foreign influence": {
        "foreign influence",
        "foreign interference",
        "foreign collusion",
    },
    "legal exposure": {"legal exposure", "legal risk", "criminal liability"},
    "political corruption": {"political corruption", "corruption", "quid pro quo"},
    "sexual misconduct": {"sexual misconduct", "sexual abuse", "sex trafficking"},
    "intelligence operation": {
        "intelligence operation",
        "intelligence activity",
        "spycraft",
        "espionage",
    },
    "national security": {"national security", "security breach"},
    "human trafficking": {
        "human trafficking",
        "trafficking",
        "exploitation",
    },
    "cover-up": {"cover-up", "cover up", "obstruction"},
    "financial fraud": {"financial fraud", "fraud"},
}

AGENCY_CANONICAL_MAP = {
    "NSA": {"nsa", "national security agency"},
    "CIA": {"cia", "central intelligence agency"},
    "FBI": {"fbi", "federal bureau of investigation"},
    "DOJ": {"doj", "department of justice", "u.s. department of justice"},
    "DHS": {"dhs", "department of homeland security"},
    "ODNI": {"odni", "office of the director of national intelligence"},
    "State Department": {"state department", "u.s. department of state", "dos"},
    "Treasury": {"treasury", "u.s. treasury", "department of the treasury"},
    "IRS": {"irs", "internal revenue service"},
    "SEC": {"sec", "securities and exchange commission"},
    "House Oversight Committee": {
        "house oversight committee",
        "house oversight",
        "house committee on oversight",
    },
    "Senate Judiciary Committee": {"senate judiciary committee", "senate judiciary"},
    "Congress": {"congress", "u.s. congress"},
    "FSB": {"fsb", "federal security service"},
    "GRU": {"gru", "main directorate"},
    "GCHQ": {"gchq", "government communications headquarters"},
    "MI6": {"mi6", "secret intelligence service", "sis"},
    "MI5": {"mi5"},
    "Mossad": {"mossad"},
    "Interpol": {"interpol"},
    "NYPD": {"nypd", "new york police department"},
}

DEFAULT_JUSTICE_FILES_BASE_URL = "https://www.justice.gov/epstein/files"
IMAGE_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
TEXT_SUFFIXES = {".txt"}
RETRIABLE_HTTP_STATUS_CODES = {408, 409, 425, 429}
