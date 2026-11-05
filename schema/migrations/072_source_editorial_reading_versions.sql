-- Additive private-source snapshot migration. Never rewrites sections or rules.
CREATE TABLE IF NOT EXISTS source_editorial_editions (
    section_id INTEGER NOT NULL REFERENCES sections(section_id),
    version INTEGER NOT NULL,
    cleaned_text TEXT NOT NULL,
    cleaned_sha256 TEXT NOT NULL,
    ledger_json TEXT NOT NULL,
    PRIMARY KEY(section_id, version)
);
CREATE TABLE IF NOT EXISTS source_collation_units (
    unit_id TEXT PRIMARY KEY,
    section_id INTEGER NOT NULL REFERENCES sections(section_id),
    payload_json TEXT NOT NULL
);
