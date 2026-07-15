CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    aliases JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS relations (
    id BIGSERIAL PRIMARY KEY,
    head_id TEXT NOT NULL REFERENCES entities(id),
    tail_id TEXT NOT NULL REFERENCES entities(id),
    relation_id TEXT NOT NULL,
    relation_name TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    split TEXT NOT NULL,
    document TEXT NOT NULL,
    sentence_id JSONB NOT NULL,
    evidence JSONB NOT NULL,
    evidence_source TEXT NOT NULL,
    is_revised BOOLEAN NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_relations_head_id ON relations(head_id);
CREATE INDEX IF NOT EXISTS idx_relations_tail_id ON relations(tail_id);
CREATE INDEX IF NOT EXISTS idx_relations_relation_id ON relations(relation_id);
CREATE INDEX IF NOT EXISTS idx_relations_confidence ON relations(confidence);
