-- Supabase Document Registry Migration
-- Run this in your Supabase SQL Editor

CREATE TABLE IF NOT EXISTS documents (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_hash     TEXT NOT NULL,
    filename      TEXT NOT NULL,
    storage_path  TEXT NOT NULL,
    user_id       TEXT NOT NULL,
    vector_ids    TEXT[] DEFAULT '{}',
    vector_count  INTEGER DEFAULT 0,
    status        TEXT DEFAULT 'pending',
    document_type TEXT,
    topic         TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Index for duplicate hash checking per user
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_hash_user
    ON documents (file_hash, user_id);

-- Index for fast user_id retrieval
CREATE INDEX IF NOT EXISTS idx_documents_user_id
    ON documents (user_id);
