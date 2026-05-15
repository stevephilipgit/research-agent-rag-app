-- Supabase Document Registry Migration
-- Run this in your Supabase SQL Editor (https://supabase.com/dashboard/project/_/sql)

-- 1. Create the Documents table
CREATE TABLE IF NOT EXISTS documents (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_hash     TEXT NOT NULL,
    filename      TEXT NOT NULL,
    storage_path  TEXT NOT NULL,
    storage_url   TEXT,              -- Public URL for the file
    user_id       TEXT NOT NULL,     -- acts as uploaded_by
    vector_count  INTEGER DEFAULT 0,
    status        TEXT DEFAULT 'pending', -- pending, indexed, failed, delete_failed
    document_type TEXT DEFAULT 'general',
    topic         TEXT DEFAULT 'general',
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT unique_hash_per_user UNIQUE (file_hash, user_id)
);

-- 2. Indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents (user_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (status);

-- 3. Storage Bucket Instructions
-- Note: Buckets are best created via the Supabase UI (Storage -> New Bucket -> 'documents')
-- Or via SQL if you have the right permissions:
-- INSERT INTO storage.buckets (id, name, public) VALUES ('documents', 'documents', true) ON CONFLICT DO NOTHING;

-- 4. RLS Policies (Optional but recommended)
-- ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY "Allow all for anon" ON documents FOR ALL USING (true) WITH CHECK (true);
-- CREATE POLICY "Allow all for anon storage" ON storage.objects FOR ALL USING (bucket_id = 'documents') WITH CHECK (bucket_id = 'documents');
