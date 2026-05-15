-- Migration: Sync `documents` table with backend document registry requirements.
-- Author: Antigravity

DO $$
BEGIN
    -- 1. storage_url
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='documents' AND column_name='storage_url') THEN
        ALTER TABLE documents ADD COLUMN storage_url TEXT;
    END IF;

    -- 2. vector_count
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='documents' AND column_name='vector_count') THEN
        ALTER TABLE documents ADD COLUMN vector_count INTEGER DEFAULT 0;
    END IF;

    -- 3. document_type
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='documents' AND column_name='document_type') THEN
        ALTER TABLE documents ADD COLUMN document_type TEXT DEFAULT 'general';
    END IF;

    -- 4. topic
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='documents' AND column_name='topic') THEN
        ALTER TABLE documents ADD COLUMN topic TEXT DEFAULT 'general';
    END IF;

    -- 5. schema_version (Task 4)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='documents' AND column_name='schema_version') THEN
        ALTER TABLE documents ADD COLUMN schema_version INTEGER DEFAULT 1;
    END IF;
    
    -- Ensure status has a default if not already present
    -- Note: 'status' should already exist based on old schemas, but we ensure it supports backend values.
    
    -- If there's an 'uploaded_by' or 'session_id', ensure it matches 'user_id' which the backend is currently using.
    -- The backend uses `user_id` as `session_id`. We'll just verify `user_id` exists.
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='documents' AND column_name='user_id') THEN
        ALTER TABLE documents ADD COLUMN user_id TEXT;
    END IF;

END $$;
