-- Migration 001: add the GET /documents sort index to a database created before it was part of
-- schema.sql. New databases already get it from schema.sql; do not run this on them before
-- schema.sql (the documents table must exist).
--
-- Safe to re-run: if an index with this name already exists, its definition is compared with the
-- expected one. A match is a no-op; a different definition (other columns, directions, method or
-- table) stops the migration with an error instead of being silently accepted.
-- Only this index is created; tables, constraints, other indexes and data are not touched.

BEGIN;

DO $$
DECLARE
    expected CONSTANT text :=
        'CREATE INDEX documents_created_at_id_idx ON public.documents USING btree (created_at DESC, id DESC)';
    actual text;
BEGIN
    SELECT pg_get_indexdef(i.indexrelid) INTO actual
    FROM pg_index i
    JOIN pg_class c ON c.oid = i.indexrelid
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public' AND c.relname = 'documents_created_at_id_idx';

    IF actual IS NULL THEN
        -- Plain CREATE INDEX (not CONCURRENTLY, which cannot run in a transaction) briefly blocks
        -- writes to documents; fine for this small table.
        CREATE INDEX documents_created_at_id_idx ON public.documents (created_at DESC, id DESC);
        RAISE NOTICE 'created documents_created_at_id_idx';
    ELSIF actual = expected THEN
        RAISE NOTICE 'documents_created_at_id_idx already exists with the expected definition; nothing to do';
    ELSE
        RAISE EXCEPTION 'documents_created_at_id_idx exists with a different definition: %', actual
            USING HINT = 'Inspect it and drop or rename it before re-running this migration.';
    END IF;
END
$$;

COMMIT;
