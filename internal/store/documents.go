// Package store reads and deletes document metadata in the PostgreSQL database shared with the
// Python RAG service. The Python service owns ingestion; this package lists and deletes.
package store

import (
	"context"
	"errors"
	"fmt"
	"net"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

var (
	// ErrUnavailable: the database could not be reached or is temporarily refusing work.
	ErrUnavailable = errors.New("database unavailable")
	// ErrTimeout: the query did not finish within the configured limit.
	ErrTimeout = errors.New("database query timed out")
	// ErrCanceled: the caller (HTTP client) went away before the query finished.
	ErrCanceled = errors.New("query canceled by caller")
)

// Document is one row of the documents table without its content (the full original text)
// and without any embeddings.
type Document struct {
	ID          int64
	Title       string
	Source      *string // NULL for text ingested without a file name
	ContentHash string
	CreatedAt   time.Time
}

// DBTX is satisfied by *pgxpool.Pool and pgx.Tx, so tests can run queries inside a
// transaction that is rolled back.
type DBTX interface {
	Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error)
	Exec(ctx context.Context, sql string, args ...any) (pgconn.CommandTag, error)
}

type Store struct {
	db      DBTX
	timeout time.Duration
}

func New(db DBTX, timeout time.Duration) *Store {
	return &Store{db: db, timeout: timeout}
}

// Newest first; id breaks ties so the order is stable when rows share a created_at.
// Selecting columns explicitly keeps content out of the query, not just out of the response.
const listDocumentsSQL = `
SELECT id, title, source, content_hash, created_at
FROM documents
ORDER BY created_at DESC, id DESC`

// ListDocuments returns every document's metadata. An empty table yields an empty, non-nil slice.
func (s *Store) ListDocuments(ctx context.Context) ([]Document, error) {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()

	rows, err := s.db.Query(ctx, listDocumentsSQL)
	if err != nil {
		return nil, classify(ctx, "list documents", err)
	}
	docs, err := pgx.CollectRows(rows, func(row pgx.CollectableRow) (Document, error) {
		var d Document
		err := row.Scan(&d.ID, &d.Title, &d.Source, &d.ContentHash, &d.CreatedAt)
		return d, err
	})
	if err != nil {
		return nil, classify(ctx, "list documents", err)
	}
	if docs == nil {
		docs = []Document{}
	}
	return docs, nil
}

// embeddings.document_id references documents(id) ON DELETE CASCADE (schema.sql), so this one
// statement removes the document and all of its chunks atomically.
const deleteDocumentSQL = `DELETE FROM documents WHERE id = $1`

// DeleteDocument removes the document with the given id and, through the cascade, its
// embeddings. It reports false when no such document exists. Existence comes from the
// affected row count, so there is no separate lookup that could race with another delete.
func (s *Store) DeleteDocument(ctx context.Context, id int64) (bool, error) {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()

	tag, err := s.db.Exec(ctx, deleteDocumentSQL, id)
	if err != nil {
		return false, classify(ctx, "delete document", err)
	}
	return tag.RowsAffected() > 0, nil
}

// SQLSTATE codes that describe a temporary condition on a reachable server.
var transientCodes = map[string]bool{
	"53300": true, // too_many_connections
	"57P01": true, // admin_shutdown
	"57P02": true, // crash_shutdown
	"57P03": true, // cannot_connect_now (starting up / in recovery)
}

// classify separates failures the caller can retry (unavailable, timeout) and a caller that
// left (canceled) from everything else, which is returned wrapped and treated as internal.
func classify(ctx context.Context, op string, err error) error {
	switch {
	case errors.Is(ctx.Err(), context.DeadlineExceeded):
		return fmt.Errorf("%w: %v", ErrTimeout, err)
	case errors.Is(ctx.Err(), context.Canceled):
		return fmt.Errorf("%w: %v", ErrCanceled, err)
	}

	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) {
		// Class 08 is connection_exception. Other classes (missing table, bad SQL, failed
		// authentication) are not transient and fall through as internal errors.
		if transientCodes[pgErr.Code] || pgErr.Code[:2] == "08" {
			return fmt.Errorf("%w: SQLSTATE %s", ErrUnavailable, pgErr.Code)
		}
		return fmt.Errorf("%s: SQLSTATE %s: %s", op, pgErr.Code, pgErr.Message)
	}

	var connectErr *pgconn.ConnectError
	var netErr net.Error
	if errors.As(err, &connectErr) || errors.As(err, &netErr) || pgconn.Timeout(err) {
		return fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	return fmt.Errorf("%s: %w", op, err)
}
