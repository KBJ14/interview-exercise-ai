package store_test

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/KBJ14/interview-exercise-ai/internal/httpapi"
	"github.com/KBJ14/interview-exercise-ai/internal/store"
)

// testVector is a valid vector(1536) literal; no embedding API is involved.
var testVector = "[" + strings.Repeat("0,", 1535) + "1]"

// createDocWithChunks inserts one marked document plus `chunks` embeddings and returns its id
// and content_hash. The model name marks the rows as test data.
func createDocWithChunks(t *testing.T, db store.DBTX, chunks int) (int64, string) {
	t.Helper()
	buf := make([]byte, 32)
	if _, err := rand.Read(buf); err != nil {
		t.Fatal(err)
	}
	hash := hex.EncodeToString(buf)
	marker := "go-delete-test-" + hash[:12]

	rows, err := db.Query(context.Background(),
		`INSERT INTO documents (title, source, content, content_hash) VALUES ($1, $2, 'temporary test body', $3) RETURNING id`,
		marker, marker+".txt", hash)
	if err != nil {
		t.Fatal(err)
	}
	var id int64
	for rows.Next() {
		if err := rows.Scan(&id); err != nil {
			t.Fatal(err)
		}
	}
	if rows.Close(); rows.Err() != nil {
		t.Fatal(rows.Err())
	}
	for i := 0; i < chunks; i++ {
		if _, err := db.Exec(context.Background(),
			`INSERT INTO embeddings (document_id, chunk_index, content, embedding, embedding_model, token_count)
			 VALUES ($1, $2, $3, $4::vector, 'go-delete-test', 1)`,
			id, i, fmt.Sprintf("%s chunk %d", marker, i), testVector); err != nil {
			t.Fatal(err)
		}
	}
	return id, hash
}

func rowsFor(t *testing.T, db store.DBTX, id int64) (docs, chunks int) {
	t.Helper()
	rows, err := db.Query(context.Background(),
		`SELECT (SELECT count(*) FROM documents WHERE id = $1), (SELECT count(*) FROM embeddings WHERE document_id = $1)`, id)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	rows.Next()
	if err := rows.Scan(&docs, &chunks); err != nil {
		t.Fatal(err)
	}
	return docs, chunks
}

// realCorpus captures the ids and hashes of every pre-existing document.
func realCorpus(t *testing.T, pool *pgxpool.Pool) string {
	t.Helper()
	var s string
	err := pool.QueryRow(context.Background(),
		`SELECT coalesce(string_agg(id || ':' || content_hash, ',' ORDER BY id), '') FROM documents WHERE title NOT LIKE 'go-delete-test-%'`).Scan(&s)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestDeleteDocumentCascadesInsideTransaction(t *testing.T) {
	pool := openPool(t)
	tx := begin(t, pool) // rolled back by Cleanup: nothing is committed
	id, _ := createDocWithChunks(t, tx, 3)
	if d, c := rowsFor(t, tx, id); d != 1 || c != 3 {
		t.Fatalf("setup: %d docs, %d chunks", d, c)
	}

	s := store.New(tx, 5*time.Second)
	deleted, err := s.DeleteDocument(context.Background(), id)
	if err != nil || !deleted {
		t.Fatalf("first delete: deleted=%v err=%v", deleted, err)
	}
	if d, c := rowsFor(t, tx, id); d != 0 || c != 0 {
		t.Errorf("after delete: %d docs, %d chunks remain (cascade failed)", d, c)
	}
	deleted, err = s.DeleteDocument(context.Background(), id)
	if err != nil || deleted {
		t.Errorf("second delete: deleted=%v err=%v, want false, nil", deleted, err)
	}
}

func TestDeleteOverHTTPRemovesDocumentAndEmbeddings(t *testing.T) {
	pool := openPool(t)
	beforeDocs, beforeChunks := counts(t, pool)
	beforeCorpus := realCorpus(t, pool)

	// Committed so that the handler's own pool connection can see and delete it.
	id, hash := createDocWithChunks(t, pool, 2)
	t.Cleanup(func() {
		// Runs on success and failure; deletes only this test's row (its chunks cascade).
		_, _ = pool.Exec(context.Background(), `DELETE FROM documents WHERE content_hash = $1`, hash)
	})

	quiet := slog.New(slog.NewTextHandler(io.Discard, nil))
	api := httpapi.NewServer(nil, store.New(pool, 5*time.Second), quiet)
	del := func() *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		api.ServeHTTP(rec, httptest.NewRequest(http.MethodDelete, fmt.Sprintf("/documents/%d", id), nil))
		return rec
	}

	if rec := del(); rec.Code != http.StatusNoContent || rec.Body.Len() != 0 {
		t.Fatalf("first DELETE: status %d body %q, want 204 and no body", rec.Code, rec.Body)
	}
	if d, c := rowsFor(t, pool, id); d != 0 || c != 0 {
		t.Errorf("after DELETE: %d docs, %d chunks remain", d, c)
	}
	if rec := del(); rec.Code != http.StatusNotFound || !strings.Contains(rec.Body.String(), `"document_not_found"`) {
		t.Errorf("second DELETE: status %d body %s, want 404 document_not_found", rec.Code, rec.Body)
	}

	if d, c := counts(t, pool); d != beforeDocs || c != beforeChunks {
		t.Errorf("table sizes changed: documents %d->%d, embeddings %d->%d", beforeDocs, d, beforeChunks, c)
	}
	if after := realCorpus(t, pool); after != beforeCorpus {
		t.Errorf("pre-existing documents changed:\nbefore %s\nafter  %s", beforeCorpus, after)
	}
}

func TestDeleteOnUnreachableDatabaseIsUnavailable(t *testing.T) {
	pool, err := pgxpool.New(context.Background(), "postgres://rag:pw@127.0.0.1:1/rag?connect_timeout=2")
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	_, err = store.New(pool, 3*time.Second).DeleteDocument(context.Background(), 1)
	if !errors.Is(err, store.ErrUnavailable) {
		t.Errorf("err = %v, want ErrUnavailable", err)
	}
}
