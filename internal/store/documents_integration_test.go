package store_test

// Integration tests against the real PostgreSQL from docker-compose. They are skipped when the
// database is unreachable. Every test works inside a transaction that is rolled back, so the
// real corpus (documents and embeddings) is never modified.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/KBJ14/interview-exercise-ai/internal/config"
	"github.com/KBJ14/interview-exercise-ai/internal/httpapi"
	"github.com/KBJ14/interview-exercise-ai/internal/store"
)

func openPool(t *testing.T) *pgxpool.Pool {
	t.Helper()
	getenv, err := config.EnvWithFile(os.Getenv, "../../.env") // same settings as the Python service
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := config.Load(getenv)
	if err != nil {
		t.Skipf("database settings unavailable: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	pool, err := pgxpool.New(ctx, cfg.DatabaseURL)
	if err == nil {
		err = pool.Ping(ctx)
	}
	if err != nil {
		t.Skipf("PostgreSQL not reachable at %s (start it with `docker compose up -d`)", cfg.DatabaseTarget)
	}
	t.Cleanup(pool.Close)
	return pool
}

// begin starts a transaction that is always rolled back at the end of the test.
func begin(t *testing.T, pool *pgxpool.Pool) pgx.Tx {
	t.Helper()
	tx, err := pool.Begin(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = tx.Rollback(context.Background()) })
	return tx
}

// isolate shadows documents with an empty temporary copy for this transaction only.
// pg_temp comes first in the search path, so the store's unqualified query reads it.
func isolate(t *testing.T, tx pgx.Tx) {
	t.Helper()
	if _, err := tx.Exec(context.Background(),
		`CREATE TEMP TABLE documents (LIKE public.documents INCLUDING ALL) ON COMMIT DROP`); err != nil {
		t.Fatal(err)
	}
}

func counts(t *testing.T, pool *pgxpool.Pool) (docs, chunks int) {
	t.Helper()
	err := pool.QueryRow(context.Background(),
		`SELECT (SELECT count(*) FROM public.documents), (SELECT count(*) FROM public.embeddings)`).Scan(&docs, &chunks)
	if err != nil {
		t.Fatal(err)
	}
	return docs, chunks
}

func insertDoc(t *testing.T, tx pgx.Tx, title string, source *string, createdAt time.Time) int64 {
	t.Helper()
	var id int64
	hash := fmt.Sprintf("%064x", time.Now().UnixNano()) // unique, matches the schema's hex check
	err := tx.QueryRow(context.Background(),
		`INSERT INTO documents (title, source, content, content_hash, created_at) VALUES ($1, $2, 'test body', $3, $4) RETURNING id`,
		title, source, hash, createdAt).Scan(&id)
	if err != nil {
		t.Fatal(err)
	}
	return id
}

func TestListDocumentsOrderAndFieldsOnRealDatabase(t *testing.T) {
	pool := openPool(t)
	beforeDocs, beforeChunks := counts(t, pool)
	tx := begin(t, pool)
	isolate(t, tx)

	marker := fmt.Sprintf("go-store-test-%d", time.Now().UnixNano())
	src := marker + ".txt"
	same := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	older := insertDoc(t, tx, marker+" older", &src, same.Add(-time.Hour))
	tieLow := insertDoc(t, tx, marker+" tie low id", nil, same)
	tieHigh := insertDoc(t, tx, marker+" tie high id", &src, same)

	docs, err := store.New(tx, 5*time.Second).ListDocuments(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	var ids []int64
	for _, d := range docs {
		ids = append(ids, d.ID)
	}
	if want := []int64{tieHigh, tieLow, older}; fmt.Sprint(ids) != fmt.Sprint(want) {
		t.Errorf("order = %v, want %v (created_at DESC, then id DESC)", ids, want)
	}
	if d := docs[1]; d.Source != nil || d.Title != marker+" tie low id" || !d.CreatedAt.Equal(same) || len(d.ContentHash) != 64 {
		t.Errorf("unexpected row: %+v", d)
	}

	if d, c := counts(t, pool); d != beforeDocs || c != beforeChunks {
		t.Errorf("real tables changed: documents %d->%d, embeddings %d->%d", beforeDocs, d, beforeChunks, c)
	}
}

func TestListDocumentsEmptyTableOnRealDatabase(t *testing.T) {
	pool := openPool(t)
	tx := begin(t, pool)
	isolate(t, tx)

	docs, err := store.New(tx, 5*time.Second).ListDocuments(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if docs == nil || len(docs) != 0 {
		t.Errorf("got %#v, want empty non-nil slice", docs)
	}
}

func TestGetDocumentsEndToEndOnRealCorpus(t *testing.T) {
	pool := openPool(t)
	beforeDocs, beforeChunks := counts(t, pool)

	quiet := slog.New(slog.NewTextHandler(io.Discard, nil))
	api := httpapi.NewServer(nil, store.New(pool, 5*time.Second), quiet)
	rec := httptest.NewRecorder()
	api.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/documents", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d body %s", rec.Code, rec.Body)
	}
	if strings.Contains(rec.Body.String(), `"content"`) {
		t.Error("response includes document content")
	}
	var body httpapi.DocumentsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if len(body.Documents) != beforeDocs {
		t.Errorf("listed %d documents, table has %d", len(body.Documents), beforeDocs)
	}
	for i := 1; i < len(body.Documents); i++ {
		prev, cur := body.Documents[i-1], body.Documents[i]
		if cur.CreatedAt.After(prev.CreatedAt) || (cur.CreatedAt.Equal(prev.CreatedAt) && cur.ID > prev.ID) {
			t.Errorf("not newest first at %d: %v then %v", i, prev, cur)
		}
	}
	if d, c := counts(t, pool); d != beforeDocs || c != beforeChunks {
		t.Errorf("real tables changed: documents %d->%d, embeddings %d->%d", beforeDocs, d, beforeChunks, c)
	}
}

func TestErrorClassificationOnRealConnections(t *testing.T) {
	t.Run("nothing listening is unavailable", func(t *testing.T) {
		pool, err := pgxpool.New(context.Background(), "postgres://rag:pw@127.0.0.1:1/rag?connect_timeout=2")
		if err != nil {
			t.Fatal(err)
		}
		defer pool.Close()
		_, err = store.New(pool, 3*time.Second).ListDocuments(context.Background())
		if !errors.Is(err, store.ErrUnavailable) {
			t.Errorf("err = %v, want ErrUnavailable", err)
		}
	})

	t.Run("missing table is internal, not unavailable", func(t *testing.T) {
		tx := begin(t, openPool(t))
		if _, err := tx.Exec(context.Background(), `SET LOCAL search_path TO pg_catalog`); err != nil {
			t.Fatal(err)
		}
		_, err := store.New(tx, 5*time.Second).ListDocuments(context.Background())
		if err == nil || errors.Is(err, store.ErrUnavailable) || errors.Is(err, store.ErrTimeout) || !strings.Contains(err.Error(), "42P01") {
			t.Errorf("err = %v, want an internal SQLSTATE 42P01 error", err)
		}
	})

	t.Run("expired deadline is timeout", func(t *testing.T) {
		_, err := store.New(openPool(t), time.Nanosecond).ListDocuments(context.Background())
		if !errors.Is(err, store.ErrTimeout) {
			t.Errorf("err = %v, want ErrTimeout", err)
		}
	})

	t.Run("caller cancel is canceled", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := store.New(openPool(t), 5*time.Second).ListDocuments(ctx)
		if !errors.Is(err, store.ErrCanceled) {
			t.Errorf("err = %v, want ErrCanceled", err)
		}
	})
}
