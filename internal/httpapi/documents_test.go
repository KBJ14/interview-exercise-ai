package httpapi_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/KBJ14/interview-exercise-ai/internal/httpapi"
	"github.com/KBJ14/interview-exercise-ai/internal/store"
)

type fakeLister struct {
	docs  []store.Document
	err   error
	calls int

	deleted     bool  // DeleteDocument result
	deleteErr   error // DeleteDocument error
	deleteCalls []int64
}

func (f *fakeLister) ListDocuments(context.Context) ([]store.Document, error) {
	f.calls++
	return f.docs, f.err
}

func (f *fakeLister) DeleteDocument(_ context.Context, id int64) (bool, error) {
	f.deleteCalls = append(f.deleteCalls, id)
	return f.deleted, f.deleteErr
}

func docsAPI(lister httpapi.DocumentStore) http.Handler {
	quiet := slog.New(slog.NewTextHandler(io.Discard, nil))
	return httpapi.NewServer(nil, lister, quiet) // /documents never calls the RAG service
}

func getDocuments(api http.Handler) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	api.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/documents", nil))
	return rec
}

func strptr(s string) *string { return &s }

func TestListDocumentsReturnsMetadataInStoreOrder(t *testing.T) {
	seoul := time.FixedZone("KST", 9*3600)
	lister := &fakeLister{docs: []store.Document{
		{ID: 16, Title: "Quillbox Account and Data Management", Source: strptr("account.txt"),
			ContentHash: strings.Repeat("a", 64), CreatedAt: time.Date(2026, 10, 30, 14, 0, 0, 0, seoul)},
		{ID: 3, Title: "Pasted note", Source: nil,
			ContentHash: strings.Repeat("b", 64), CreatedAt: time.Date(2026, 10, 30, 12, 0, 0, 0, time.UTC)},
	}}
	rec := getDocuments(docsAPI(lister))

	if rec.Code != http.StatusOK || rec.Header().Get("Content-Type") != "application/json" {
		t.Fatalf("status %d content-type %q body %s", rec.Code, rec.Header().Get("Content-Type"), rec.Body)
	}
	var body struct {
		Documents []map[string]any `json:"documents"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if len(body.Documents) != 2 {
		t.Fatalf("got %d documents", len(body.Documents))
	}
	first, second := body.Documents[0], body.Documents[1]
	want := map[string]any{
		"id": 16.0, "title": "Quillbox Account and Data Management", "source": "account.txt",
		"content_hash": strings.Repeat("a", 64), "created_at": "2026-10-30T05:00:00Z", // converted to UTC
	}
	if fmt.Sprint(first) != fmt.Sprint(want) {
		t.Errorf("first document = %v, want %v", first, want)
	}
	if second["id"] != 3.0 || second["source"] != nil {
		t.Errorf("order or null source not preserved: %v", second)
	}
	for _, field := range []string{"content", "embedding", "embeddings"} {
		if _, ok := first[field]; ok {
			t.Errorf("listing exposes %q", field)
		}
	}
}

func TestListDocumentsEmptyIsArray(t *testing.T) {
	for name, docs := range map[string][]store.Document{"empty slice": {}, "nil slice": nil} {
		t.Run(name, func(t *testing.T) {
			rec := getDocuments(docsAPI(&fakeLister{docs: docs}))
			if rec.Code != http.StatusOK || strings.TrimSpace(rec.Body.String()) != `{"documents":[]}` {
				t.Errorf("status %d body %s", rec.Code, rec.Body)
			}
		})
	}
}

func TestListDocumentsErrors(t *testing.T) {
	cases := []struct {
		name       string
		err        error
		wantStatus int
		wantCode   string
	}{
		{"unavailable", fmt.Errorf("%w: dial tcp 127.0.0.1:5432: connect: connection refused", store.ErrUnavailable), 503, "database_unavailable"},
		{"timeout", fmt.Errorf("%w: context deadline exceeded", store.ErrTimeout), 503, "database_timeout"},
		{"internal", errors.New(`query documents: SQLSTATE 42P01: relation "documents" does not exist`), 500, "internal_error"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			msg := assertError(t, getDocuments(docsAPI(&fakeLister{err: tc.err})), tc.wantStatus, tc.wantCode)
			for _, leak := range []string{"SQLSTATE", "127.0.0.1", "relation", "refused", "deadline"} {
				if strings.Contains(msg, leak) {
					t.Errorf("message leaks %q: %q", leak, msg)
				}
			}
		})
	}
}

func TestListDocumentsCanceledWritesNothing(t *testing.T) {
	rec := getDocuments(docsAPI(&fakeLister{err: fmt.Errorf("%w: context canceled", store.ErrCanceled)}))
	if rec.Body.Len() != 0 {
		t.Errorf("wrote a response to a client that is gone: %s", rec.Body)
	}
}

func TestDocumentsWrongMethodIs405(t *testing.T) {
	lister := &fakeLister{}
	rec := httptest.NewRecorder()
	docsAPI(lister).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/documents", strings.NewReader(`{}`)))
	assertError(t, rec, http.StatusMethodNotAllowed, "method_not_allowed")
	if rec.Header().Get("Allow") != "GET, HEAD" {
		t.Errorf("Allow = %q", rec.Header().Get("Allow"))
	}
	if lister.calls != 0 {
		t.Error("store was queried for a rejected method")
	}
}
