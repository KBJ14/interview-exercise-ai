package httpapi_test

import (
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/KBJ14/interview-exercise-ai/internal/store"
)

func deleteDocument(api http.Handler, rawID string) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	api.ServeHTTP(rec, httptest.NewRequest(http.MethodDelete, "/documents/"+rawID, nil))
	return rec
}

func TestDeleteExistingDocumentIs204WithoutBody(t *testing.T) {
	fake := &fakeLister{deleted: true}
	rec := deleteDocument(docsAPI(fake), "16")

	if rec.Code != http.StatusNoContent {
		t.Fatalf("status = %d, want 204; body %s", rec.Code, rec.Body)
	}
	if rec.Body.Len() != 0 || rec.Header().Get("Content-Type") != "" {
		t.Errorf("204 must have no body or Content-Type: %q %q", rec.Body, rec.Header().Get("Content-Type"))
	}
	if fmt.Sprint(fake.deleteCalls) != "[16]" {
		t.Errorf("store called with %v, want [16]", fake.deleteCalls)
	}
}

func TestDeleteLargestBigintIsAccepted(t *testing.T) {
	fake := &fakeLister{deleted: false}
	assertError(t, deleteDocument(docsAPI(fake), "9223372036854775807"), http.StatusNotFound, "document_not_found")
	if fmt.Sprint(fake.deleteCalls) != "[9223372036854775807]" {
		t.Errorf("store called with %v", fake.deleteCalls)
	}
}

func TestDeleteMissingDocumentIs404(t *testing.T) {
	fake := &fakeLister{deleted: false}
	assertError(t, deleteDocument(docsAPI(fake), "999"), http.StatusNotFound, "document_not_found")
	if len(fake.deleteCalls) != 1 {
		t.Errorf("store called %d times, want 1 (no separate existence lookup)", len(fake.deleteCalls))
	}
	if fake.calls != 0 {
		t.Error("delete listed documents to check existence")
	}
}

func TestDeleteInvalidIDIs400AndNeverTouchesStore(t *testing.T) {
	fake := &fakeLister{deleted: true}
	api := docsAPI(fake)
	for _, raw := range []string{
		"0", "-1", "+1", "007", "1.0", "1e3", "abc", "12a", "%20%31", "0x10",
		"9223372036854775808",  // BIGINT max + 1
		"99999999999999999999", // far beyond BIGINT
	} {
		t.Run(raw, func(t *testing.T) {
			assertError(t, deleteDocument(api, raw), http.StatusBadRequest, "invalid_document_id")
		})
	}
	if len(fake.deleteCalls) != 0 {
		t.Errorf("store was called for invalid ids: %v", fake.deleteCalls)
	}
}

func TestDeleteRoutingEdges(t *testing.T) {
	fake := &fakeLister{deleted: true}
	api := docsAPI(fake)

	assertError(t, deleteDocument(api, ""), http.StatusNotFound, "not_found")        // DELETE /documents/
	assertError(t, deleteDocument(api, "3/extra"), http.StatusNotFound, "not_found") // extra segment
	rec := httptest.NewRecorder()
	api.ServeHTTP(rec, httptest.NewRequest(http.MethodDelete, "/documents", nil)) // collection, not an item
	assertError(t, rec, http.StatusMethodNotAllowed, "method_not_allowed")

	rec = httptest.NewRecorder()
	api.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/documents/3", nil))
	assertError(t, rec, http.StatusMethodNotAllowed, "method_not_allowed")
	if rec.Header().Get("Allow") != http.MethodDelete {
		t.Errorf("Allow = %q, want DELETE", rec.Header().Get("Allow"))
	}
	if len(fake.deleteCalls) != 0 {
		t.Errorf("store was called: %v", fake.deleteCalls)
	}
}

func TestDeleteStoreErrors(t *testing.T) {
	cases := []struct {
		name       string
		err        error
		wantStatus int
		wantCode   string
	}{
		{"unavailable", fmt.Errorf("%w: dial tcp 127.0.0.1:5432: connect: connection refused", store.ErrUnavailable), 503, "database_unavailable"},
		{"timeout", fmt.Errorf("%w: context deadline exceeded", store.ErrTimeout), 503, "database_timeout"},
		{"internal", errors.New("delete document: SQLSTATE 42501: permission denied for table documents"), 500, "internal_error"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			msg := assertError(t, deleteDocument(docsAPI(&fakeLister{deleteErr: tc.err}), "16"), tc.wantStatus, tc.wantCode)
			for _, leak := range []string{"SQLSTATE", "permission", "127.0.0.1", "refused"} {
				if strings.Contains(msg, leak) {
					t.Errorf("message leaks %q: %q", leak, msg)
				}
			}
		})
	}

	rec := deleteDocument(docsAPI(&fakeLister{deleteErr: fmt.Errorf("%w: context canceled", store.ErrCanceled)}), "16")
	if rec.Body.Len() != 0 {
		t.Errorf("wrote a response to a client that is gone: %s", rec.Body)
	}
}
