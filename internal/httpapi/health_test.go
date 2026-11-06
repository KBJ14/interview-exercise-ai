package httpapi_test

import (
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/KBJ14/interview-exercise-ai/internal/httpapi"
	"github.com/KBJ14/interview-exercise-ai/internal/ragclient"
)

// healthAPI has no RAG client and no store: if /health touched either, the nil interface call
// would panic and fail the test.
func healthAPI() http.Handler {
	return httpapi.NewServer(nil, nil, slog.New(slog.NewTextHandler(io.Discard, nil)))
}

func TestHealthIsOKWithoutDependencies(t *testing.T) {
	rec := httptest.NewRecorder()
	healthAPI().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/health", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q", ct)
	}
	if body := strings.TrimSpace(rec.Body.String()); body != `{"status":"ok"}` {
		t.Errorf("body = %s", body)
	}
}

func TestHealthWithoutPythonServiceStillOK(t *testing.T) {
	// A real client pointing at a closed port: /health must not depend on Python being up.
	quiet := slog.New(slog.NewTextHandler(io.Discard, nil))
	api := httpapi.NewServer(ragclient.New("http://127.0.0.1:1", time.Second), nil, quiet)
	rec := httptest.NewRecorder()
	api.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/health", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
}

func TestHealthOtherMethodsUseTheErrorShape(t *testing.T) {
	rec := httptest.NewRecorder()
	healthAPI().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/health", nil))
	assertError(t, rec, http.StatusMethodNotAllowed, "method_not_allowed")
	if rec.Header().Get("Allow") != "GET, HEAD" {
		t.Errorf("Allow = %q", rec.Header().Get("Allow"))
	}
}
