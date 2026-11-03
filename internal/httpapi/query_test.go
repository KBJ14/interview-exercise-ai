package httpapi_test

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/KBJ14/interview-exercise-ai/internal/httpapi"
	"github.com/KBJ14/interview-exercise-ai/internal/ragclient"
)

// A /internal/query success body in the exact shape rag/api.py produces.
const pythonOK = `{
  "question": "Can I get a refund for a monthly renewal charge?",
  "answerable": true,
  "answer": "Monthly renewals are not refundable. [S1]",
  "citations": ["S1"],
  "sources": [
    {"label": "S1", "document_id": 3, "title": "Quillbox Billing", "section": "Refund Policy",
     "chunk_index": 3, "chunk": "Monthly renewals are not refundable.", "cosine_distance": 0.3775, "cited": true},
    {"label": "S2", "document_id": 3, "title": "Quillbox Billing", "section": "Failed Payments",
     "chunk_index": 2, "chunk": "Retries happen 3 times.", "cosine_distance": 0.4953, "cited": false}
  ],
  "model": "gpt-6-luna",
  "usage": {"input_tokens": 723, "output_tokens": 58, "reasoning_tokens": 0, "total_tokens": 781}
}`

// fakePython stands in for the FastAPI service and records what the Go API sent it.
type fakePython struct {
	*httptest.Server
	calls atomic.Int32
	mu    sync.Mutex
	last  struct {
		method, path, contentType string
		body                      map[string]any
	}
}

func newFakePython(t *testing.T, respond http.HandlerFunc) *fakePython {
	t.Helper()
	f := &fakePython{}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.calls.Add(1)
		f.mu.Lock()
		f.last.method, f.last.path, f.last.contentType = r.Method, r.URL.Path, r.Header.Get("Content-Type")
		f.last.body = nil
		_ = json.NewDecoder(r.Body).Decode(&f.last.body)
		f.mu.Unlock()
		respond(w, r)
	}))
	t.Cleanup(f.Close)
	return f
}

func reply(status int, body string) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}
}

func newAPI(ragURL string, timeout time.Duration) http.Handler {
	quiet := slog.New(slog.NewTextHandler(io.Discard, nil))
	return httpapi.NewServer(ragclient.New(ragURL, timeout), nil, quiet) // /query never touches documents
}

func post(api http.Handler, body string) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	api.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/query", strings.NewReader(body)))
	return rec
}

// assertError checks the shared error shape and returns the public message.
func assertError(t *testing.T, rec *httptest.ResponseRecorder, status int, code string) string {
	t.Helper()
	if rec.Code != status {
		t.Fatalf("status = %d, want %d; body %s", rec.Code, status, rec.Body)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}
	var body map[string]map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil || len(body) != 1 || len(body["error"]) != 2 {
		t.Fatalf("body is not {\"error\":{\"code\",\"message\"}}: %s", rec.Body)
	}
	if body["error"]["code"] != code {
		t.Errorf("code = %q, want %q", body["error"]["code"], code)
	}
	if body["error"]["message"] == "" {
		t.Error("message is empty")
	}
	return body["error"]["message"]
}

func TestQuerySuccessReturnsPublicResponse(t *testing.T) {
	py := newFakePython(t, reply(http.StatusOK, pythonOK))
	rec := post(newAPI(py.URL, time.Second), `{"question": "  Can I get a refund for a monthly renewal charge?  ", "k": 2}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d; body %s", rec.Code, rec.Body)
	}
	if py.last.method != http.MethodPost || py.last.path != "/internal/query" || py.last.contentType != "application/json" {
		t.Errorf("python got %s %s (%s)", py.last.method, py.last.path, py.last.contentType)
	}
	if q, k := py.last.body["question"], py.last.body["k"]; q != "Can I get a refund for a monthly renewal charge?" || k != 2.0 {
		t.Errorf("python got question=%q k=%v; want trimmed question and k=2", q, k)
	}

	var got map[string]json.RawMessage
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	for _, internal := range []string{"model", "usage", "question"} {
		if _, ok := got[internal]; ok {
			t.Errorf("public response exposes internal field %q", internal)
		}
	}
	var resp httpapi.QueryResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	want := httpapi.Source{Label: "S1", DocumentID: 3, Title: "Quillbox Billing", Section: "Refund Policy",
		ChunkIndex: 3, Chunk: "Monthly renewals are not refundable.", CosineDistance: 0.3775, Cited: true}
	if !resp.Answerable || resp.Answer != "Monthly renewals are not refundable. [S1]" ||
		len(resp.Citations) != 1 || resp.Citations[0] != "S1" || len(resp.Sources) != 2 || resp.Sources[0] != want {
		t.Errorf("unexpected response: %+v", resp)
	}
	if !strings.Contains(rec.Body.String(), `"cosine_distance":0.3775`) {
		t.Errorf("distance not exposed as cosine_distance: %s", rec.Body)
	}
}

func TestQueryDefaultsKToThree(t *testing.T) {
	py := newFakePython(t, reply(http.StatusOK, pythonOK))
	post(newAPI(py.URL, time.Second), `{"question": "refund?"}`)
	if py.last.body["k"] != 3.0 {
		t.Errorf("k sent to python = %v, want 3", py.last.body["k"])
	}
}

func TestQueryUnanswerableKeepsEmptyArrays(t *testing.T) {
	py := newFakePython(t, reply(http.StatusOK,
		`{"answerable": false, "answer": "The provided documents do not contain this information.", "citations": [], "sources": [], "model": null, "usage": null}`))
	rec := post(newAPI(py.URL, time.Second), `{"question": "discount?"}`)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"citations":[]`) || !strings.Contains(rec.Body.String(), `"sources":[]`) {
		t.Errorf("status %d body %s", rec.Code, rec.Body)
	}
}

func TestInvalidInputIs400AndNeverCallsPython(t *testing.T) {
	py := newFakePython(t, reply(http.StatusOK, pythonOK))
	api := newAPI(py.URL, time.Second)
	cases := map[string]string{
		"empty body":       ``,
		"malformed json":   `{"question": `,
		"array body":       `["refund?"]`,
		"two json values":  `{"question": "a"} {"question": "b"}`,
		"missing question": `{}`,
		"null question":    `{"question": null}`,
		"empty question":   `{"question": ""}`,
		"blank question":   `{"question": " \n\t "}`,
		"number question":  `{"question": 42}`,
		"k zero":           `{"question": "ok", "k": 0}`,
		"k negative":       `{"question": "ok", "k": -1}`,
		"k above max":      `{"question": "ok", "k": 21}`,
		"k float":          `{"question": "ok", "k": 2.5}`,
		"k float integral": `{"question": "ok", "k": 3.0}`,
		"k exponent":       `{"question": "ok", "k": 1e1}`,
		"k string":         `{"question": "ok", "k": "3"}`,
		"k bool":           `{"question": "ok", "k": true}`,
		"k null":           `{"question": "ok", "k": null}`,
		"k huge":           `{"question": "ok", "k": 99999999999999999999}`,
		"unknown field":    `{"question": "ok", "extra": 1}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			assertError(t, post(api, body), http.StatusBadRequest, "invalid_request")
		})
	}
	if n := py.calls.Load(); n != 0 {
		t.Errorf("python was called %d times for invalid input", n)
	}
}

func TestOversizedBodyIs413(t *testing.T) {
	py := newFakePython(t, reply(http.StatusOK, pythonOK))
	big := `{"question": "` + strings.Repeat("a", 70<<10) + `"}`
	assertError(t, post(newAPI(py.URL, time.Second), big), http.StatusRequestEntityTooLarge, "request_too_large")
	if py.calls.Load() != 0 {
		t.Error("python was called for an oversized body")
	}
}

func TestRoutingErrorsUseTheSameShape(t *testing.T) {
	api := newAPI("http://127.0.0.1:1", time.Second)

	rec := httptest.NewRecorder()
	api.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/query", nil))
	assertError(t, rec, http.StatusMethodNotAllowed, "method_not_allowed")
	if rec.Header().Get("Allow") != http.MethodPost {
		t.Errorf("Allow = %q, want POST", rec.Header().Get("Allow"))
	}

	rec = httptest.NewRecorder()
	api.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/nope", nil))
	assertError(t, rec, http.StatusNotFound, "not_found")
}

func TestPythonErrorResponsesAreTranslated(t *testing.T) {
	cases := []struct {
		name         string
		status       int
		body         string
		wantStatus   int
		wantCode     string
		forwardedMsg bool // only input-validation messages may reach the client
	}{
		{"input error", 400, `{"error":{"code":"invalid_request","message":"question must be a non-empty string"}}`, 400, "invalid_request", true},
		{"llm unavailable", 503, `{"error":{"code":"llm_unavailable","message":"could not reach OpenAI API"}}`, 503, "llm_unavailable", false},
		{"llm rate limited", 503, `{"error":{"code":"llm_rate_limited","message":"OpenAI quota reached"}}`, 503, "llm_rate_limited", false},
		{"llm timeout", 504, `{"error":{"code":"llm_timeout","message":"OpenAI API did not respond in time"}}`, 504, "llm_timeout", false},
		{"llm invalid answer", 502, `{"error":{"code":"llm_invalid_response","message":"citations to sources that were not provided: ['S4'] sk-proj-secret"}}`, 502, "llm_invalid_response", false},
		{"llm rejected", 502, `{"error":{"code":"llm_request_rejected","message":"OpenAI API rejected the request (HTTP 401)"}}`, 502, "llm_request_rejected", false},
		{"embedding invalid", 502, `{"error":{"code":"embedding_invalid_response","message":"expected 1536 dimensions"}}`, 502, "embedding_invalid_response", false},
		{"database down", 503, `{"error":{"code":"database_unavailable","message":"database is unreachable"}}`, 503, "database_unavailable", false},
		{"python config error", 500, `{"error":{"code":"configuration_error","message":"service is not configured for the language model"}}`, 502, "rag_service_error", false},
		{"python internal error", 500, `{"error":{"code":"internal_error","message":"unexpected internal error"}}`, 502, "rag_service_error", false},
		{"python route missing", 404, `{"error":{"code":"not_found","message":"Not Found"}}`, 502, "rag_service_error", false},
		{"unknown code", 503, `{"error":{"code":"something_new","message":"internal detail"}}`, 502, "rag_service_error", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			py := newFakePython(t, reply(tc.status, tc.body))
			msg := assertError(t, post(newAPI(py.URL, time.Second), `{"question": "refund?"}`), tc.wantStatus, tc.wantCode)

			var upstream struct{ Error struct{ Message string } }
			_ = json.Unmarshal([]byte(tc.body), &upstream)
			if forwarded := msg == upstream.Error.Message; forwarded != tc.forwardedMsg {
				t.Errorf("python message forwarded = %v, want %v (public message %q)", forwarded, tc.forwardedMsg, msg)
			}
			if strings.Contains(msg, "sk-") || strings.Contains(msg, "OpenAI API rejected") {
				t.Errorf("internal detail leaked: %q", msg)
			}
		})
	}
}

func TestUnusablePythonResponsesAre502(t *testing.T) {
	cases := map[string]struct {
		status int
		body   string
	}{
		"not json":                {200, `Internal Server Error`},
		"empty body":              {200, ``},
		"empty object":            {200, `{}`},
		"missing sources":         {200, `{"answerable": true, "answer": "x [S1]", "citations": ["S1"]}`},
		"null answer":             {200, `{"answerable": true, "answer": null, "citations": [], "sources": []}`},
		"wrong type":              {200, `{"answerable": "yes", "answer": "x", "citations": [], "sources": []}`},
		"source missing chunk":    {200, `{"answerable": true, "answer": "x [S1]", "citations": ["S1"], "sources": [{"label": "S1", "document_id": 3, "chunk_index": 0, "cosine_distance": 0.4}]}`},
		"citation without source": {200, `{"answerable": true, "answer": "x [S9]", "citations": ["S9"], "sources": []}`},
		"error without json body": {503, `Service Unavailable`},
		"error without code":      {500, `{"error": {"message": "boom"}}`},
		"oversized body":          {200, `{"answer": "` + strings.Repeat("a", 2<<20) + `"}`},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			py := newFakePython(t, reply(tc.status, tc.body))
			assertError(t, post(newAPI(py.URL, time.Second), `{"question": "refund?"}`), http.StatusBadGateway, "rag_invalid_response")
		})
	}
}

func TestConnectionFailureIs502(t *testing.T) {
	py := newFakePython(t, reply(http.StatusOK, pythonOK))
	url := py.URL
	py.Close() // nothing listens on this address any more
	msg := assertError(t, post(newAPI(url, time.Second), `{"question": "refund?"}`), http.StatusBadGateway, "rag_unavailable")
	if strings.Contains(msg, "127.0.0.1") || strings.Contains(msg, "refused") {
		t.Errorf("transport detail leaked: %q", msg)
	}
}

func TestTimeoutIs504AndCancelsThePythonCall(t *testing.T) {
	pythonSawCancel := make(chan struct{})
	py := newFakePython(t, func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
			close(pythonSawCancel)
		case <-time.After(5 * time.Second):
			reply(http.StatusOK, pythonOK)(w, r)
		}
	})

	start := time.Now()
	assertError(t, post(newAPI(py.URL, 100*time.Millisecond), `{"question": "refund?"}`), http.StatusGatewayTimeout, "rag_timeout")
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Errorf("timeout took %v", elapsed)
	}
	select {
	case <-pythonSawCancel:
	case <-time.After(2 * time.Second):
		t.Error("python request was not canceled after the timeout")
	}
}

func TestClientCancellationReachesPython(t *testing.T) {
	started, pythonSawCancel := make(chan struct{}), make(chan struct{})
	py := newFakePython(t, func(_ http.ResponseWriter, r *http.Request) {
		close(started)
		select {
		case <-r.Context().Done():
			close(pythonSawCancel)
		case <-time.After(5 * time.Second):
		}
	})
	api := newAPI(py.URL, 10*time.Second) // long timeout: only the client cancel can end this

	ctx, cancel := context.WithCancel(context.Background())
	rec := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		defer close(done)
		req := httptest.NewRequest(http.MethodPost, "/query", strings.NewReader(`{"question": "refund?"}`)).WithContext(ctx)
		api.ServeHTTP(rec, req)
	}()

	<-started
	cancel()
	select {
	case <-pythonSawCancel:
	case <-time.After(2 * time.Second):
		t.Fatal("client cancellation did not reach the python request")
	}
	<-done
	if rec.Body.Len() != 0 {
		t.Errorf("wrote a response to a client that is gone: %s", rec.Body)
	}
}
