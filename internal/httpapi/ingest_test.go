package httpapi_test

import (
	"bytes"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/textproto"
	"strings"
	"testing"
	"time"

	"github.com/KBJ14/interview-exercise-ai/internal/httpapi"
)

const pythonInserted = `{"status": "inserted", "document_id": 41, "chunk_count": 2}`

func postIngest(api http.Handler, contentType string, body []byte) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/ingest", bytes.NewReader(body))
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	api.ServeHTTP(rec, req)
	return rec
}

func postJSON(api http.Handler, body string) *httptest.ResponseRecorder {
	return postIngest(api, "application/json", []byte(body))
}

type formPart struct {
	field, filename string
	content         []byte
}

// multipartBody builds a form; parts with a filename are file parts.
func multipartBody(t *testing.T, parts ...formPart) (string, []byte) {
	t.Helper()
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	for _, p := range parts {
		h := textproto.MIMEHeader{}
		disposition := `form-data; name="` + p.field + `"`
		if p.filename != "" {
			// Quote the way multipart.Writer.CreateFormFile does, so backslashes survive parsing.
			disposition += `; filename="` + strings.NewReplacer(`\`, `\\`, `"`, `\"`).Replace(p.filename) + `"`
			h.Set("Content-Type", "text/plain")
		}
		h.Set("Content-Disposition", disposition)
		w, err := mw.CreatePart(h)
		if err != nil {
			t.Fatal(err)
		}
		_, _ = w.Write(p.content)
	}
	if err := mw.Close(); err != nil {
		t.Fatal(err)
	}
	return mw.FormDataContentType(), buf.Bytes()
}

func decodeIngestResponse(t *testing.T, rec *httptest.ResponseRecorder) httpapi.IngestResponse {
	t.Helper()
	var got map[string]json.RawMessage
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil || len(got) != 3 {
		t.Fatalf("body is not {status, document_id, chunk_count}: %s", rec.Body)
	}
	var resp httpapi.IngestResponse
	_ = json.Unmarshal(rec.Body.Bytes(), &resp)
	return resp
}

func TestIngestJSONTextIsForwardedAndReturns201(t *testing.T) {
	py := newFakePython(t, reply(http.StatusCreated, pythonInserted))
	rec := postJSON(newAPI(py.URL, time.Second),
		`{"text": "Refunds take 5 days. <b>Really</b>.", "title": "Refund Note", "source": "note.txt", "format": "plain"}`)

	if rec.Code != http.StatusCreated {
		t.Fatalf("status = %d; body %s", rec.Code, rec.Body)
	}
	if resp := decodeIngestResponse(t, rec); resp != (httpapi.IngestResponse{Status: "inserted", DocumentID: 41, ChunkCount: 2}) {
		t.Errorf("response = %+v", resp)
	}
	if py.last.path != "/internal/ingest" || py.last.method != http.MethodPost || py.last.contentType != "application/json" {
		t.Errorf("python got %s %s (%s)", py.last.method, py.last.path, py.last.contentType)
	}
	want := map[string]any{"text": "Refunds take 5 days. <b>Really</b>.", "title": "Refund Note", "source": "note.txt", "format": "plain"}
	for k, v := range want {
		if py.last.body[k] != v {
			t.Errorf("python got %s = %#v, want %#v", k, py.last.body[k], v)
		}
	}
}

func TestIngestJSONDefaultsFormatAndSendsNullForMissingFields(t *testing.T) {
	py := newFakePython(t, reply(http.StatusCreated, pythonInserted))
	postJSON(newAPI(py.URL, time.Second), `{"text": "Hello world.", "title": null}`)
	if py.last.body["format"] != "plain" {
		t.Errorf("format = %#v, want plain", py.last.body["format"])
	}
	for _, k := range []string{"source", "title"} {
		if v, ok := py.last.body[k]; !ok || v != nil {
			t.Errorf("%s = %#v (present %v), want JSON null", k, v, ok)
		}
	}
}

func TestIngestDuplicateIs200Skipped(t *testing.T) {
	py := newFakePython(t, reply(http.StatusOK, `{"status": "skipped", "document_id": 3, "chunk_count": 5}`))
	rec := postJSON(newAPI(py.URL, time.Second), `{"text": "already stored", "title": "T"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d; body %s", rec.Code, rec.Body)
	}
	if resp := decodeIngestResponse(t, rec); resp != (httpapi.IngestResponse{Status: "skipped", DocumentID: 3, ChunkCount: 5}) {
		t.Errorf("response = %+v", resp)
	}
}

func TestIngestFileUploadUsesFileNameOnlyAsSource(t *testing.T) {
	cases := map[string]string{
		"plain name":        "returns.txt",
		"unix traversal":    "../../etc/returns.txt",
		"windows path":      `C:\Users\me\returns.txt`,
		"uppercase txt ext": "RETURNS.TXT",
	}
	for name, filename := range cases {
		t.Run(name, func(t *testing.T) {
			py := newFakePython(t, reply(http.StatusCreated, pythonInserted))
			ct, body := multipartBody(t,
				formPart{field: "file", filename: filename, content: []byte("Returns are accepted within 30 days.\n")},
				formPart{field: "title", content: []byte("Returns")},
				formPart{field: "format", content: []byte("plain")},
			)
			rec := postIngest(newAPI(py.URL, time.Second), ct, body)
			if rec.Code != http.StatusCreated {
				t.Fatalf("status = %d; body %s", rec.Code, rec.Body)
			}
			wantSource := filename[strings.LastIndexAny(filename, `/\`)+1:]
			if py.last.body["source"] != wantSource || py.last.body["text"] != "Returns are accepted within 30 days.\n" ||
				py.last.body["title"] != "Returns" || py.last.body["format"] != "plain" {
				t.Errorf("python got %#v, want source %q", py.last.body, wantSource)
			}
		})
	}
}

func TestIngestUploadWithoutTitleOrFormatSendsDefaults(t *testing.T) {
	py := newFakePython(t, reply(http.StatusCreated, pythonInserted))
	ct, body := multipartBody(t, formPart{field: "file", filename: "billing.txt", content: []byte("[FICTIONAL]\n\nTitle\n\nH\nBody.\n")})
	postIngest(newAPI(py.URL, time.Second), ct, body)
	if py.last.body["format"] != "plain" || py.last.body["title"] != nil || py.last.body["source"] != "billing.txt" {
		t.Errorf("python got %#v", py.last.body)
	}
}

func TestIngestInvalidInputNeverCallsPython(t *testing.T) {
	py := newFakePython(t, reply(http.StatusCreated, pythonInserted))
	api := newAPI(py.URL, time.Second)
	file := func(name string, content []byte) formPart {
		return formPart{field: "file", filename: name, content: content}
	}

	jsonCases := map[string]string{
		"empty body":       ``,
		"malformed":        `{"text": `,
		"array":            `["text"]`,
		"two values":       `{"text": "a"} {"text": "b"}`,
		"missing text":     `{"title": "T"}`,
		"null text":        `{"text": null}`,
		"empty text":       `{"text": ""}`,
		"blank text":       `{"text": " \n\t "}`,
		"number text":      `{"text": 42}`,
		"nul in text":      `{"text": "a\u0000b"}`,
		"bad format":       `{"text": "ok", "format": "pdf"}`,
		"number format":    `{"text": "ok", "format": 1}`,
		"blank title":      `{"text": "ok", "title": "  "}`,
		"long title":       `{"text": "ok", "title": "` + strings.Repeat("t", 301) + `"}`,
		"number source":    `{"text": "ok", "source": 7}`,
		"long source":      `{"text": "ok", "source": "` + strings.Repeat("s", 256) + `"}`,
		"unknown field":    `{"text": "ok", "path": "/etc/passwd"}`,
		"invalid utf8 raw": "{\"text\": \"caf\xe9\"}",
	}
	for name, body := range jsonCases {
		t.Run("json "+name, func(t *testing.T) {
			assertError(t, postJSON(api, body), http.StatusBadRequest, "invalid_request")
		})
	}

	uploadCases := map[string][]formPart{
		"no file":           {{field: "title", content: []byte("T")}},
		"two files":         {file("a.txt", []byte("a")), file("b.txt", []byte("b"))},
		"empty file":        {file("a.txt", nil)},
		"blank file":        {file("a.txt", []byte("  \n"))},
		"invalid utf8 file": {file("a.txt", []byte{0xff, 0xfe, 'h', 'i'})},
		"nul in file":       {file("a.txt", []byte("bin\x00ary"))},
		"unknown field":     {file("a.txt", []byte("ok")), {field: "source", content: []byte("x")}},
		"title as file":     {file("a.txt", []byte("ok")), {field: "title", filename: "t.txt", content: []byte("T")}},
		"bad format field":  {file("a.txt", []byte("ok")), {field: "format", content: []byte("markdown")}},
		"dot-dot file name": {file("..", []byte("ok"))},
	}
	for name, parts := range uploadCases {
		t.Run("upload "+name, func(t *testing.T) {
			ct, body := multipartBody(t, parts...)
			assertError(t, postIngest(api, ct, body), http.StatusBadRequest, "invalid_request")
		})
	}

	t.Run("upload non-txt file", func(t *testing.T) {
		for _, name := range []string{"doc.pdf", "notes.md", "archive.txt.zip", "noext"} {
			ct, body := multipartBody(t, file(name, []byte("ok")))
			assertError(t, postIngest(api, ct, body), http.StatusUnsupportedMediaType, "unsupported_file_type")
		}
	})
	t.Run("unsupported content type", func(t *testing.T) {
		for _, ct := range []string{"", "text/plain", "application/x-www-form-urlencoded"} {
			assertError(t, postIngest(api, ct, []byte(`text=hi`)), http.StatusUnsupportedMediaType, "unsupported_media_type")
		}
	})
	t.Run("multipart without boundary", func(t *testing.T) {
		assertError(t, postIngest(api, "multipart/form-data", []byte("x")), http.StatusBadRequest, "invalid_request")
	})

	if n := py.calls.Load(); n != 0 {
		t.Errorf("python was called %d times for invalid input", n)
	}
}

func TestIngestSizeLimitsAre413(t *testing.T) {
	py := newFakePython(t, reply(http.StatusCreated, pythonInserted))
	api := newAPI(py.URL, time.Second)
	over := strings.Repeat("a", httpapi.MaxDocumentBytes+1)

	assertError(t, postJSON(api, `{"text": "`+over+`", "title": "T"}`), http.StatusRequestEntityTooLarge, "document_too_large")
	// 3-byte characters: under the limit in characters, over it in UTF-8 bytes.
	assertError(t, postJSON(api, `{"text": "`+strings.Repeat("가", httpapi.MaxDocumentBytes/3+1)+`", "title": "T"}`),
		http.StatusRequestEntityTooLarge, "document_too_large")
	assertError(t, postJSON(api, `{"text": "`+strings.Repeat(`\u0001`, (2<<20)/6+1)+`"}`), // escapes inflate the body
		http.StatusRequestEntityTooLarge, "request_too_large")

	ct, body := multipartBody(t, formPart{field: "file", filename: "big.txt", content: []byte(over)})
	assertError(t, postIngest(api, ct, body), http.StatusRequestEntityTooLarge, "document_too_large")
	ct, body = multipartBody(t, formPart{field: "file", filename: "huge.txt", content: bytes.Repeat([]byte("a"), 2*httpapi.MaxDocumentBytes)})
	if rec := postIngest(api, ct, body); rec.Code != http.StatusRequestEntityTooLarge {
		t.Errorf("huge upload: status %d body %s", rec.Code, rec.Body)
	}

	if n := py.calls.Load(); n != 0 {
		t.Errorf("python was called %d times for oversized input", n)
	}

	// Exactly at the limit is accepted.
	atLimit := strings.Repeat("a", httpapi.MaxDocumentBytes)
	if rec := postJSON(api, `{"text": "`+atLimit+`", "title": "T"}`); rec.Code != http.StatusCreated {
		t.Errorf("document at the limit: status %d body %s", rec.Code, rec.Body)
	}
	ct, body = multipartBody(t, formPart{field: "file", filename: "max.txt", content: []byte(atLimit)})
	if rec := postIngest(api, ct, body); rec.Code != http.StatusCreated {
		t.Errorf("upload at the limit: status %d body %s", rec.Code, rec.Body)
	}
}

func TestIngestPythonErrorsAreTranslated(t *testing.T) {
	cases := []struct {
		name         string
		status       int
		body         string
		wantStatus   int
		wantCode     string
		forwardedMsg bool
	}{
		{"document rejected", 400, `{"error":{"code":"invalid_document","message":"text: plain text needs a title or a source file name to derive it from"}}`, 400, "invalid_document", true},
		{"python size check", 413, `{"error":{"code":"document_too_large","message":"document must be at most 262144 bytes of UTF-8 text"}}`, 413, "document_too_large", true},
		{"embedding outage", 503, `{"error":{"code":"llm_unavailable","message":"could not reach OpenAI API"}}`, 503, "llm_unavailable", false},
		{"database outage", 503, `{"error":{"code":"database_unavailable","message":"database is unreachable"}}`, 503, "database_unavailable", false},
		{"storage bug", 500, `{"error":{"code":"internal_error","message":"unexpected internal error"}}`, 502, "rag_service_error", false},
		{"status/code mismatch", 500, `{"error":{"code":"invalid_document","message":"traceback detail"}}`, 502, "rag_service_error", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			py := newFakePython(t, reply(tc.status, tc.body))
			msg := assertError(t, postJSON(newAPI(py.URL, time.Second), `{"text": "x", "title": "T"}`), tc.wantStatus, tc.wantCode)
			var upstream struct{ Error struct{ Message string } }
			_ = json.Unmarshal([]byte(tc.body), &upstream)
			if forwarded := msg == upstream.Error.Message; forwarded != tc.forwardedMsg {
				t.Errorf("python message forwarded = %v, want %v (public %q)", forwarded, tc.forwardedMsg, msg)
			}
		})
	}
}

func TestIngestUnusablePythonResponsesAre502(t *testing.T) {
	cases := map[string]struct {
		status int
		body   string
	}{
		"not json":            {201, `created`},
		"missing document_id": {201, `{"status": "inserted", "chunk_count": 2}`},
		"zero document_id":    {201, `{"status": "inserted", "document_id": 0, "chunk_count": 2}`},
		"201 but skipped":     {201, `{"status": "skipped", "document_id": 3, "chunk_count": 5}`},
		"200 but inserted":    {200, `{"status": "inserted", "document_id": 3, "chunk_count": 5}`},
		"unknown status":      {200, `{"status": "queued", "document_id": 3, "chunk_count": 5}`},
		"negative chunks":     {201, `{"status": "inserted", "document_id": 3, "chunk_count": -1}`},
		"204 no body":         {204, ``},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			py := newFakePython(t, reply(tc.status, tc.body))
			assertError(t, postJSON(newAPI(py.URL, time.Second), `{"text": "x", "title": "T"}`), http.StatusBadGateway, "rag_invalid_response")
		})
	}
}

func TestIngestTransportFailures(t *testing.T) {
	py := newFakePython(t, reply(http.StatusCreated, pythonInserted))
	url := py.URL
	py.Close()
	assertError(t, postJSON(newAPI(url, time.Second), `{"text": "x", "title": "T"}`), http.StatusBadGateway, "rag_unavailable")

	pythonSawCancel := make(chan struct{})
	slow := newFakePython(t, func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
			close(pythonSawCancel)
		case <-time.After(5 * time.Second):
			reply(http.StatusCreated, pythonInserted)(w, r)
		}
	})
	assertError(t, postJSON(newAPI(slow.URL, 100*time.Millisecond), `{"text": "x", "title": "T"}`), http.StatusGatewayTimeout, "rag_timeout")
	select {
	case <-pythonSawCancel:
	case <-time.After(2 * time.Second):
		t.Error("python ingest request was not canceled after the timeout")
	}
}

func TestIngestWrongMethodIs405(t *testing.T) {
	rec := httptest.NewRecorder()
	newAPI("http://127.0.0.1:1", time.Second).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/ingest", nil))
	assertError(t, rec, http.StatusMethodNotAllowed, "method_not_allowed")
	if rec.Header().Get("Allow") != http.MethodPost {
		t.Errorf("Allow = %q", rec.Header().Get("Allow"))
	}
}
