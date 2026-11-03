package httpapi

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"path"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/KBJ14/interview-exercise-ai/internal/ragclient"
)

// Size limits. MaxDocumentBytes must match MAX_DOCUMENT_BYTES in rag/api.py.
const (
	MaxDocumentBytes = 256 << 10 // UTF-8 bytes of document text; all chunks go in one embeddings request
	// JSON escaping can grow text up to 6x (\u0001), so the body limit is larger than the text limit.
	maxIngestJSONBytes = 2 << 20
	// A multipart upload is the file plus a few small fields and part headers.
	maxIngestMultipartBytes = MaxDocumentBytes + 64<<10
	maxFormFieldBytes       = 1 << 10
	maxSourceChars          = 255
	maxTitleChars           = 300
)

// IngestResponse is the public body of POST /ingest.
//
// Status "inserted" (HTTP 201): a new document was stored; DocumentID and ChunkCount describe it.
// Status "skipped"  (HTTP 200): identical content (same SHA-256) was already stored; DocumentID and
// ChunkCount describe that existing document, nothing was embedded or written, and the submitted
// title, source and format were ignored.
type IngestResponse struct {
	Status     string `json:"status"`
	DocumentID int64  `json:"document_id"`
	ChunkCount int    `json:"chunk_count"`
}

func (s *Server) handleIngest(w http.ResponseWriter, r *http.Request) {
	req, reqErr := decodeIngest(w, r)
	if reqErr != nil {
		writeError(w, reqErr.status, reqErr.code, reqErr.message)
		return
	}

	// Parsing, chunking, embedding and the single-transaction write all happen in Python.
	res, err := s.rag.Ingest(r.Context(), req)
	if err != nil {
		s.writeRAGError(w, r, err)
		return
	}
	status := http.StatusOK
	if res.Status == "inserted" {
		status = http.StatusCreated
	}
	writeJSON(w, status, IngestResponse{Status: res.Status, DocumentID: res.DocumentID, ChunkCount: res.ChunkCount})
}

// decodeIngest turns either accepted body type into the one request sent to Python.
func decodeIngest(w http.ResponseWriter, r *http.Request) (ragclient.IngestRequest, *requestError) {
	mediaType, _, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	switch {
	case err == nil && mediaType == "application/json":
		return decodeIngestJSON(w, r)
	case err == nil && mediaType == "multipart/form-data":
		return decodeIngestUpload(w, r)
	default:
		return ragclient.IngestRequest{}, &requestError{http.StatusUnsupportedMediaType, "unsupported_media_type",
			"Content-Type must be application/json or multipart/form-data"}
	}
}

func tooLarge(code string, limit int) *requestError {
	what := "request body"
	if code == "document_too_large" {
		what = "document"
	}
	return &requestError{http.StatusRequestEntityTooLarge, code, fmt.Sprintf("%s must be at most %d bytes", what, limit)}
}

// decodeIngestJSON reads {"text", "source"?, "title"?, "format"?}.
func decodeIngestJSON(w http.ResponseWriter, r *http.Request) (ragclient.IngestRequest, *requestError) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxIngestJSONBytes))
	if err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			return ragclient.IngestRequest{}, tooLarge("request_too_large", maxIngestJSONBytes)
		}
		return ragclient.IngestRequest{}, badRequest("could not read request body")
	}
	// encoding/json silently replaces invalid UTF-8 with U+FFFD, so check the raw bytes first.
	if !utf8.Valid(body) {
		return ragclient.IngestRequest{}, badRequest("request body must be valid UTF-8")
	}

	dec := json.NewDecoder(bytes.NewReader(body))
	dec.DisallowUnknownFields()
	var raw struct {
		Text   json.RawMessage `json:"text"`
		Source json.RawMessage `json:"source"`
		Title  json.RawMessage `json:"title"`
		Format json.RawMessage `json:"format"`
	}
	if err := dec.Decode(&raw); err != nil {
		if strings.HasPrefix(err.Error(), "json: unknown field ") {
			return ragclient.IngestRequest{}, badRequest(fmt.Sprintf("unknown field %s; allowed fields are text, source, title and format",
				strings.TrimPrefix(err.Error(), "json: unknown field ")))
		}
		return ragclient.IngestRequest{}, badRequest("request body must be a JSON object with a text field")
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		return ragclient.IngestRequest{}, badRequest("request body must contain a single JSON object")
	}

	text, ok := jsonString(raw.Text)
	if !ok {
		return ragclient.IngestRequest{}, badRequest("text must be a non-empty string")
	}
	source, reqErr := optionalJSONString(raw.Source, "source")
	if reqErr != nil {
		return ragclient.IngestRequest{}, reqErr
	}
	title, reqErr := optionalJSONString(raw.Title, "title")
	if reqErr != nil {
		return ragclient.IngestRequest{}, reqErr
	}
	format, reqErr := optionalJSONString(raw.Format, "format")
	if reqErr != nil {
		return ragclient.IngestRequest{}, reqErr
	}
	return buildIngestRequest(text, source, title, format)
}

func jsonString(raw json.RawMessage) (string, bool) {
	var s string
	if len(raw) == 0 || raw[0] != '"' || json.Unmarshal(raw, &s) != nil {
		return "", false
	}
	return s, true
}

// optionalJSONString treats a missing field and JSON null the same way, as Python does.
func optionalJSONString(raw json.RawMessage, name string) (*string, *requestError) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	s, ok := jsonString(raw)
	if !ok {
		return nil, badRequest(name + " must be a string")
	}
	return &s, nil
}

// decodeIngestUpload reads multipart fields "file" (required), "title" and "format" (optional).
// The file is streamed into memory under a size cap; nothing is written to disk, and the
// client's file name is used only as the document's source label.
func decodeIngestUpload(w http.ResponseWriter, r *http.Request) (ragclient.IngestRequest, *requestError) {
	r.Body = http.MaxBytesReader(w, r.Body, maxIngestMultipartBytes)
	mr, err := r.MultipartReader()
	if err != nil {
		return ragclient.IngestRequest{}, badRequest("malformed multipart/form-data body")
	}

	var (
		text, source  *string
		title, format *string
	)
	for {
		part, err := mr.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			return ragclient.IngestRequest{}, multipartError(err)
		}
		name := part.FormName()
		switch name {
		case "file":
			if text != nil {
				return ragclient.IngestRequest{}, badRequest("exactly one file may be uploaded")
			}
			content, label, reqErr := readUploadedFile(part)
			if reqErr != nil {
				return ragclient.IngestRequest{}, reqErr
			}
			text, source = &content, &label
		case "title", "format":
			if (name == "title" && title != nil) || (name == "format" && format != nil) || part.FileName() != "" {
				return ragclient.IngestRequest{}, badRequest(name + " must be a single text field")
			}
			value, reqErr := readFormField(part, name)
			if reqErr != nil {
				return ragclient.IngestRequest{}, reqErr
			}
			if name == "title" {
				title = &value
			} else {
				format = &value
			}
		default:
			return ragclient.IngestRequest{}, badRequest(fmt.Sprintf("unknown form field %q; allowed fields are file, title and format", name))
		}
	}
	if text == nil {
		return ragclient.IngestRequest{}, badRequest("a file field is required")
	}
	return buildIngestRequest(*text, source, title, format)
}

func multipartError(err error) *requestError {
	var tooBig *http.MaxBytesError
	if errors.As(err, &tooBig) {
		return tooLarge("request_too_large", maxIngestMultipartBytes)
	}
	return badRequest("malformed multipart/form-data body")
}

func readUploadedFile(part *multipart.Part) (string, string, *requestError) {
	label := uploadName(part.FileName())
	if label == "" {
		return "", "", badRequest("the file part must have a file name")
	}
	// The loader reads plain-text .txt documents; other types would need their own parser (e.g. PDF).
	if !strings.EqualFold(path.Ext(label), ".txt") {
		return "", "", &requestError{http.StatusUnsupportedMediaType, "unsupported_file_type", "only .txt files are supported"}
	}
	data, err := io.ReadAll(io.LimitReader(part, MaxDocumentBytes+1))
	if err != nil {
		return "", "", multipartError(err)
	}
	if len(data) > MaxDocumentBytes {
		return "", "", tooLarge("document_too_large", MaxDocumentBytes)
	}
	if !utf8.Valid(data) {
		return "", "", badRequest("file must be UTF-8 text")
	}
	return string(data), label, nil
}

// uploadName keeps only the last path element of a client-supplied file name, for either
// separator. multipart.Part.FileName applies filepath.Base, which on Linux ignores backslashes.
func uploadName(name string) string {
	if i := strings.LastIndexAny(name, `/\`); i >= 0 {
		name = name[i+1:]
	}
	if name == "." || name == ".." || strings.IndexFunc(name, unicode.IsControl) >= 0 {
		return ""
	}
	return name
}

func readFormField(part *multipart.Part, name string) (string, *requestError) {
	data, err := io.ReadAll(io.LimitReader(part, maxFormFieldBytes+1))
	if err != nil {
		return "", multipartError(err)
	}
	if len(data) > maxFormFieldBytes || !utf8.Valid(data) {
		return "", badRequest(fmt.Sprintf("%s must be UTF-8 text of at most %d bytes", name, maxFormFieldBytes))
	}
	return string(data), nil
}

// buildIngestRequest applies the checks shared by both body types. Rules that depend on the
// document layout (a title is required for plain text, sectioned text must parse) stay in
// Python, which owns parsing; its 400 invalid_document responses are forwarded as-is.
func buildIngestRequest(text string, source, title, format *string) (ragclient.IngestRequest, *requestError) {
	if strings.TrimSpace(text) == "" {
		return ragclient.IngestRequest{}, badRequest("text must be a non-empty string")
	}
	if len(text) > MaxDocumentBytes {
		return ragclient.IngestRequest{}, tooLarge("document_too_large", MaxDocumentBytes)
	}
	if strings.ContainsRune(text, 0) {
		// PostgreSQL TEXT cannot store NUL, and it usually means a binary file.
		return ragclient.IngestRequest{}, badRequest("text must not contain NUL characters")
	}
	for _, f := range []struct {
		name  string
		value *string
		limit int
	}{{"source", source, maxSourceChars}, {"title", title, maxTitleChars}} {
		if f.value != nil && (strings.TrimSpace(*f.value) == "" || utf8.RuneCountInString(*f.value) > f.limit) {
			return ragclient.IngestRequest{}, badRequest(fmt.Sprintf("%s must be a non-empty string of at most %d characters", f.name, f.limit))
		}
	}
	fmtValue := "plain"
	if format != nil {
		fmtValue = *format
	}
	if fmtValue != "plain" && fmtValue != "sectioned" {
		return ragclient.IngestRequest{}, badRequest("format must be plain or sectioned")
	}
	return ragclient.IngestRequest{Text: text, Source: source, Title: title, Format: fmtValue}, nil
}
