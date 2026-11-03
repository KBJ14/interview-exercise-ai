// Package ragclient calls the internal Python RAG service (rag/api.py) over HTTP.
//
// Every failure is reported as one of the sentinel errors below or as *UpstreamError,
// so callers can map it to a public response without inspecting transport details.
package ragclient

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

var (
	ErrUnavailable     = errors.New("rag service unreachable")
	ErrTimeout         = errors.New("rag service timed out")
	ErrCanceled        = errors.New("request canceled by caller")
	ErrInvalidResponse = errors.New("rag service returned an unusable response")
)

// UpstreamError is a structured error returned by the RAG service: {"error":{"code","message"}}.
type UpstreamError struct {
	Status  int
	Code    string
	Message string
}

func (e *UpstreamError) Error() string {
	return fmt.Sprintf("rag service HTTP %d %s: %s", e.Status, e.Code, e.Message)
}

type Source struct {
	Label          string
	DocumentID     int64
	Title          string
	Section        string
	ChunkIndex     int
	Chunk          string
	CosineDistance float64
	Cited          bool
}

type Answer struct {
	Answer     string
	Answerable bool
	Citations  []string
	Sources    []Source
}

// Responses are a few KB; the cap stops a misbehaving service from exhausting memory.
const maxResponseBytes = 1 << 20

type Client struct {
	baseURL string
	timeout time.Duration
	http    *http.Client
}

// New returns a client for the service at baseURL. timeout bounds each Query call,
// on top of any deadline or cancellation already carried by the caller's context.
func New(baseURL string, timeout time.Duration) *Client {
	return &Client{baseURL: strings.TrimRight(baseURL, "/"), timeout: timeout, http: &http.Client{}}
}

func (c *Client) Query(ctx context.Context, question string, k int) (*Answer, error) {
	status, raw, err := c.post(ctx, "/internal/query", map[string]any{"question": question, "k": k})
	if err != nil {
		return nil, err
	}
	if status != http.StatusOK {
		return nil, parseError(status, raw)
	}
	return parseAnswer(raw)
}

// post sends payload as JSON and returns the status and the (size-capped) body.
// Transport failures are already mapped to ErrTimeout, ErrCanceled or ErrUnavailable.
func (c *Client) post(ctx context.Context, path string, payload any) (int, []byte, error) {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	var body bytes.Buffer
	enc := json.NewEncoder(&body)
	enc.SetEscapeHTML(false) // document text is data; <-style escaping only inflates it
	if err := enc.Encode(payload); err != nil {
		return 0, nil, fmt.Errorf("encode request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, &body)
	if err != nil {
		return 0, nil, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return 0, nil, transportError(ctx, err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes+1))
	if err != nil {
		// The deadline can also expire while the body is still streaming.
		return 0, nil, transportError(ctx, err)
	}
	if len(raw) > maxResponseBytes {
		return 0, nil, fmt.Errorf("%w: body larger than %d bytes", ErrInvalidResponse, maxResponseBytes)
	}
	return resp.StatusCode, raw, nil
}

// transportError tells our timeout apart from the caller going away and from a dead service.
func transportError(ctx context.Context, err error) error {
	switch {
	case errors.Is(ctx.Err(), context.DeadlineExceeded):
		return fmt.Errorf("%w: %v", ErrTimeout, err)
	case errors.Is(ctx.Err(), context.Canceled):
		return fmt.Errorf("%w: %v", ErrCanceled, err)
	default:
		return fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
}

func parseError(status int, raw []byte) error {
	var body struct {
		Error *struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(raw, &body); err != nil || body.Error == nil || body.Error.Code == "" {
		return fmt.Errorf("%w: HTTP %d without a structured error body", ErrInvalidResponse, status)
	}
	return &UpstreamError{Status: status, Code: body.Error.Code, Message: body.Error.Message}
}

// Wire format of a successful /internal/query response. Pointers distinguish a missing
// field from a zero value, so a truncated or changed contract is rejected, not passed on.
type wireAnswer struct {
	Answerable *bool        `json:"answerable"`
	Answer     *string      `json:"answer"`
	Citations  []string     `json:"citations"`
	Sources    []wireSource `json:"sources"`
}

type wireSource struct {
	Label          string   `json:"label"`
	DocumentID     *int64   `json:"document_id"`
	Title          string   `json:"title"`
	Section        string   `json:"section"`
	ChunkIndex     *int     `json:"chunk_index"`
	Chunk          *string  `json:"chunk"`
	CosineDistance *float64 `json:"cosine_distance"`
	Cited          bool     `json:"cited"`
}

func parseAnswer(raw []byte) (*Answer, error) {
	var w wireAnswer
	if err := json.Unmarshal(raw, &w); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidResponse, err)
	}
	if w.Answerable == nil || w.Answer == nil || w.Citations == nil || w.Sources == nil {
		return nil, fmt.Errorf("%w: missing answerable, answer, citations or sources", ErrInvalidResponse)
	}

	out := &Answer{Answer: *w.Answer, Answerable: *w.Answerable, Citations: w.Citations, Sources: make([]Source, 0, len(w.Sources))}
	labels := make(map[string]bool, len(w.Sources))
	for i, s := range w.Sources {
		if s.Label == "" || s.DocumentID == nil || s.ChunkIndex == nil || s.Chunk == nil || s.CosineDistance == nil {
			return nil, fmt.Errorf("%w: source %d is missing required fields", ErrInvalidResponse, i)
		}
		labels[s.Label] = true
		out.Sources = append(out.Sources, Source{
			Label: s.Label, DocumentID: *s.DocumentID, Title: s.Title, Section: s.Section,
			ChunkIndex: *s.ChunkIndex, Chunk: *s.Chunk, CosineDistance: *s.CosineDistance, Cited: s.Cited,
		})
	}
	for _, c := range w.Citations {
		if !labels[c] {
			return nil, fmt.Errorf("%w: citation %q has no matching source", ErrInvalidResponse, c)
		}
	}
	return out, nil
}
