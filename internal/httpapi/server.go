// Package httpapi implements the public REST API.
package httpapi

import (
	"context"
	"errors"
	"log/slog"
	"net/http"

	"github.com/KBJ14/interview-exercise-ai/internal/ragclient"
)

// Querier is the RAG backend the handlers depend on; *ragclient.Client implements it.
type Querier interface {
	Query(ctx context.Context, question string, k int) (*ragclient.Answer, error)
}

type Server struct {
	rag  Querier
	docs DocumentStore
	log  *slog.Logger
}

// NewServer returns the API's HTTP handler.
func NewServer(rag Querier, docs DocumentStore, log *slog.Logger) http.Handler {
	s := &Server{rag: rag, docs: docs, log: log}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /query", s.handleQuery)
	mux.HandleFunc("GET /documents", s.handleListDocuments)
	mux.HandleFunc("DELETE /documents/{id}", s.handleDeleteDocument)
	// Without these, ServeMux answers 404/405 in plain text; keep the JSON error shape everywhere.
	mux.HandleFunc("/query", methodNotAllowed(http.MethodPost))
	mux.HandleFunc("/documents", methodNotAllowed(http.MethodGet, http.MethodHead))
	mux.HandleFunc("/documents/{id}", methodNotAllowed(http.MethodDelete))
	mux.HandleFunc("/", notFound)
	return mux
}

// QueryResponse is the public body of a successful POST /query.
type QueryResponse struct {
	Answer     string   `json:"answer"`
	Answerable bool     `json:"answerable"`
	Citations  []string `json:"citations"` // labels of the sources the answer cites, e.g. ["S1"]
	Sources    []Source `json:"sources"`   // every retrieved chunk, nearest first
}

type Source struct {
	Label      string `json:"label"`
	DocumentID int64  `json:"document_id"`
	Title      string `json:"title"`
	Section    string `json:"section"`
	ChunkIndex int    `json:"chunk_index"`
	Chunk      string `json:"chunk"`
	// Distance between question and chunk embeddings: 0 is the same direction, smaller is
	// closer. It is not a probability or confidence score.
	CosineDistance float64 `json:"cosine_distance"`
	Cited          bool    `json:"cited"`
}

func (s *Server) handleQuery(w http.ResponseWriter, r *http.Request) {
	question, k, reqErr := decodeQuery(w, r)
	if reqErr != nil {
		writeError(w, reqErr.status, reqErr.code, reqErr.message)
		return
	}

	// r.Context() is canceled when the client disconnects, which aborts the Python call too.
	ans, err := s.rag.Query(r.Context(), question, k)
	if err != nil {
		s.writeRAGError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, toResponse(ans))
}

func toResponse(a *ragclient.Answer) QueryResponse {
	sources := make([]Source, 0, len(a.Sources))
	for _, s := range a.Sources {
		sources = append(sources, Source{
			Label: s.Label, DocumentID: s.DocumentID, Title: s.Title, Section: s.Section,
			ChunkIndex: s.ChunkIndex, Chunk: s.Chunk, CosineDistance: s.CosineDistance, Cited: s.Cited,
		})
	}
	citations := a.Citations
	if citations == nil {
		citations = []string{} // always an array in JSON, never null
	}
	return QueryResponse{Answer: a.Answer, Answerable: a.Answerable, Citations: citations, Sources: sources}
}

// Python error codes that are safe to expose as-is, with our own public message.
// Python's messages are never forwarded for these: they can carry model or SDK details.
var upstreamErrors = map[string]struct {
	status  int
	message string
}{
	"llm_unavailable":            {http.StatusServiceUnavailable, "language model service is unavailable"},
	"llm_rate_limited":           {http.StatusServiceUnavailable, "language model rate limit reached; try again later"},
	"llm_timeout":                {http.StatusGatewayTimeout, "language model did not respond in time"},
	"llm_invalid_response":       {http.StatusBadGateway, "language model returned an unusable answer"},
	"llm_request_rejected":       {http.StatusBadGateway, "language model rejected the request"},
	"embedding_invalid_response": {http.StatusBadGateway, "embedding service returned an unexpected result"},
	"database_unavailable":       {http.StatusServiceUnavailable, "database is unavailable"},
}

func (s *Server) writeRAGError(w http.ResponseWriter, r *http.Request, err error) {
	var up *ragclient.UpstreamError
	switch {
	case errors.Is(err, ragclient.ErrCanceled):
		// The client is gone; there is nobody to send a response to.
		s.log.Info("query canceled by client", "err", err)
	case errors.Is(err, ragclient.ErrTimeout):
		s.log.Warn("rag service timed out", "err", err)
		writeError(w, http.StatusGatewayTimeout, "rag_timeout", "RAG service did not respond in time")
	case errors.Is(err, ragclient.ErrUnavailable):
		s.log.Error("rag service unreachable", "err", err)
		writeError(w, http.StatusBadGateway, "rag_unavailable", "RAG service is unreachable")
	case errors.As(err, &up):
		s.writeUpstreamError(w, up)
	case errors.Is(err, ragclient.ErrInvalidResponse):
		s.log.Error("invalid rag service response", "err", err)
		writeError(w, http.StatusBadGateway, "rag_invalid_response", "RAG service returned an invalid response")
	default:
		s.log.Error("unexpected error calling rag service", "err", err, "path", r.URL.Path)
		writeError(w, http.StatusInternalServerError, "internal_error", "unexpected internal error")
	}
}

func (s *Server) writeUpstreamError(w http.ResponseWriter, up *ragclient.UpstreamError) {
	if up.Status == http.StatusBadRequest && up.Code == "invalid_request" {
		// Only input-validation text reaches this branch; it describes the caller's own request.
		writeError(w, http.StatusBadRequest, "invalid_request", up.Message)
		return
	}
	if known, ok := upstreamErrors[up.Code]; ok {
		s.log.Warn("rag dependency failure", "status", up.Status, "code", up.Code, "detail", up.Message)
		writeError(w, known.status, up.Code, known.message)
		return
	}
	// Python 500s (internal_error, configuration_error), 404s and unknown codes are failures of the
	// internal service itself; from the caller's point of view the gateway's upstream broke.
	s.log.Error("rag service error", "status", up.Status, "code", up.Code, "detail", up.Message)
	writeError(w, http.StatusBadGateway, "rag_service_error", "RAG service failed to process the request")
}
