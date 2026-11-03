package httpapi

import (
	"context"
	"errors"
	"net/http"
	"regexp"
	"strconv"
	"time"

	"github.com/KBJ14/interview-exercise-ai/internal/store"
)

// DocumentStore is the metadata store the handlers depend on; *store.Store implements it.
type DocumentStore interface {
	ListDocuments(ctx context.Context) ([]store.Document, error)
	DeleteDocument(ctx context.Context, id int64) (bool, error)
}

// Canonical positive decimal only: rejects 0, -1, +1, 007, 1.0 and 1e3 before any parsing.
var documentID = regexp.MustCompile(`^[1-9][0-9]*$`)

// DocumentsResponse is the public body of GET /documents.
type DocumentsResponse struct {
	Documents []Document `json:"documents"` // newest first; always an array, never null
}

// Document is the public metadata of one ingested document.
// The original text and the embeddings are deliberately not part of the listing.
type Document struct {
	ID          int64     `json:"id"`
	Title       string    `json:"title"`
	Source      *string   `json:"source"` // null when the text was ingested without a file name
	ContentHash string    `json:"content_hash"`
	CreatedAt   time.Time `json:"created_at"` // RFC 3339, UTC
}

func (s *Server) handleListDocuments(w http.ResponseWriter, r *http.Request) {
	docs, err := s.docs.ListDocuments(r.Context())
	if err != nil {
		s.writeStoreError(w, r, err)
		return
	}
	out := DocumentsResponse{Documents: make([]Document, 0, len(docs))}
	for _, d := range docs {
		out.Documents = append(out.Documents, Document{
			ID: d.ID, Title: d.Title, Source: d.Source, ContentHash: d.ContentHash, CreatedAt: d.CreatedAt.UTC(),
		})
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *Server) handleDeleteDocument(w http.ResponseWriter, r *http.Request) {
	raw := r.PathValue("id")
	// documents.id is BIGINT; ParseInt with bitSize 64 rejects anything above its maximum.
	id, err := strconv.ParseInt(raw, 10, 64)
	if !documentID.MatchString(raw) || err != nil {
		writeError(w, http.StatusBadRequest, "invalid_document_id",
			"document id must be a positive integer no greater than 9223372036854775807")
		return
	}

	deleted, err := s.docs.DeleteDocument(r.Context(), id)
	if err != nil {
		s.writeStoreError(w, r, err)
		return
	}
	if !deleted {
		writeError(w, http.StatusNotFound, "document_not_found", "no document with this id")
		return
	}
	s.log.Info("document deleted", "id", id)
	w.WriteHeader(http.StatusNoContent) // no body, no Content-Type
}

func (s *Server) writeStoreError(w http.ResponseWriter, r *http.Request, err error) {
	switch {
	case errors.Is(err, store.ErrCanceled):
		s.log.Info("documents query canceled by client", "err", err)
	case errors.Is(err, store.ErrTimeout):
		s.log.Warn("documents query timed out", "err", err)
		writeError(w, http.StatusServiceUnavailable, "database_timeout", "database did not respond in time")
	case errors.Is(err, store.ErrUnavailable):
		s.log.Error("database unavailable", "err", err)
		writeError(w, http.StatusServiceUnavailable, "database_unavailable", "database is unavailable")
	default:
		// Details (SQLSTATE, driver messages) stay in the log.
		s.log.Error("unexpected database error", "err", err, "path", r.URL.Path)
		writeError(w, http.StatusInternalServerError, "internal_error", "unexpected internal error")
	}
}
