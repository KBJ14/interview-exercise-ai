package ragclient

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

// IngestRequest is the body of POST /internal/ingest. Nil Source/Title are sent as JSON null.
type IngestRequest struct {
	Text   string  `json:"text"`
	Source *string `json:"source"`
	Title  *string `json:"title"`
	Format string  `json:"format"` // "plain" or "sectioned"
}

// IngestResult describes the stored document. For Status "skipped" the content was already
// stored: DocumentID and ChunkCount describe that existing row and nothing new was written.
type IngestResult struct {
	Status     string // "inserted" or "skipped"
	DocumentID int64
	ChunkCount int
}

// Ingest asks the Python service to parse, embed and store one document. Embedding and
// storage happen there in a single transaction; this client never writes to the database.
func (c *Client) Ingest(ctx context.Context, req IngestRequest) (*IngestResult, error) {
	status, raw, err := c.post(ctx, "/internal/ingest", req)
	if err != nil {
		return nil, err
	}
	if status != http.StatusCreated && status != http.StatusOK {
		return nil, parseError(status, raw)
	}

	var w struct {
		Status     *string `json:"status"`
		DocumentID *int64  `json:"document_id"`
		ChunkCount *int    `json:"chunk_count"`
	}
	if err := json.Unmarshal(raw, &w); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidResponse, err)
	}
	if w.Status == nil || w.DocumentID == nil || w.ChunkCount == nil || *w.DocumentID <= 0 || *w.ChunkCount < 0 {
		return nil, fmt.Errorf("%w: missing or invalid status, document_id or chunk_count", ErrInvalidResponse)
	}
	// The status code and the body must agree, so a changed contract cannot slip through.
	if (status == http.StatusCreated) != (*w.Status == "inserted") || (*w.Status != "inserted" && *w.Status != "skipped") {
		return nil, fmt.Errorf("%w: HTTP %d with status %q", ErrInvalidResponse, status, *w.Status)
	}
	return &IngestResult{Status: *w.Status, DocumentID: *w.DocumentID, ChunkCount: *w.ChunkCount}, nil
}
