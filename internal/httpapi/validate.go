package httpapi

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"
)

// Same rules and default as rag/retriever.py (DEFAULT_K, MAX_K, validate_query), so a request
// the Go API accepts is never rejected by the Python service for a different reason.
const (
	DefaultK = 3
	MaxK     = 20

	maxRequestBytes = 64 << 10
)

// Only plain integer literals: 2.5, 3.0 and 1e1 are rejected, as Python rejects floats.
var integerLiteral = regexp.MustCompile(`^-?[0-9]+$`)

type requestError struct {
	status  int
	code    string
	message string
}

func badRequest(message string) *requestError {
	return &requestError{http.StatusBadRequest, "invalid_request", message}
}

// rawQuery keeps fields as raw JSON so that types are checked here, not coerced by encoding/json.
type rawQuery struct {
	Question json.RawMessage `json:"question"`
	K        json.RawMessage `json:"k"`
}

// decodeQuery parses and validates a POST /query body, returning the trimmed question and k.
func decodeQuery(w http.ResponseWriter, r *http.Request) (string, int, *requestError) {
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxRequestBytes))
	dec.DisallowUnknownFields() // matches extra="forbid" on the Python side

	var raw rawQuery
	if err := dec.Decode(&raw); err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			return "", 0, &requestError{http.StatusRequestEntityTooLarge, "request_too_large",
				fmt.Sprintf("request body must be at most %d bytes", maxRequestBytes)}
		}
		if strings.HasPrefix(err.Error(), "json: unknown field ") {
			return "", 0, badRequest(fmt.Sprintf("unknown field %s; allowed fields are question and k",
				strings.TrimPrefix(err.Error(), "json: unknown field ")))
		}
		return "", 0, badRequest("request body must be a JSON object with a question field")
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		return "", 0, badRequest("request body must contain a single JSON object")
	}

	var question string
	if len(raw.Question) == 0 || raw.Question[0] != '"' || json.Unmarshal(raw.Question, &question) != nil ||
		strings.TrimSpace(question) == "" {
		return "", 0, badRequest("question must be a non-empty string")
	}

	k := DefaultK
	if raw.K != nil {
		literal := string(bytes.TrimSpace(raw.K))
		n, err := strconv.Atoi(literal)
		if !integerLiteral.MatchString(literal) || err != nil || n < 1 || n > MaxK {
			return "", 0, badRequest(fmt.Sprintf("k must be an integer between 1 and %d, got %s", MaxK, literal))
		}
		k = n
	}
	return strings.TrimSpace(question), k, nil
}
