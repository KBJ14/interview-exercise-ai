// Package config reads the API server settings from environment variables.
package config

import (
	"fmt"
	"net/url"
	"strings"
	"time"
)

const (
	DefaultAddr   = ":8080"
	DefaultRAGURL = "http://127.0.0.1:8000"
	// One query is an embedding call plus a generation call on the Python side, each with
	// its own 30s timeout and retries, so the default leaves room for a slow but healthy answer.
	DefaultRAGTimeout = 60 * time.Second
)

// Config holds the server settings. Zero values are never used; Load fills in defaults.
type Config struct {
	Addr       string        // API_ADDR: listen address
	RAGURL     string        // RAG_SERVICE_URL: base URL of the internal Python service
	RAGTimeout time.Duration // RAG_TIMEOUT: limit for one call to the Python service
}

// Load reads settings through getenv (os.Getenv in production) and validates them.
func Load(getenv func(string) string) (Config, error) {
	cfg := Config{Addr: DefaultAddr, RAGURL: DefaultRAGURL, RAGTimeout: DefaultRAGTimeout}

	if v := getenv("API_ADDR"); v != "" {
		cfg.Addr = v
	}
	if v := getenv("RAG_SERVICE_URL"); v != "" {
		cfg.RAGURL = strings.TrimRight(v, "/")
	}
	u, err := url.Parse(cfg.RAGURL)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return Config{}, fmt.Errorf("RAG_SERVICE_URL must be an absolute http(s) URL, got %q", cfg.RAGURL)
	}
	if v := getenv("RAG_TIMEOUT"); v != "" {
		d, err := time.ParseDuration(v)
		if err != nil || d <= 0 {
			return Config{}, fmt.Errorf("RAG_TIMEOUT must be a positive duration such as 60s, got %q", v)
		}
		cfg.RAGTimeout = d
	}
	return cfg, nil
}
