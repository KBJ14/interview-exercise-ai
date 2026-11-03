// Package config reads the API server settings from environment variables.
package config

import (
	"bufio"
	"errors"
	"fmt"
	"io/fs"
	"net"
	"net/url"
	"os"
	"strings"
	"time"
)

const (
	DefaultAddr   = ":8080"
	DefaultRAGURL = "http://127.0.0.1:8000"
	// One query is an embedding call plus a generation call on the Python side, each with
	// its own 30s timeout and retries, so the default leaves room for a slow but healthy answer.
	DefaultRAGTimeout = 60 * time.Second
	// Metadata lookups are single indexed-table reads; anything slower means the DB is in trouble.
	DefaultDBTimeout = 5 * time.Second
)

// Config holds the server settings. Zero values are never used; Load fills in defaults.
type Config struct {
	Addr       string        // API_ADDR: listen address
	RAGURL     string        // RAG_SERVICE_URL: base URL of the internal Python service
	RAGTimeout time.Duration // RAG_TIMEOUT: limit for one call to the Python service

	// DatabaseURL contains the password. Never log it; log DatabaseTarget instead.
	DatabaseURL    string
	DatabaseTarget string        // host:port/dbname, safe to log
	DBTimeout      time.Duration // DB_QUERY_TIMEOUT: limit for one database query
}

// Load reads settings through getenv and validates them.
// In production getenv is EnvWithFile(os.Getenv, ".env"); tests pass a map lookup.
func Load(getenv func(string) string) (Config, error) {
	cfg := Config{Addr: DefaultAddr, RAGURL: DefaultRAGURL, RAGTimeout: DefaultRAGTimeout, DBTimeout: DefaultDBTimeout}

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
	if cfg.RAGTimeout, err = duration(getenv, "RAG_TIMEOUT", cfg.RAGTimeout); err != nil {
		return Config{}, err
	}
	if cfg.DBTimeout, err = duration(getenv, "DB_QUERY_TIMEOUT", cfg.DBTimeout); err != nil {
		return Config{}, err
	}
	if cfg.DatabaseURL, cfg.DatabaseTarget, err = databaseURL(getenv); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func duration(getenv func(string) string, key string, fallback time.Duration) (time.Duration, error) {
	v := getenv(key)
	if v == "" {
		return fallback, nil
	}
	d, err := time.ParseDuration(v)
	if err != nil || d <= 0 {
		return 0, fmt.Errorf("%s must be a positive duration such as 5s, got %q", key, v)
	}
	return d, nil
}

// databaseURL follows rag/db.py so both services reach the same database from the same settings:
// DATABASE_URL wins; otherwise POSTGRES_USER/PASSWORD/DB are required and
// POSTGRES_HOST (default localhost) and POSTGRES_PORT (default 5432) are optional.
func databaseURL(getenv func(string) string) (dsn, target string, err error) {
	if v := getenv("DATABASE_URL"); v != "" {
		u, err := url.Parse(v)
		if err != nil || (u.Scheme != "postgres" && u.Scheme != "postgresql") {
			// The value may hold a password, so it is not echoed back.
			return "", "", errors.New("DATABASE_URL must be a postgres:// URL")
		}
		return v, u.Host + u.Path, nil
	}

	var missing []string
	for _, key := range []string{"POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"} {
		if getenv(key) == "" {
			missing = append(missing, key)
		}
	}
	if len(missing) > 0 {
		return "", "", fmt.Errorf("database settings missing: %s (set DATABASE_URL or fill .env)", strings.Join(missing, ", "))
	}
	host, port := getenv("POSTGRES_HOST"), getenv("POSTGRES_PORT")
	if host == "" {
		host = "localhost"
	}
	if port == "" {
		port = "5432"
	}
	u := url.URL{
		Scheme: "postgres",
		User:   url.UserPassword(getenv("POSTGRES_USER"), getenv("POSTGRES_PASSWORD")), // escapes special characters
		Host:   net.JoinHostPort(host, port),
		Path:   "/" + getenv("POSTGRES_DB"),
	}
	return u.String(), u.Host + u.Path, nil
}

// EnvWithFile returns a getenv that prefers real environment variables and falls back to
// KEY=VALUE lines in path, mirroring how rag/db.py reads the repo's .env for local runs.
// A missing file is not an error; an unreadable one is.
func EnvWithFile(getenv func(string) string, path string) (func(string) string, error) {
	file := map[string]string{}
	f, err := os.Open(path)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		return getenv, nil
	case err != nil:
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if key, value, ok := strings.Cut(line, "="); ok {
			file[strings.TrimSpace(key)] = strings.TrimSpace(value)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	return func(key string) string {
		if v := getenv(key); v != "" {
			return v
		}
		return file[key]
	}, nil
}
