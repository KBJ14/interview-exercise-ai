// Command api serves the public REST API. Queries are forwarded to the internal Python RAG
// service; document metadata is read directly from the shared PostgreSQL database.
//
// Environment (defaults in internal/config). Real environment variables win over ./.env:
//
//	API_ADDR          listen address                 (default ":8080")
//	RAG_SERVICE_URL   Python service base URL        (default "http://127.0.0.1:8000")
//	RAG_TIMEOUT       limit per call to Python       (default "60s")
//	DB_QUERY_TIMEOUT  limit per database query       (default "5s")
//	DATABASE_URL      postgres:// URL; if unset, built from POSTGRES_USER, POSTGRES_PASSWORD,
//	                  POSTGRES_DB, POSTGRES_HOST (default localhost), POSTGRES_PORT (default 5432)
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/KBJ14/interview-exercise-ai/internal/config"
	"github.com/KBJ14/interview-exercise-ai/internal/httpapi"
	"github.com/KBJ14/interview-exercise-ai/internal/ragclient"
	"github.com/KBJ14/interview-exercise-ai/internal/store"
)

func main() {
	log := slog.New(slog.NewTextHandler(os.Stderr, nil))
	if err := run(log); err != nil {
		log.Error("server stopped", "err", err)
		os.Exit(1)
	}
}

func run(log *slog.Logger) error {
	getenv, err := config.EnvWithFile(os.Getenv, ".env")
	if err != nil {
		return err
	}
	cfg, err := config.Load(getenv)
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	pool, err := newPool(ctx, cfg)
	if err != nil {
		return err
	}
	defer pool.Close() // runs after the HTTP server has drained, so no handler uses a closed pool

	// Not fatal: /query works without the database, and /documents reports 503 until it is back.
	pingCtx, cancel := context.WithTimeout(ctx, cfg.DBTimeout)
	if err := pool.Ping(pingCtx); err != nil {
		log.Warn("database not reachable at startup", "target", cfg.DatabaseTarget, "err", err)
	}
	cancel()

	srv := &http.Server{
		Addr:              cfg.Addr,
		Handler:           httpapi.NewServer(ragclient.New(cfg.RAGURL, cfg.RAGTimeout), store.New(pool, cfg.DBTimeout), log),
		ReadHeaderTimeout: 10 * time.Second,
		// Must outlast the RAG call, or slow but successful answers would be cut off.
		WriteTimeout: cfg.RAGTimeout + 10*time.Second,
	}

	serveErr := make(chan error, 1)
	go func() { serveErr <- srv.ListenAndServe() }()
	log.Info("listening", "addr", cfg.Addr, "rag_service_url", cfg.RAGURL, "rag_timeout", cfg.RAGTimeout,
		"database", cfg.DatabaseTarget, "db_query_timeout", cfg.DBTimeout)

	select {
	case err := <-serveErr:
		return err // failed to start (e.g. port in use)
	case <-ctx.Done():
	}

	log.Info("shutting down")
	shutdownCtx, cancelShutdown := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancelShutdown()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		return err
	}
	if err := <-serveErr; !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

func newPool(ctx context.Context, cfg config.Config) (*pgxpool.Pool, error) {
	poolCfg, err := pgxpool.ParseConfig(cfg.DatabaseURL)
	if err != nil {
		// pgx's parse errors can quote the connection string, so report only that it was invalid.
		return nil, errors.New("invalid database connection settings")
	}
	poolCfg.ConnConfig.ConnectTimeout = cfg.DBTimeout
	// Connections are opened lazily, so a database that is down does not prevent startup.
	return pgxpool.NewWithConfig(ctx, poolCfg)
}
