// Command api serves the public REST API and forwards queries to the internal Python RAG service.
//
// Environment (defaults in internal/config):
//
//	API_ADDR         listen address              (default ":8080")
//	RAG_SERVICE_URL  Python service base URL     (default "http://127.0.0.1:8000")
//	RAG_TIMEOUT      limit per call to Python    (default "60s")
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

	"github.com/KBJ14/interview-exercise-ai/internal/config"
	"github.com/KBJ14/interview-exercise-ai/internal/httpapi"
	"github.com/KBJ14/interview-exercise-ai/internal/ragclient"
)

func main() {
	log := slog.New(slog.NewTextHandler(os.Stderr, nil))

	cfg, err := config.Load(os.Getenv)
	if err != nil {
		log.Error("invalid configuration", "err", err)
		os.Exit(1)
	}

	srv := &http.Server{
		Addr:              cfg.Addr,
		Handler:           httpapi.NewServer(ragclient.New(cfg.RAGURL, cfg.RAGTimeout), log),
		ReadHeaderTimeout: 10 * time.Second,
		// Must outlast the RAG call, or slow but successful answers would be cut off.
		WriteTimeout: cfg.RAGTimeout + 10*time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := srv.Shutdown(shutdownCtx); err != nil {
			log.Error("shutdown", "err", err)
		}
	}()

	log.Info("listening", "addr", cfg.Addr, "rag_service_url", cfg.RAGURL, "rag_timeout", cfg.RAGTimeout)
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Error("server stopped", "err", err)
		os.Exit(1)
	}
}
