package config

import (
	"testing"
	"time"
)

func env(values map[string]string) func(string) string {
	return func(key string) string { return values[key] }
}

func TestDefaults(t *testing.T) {
	cfg, err := Load(env(nil))
	if err != nil {
		t.Fatal(err)
	}
	want := Config{Addr: ":8080", RAGURL: "http://127.0.0.1:8000", RAGTimeout: 60 * time.Second}
	if cfg != want {
		t.Errorf("got %+v, want %+v", cfg, want)
	}
}

func TestOverrides(t *testing.T) {
	cfg, err := Load(env(map[string]string{
		"API_ADDR": "127.0.0.1:9090", "RAG_SERVICE_URL": "http://rag:8000/", "RAG_TIMEOUT": "1500ms",
	}))
	if err != nil {
		t.Fatal(err)
	}
	want := Config{Addr: "127.0.0.1:9090", RAGURL: "http://rag:8000", RAGTimeout: 1500 * time.Millisecond}
	if cfg != want {
		t.Errorf("got %+v, want %+v", cfg, want)
	}
}

func TestInvalidValuesAreRejected(t *testing.T) {
	cases := map[string]map[string]string{
		"relative url":     {"RAG_SERVICE_URL": "rag:8000"},
		"non-http url":     {"RAG_SERVICE_URL": "ftp://rag:8000"},
		"timeout no unit":  {"RAG_TIMEOUT": "60"},
		"timeout zero":     {"RAG_TIMEOUT": "0s"},
		"timeout negative": {"RAG_TIMEOUT": "-5s"},
	}
	for name, values := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := Load(env(values)); err == nil {
				t.Error("expected an error")
			}
		})
	}
}
