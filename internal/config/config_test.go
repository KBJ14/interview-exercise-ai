package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func env(values map[string]string) func(string) string {
	return func(key string) string { return values[key] }
}

// dbEnv is the minimum database setting Load requires, as in rag/db.py.
func dbEnv(extra map[string]string) map[string]string {
	m := map[string]string{"POSTGRES_USER": "rag", "POSTGRES_PASSWORD": "pw", "POSTGRES_DB": "rag"}
	for k, v := range extra {
		m[k] = v
	}
	return m
}

func TestDefaults(t *testing.T) {
	cfg, err := Load(env(dbEnv(nil)))
	if err != nil {
		t.Fatal(err)
	}
	want := Config{
		Addr: ":8080", RAGURL: "http://127.0.0.1:8000", RAGTimeout: 60 * time.Second,
		DatabaseURL: "postgres://rag:pw@localhost:5432/rag", DatabaseTarget: "localhost:5432/rag", DBTimeout: 5 * time.Second,
	}
	if cfg != want {
		t.Errorf("got %+v, want %+v", cfg, want)
	}
}

func TestOverrides(t *testing.T) {
	cfg, err := Load(env(dbEnv(map[string]string{
		"API_ADDR": "127.0.0.1:9090", "RAG_SERVICE_URL": "http://rag:8000/", "RAG_TIMEOUT": "1500ms",
		"DB_QUERY_TIMEOUT": "2s", "POSTGRES_HOST": "db", "POSTGRES_PORT": "55432",
	})))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Addr != "127.0.0.1:9090" || cfg.RAGURL != "http://rag:8000" || cfg.RAGTimeout != 1500*time.Millisecond ||
		cfg.DBTimeout != 2*time.Second || cfg.DatabaseURL != "postgres://rag:pw@db:55432/rag" || cfg.DatabaseTarget != "db:55432/rag" {
		t.Errorf("got %+v", cfg)
	}
}

func TestDatabaseURLWinsOverPostgresVariables(t *testing.T) {
	cfg, err := Load(env(dbEnv(map[string]string{"DATABASE_URL": "postgres://u:secret@other:6543/x"})))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.DatabaseURL != "postgres://u:secret@other:6543/x" || cfg.DatabaseTarget != "other:6543/x" {
		t.Errorf("got url %q target %q", cfg.DatabaseURL, cfg.DatabaseTarget)
	}
}

func TestPasswordIsEscapedAndNeverInTarget(t *testing.T) {
	cfg, err := Load(env(dbEnv(map[string]string{"POSTGRES_PASSWORD": "p@ss:w/rd?#"})))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(cfg.DatabaseURL, "p%40ss%3Aw%2Frd%3F%23@localhost") {
		t.Errorf("password not URL-escaped: %q", cfg.DatabaseURL)
	}
	if strings.Contains(cfg.DatabaseTarget, "p@ss") || strings.Contains(cfg.DatabaseTarget, "rag@") {
		t.Errorf("target exposes credentials: %q", cfg.DatabaseTarget)
	}
}

func TestInvalidValuesAreRejected(t *testing.T) {
	cases := map[string]map[string]string{
		"relative url":        dbEnv(map[string]string{"RAG_SERVICE_URL": "rag:8000"}),
		"non-http url":        dbEnv(map[string]string{"RAG_SERVICE_URL": "ftp://rag:8000"}),
		"timeout no unit":     dbEnv(map[string]string{"RAG_TIMEOUT": "60"}),
		"timeout zero":        dbEnv(map[string]string{"RAG_TIMEOUT": "0s"}),
		"timeout negative":    dbEnv(map[string]string{"RAG_TIMEOUT": "-5s"}),
		"db timeout bad":      dbEnv(map[string]string{"DB_QUERY_TIMEOUT": "soon"}),
		"database url scheme": dbEnv(map[string]string{"DATABASE_URL": "mysql://u:secret@h/db"}),
	}
	for name, values := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := Load(env(values))
			if err == nil {
				t.Fatal("expected an error")
			}
			if strings.Contains(err.Error(), "secret") {
				t.Errorf("error echoes a password: %v", err)
			}
		})
	}
}

func TestMissingDatabaseSettingsNameTheKeys(t *testing.T) {
	_, err := Load(env(map[string]string{"POSTGRES_PASSWORD": "secret"}))
	if err == nil || !strings.Contains(err.Error(), "POSTGRES_USER, POSTGRES_DB") || strings.Contains(err.Error(), "secret") {
		t.Errorf("err = %v", err)
	}
}

func TestEnvWithFileFallsBackToFileValues(t *testing.T) {
	path := filepath.Join(t.TempDir(), ".env")
	content := "# comment\nPOSTGRES_USER=fromfile\n\nPOSTGRES_DB = rag \nPOSTGRES_PASSWORD=filepw\n"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	getenv, err := EnvWithFile(env(map[string]string{"POSTGRES_PASSWORD": "realenv"}), path)
	if err != nil {
		t.Fatal(err)
	}
	if getenv("POSTGRES_USER") != "fromfile" || getenv("POSTGRES_DB") != "rag" || getenv("POSTGRES_PASSWORD") != "realenv" {
		t.Errorf("user=%q db=%q password from env=%v", getenv("POSTGRES_USER"), getenv("POSTGRES_DB"), getenv("POSTGRES_PASSWORD") == "realenv")
	}

	same, err := EnvWithFile(env(map[string]string{"A": "1"}), filepath.Join(t.TempDir(), "missing.env"))
	if err != nil || same("A") != "1" {
		t.Errorf("missing file should be ignored: err=%v", err)
	}
}
