"""Ingest one or more document files into PostgreSQL. Prints a summary only (no vectors, no secrets).

Usage:
    python scripts/ingest.py data/documents/billing.txt [more files...]

Relative paths resolve against the current directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Lets the script run by path from any working directory without installing the package.
sys.path.insert(0, str(REPO_ROOT))

import openai  # noqa: E402
import psycopg  # noqa: E402

from rag.chunker import ChunkingError  # noqa: E402
from rag.embedder import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, EmbeddingError  # noqa: E402
from rag.ingest import ingest_file  # noqa: E402
from rag.loader import DocumentFormatError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)

    print(f"embedding model: {EMBEDDING_MODEL} ({EMBEDDING_DIMENSIONS} dims)")
    failed = 0
    for path in args.files:
        try:
            r = ingest_file(path)
        except (FileNotFoundError, DocumentFormatError, ChunkingError, EmbeddingError, RuntimeError) as exc:
            print(f"{path}: error: {exc}", file=sys.stderr)
        except openai.APIStatusError as exc:
            # Status and type only; the SDK message can echo a masked key fragment.
            print(f"{path}: error: OpenAI API returned HTTP {exc.status_code} ({type(exc).__name__})", file=sys.stderr)
        except openai.APIConnectionError as exc:
            print(f"{path}: error: could not reach OpenAI API ({type(exc).__name__})", file=sys.stderr)
        except psycopg.Error as exc:
            print(f"{path}: error: database error ({type(exc).__name__}): {exc}", file=sys.stderr)
        else:
            print(f"{r.source}: {r.status} document_id={r.document_id} chunks={r.chunk_count} "
                  f"api_calls={r.api_calls} embedding_tokens={r.embedding_tokens}")
            continue
        failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
