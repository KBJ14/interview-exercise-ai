"""Search stored chunks for a question and print the nearest ones. No answer generation.

Usage:
    python scripts/search.py "Can I get a refund for a monthly renewal charge?" [--k 3]

Makes one embeddings request for the question. Read-only on the database.
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

from rag.embedder import EMBEDDING_MODEL, EmbeddingError  # noqa: E402
from rag.retriever import DEFAULT_K, search  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args(argv)

    try:
        results = search(args.question, args.k)
    except (ValueError, EmbeddingError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except openai.APIStatusError as exc:
        # Status and type only; the SDK message can echo a masked key fragment.
        print(f"error: OpenAI API returned HTTP {exc.status_code} ({type(exc).__name__})", file=sys.stderr)
        return 1
    except openai.APIConnectionError as exc:
        print(f"error: could not reach OpenAI API ({type(exc).__name__})", file=sys.stderr)
        return 1
    except psycopg.Error as exc:
        print(f"error: database error ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1

    print(f"question: {args.question}")
    print(f"model: {EMBEDDING_MODEL} | top {args.k} by cosine_distance "
          "(0 = same direction, smaller = closer; a distance, not a probability)")
    if not results:
        print("no chunks stored for this model")
    for rank, r in enumerate(results, start=1):
        header, _, body = r.content.partition("\n")
        section = header.rsplit(" > ", 1)[-1]
        print(f"\n#{rank} cosine_distance={r.cosine_distance:.4f}  document_id={r.document_id} "
              f"chunk_index={r.chunk_index}  section: {section}")
        print(f"    {body}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
