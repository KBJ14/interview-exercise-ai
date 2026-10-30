"""Answer a question from the stored documents (retrieve, then generate with citations).

Usage:
    python scripts/ask.py "Can I get a refund for a monthly renewal charge?" [--k 3]

Model settings: RAG_GENERATION_MODEL (default gpt-6-luna), RAG_REASONING_EFFORT (default low).
Makes one embeddings request and at most one generation request.
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

from rag.embedder import EmbeddingError  # noqa: E402
from rag.generator import GenerationError, answer, generation_model, reasoning_effort  # noqa: E402
from rag.retriever import DEFAULT_K  # noqa: E402


def _api_error(exc: openai.APIStatusError) -> str:
    detail = ""
    # 401 bodies can echo a masked key fragment; other errors (model access, bad parameter) are safe and useful.
    if exc.status_code != 401 and isinstance(exc.body, dict):
        detail = f" code={exc.body.get('code')} param={exc.body.get('param')} message={exc.body.get('message')}"
    return f"OpenAI API returned HTTP {exc.status_code} ({type(exc).__name__}){detail}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args(argv)

    try:
        result = answer(args.question, args.k)
    except (ValueError, EmbeddingError, GenerationError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except openai.APIStatusError as exc:
        print(f"error: {_api_error(exc)}", file=sys.stderr)
        return 1
    except openai.APIConnectionError as exc:
        print(f"error: could not reach OpenAI API ({type(exc).__name__})", file=sys.stderr)
        return 1
    except psycopg.Error as exc:
        print(f"error: database error ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1

    print(f"question:   {result.question}")
    print(f"model:      {result.model or '(not called)'} | reasoning effort: {reasoning_effort()} "
          f"| configured: {generation_model()}")
    print(f"answerable: {result.answerable}")
    print(f"answer:\n  {result.answer}")
    print("sources (cosine_distance: smaller = closer; not a probability):")
    for s in result.sources:
        mark = "cited" if s.label in result.citations else "not cited"
        print(f"  [{s.label}] {mark:<9} distance={s.cosine_distance:.4f} document_id={s.document_id} "
              f"chunk_index={s.chunk_index} {s.title} > {s.section}")
    if result.usage:
        u = result.usage
        print(f"usage:      input={u.input_tokens} output={u.output_tokens} "
              f"(reasoning={u.reasoning_tokens}) total={u.total_tokens}")
    print("note: format and citation labels were validated; factual accuracy was not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
