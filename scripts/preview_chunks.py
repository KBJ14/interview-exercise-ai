"""Print the chunks produced from a document directory. No DB writes, no API calls.

Usage:
    python scripts/preview_chunks.py [DIRECTORY] [--show SECTION]

DIRECTORY defaults to <repo>/data/documents; a relative path resolves against the current directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Lets the script run by path from any working directory without installing the package.
sys.path.insert(0, str(REPO_ROOT))

from rag.chunker import DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP_TOKENS, ChunkingError, chunk_document  # noqa: E402
from rag.loader import DocumentFormatError, load_documents  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", nargs="?", type=Path, default=REPO_ROOT / "data" / "documents")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=DEFAULT_OVERLAP_TOKENS)
    parser.add_argument("--show", metavar="SECTION", help="print the full content of chunks in this section")
    args = parser.parse_args(argv)

    try:
        docs = load_documents(args.directory)
        results = [(doc, chunk_document(doc, args.max_tokens, args.overlap_tokens)) for doc in docs]
    except (FileNotFoundError, DocumentFormatError, ChunkingError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"input: {args.directory.expanduser().resolve()}")
    print(f"max_tokens={args.max_tokens} overlap_tokens={args.overlap_tokens}")
    for doc, chunks in results:
        print(f"\n{doc.source}: {len(chunks)} chunks")
        for c in chunks:
            print(f"  [{c.chunk_index}] {c.token_count:>3} tokens  {c.section}")

    if args.show:
        for doc, chunks in results:
            for c in chunks:
                if c.section == args.show:
                    print(f"\n--- {c.source_file} [{c.chunk_index}] {c.token_count} tokens ---\n{c.content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
