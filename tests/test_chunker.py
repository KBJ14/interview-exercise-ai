import unittest
from pathlib import Path

from rag.chunker import ChunkingError, Chunk, chunk_document, count_tokens
from rag.loader import NOTICE_PREFIX, Document, Section, load_documents

REPO_ROOT = Path(__file__).resolve().parents[1]
MAX, OVERLAP = 300, 50


def make_doc(body: str, title: str = "Test Doc", heading: str = "Long Section") -> Document:
    return Document(source="test.txt", title=title, raw_text=body, sections=(Section(heading, body),))


def body_words(chunk: Chunk) -> list[str]:
    return chunk.content.split("\n", 1)[1].split()


def rebuild(chunks: list[Chunk]) -> tuple[list[str], list[int]]:
    """Merge chunk bodies back into one word list, removing the overlap between neighbours.

    Returns the merged words and the number of overlapping words found at each boundary.
    Assumes the source text has no repeated word runs at chunk boundaries (true for test data).
    """
    merged = body_words(chunks[0])
    overlaps = []
    for chunk in chunks[1:]:
        words = body_words(chunk)
        k = next(k for k in range(min(len(merged), len(words)), -1, -1) if merged[len(merged) - k:] == words[:k])
        overlaps.append(k)
        merged += words[k:]
    return merged, overlaps


class LongSectionTest(unittest.TestCase):
    def setUp(self):
        self.body = " ".join(f"Rule {i} says invoice item {i} is billed on day {i} of the cycle." for i in range(60))
        self.chunks = chunk_document(make_doc(self.body), MAX, OVERLAP)

    def test_long_section_is_split_into_several_chunks(self):
        self.assertGreater(len(self.chunks), 1)
        self.assertEqual([c.chunk_index for c in self.chunks], list(range(len(self.chunks))))

    def test_every_chunk_including_header_and_overlap_is_within_max_tokens(self):
        for c in self.chunks:
            self.assertLessEqual(c.token_count, MAX)
            self.assertEqual(c.token_count, count_tokens(c.content))
            self.assertTrue(c.content.startswith("Test Doc > Long Section\n"))

    def test_neighbouring_chunks_share_whole_sentences_up_to_the_overlap_target(self):
        _, overlaps = rebuild(self.chunks)
        for prev, nxt, k in zip(self.chunks, self.chunks[1:], overlaps):
            self.assertGreater(k, 0)
            shared = " ".join(body_words(nxt)[:k])
            self.assertLessEqual(count_tokens(shared), OVERLAP)
            self.assertTrue(shared.endswith("."), "overlap should end on a sentence boundary")

    def test_no_text_is_lost_or_reordered(self):
        merged, _ = rebuild(self.chunks)
        self.assertEqual(merged, self.body.split())


class OversizedInputTest(unittest.TestCase):
    def test_sentence_longer_than_max_is_split_at_words_without_loss(self):
        body = " ".join(f"term{i}" for i in range(400)) + "."
        chunks = chunk_document(make_doc(body), MAX, OVERLAP)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c.token_count <= MAX for c in chunks))
        merged, _ = rebuild(chunks)
        self.assertEqual(merged, body.split())

    def test_single_word_longer_than_max_raises_instead_of_truncating(self):
        with self.assertRaises(ChunkingError):
            chunk_document(make_doc("x" * 5000), MAX, OVERLAP)

    def test_heading_longer_than_max_raises(self):
        with self.assertRaises(ChunkingError):
            chunk_document(make_doc("Short body.", title=" ".join(["Title"] * 400)), MAX, OVERLAP)

    def test_overlap_must_be_smaller_than_max(self):
        with self.assertRaises(ValueError):
            chunk_document(make_doc("Short body."), max_tokens=50, overlap_tokens=50)


class PracticeCorpusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = load_documents(REPO_ROOT / "data" / "documents")

    def test_each_section_fits_in_one_chunk_within_the_limit(self):
        for doc in self.docs:
            chunks = chunk_document(doc, MAX, OVERLAP)
            self.assertEqual(len(chunks), len(doc.sections), doc.source)
            self.assertTrue(all(c.token_count <= MAX for c in chunks), doc.source)

    def test_chunk_bodies_match_section_text_exactly(self):
        for doc in self.docs:
            for section, chunk in zip(doc.sections, chunk_document(doc, MAX, OVERLAP)):
                self.assertEqual(chunk.content, f"{doc.title} > {section.heading}\n{section.body}")

    def test_every_source_word_appears_in_chunks(self):
        # Guards the loader too: compares chunk text against the raw file, minus notice/title/headings.
        for doc in self.docs:
            chunks = chunk_document(doc, MAX, OVERLAP)
            expected = " ".join(s.body for s in doc.sections).split()
            actual = [w for c in chunks for w in body_words(c)]
            self.assertEqual(actual, expected, doc.source)
            raw_words = doc.raw_text.split()
            self.assertTrue(set(expected) <= set(raw_words))

    def test_fictional_notice_is_not_embedded(self):
        for doc in self.docs:
            for c in chunk_document(doc, MAX, OVERLAP):
                self.assertNotIn(NOTICE_PREFIX, c.content)


if __name__ == "__main__":
    unittest.main()
