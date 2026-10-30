import os
import tempfile
import unittest
from pathlib import Path

from rag.loader import DocumentFormatError, load_documents, parse_document

REPO_ROOT = Path(__file__).resolve().parents[1]

SAMPLE = """[FICTIONAL DOCUMENT - test]

Sample Title

First Heading
First body line one.
First body line two.

Second Heading
Second body.
"""


class LoadDocumentsTest(unittest.TestCase):
    def test_practice_corpus_has_three_documents_with_five_sections(self):
        docs = load_documents(REPO_ROOT / "data" / "documents")
        self.assertEqual([d.source for d in docs], ["account.txt", "billing.txt", "plans.txt"])
        self.assertTrue(all(len(d.sections) == 5 for d in docs))

    def test_only_top_level_txt_files_are_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text(SAMPLE, encoding="utf-8")
            (root / "rag-checks.md").write_text(SAMPLE, encoding="utf-8")
            (root / "sub").mkdir()
            (root / "sub" / "b.txt").write_text(SAMPLE, encoding="utf-8")
            self.assertEqual([d.source for d in load_documents(root)], ["a.txt"])

    def test_relative_path_resolves_against_current_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "docs_in").mkdir()
            (Path(tmp) / "docs_in" / "a.txt").write_text(SAMPLE, encoding="utf-8")
            previous = os.getcwd()
            os.chdir(tmp)
            try:
                self.assertEqual([d.source for d in load_documents("docs_in")], ["a.txt"])
            finally:
                os.chdir(previous)

    def test_missing_directory_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_documents(REPO_ROOT / "no-such-dir")


class ParseDocumentTest(unittest.TestCase):
    def test_notice_is_dropped_and_body_lines_are_joined(self):
        doc = parse_document("a.txt", SAMPLE)
        self.assertEqual(doc.title, "Sample Title")
        self.assertEqual([s.heading for s in doc.sections], ["First Heading", "Second Heading"])
        self.assertEqual(doc.sections[0].body, "First body line one. First body line two.")
        self.assertEqual(doc.raw_text, SAMPLE)

    def test_section_without_body_raises(self):
        with self.assertRaises(DocumentFormatError):
            parse_document("a.txt", "Title\n\nHeading only\n")

    def test_title_only_raises(self):
        with self.assertRaises(DocumentFormatError):
            parse_document("a.txt", "Title\n")


if __name__ == "__main__":
    unittest.main()
