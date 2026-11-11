"""Tests for loading a run's input as a corpus, parsing it, and guarding its outputs."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import itertool.logger as logger
from itertool.corpus import check_output_location, load_corpus, natural_sort_key, parse_corpus


def setUpModule():
    logger.set_backend("silent")


def tearDownModule():
    logger.set_backend("console")


class _Folder(unittest.TestCase):
    """Each test gets an empty folder to fill with input files."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.docs = self.dir / "docs"
        self.docs.mkdir()

    def _write(self, name: str, text: str = "Text.\n") -> Path:
        path = self.docs / name
        path.write_text(text, encoding="utf-8")
        return path


class TestLoadCorpus(_Folder):
    """A folder contributes its own Markdown files, in natural order."""

    def test_files_are_in_natural_order(self):
        for name in ("10.md", "2.md", "1.md"):
            self._write(name)
        corpus = load_corpus(str(self.docs))
        self.assertEqual([f.name for f in corpus.files], ["1.md", "2.md", "10.md"])
        self.assertTrue(corpus.is_folder)
        self.assertEqual(corpus.root, self.docs.resolve())

    def test_only_markdown_files_directly_inside_are_read(self):
        self._write("a.md")
        self._write("b.MARKDOWN")
        self._write("notes.txt")
        (self.docs / "sub").mkdir()
        (self.docs / "sub" / "c.md").write_text("Nested.\n", encoding="utf-8")
        corpus = load_corpus(str(self.docs))
        self.assertEqual([f.name for f in corpus.files], ["a.md", "b.MARKDOWN"])

    def test_folder_without_markdown_is_an_error(self):
        self._write("notes.txt")
        with self.assertRaises(ValueError):
            load_corpus(str(self.docs))

    def test_a_file_is_a_corpus_of_one(self):
        path = self._write("doc.md", "Alpha.\n")
        corpus = load_corpus(str(path))
        self.assertFalse(corpus.is_folder)
        self.assertEqual(corpus.root, self.docs.resolve())
        self.assertEqual([(f.name, f.text) for f in corpus.files], [("doc.md", "Alpha.\n")])

    def test_raw_text_resolves_against_the_working_directory(self):
        corpus = load_corpus(None, "Alpha.")
        self.assertEqual(corpus.root, Path.cwd())
        self.assertEqual(corpus.files[0].name, "")

    def test_missing_path_is_an_error(self):
        with self.assertRaises(ValueError):
            load_corpus(str(self.docs / "absent.md"))

    def test_natural_sort_key_ignores_case(self):
        self.assertEqual(sorted(["b.md", "A.md"], key=natural_sort_key), ["A.md", "b.md"])


class TestParseCorpus(_Folder):
    """Files are parsed one at a time; indices run across the whole corpus."""

    def _corpus(self, **files: str):
        for name, text in files.items():
            self._write(f"{name}.md", text)
        return load_corpus(str(self.docs))

    def test_index_is_global_and_source_is_tagged(self):
        corpus = self._corpus(a="One.\n\nTwo.\n", b="Three.\n")
        nodes = parse_corpus(corpus, "paragraph")
        self.assertEqual([n.index for n in nodes], [1, 2, 3])
        self.assertEqual([n.source for n in nodes], ["a.md", "a.md", "b.md"])

    def test_a_section_never_spans_two_files(self):
        corpus = self._corpus(a="One.\n\nTwo.\n", b="Three.\n")
        nodes = parse_corpus(corpus, "section")
        self.assertEqual([n.content for n in nodes], ["One.\n\nTwo.", "Three."])

    def test_file_mode_makes_one_text_node_per_file(self):
        corpus = self._corpus(a="# Title\n\nOne.\n\n| x |\n| - |\n", b="Two.\n", c="  \n")
        nodes = parse_corpus(corpus, "file")
        self.assertEqual([(n.index, n.node_type, n.source) for n in nodes], [(1, "text", "a.md"), (2, "text", "b.md")])
        self.assertEqual(nodes[0].content, "# Title\n\nOne.\n\n| x |\n| - |")


class TestCheckOutputLocation(_Folder):
    """The output of a folder never overwrites it and never joins it on the next run."""

    def setUp(self):
        super().setUp()
        self._write("a.md")
        self.corpus = load_corpus(str(self.docs))

    def test_mirror_into_the_input_folder_is_refused(self):
        with self.assertRaises(ValueError):
            check_output_location(self.corpus, str(self.docs), "mirror")

    def test_output_directly_inside_the_input_folder_is_refused(self):
        for layout, name in (("merged", "all.md"), ("mirror", "out")):
            with self.subTest(layout=layout), self.assertRaises(ValueError):
                check_output_location(self.corpus, str(self.docs / name), layout)

    def test_output_beside_the_input_folder_is_accepted(self):
        check_output_location(self.corpus, str(self.dir / "out"), "mirror")
        check_output_location(self.corpus, str(self.dir / "all.md"), "merged")

    def test_a_relative_path_is_resolved_before_comparing(self):
        cwd = os.getcwd()
        os.chdir(self.dir)
        try:
            with self.assertRaises(ValueError):
                check_output_location(self.corpus, "docs", "mirror")
        finally:
            os.chdir(cwd)

    def test_mirror_needs_a_folder_and_merged_a_file(self):
        (self.dir / "taken.md").write_text("x", encoding="utf-8")
        (self.dir / "taken").mkdir()
        with self.assertRaises(ValueError):
            check_output_location(self.corpus, str(self.dir / "taken.md"), "mirror")
        with self.assertRaises(ValueError):
            check_output_location(self.corpus, str(self.dir / "taken"), "merged")

    def test_a_single_file_input_is_not_checked(self):
        corpus = load_corpus(str(self.docs / "a.md"))
        check_output_location(corpus, str(self.docs / "a_processed.md"), "mirror")


if __name__ == "__main__":
    unittest.main()
