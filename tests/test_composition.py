"""Tests for output composition: joining, layouts and the default output name.

The layouts are also covered end to end, through run_iter, in test_processor.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import itertool.logger as logger
from itertool.models import FileResult, Preset, ProcessingResult
from itertool.output import compose_output, default_output_path, join_by_file, write_outputs


def setUpModule():
    logger.set_backend("silent")


def tearDownModule():
    logger.set_backend("console")


class TestJoinByFile(unittest.TestCase):
    """Per-file texts are joined in order, under their headers when asked."""

    def test_with_headers(self):
        self.assertEqual(
            join_by_file([("a.md", "A"), ("b.md", "B")], headers=True),
            "# File: a.md\n\nA\n\n# File: b.md\n\nB",
        )

    def test_without_headers_a_single_text_is_unchanged(self):
        self.assertEqual(join_by_file([("", "A")], headers=False), "A")

    def test_empty_texts_are_left_out_with_their_header(self):
        self.assertEqual(join_by_file([("a.md", ""), ("b.md", "B")], headers=True), "# File: b.md\n\nB")


class TestComposeOutput(unittest.TestCase):
    """The output of one document holds the selected phases, separated by rules."""

    def test_selected_phases_in_order(self):
        preset = Preset(output_include_preprocessed=True, output_include_postprocessed=True)
        self.assertEqual(compose_output("P", "R", "Q", preset), "P\n\n---\n\nR\n\n---\n\nQ")

    def test_nothing_selected_is_empty(self):
        self.assertEqual(compose_output("P", "R", "Q", Preset(output_include_processed=False)), "")


class _OutputDir(unittest.TestCase):
    """An empty working directory for the written outputs."""

    def setUp(self):
        self._cwd = os.getcwd()
        self.dir = Path(tempfile.mkdtemp())
        os.chdir(self.dir)

    def tearDown(self):
        os.chdir(self._cwd)


class TestWriteOutputs(_OutputDir):
    """The layout is chosen by the input kind and the preset."""

    @staticmethod
    def _result(*names: str) -> ProcessingResult:
        files = [FileResult(source=n, preprocessed="", node_results=[], results=f"R:{n}", postprocessed="")
                 for n in names]
        return ProcessingResult(preprocessed="", node_results=[], results="", postprocessed="", files=files)

    def test_folder_in_the_mirror_layout_writes_one_file_per_document(self):
        write_outputs(self._result("a.md", "b.md"), "", Path("out"), Preset(), is_folder=True)
        self.assertEqual((self.dir / "out" / "b.md").read_text(encoding="utf-8"), "R:b.md")

    def test_folder_in_the_merged_layout_writes_one_file(self):
        write_outputs(self._result("a.md", "b.md"), "", Path("all.md"), Preset(output_layout="merged"), True)
        self.assertEqual(
            (self.dir / "all.md").read_text(encoding="utf-8"),
            "# File: a.md\n\nR:a.md\n\n# File: b.md\n\nR:b.md",
        )

    def test_single_document_ignores_the_mirror_layout(self):
        write_outputs(self._result(""), "", Path("out.md"), Preset(), is_folder=False)
        self.assertEqual((self.dir / "out.md").read_text(encoding="utf-8"), "R:")


class TestDefaultOutputPath(_OutputDir):
    """Without an explicit output, the name derives from the input and the layout."""

    def setUp(self):
        super().setUp()
        (self.dir / "docs").mkdir()

    def test_file(self):
        self.assertEqual(
            default_output_path(str(Path("docs") / "a.md"), Preset()),
            str(Path("docs") / "a_processed.md"),
        )

    def test_folder_mirror(self):
        self.assertEqual(default_output_path("docs", Preset()), str(self.dir.resolve() / "docs_processed"))

    def test_folder_merged(self):
        self.assertEqual(
            default_output_path("docs", Preset(output_layout="merged")),
            str(self.dir.resolve() / "docs_processed.md"),
        )


if __name__ == "__main__":
    unittest.main()
