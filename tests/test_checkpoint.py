"""Tests for resumable runs: the progress file and how run_iter uses it.

Every AI call is stubbed: no test in this suite may reach call_ai.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import itertool.logger as logger
import itertool.processor as processor
from itertool.checkpoint import (
    PROGRESS_SUFFIX,
    Checkpoint,
    CheckpointMismatchError,
    fingerprint,
    progress_path,
)
from itertool.config import AppConfig
from itertool.corpus import load_corpus
from itertool.models import Preset

DOC = "Alpha paragraph.\n\nBeta paragraph.\n\nGamma paragraph.\n"


def setUpModule():
    logger.set_backend("silent")


def tearDownModule():
    logger.set_backend("console")


class _StubAi:
    """Replaces _call_ai; raises on the call numbered fail_on (1-based), if any."""

    def __init__(self, fail_on: int | None = None):
        self.fail_on = fail_on
        self.prompts: list[str] = []
        self._original = None

    def __enter__(self):
        self._original = processor._call_ai

        def fake(prompt, config, preset, messages=None, file_path=None):
            if self.fail_on is not None and len(self.prompts) + 1 == self.fail_on:
                raise ConnectionError("provider went away")
            self.prompts.append(prompt)
            return processor._AiReply(text=f"[AI {len(self.prompts)}]", elapsed=1.0)

        processor._call_ai = fake
        return self

    def __exit__(self, *exc):
        processor._call_ai = self._original


class _Workspace(unittest.TestCase):
    """Runs each test in an empty working directory, with provider setup stubbed."""

    def setUp(self):
        self._cwd = os.getcwd()
        self.dir = Path(tempfile.mkdtemp())
        os.chdir(self.dir)
        self._orig_apply = AppConfig.apply_to_client
        AppConfig.apply_to_client = lambda self: None
        self.input = self.dir / "doc.md"
        self.input.write_text(DOC, encoding="utf-8")
        self.output = self.dir / "out.md"
        self.progress = self.output.with_name(self.output.name + PROGRESS_SUFFIX)

    def tearDown(self):
        AppConfig.apply_to_client = self._orig_apply
        os.chdir(self._cwd)

    def _run(self, preset: Preset | None = None, ai: _StubAi | None = None, **kwargs):
        preset = preset or Preset(node_prompt="Condense.")
        with ai or _StubAi() as stub:
            run_iter_kwargs = dict(input_path=str(self.input), output_path=str(self.output),
                                   preset=preset, config=AppConfig())
            run_iter_kwargs.update(kwargs)
            processor.run_iter(**run_iter_kwargs)
        return stub

    def _interrupted_run(self, fail_on: int, preset: Preset | None = None):
        with self.assertRaises(ConnectionError):
            self._run(preset, _StubAi(fail_on=fail_on))


class TestProgressPath(unittest.TestCase):
    """The file sits next to the output, or in the working directory without one."""

    def test_next_to_the_output_file(self):
        path = progress_path("results/out.md", "doc.md", None)
        self.assertEqual(path, Path("results/out.md" + PROGRESS_SUFFIX))

    def test_in_the_working_directory_without_an_output(self):
        path = progress_path(None, "some/dir/doc.md", None)
        self.assertEqual(path.parent, Path.cwd())
        self.assertTrue(path.name.startswith("doc-"))

    def test_same_name_in_two_folders_gives_two_files(self):
        self.assertNotEqual(progress_path(None, "a/doc.md", None), progress_path(None, "b/doc.md", None))

    def test_text_input(self):
        path = progress_path(None, None, "some text")
        self.assertEqual(path.parent, Path.cwd())
        self.assertTrue(path.name.startswith("text-"))


class TestResume(_Workspace):
    """An interrupted run resumes where it stopped."""

    def test_file_is_deleted_after_a_complete_run(self):
        self._run()
        self.assertFalse(self.progress.exists())
        self.assertTrue(self.output.exists())

    def test_file_is_kept_after_a_failure(self):
        self._interrupted_run(fail_on=2)
        self.assertTrue(self.progress.exists())

    def test_resumed_run_calls_the_ai_only_for_what_is_missing(self):
        self._interrupted_run(fail_on=2)
        stub = self._run()
        self.assertEqual(len(stub.prompts), 2)
        self.assertNotIn("Alpha", " ".join(stub.prompts))
        output = self.output.read_text(encoding="utf-8")
        self.assertEqual(output.count("[AI"), 3)

    def test_saved_preprocessing_is_reused(self):
        preset = Preset(preprocessing_prompt="Outline.", node_prompt="Condense.")
        self._interrupted_run(fail_on=3, preset=preset)
        stub = self._run(preset)
        self.assertFalse(any(p.startswith("Outline.") for p in stub.prompts))

    def test_truncated_last_line_is_ignored_and_the_run_completes(self):
        self._interrupted_run(fail_on=2)
        with open(self.progress, "a", encoding="utf-8") as f:
            f.write('{"type": "node", "ke')
        stub = self._run()
        self.assertEqual(len(stub.prompts), 2)
        self.assertFalse(self.progress.exists())

    def test_records_after_a_truncated_line_are_not_lost(self):
        self._interrupted_run(fail_on=2)
        with open(self.progress, "a", encoding="utf-8") as f:
            f.write('{"type": "node", "ke')
        self._interrupted_run(fail_on=2)
        stub = self._run()
        self.assertEqual(len(stub.prompts), 1)


class TestMismatch(_Workspace):
    """A progress file from a different run is never reused silently."""

    def test_changed_prompt_asks(self):
        self._interrupted_run(fail_on=2)
        with self.assertRaises(CheckpointMismatchError) as ctx:
            self._run(Preset(node_prompt="Translate."))
        self.assertEqual(ctx.exception.differences, ["preset"])
        self.assertEqual(ctx.exception.path, self.progress)

    def test_changed_input_asks(self):
        self._interrupted_run(fail_on=2)
        self.input.write_text(DOC.replace("Beta", "Delta"), encoding="utf-8")
        with self.assertRaises(CheckpointMismatchError) as ctx:
            self._run()
        self.assertEqual(ctx.exception.differences, ["input document"])

    def test_output_only_fields_do_not_count_as_a_change(self):
        self._interrupted_run(fail_on=2)
        stub = self._run(Preset(node_prompt="Condense.", interlace_input_output=True, name="renamed"))
        self.assertEqual(len(stub.prompts), 2)

    def test_continue_after_an_input_edit_redoes_only_the_changed_nodes(self):
        self._interrupted_run(fail_on=3)
        self.input.write_text(DOC.replace("Beta", "Delta"), encoding="utf-8")
        stub = self._run(on_mismatch="continue")
        self.assertEqual(len(stub.prompts), 2)
        self.assertIn("Delta", stub.prompts[0])
        self.assertIn("Gamma", stub.prompts[1])

    def test_restart_redoes_everything(self):
        self._interrupted_run(fail_on=3)
        stub = self._run(Preset(node_prompt="Translate."), on_mismatch="restart")
        self.assertEqual(len(stub.prompts), 3)

    def test_continue_rewrites_the_header_so_the_next_run_does_not_ask(self):
        self._interrupted_run(fail_on=2)
        changed = Preset(node_prompt="Translate.")
        self._interrupted_run_with(changed, on_mismatch="continue", fail_on=1)
        stub = self._run(changed)
        self.assertEqual(len(stub.prompts), 2)

    def _interrupted_run_with(self, preset, on_mismatch, fail_on):
        with self.assertRaises(ConnectionError):
            self._run(preset, _StubAi(fail_on=fail_on), on_mismatch=on_mismatch)

    def test_unknown_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            self._run(on_mismatch="maybe")


class TestPostprocessingReuse(_Workspace):
    """A saved Phase 3 result is reused only while the node results are unchanged."""

    PRESET = Preset(node_prompt="Condense.", postprocessing_prompt="Review.",
                    output_include_postprocessed=True)

    def _save_postprocessed_without_finishing(self):
        """Leave a progress file holding every phase, as if the output write had failed."""
        path = progress_path(str(self.output), str(self.input), None)
        corpus = load_corpus(str(self.input))
        cp = Checkpoint(path, fingerprint(corpus, self.PRESET, AppConfig()))
        with _StubAi():
            processor._run_node_processing(corpus, self.PRESET, AppConfig(), {}, cp)
        cp.record_postprocessed("doc.md", "[SAVED REVIEW]", 1.0)

    def test_saved_phase_three_is_reused_when_no_node_was_redone(self):
        self._save_postprocessed_without_finishing()
        stub = self._run(self.PRESET)
        self.assertEqual(stub.prompts, [])
        self.assertIn("[SAVED REVIEW]", self.output.read_text(encoding="utf-8"))

    def test_saved_phase_three_is_redone_when_a_node_was_redone(self):
        self._save_postprocessed_without_finishing()
        self.input.write_text(DOC.replace("Beta", "Delta"), encoding="utf-8")
        stub = self._run(self.PRESET, on_mismatch="continue")
        self.assertEqual(len(stub.prompts), 2)
        self.assertTrue(stub.prompts[-1].startswith("Review."))


class TestImageNodeKeys(_Workspace):
    """An attached image is part of its node's key, so a changed file is resent."""

    IMAGE_DOC = "Alpha paragraph.\n\n![chart](chart.png)\n"
    # The run stops at Phase 3, after both nodes are saved.
    PRESET = Preset(node_prompt="Describe.", image_handling="standalone_nodes",
                    postprocessing_prompt="Review.")

    def setUp(self):
        super().setUp()
        self.input.write_text(self.IMAGE_DOC, encoding="utf-8")
        self.image = self.dir / "chart.png"
        self.image.write_bytes(b"first")

    def test_changed_image_redoes_only_its_node(self):
        self._interrupted_run(fail_on=3, preset=self.PRESET)
        self.image.write_bytes(b"second")
        stub = self._run(self.PRESET)
        self.assertEqual(len(stub.prompts), 2)
        self.assertIn("![chart](chart.png)", stub.prompts[0])
        self.assertTrue(stub.prompts[1].startswith("Review."))

    def test_unchanged_image_is_restored(self):
        self._interrupted_run(fail_on=3, preset=self.PRESET)
        stub = self._run(self.PRESET)
        self.assertEqual(len(stub.prompts), 1)
        self.assertTrue(stub.prompts[0].startswith("Review."))


class TestFormatChange(_Workspace):
    """A progress file of another format cannot be read back, so the run starts over."""

    def test_old_format_is_discarded_without_asking(self):
        self._interrupted_run(fail_on=2)
        lines = self.progress.read_text(encoding="utf-8").splitlines()
        header = json.loads(lines[0])
        header["fingerprint"]["version"] = 1
        self.progress.write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n", encoding="utf-8")
        stub = self._run()
        self.assertEqual(len(stub.prompts), 3)


class TestContextChange(_Workspace):
    """The reference context is part of what a saved run was computed with."""

    def test_changed_context_asks(self):
        context = self.dir / "glossary.md"
        context.write_text("First.\n", encoding="utf-8")
        with self.assertRaises(ConnectionError):
            self._run(ai=_StubAi(fail_on=2), context_file=str(context))
        context.write_text("Second.\n", encoding="utf-8")
        with self.assertRaises(CheckpointMismatchError) as ctx:
            self._run(context_file=str(context))
        self.assertEqual(ctx.exception.differences, ["reference context"])


class TestFolderResume(_Workspace):
    """A folder run resumes per file: an edited file redoes only its own work."""

    PRESET = Preset(preprocessing_prompt="Outline.", node_prompt="Condense.",
                    postprocessing_prompt="Review.")

    def setUp(self):
        super().setUp()
        self.docs = self.dir / "docs"
        self.docs.mkdir()
        (self.docs / "a.md").write_text("Alpha.\n", encoding="utf-8")
        (self.docs / "b.md").write_text("Beta.\n", encoding="utf-8")
        self.output = self.dir / "out"
        self.progress = self.output.with_name(self.output.name + PROGRESS_SUFFIX)

    def _run_folder(self, ai: _StubAi | None = None, **kwargs):
        return self._run(self.PRESET, ai, input_path=str(self.docs), **kwargs)

    def _interrupted_at_the_last_call(self):
        """Stop on the last call: Phase 3 of b.md, after Phase 3 of a.md was saved."""
        with self.assertRaises(ConnectionError):
            # Order: pre a, pre b, node a, node b, post a, post b.
            self._run_folder(_StubAi(fail_on=6))

    def test_resumed_run_calls_the_ai_only_for_what_is_missing(self):
        with self.assertRaises(ConnectionError):
            self._run_folder(_StubAi(fail_on=4))
        stub = self._run_folder()
        self.assertEqual(len(stub.prompts), 3)
        self.assertTrue(stub.prompts[0].startswith("Condense.\n\nBeta."))
        self.assertFalse(self.progress.exists())
        self.assertTrue((self.output / "a.md").exists())

    def test_edited_file_redoes_its_own_phases_only(self):
        self._interrupted_at_the_last_call()
        (self.docs / "b.md").write_text("Gamma.\n", encoding="utf-8")
        stub = self._run_folder(on_mismatch="continue")
        # Phase 1, the node and Phase 3 of b.md; a.md is restored, its Phase 3 included.
        self.assertEqual(len(stub.prompts), 3)
        self.assertEqual(stub.prompts[0], "Outline.\n\nGamma.\n")
        self.assertEqual(stub.prompts[1], "Condense.\n\nGamma.")
        self.assertTrue(stub.prompts[2].startswith("Review."))

    def test_corpus_phase_three_is_redone_when_any_node_was(self):
        preset = Preset(node_prompt="Condense.", postprocessing_prompt="Review.",
                        postprocessing_scope="corpus")
        with self.assertRaises(ConnectionError):
            # 2 nodes + 1 corpus post: the run stops at the post.
            self._run(preset, _StubAi(fail_on=3), input_path=str(self.docs))
        (self.docs / "b.md").write_text("Gamma.\n", encoding="utf-8")
        stub = self._run(preset, input_path=str(self.docs), on_mismatch="continue")
        self.assertEqual(len(stub.prompts), 2)
        self.assertTrue(stub.prompts[-1].startswith("Review."))


if __name__ == "__main__":
    unittest.main()
