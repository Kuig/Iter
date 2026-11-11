"""Tests for the node-processing pipeline.

Every AI call is stubbed: no test in this suite may reach call_ai.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import itertool.images as images
import itertool.logger as logger
import itertool.processor as processor
from itertool.config import AppConfig
from itertool.models import Corpus, Node, Preset, SourceFile

FENCE = "```"
EM_DASH = "\u2014"  # what the separator used before 0.1.2, and must never contain again


def _corpus(text: str, root: Path | None = None) -> Corpus:
    """A one-document corpus, as run_iter builds it for raw text."""
    return Corpus(root=root or Path.cwd(), files=[SourceFile(name="", text=text)])


def setUpModule():
    """Run the pipeline quietly: this is what the silent backend is for."""
    logger.set_backend("silent")


def tearDownModule():
    logger.set_backend("console")


class _StubAi:
    """Replaces _call_ai for the duration of a test and records its prompts and files."""

    def __init__(self, elapsed: float = 0.0):
        self.prompts: list[str] = []
        self.files: list[Path | None] = []
        self.elapsed = elapsed
        self._original = None

    def __enter__(self):
        self._original = processor._call_ai

        def fake(prompt, config, preset, messages=None, file_path=None):
            self.prompts.append(prompt)
            self.files.append(file_path)
            return processor._AiReply(text="[AI]", elapsed=self.elapsed)

        processor._call_ai = fake
        return self

    def __exit__(self, *exc):
        processor._call_ai = self._original


class _CaptureWarnings:
    """Collects logger.log_warning calls made through the processor module."""

    def __init__(self):
        self.messages: list[str] = []
        self._original = None

    def __enter__(self):
        self._original = processor.logger.log_warning
        processor.logger.log_warning = self.messages.append
        return self

    def __exit__(self, *exc):
        processor.logger.log_warning = self._original


class _StubEmbedding:
    """Replaces _get_embedding; fails on every call when failing is True."""

    def __init__(self, failing: bool):
        self.failing = failing
        self.calls = 0
        self.files: list[Path | None] = []
        self._original = None

    def __enter__(self):
        self._original = processor._get_embedding

        def fake(text, config, file_path=None):
            self.calls += 1
            self.files.append(file_path)
            if self.failing:
                raise ConnectionError("embedding server down")
            return [1.0, 0.0]

        processor._get_embedding = fake
        return self

    def __exit__(self, *exc):
        processor._get_embedding = self._original


class _CaptureLog:
    """Collects the messages of one logger function used by the processor."""

    def __init__(self, name: str):
        self.name = name
        self.messages: list[str] = []
        self._original = None

    def __enter__(self):
        self._original = getattr(processor.logger, self.name)
        setattr(processor.logger, self.name, self.messages.append)
        return self

    def __exit__(self, *exc):
        setattr(processor.logger, self.name, self._original)


class TestFormatDuration(unittest.TestCase):
    """Durations read as seconds below a minute, minutes and seconds above."""

    def test_below_a_minute(self):
        self.assertEqual(processor._format_duration(12.34), "12.3 s")

    def test_above_a_minute(self):
        self.assertEqual(processor._format_duration(245), "4 min 05 s")


class TestCallTiming(unittest.TestCase):
    """Every AI call reports how long it took."""

    DOC = "First paragraph.\n\nSecond paragraph.\n"

    def test_node_log_line_carries_the_duration(self):
        preset = Preset(node_prompt="p")
        with _StubAi(elapsed=14.2), _CaptureLog("log_ai") as ai_log:
            results, _ = processor._run_node_processing(_corpus(self.DOC), preset, AppConfig(), {})
        self.assertEqual(len(ai_log.messages), 2)
        self.assertIn("14.2 s", ai_log.messages[0])
        self.assertEqual(results[0].elapsed, 14.2)

    def test_phase_two_summary_carries_total_and_average(self):
        preset = Preset(node_prompt="p")
        with _StubAi(elapsed=30.0), _CaptureLog("log_success") as success:
            processor._run_node_processing(_corpus(self.DOC), preset, AppConfig(), {})
        self.assertIn("AI time: 1 min 00 s (avg 30.0 s per node)", success.messages[-1])

    def test_tokens_are_shown_only_when_reported(self):
        without = processor._reply_stats(10, processor._AiReply("abc", 1.0))
        with_tokens = processor._reply_stats(10, processor._AiReply("abc", 1.0, 7, 3))
        self.assertNotIn("tokens", without)
        self.assertIn("tokens in/out: 7/3", with_tokens)


class TestSimilarity(unittest.TestCase):
    """A failing embedding server is reported once, not once per node."""

    DOC = "One.\n\nTwo.\n\nThree.\n"

    def _run(self, failing: bool):
        preset = Preset(node_prompt="p", compute_similarity=True)
        with _StubAi(), _StubEmbedding(failing) as embedding, _CaptureWarnings() as warnings:
            results, _ = processor._run_node_processing(_corpus(self.DOC), preset, AppConfig(), {})
        return results, embedding.calls, warnings.messages

    def test_first_failure_disables_similarity_for_the_run(self):
        results, calls, warnings = self._run(failing=True)
        self.assertEqual(calls, 1)
        self.assertEqual(len(warnings), 1)
        self.assertIn("disabled", warnings[0])
        self.assertTrue(all(r.similarity is None for r in results))

    def test_working_embeddings_give_a_score_per_node(self):
        results, calls, warnings = self._run(failing=False)
        self.assertEqual(calls, 6)
        self.assertEqual(warnings, [])
        self.assertTrue(all(r.similarity == 1.0 for r in results))


class TestPostprocessingPrompt(unittest.TestCase):
    """The composed Phase 3 prompt is documented, so its shape is pinned."""

    def test_attachments_follow_the_prompt_in_order(self):
        preset = Preset(
            postprocessing_prompt="Review.",
            postprocess_attach_input=True,
            postprocess_attach_preprocessed=True,
            postprocess_attach_results=True,
        )
        with _StubAi() as ai:
            processor._postprocess("DOC", "PRE", "RES", preset, AppConfig())
        self.assertEqual(
            ai.prompts[0],
            "Review.\n\n\n=== ORIGINAL DOCUMENT ===\nDOC"
            "\n\n\n=== PREPROCESSED ===\nPRE"
            "\n\n\n=== NODE RESULTS ===\nRES",
        )

    def test_empty_attachments_are_left_out(self):
        preset = Preset(postprocessing_prompt="Review.", postprocess_attach_input=True)
        with _StubAi() as ai:
            processor._postprocess("", "PRE", "RES", preset, AppConfig())
        self.assertEqual(ai.prompts[0], "Review.\n\n\n=== NODE RESULTS ===\nRES")


class TestProgressCounter(unittest.TestCase):
    """The counter runs over the nodes actually sent, after both pre-passes."""

    def _counter_lines(self, doc: str, **preset_fields) -> list[str]:
        preset = Preset(node_prompt="p", **preset_fields)
        with _StubAi(), _CaptureLog("log_action") as actions:
            processor._run_node_processing(_corpus(doc), preset, AppConfig(), {})
        return [m for m in actions.messages if m.startswith("Node ")]

    def test_merged_heading_counts_once(self):
        lines = self._counter_lines(
            "# Title\n\nText.\n\nMore text.\n", node_mode="paragraph", heading_handling="merge_next"
        )
        self.assertEqual(lines, ["Node 1/2 (§1) [text]", "Node 2/2 (§3) [text]"])

    def test_absorbed_non_text_block_counts_once(self):
        lines = self._counter_lines(
            "Text.\n\n| a |\n| - |\n\n# Next\n\nMore text.\n",
            node_mode="section", non_text_handling="part_of_section",
        )
        self.assertEqual([line.split(" (")[0] for line in lines], ["Node 1/3", "Node 2/3", "Node 3/3"])

    def test_discarded_nodes_are_still_counted(self):
        lines = self._counter_lines("# Title\n\nText.\n", heading_handling="discard")
        self.assertEqual(lines, ["Node 1/2 (§1) [heading]", "Node 2/2 (§2) [text]"])

    def test_found_count_matches_the_counter(self):
        preset = Preset(node_prompt="p", heading_handling="merge_next")
        with _StubAi(), _CaptureLog("log_action") as actions:
            processor._run_node_processing(_corpus("# Title\n\nText.\n"), preset, AppConfig(), {})
        self.assertIn("Found 1 nodes. Processing...", actions.messages)


class TestBuildSeparator(unittest.TestCase):
    """The separator is part of the output contract, so its shape is pinned."""

    NODE = Node(index=3, node_type="text", content="x")

    def test_hidden_when_not_requested(self):
        self.assertEqual(processor._build_separator(self.NODE, 0.9, False), "")

    def test_without_similarity(self):
        self.assertEqual(processor._build_separator(self.NODE, None, True), "# §3")

    def test_with_similarity_uses_a_plain_hyphen(self):
        out = processor._build_separator(self.NODE, 0.842, True)
        self.assertTrue(out.startswith("# §3 - Cosine similarity: 0.842"))
        self.assertNotIn(EM_DASH, out)


class TestMergeNextPrePass(unittest.TestCase):
    """The pre-pass merges what it can; the rest reaches _process_node."""

    def _run(self, doc: str):
        preset = Preset(node_prompt="p", node_mode="paragraph", heading_handling="merge_next")
        with _StubAi():
            results, _ = processor._run_node_processing(_corpus(doc), preset, AppConfig(), {})
        return results

    def test_heading_followed_by_text_is_merged(self):
        results = self._run("# Title\n\nSome text.\n")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].node.node_type, "text")
        self.assertIn("# Title", results[0].node.content)
        self.assertIn("Some text.", results[0].node.content)

    def test_trailing_heading_reaches_the_residual_branch(self):
        results = self._run("Some text.\n\n# Trailing\n")
        heading = [r for r in results if r.node.node_type == "heading"]
        self.assertEqual(len(heading), 1)
        self.assertFalse(heading[0].was_processed)
        self.assertEqual(heading[0].output, "# Trailing")

    def test_heading_followed_by_a_table_reaches_the_residual_branch(self):
        results = self._run("# Title\n\n| a |\n| - |\n")
        heading = [r for r in results if r.node.node_type == "heading"]
        self.assertEqual(len(heading), 1)
        self.assertFalse(heading[0].was_processed)


class TestPartOfSection(unittest.TestCase):
    """part_of_section needs section mode, and says so when it does not have it."""

    DOC_LEADING_TABLE = "| a |\n| - |\n\nSome text.\n"
    DOC_TRAILING_TABLE = "Some text.\n\n| a |\n| - |\n"

    def _run(self, doc: str, node_mode: str):
        preset = Preset(
            node_prompt="p",
            node_mode=node_mode,
            non_text_handling="part_of_section",
        )
        with _CaptureWarnings() as warnings, _StubAi():
            results, _ = processor._run_node_processing(_corpus(doc), preset, AppConfig(), {})
        return results, warnings.messages

    def test_paragraph_mode_warns(self):
        _, warnings = self._run(self.DOC_TRAILING_TABLE, "paragraph")
        self.assertEqual(len(warnings), 1)
        self.assertIn("part_of_section", warnings[0])
        self.assertIn("section mode", warnings[0])

    def test_section_mode_does_not_warn(self):
        _, warnings = self._run(self.DOC_TRAILING_TABLE, "section")
        self.assertEqual(warnings, [])

    def test_section_mode_absorbs_a_following_non_text_block(self):
        results, _ = self._run(self.DOC_TRAILING_TABLE, "section")
        self.assertEqual(len(results), 1)
        self.assertIn("| a |", results[0].node.content)

    def test_leading_non_text_block_reaches_the_residual_branch(self):
        results, _ = self._run(self.DOC_LEADING_TABLE, "section")
        table = [r for r in results if r.node.node_type == "table"]
        self.assertEqual(len(table), 1)
        self.assertFalse(table[0].was_processed)

    def test_paragraph_mode_passes_non_text_through_unchanged(self):
        results, _ = self._run(self.DOC_TRAILING_TABLE, "paragraph")
        table = [r for r in results if r.node.node_type == "table"]
        self.assertEqual(len(table), 1)
        self.assertFalse(table[0].was_processed)
        self.assertIn("| a |", table[0].output)


class TestOtherHandlingRules(unittest.TestCase):
    """discard, report_unprocessed and standalone_nodes stay in _process_node."""

    DOC = "# Title\n\nSome text.\n"

    def _run(self, heading_handling: str):
        preset = Preset(node_prompt="p", node_mode="paragraph", heading_handling=heading_handling)
        with _StubAi():
            results, _ = processor._run_node_processing(_corpus(self.DOC), preset, AppConfig(), {})
        return results

    def test_discard_drops_the_heading_output(self):
        results = self._run("discard")
        heading = [r for r in results if r.node.node_type == "heading"][0]
        self.assertEqual(heading.output, "")
        self.assertFalse(heading.was_processed)

    def test_report_unprocessed_keeps_it_verbatim(self):
        results = self._run("report_unprocessed")
        heading = [r for r in results if r.node.node_type == "heading"][0]
        self.assertEqual(heading.output, "# Title")
        self.assertFalse(heading.was_processed)

    def test_standalone_nodes_sends_it_to_the_ai(self):
        results = self._run("standalone_nodes")
        heading = [r for r in results if r.node.node_type == "heading"][0]
        self.assertTrue(heading.was_processed)
        self.assertEqual(heading.output, "[AI]")


class TestImageHandling(unittest.TestCase):
    """Images follow image_handling, and standalone_nodes attaches the file.

    How a source is resolved to a file is tested in test_images.
    """

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        (self.dir / "pics").mkdir()
        self.image = self.dir / "pics" / "chart.png"
        self.image.write_bytes(b"png")

    def _run(self, doc: str, **preset_fields):
        preset = Preset(node_prompt="p", **preset_fields)
        with _StubAi() as ai:
            results, _ = processor._run_node_processing(
                _corpus(doc, self.dir), preset, AppConfig(), {}
            )
        return results, ai

    def test_standalone_attaches_the_file_resolved_from_the_document_folder(self):
        results, ai = self._run("Text.\n\n![c](pics/chart.png)\n", image_handling="standalone_nodes")
        self.assertEqual(ai.files, [None, self.dir / "pics" / "chart.png"])
        self.assertTrue(results[1].was_processed)

    def test_missing_image_stops_before_any_node_call(self):
        with _StubAi() as ai, self.assertRaises(FileNotFoundError):
            processor._run_node_processing(
                _corpus("Text.\n\n![c](pics/missing.png)\n", self.dir),
                Preset(node_prompt="p", image_handling="standalone_nodes"),
                AppConfig(), {},
            )
        self.assertEqual(ai.prompts, [])

    def test_downloaded_image_is_attached_then_deleted(self):
        seen: list[Path] = []

        def fake_fetch(url, dest_dir, index):
            path = dest_dir / "image.jpg"
            path.write_bytes(b"remote")
            return path

        def fake_call(prompt, config, preset, messages=None, file_path=None):
            seen.append(file_path)
            self.assertEqual(file_path.read_bytes(), b"remote")
            return processor._AiReply(text="[AI]", elapsed=0.0)

        preset = Preset(node_prompt="p", image_handling="standalone_nodes")
        with patch.object(images, "fetch_image", fake_fetch), \
                patch.object(processor, "_call_ai", fake_call):
            processor._run_node_processing(
                _corpus("![c](https://example.com/chart.jpg)\n"), preset, AppConfig(), {}
            )
        self.assertEqual(len(seen), 1)
        self.assertFalse(seen[0].exists())

    def test_non_text_handling_does_not_govern_images(self):
        results, ai = self._run(
            "| a |\n| - |\n\n![c](pics/chart.png)\n",
            non_text_handling="standalone_nodes", image_handling="report_unprocessed",
        )
        image = [r for r in results if r.node.node_type == "image"][0]
        self.assertFalse(image.was_processed)
        self.assertEqual(image.output, "![c](pics/chart.png)")
        self.assertEqual(ai.files, [None])

    def test_discard_drops_the_image(self):
        results, ai = self._run("![c](pics/chart.png)\n", image_handling="discard")
        self.assertEqual(results[0].output, "")
        self.assertEqual(ai.prompts, [])

    def test_part_of_section_absorbs_only_what_asks_for_it(self):
        results, ai = self._run(
            "Text.\n\n![c](pics/chart.png)\n\n| a |\n| - |\n",
            node_mode="section", image_handling="part_of_section",
        )
        self.assertIn("![c](pics/chart.png)", results[0].node.content)
        self.assertNotIn("| a |", results[0].node.content)
        self.assertEqual(results[1].node.node_type, "table")
        self.assertEqual(ai.files, [None])

    def test_image_part_of_section_warns_in_paragraph_mode(self):
        with _CaptureWarnings() as warnings:
            self._run("Text.\n", image_handling="part_of_section")
        self.assertEqual(len(warnings.messages), 1)
        self.assertIn("image_handling", warnings.messages[0])

    def test_similarity_embeds_the_image_with_the_original(self):
        preset = Preset(node_prompt="p", image_handling="standalone_nodes", compute_similarity=True)
        with _StubAi(), _StubEmbedding(failing=False) as embedding:
            processor._run_node_processing(
                _corpus("![c](pics/chart.png)\n", self.dir), preset, AppConfig(), {}
            )
        self.assertEqual(embedding.files, [self.image, None])


class TestPhaseSkipping(unittest.TestCase):
    """A phase is skipped purely by its prompt being empty."""

    def test_empty_node_prompt_skips_phase_two(self):
        preset = Preset(node_prompt="   ")
        with _StubAi() as ai:
            results, text = processor._run_node_processing(_corpus("Some text."), preset, AppConfig(), {})
        self.assertEqual(results, [])
        self.assertEqual(text, {})
        self.assertEqual(ai.prompts, [])


class _EchoAi:
    """Replaces _call_ai, answering '<first line of the prompt>:<last line>'.

    The answer tells which prompt and which input produced it, so a test can
    check where each result lands. Records every call's prompt, system prompt
    and prior messages.
    """

    def __init__(self):
        self.calls: list[dict] = []
        self._original = None

    def __enter__(self):
        self._original = processor._call_ai

        def fake(prompt, config, preset, messages=None, file_path=None):
            self.calls.append({"prompt": prompt, "system": preset.system_prompt, "messages": messages})
            lines = prompt.strip().splitlines()
            return processor._AiReply(text=f"{lines[0]}:{lines[-1]}", elapsed=0.0)

        processor._call_ai = fake
        return self

    def __exit__(self, *exc):
        processor._call_ai = self._original

    @property
    def prompts(self) -> list[str]:
        return [call["prompt"] for call in self.calls]


class _FolderRun(unittest.TestCase):
    """Runs run_iter on a folder of two documents, in an empty working directory."""

    def setUp(self):
        self._cwd = os.getcwd()
        self.dir = Path(tempfile.mkdtemp())
        os.chdir(self.dir)
        self._orig_apply = AppConfig.apply_to_client
        AppConfig.apply_to_client = lambda self: None
        self.docs = self.dir / "docs"
        self.docs.mkdir()
        self._write("a.md", "Alpha.\n")
        self._write("b.md", "Beta.\n")

    def tearDown(self):
        AppConfig.apply_to_client = self._orig_apply
        os.chdir(self._cwd)

    def _write(self, name: str, text: str) -> None:
        (self.docs / name).write_text(text, encoding="utf-8")

    def _run(self, preset: Preset, output: str | None = None, input_path: Path | None = None, **kwargs):
        # The §N separators are tested on their own; here they would only pad the outputs.
        preset = replace(preset, show_similarity_separator=False)
        with _EchoAi() as ai:
            result = processor.run_iter(
                input_path=str(input_path or self.docs), output_path=output,
                preset=preset, config=AppConfig(), **kwargs,
            )
        return result, ai

    def _read(self, path: str) -> str:
        return (self.dir / path).read_text(encoding="utf-8")


class TestFolderBoundaries(_FolderRun):
    """No pre-pass joins the end of one file to the start of the next."""

    def test_merge_next_does_not_reach_into_the_next_file(self):
        self._write("a.md", "Alpha.\n\n# Tail\n")
        result, ai = self._run(Preset(node_prompt="Condense.", heading_handling="merge_next"))
        heading = [r for r in result.node_results if r.node.node_type == "heading"]
        self.assertEqual(len(heading), 1)
        self.assertFalse(heading[0].was_processed)
        self.assertFalse(any("# Tail" in p for p in ai.prompts))

    def test_part_of_section_does_not_reach_into_the_previous_file(self):
        self._write("b.md", "| x |\n| - |\n\nBeta.\n")
        result, ai = self._run(Preset(
            node_prompt="Condense.", node_mode="section", non_text_handling="part_of_section"
        ))
        table = [r for r in result.node_results if r.node.node_type == "table"]
        self.assertEqual(len(table), 1)
        self.assertFalse(table[0].was_processed)
        self.assertFalse(any("| x |" in p for p in ai.prompts))


class TestFolderPhases(_FolderRun):
    """Phase 1 runs per file, Phase 3 per file or for the whole folder."""

    def test_each_node_gets_the_preprocessing_of_its_own_file(self):
        preset = Preset(preprocessing_prompt="Outline.", node_prompt="Condense.",
                        include_preprocessed_context=True)
        result, ai = self._run(preset)
        self.assertEqual(len(ai.calls), 4)
        node_calls = [c for c in ai.calls if c["prompt"].startswith("Condense.")]
        self.assertEqual(
            [c["messages"][1]["content"] for c in node_calls], ["Outline.:Alpha.", "Outline.:Beta."]
        )
        self.assertEqual([f.preprocessed for f in result.files], ["Outline.:Alpha.", "Outline.:Beta."])

    def test_file_mode_sends_each_file_whole(self):
        self._write("a.md", "# Title\n\nAlpha.\n")
        result, ai = self._run(Preset(node_prompt="Condense.", node_mode="file", heading_handling="discard"))
        self.assertEqual(ai.prompts, ["Condense.\n\n# Title\n\nAlpha.", "Condense.\n\nBeta."])
        self.assertEqual([r.node.source for r in result.node_results], ["a.md", "b.md"])

    def test_file_scope_makes_one_call_per_file(self):
        preset = Preset(node_prompt="Condense.", postprocessing_prompt="Review.")
        result, ai = self._run(preset)
        post_prompts = [p for p in ai.prompts if p.startswith("Review.")]
        self.assertEqual(post_prompts, [
            "Review.\n\n\n=== NODE RESULTS ===\nCondense.:Alpha.",
            "Review.\n\n\n=== NODE RESULTS ===\nCondense.:Beta.",
        ])
        self.assertEqual([f.postprocessed for f in result.files],
                         ["Review.:Condense.:Alpha.", "Review.:Condense.:Beta."])

    def test_corpus_scope_makes_one_call_with_every_file_under_its_header(self):
        preset = Preset(node_prompt="Condense.", postprocessing_prompt="Review.",
                        postprocessing_scope="corpus")
        result, ai = self._run(preset)
        post_prompts = [p for p in ai.prompts if p.startswith("Review.")]
        self.assertEqual(post_prompts, [
            "Review.\n\n\n=== NODE RESULTS ===\n"
            "# File: a.md\n\nCondense.:Alpha.\n\n# File: b.md\n\nCondense.:Beta."
        ])
        self.assertEqual(result.postprocessed, "Review.:Condense.:Beta.")
        self.assertEqual([f.postprocessed for f in result.files], ["", ""])

    def test_corpus_scope_with_the_input_attached_warns(self):
        preset = Preset(node_prompt="Condense.", postprocessing_prompt="Review.",
                        postprocessing_scope="corpus", postprocess_attach_input=True)
        with _CaptureWarnings() as warnings:
            self._run(preset)
        self.assertEqual(len(warnings.messages), 1)
        self.assertIn("postprocess_attach_input", warnings.messages[0])

    def test_single_file_results_carry_no_header(self):
        result, _ = self._run(Preset(node_prompt="Condense."), input_path=self.docs / "a.md")
        self.assertEqual(result.results, "Condense.:Alpha.")
        self.assertEqual([f.source for f in result.files], ["a.md"])

    def test_aggregate_results_join_the_files_under_their_headers(self):
        result, _ = self._run(Preset(node_prompt="Condense."))
        self.assertEqual(result.results, "# File: a.md\n\nCondense.:Alpha.\n\n# File: b.md\n\nCondense.:Beta.")


class TestFolderLayouts(_FolderRun):
    """The mirror layout writes one file per input, the merged layout a single file."""

    def test_mirror_writes_one_output_per_input(self):
        self._run(Preset(node_prompt="Condense."), output="out")
        self.assertEqual(self._read("out/a.md"), "Condense.:Alpha.")
        self.assertEqual(self._read("out/b.md"), "Condense.:Beta.")

    def test_mirror_writes_side_files_next_to_each_output(self):
        preset = Preset(preprocessing_prompt="Outline.", node_prompt="Condense.",
                        save_preprocessed_separately=True)
        self._run(preset, output="out")
        self.assertEqual(self._read("out/a_preprocessed.md"), "Outline.:Alpha.")

    def test_mirror_writes_the_corpus_phase_three_beside_the_folder(self):
        preset = Preset(node_prompt="Condense.", postprocessing_prompt="Review.",
                        postprocessing_scope="corpus", output_include_postprocessed=True)
        self._run(preset, output="out")
        self.assertEqual(self._read("out_postprocessed.md"), "Review.:Condense.:Beta.")
        self.assertEqual(self._read("out/a.md"), "Condense.:Alpha.")

    def test_merged_puts_each_file_under_its_header(self):
        self._run(Preset(node_prompt="Condense.", output_layout="merged"), output="all.md")
        self.assertEqual(
            self._read("all.md"),
            "# File: a.md\n\nCondense.:Alpha.\n\n# File: b.md\n\nCondense.:Beta.",
        )

    def test_merged_puts_each_file_phase_three_right_after_its_results(self):
        preset = Preset(node_prompt="Condense.", postprocessing_prompt="Review.",
                        output_layout="merged", output_include_postprocessed=True)
        self._run(preset, output="all.md")
        self.assertEqual(
            self._read("all.md"),
            "# File: a.md\n\nCondense.:Alpha.\n\n---\n\nReview.:Condense.:Alpha."
            "\n\n# File: b.md\n\nCondense.:Beta.\n\n---\n\nReview.:Condense.:Beta.",
        )

    def test_merged_with_only_the_corpus_phase_three_holds_just_that(self):
        preset = Preset(node_prompt="Extract.", postprocessing_prompt="Consolidate.",
                        postprocessing_scope="corpus", output_layout="merged",
                        output_include_processed=False, output_include_postprocessed=True)
        self._run(preset, output="glossary.md")
        self.assertEqual(self._read("glossary.md"), "Consolidate.:Extract.:Beta.")

    def test_merged_appends_the_corpus_phase_three_after_every_file(self):
        preset = Preset(node_prompt="Condense.", postprocessing_prompt="Review.",
                        postprocessing_scope="corpus", output_layout="merged",
                        output_include_postprocessed=True)
        self._run(preset, output="all.md")
        self.assertTrue(self._read("all.md").endswith("Condense.:Beta.\n\n---\n\nReview.:Condense.:Beta."))

    def test_merged_leaves_out_a_file_with_nothing_to_show(self):
        self._write("c.md", "# Only a heading\n")
        preset = Preset(node_prompt="Condense.", heading_handling="discard", output_layout="merged")
        self._run(preset, output="all.md")
        self.assertNotIn("c.md", self._read("all.md"))

    def test_output_inside_the_input_folder_is_refused_before_any_call(self):
        with self.assertRaises(ValueError):
            self._run(Preset(node_prompt="Condense.", output_layout="merged"), output="docs/all.md")


class TestReferenceContext(_FolderRun):
    """The context file closes the system prompt of every call of the run."""

    PRESET = Preset(preprocessing_prompt="Outline.", node_prompt="Condense.",
                    postprocessing_prompt="Review.")

    def setUp(self):
        super().setUp()
        (self.dir / "glossary.md").write_text("GLOSSARY\n", encoding="utf-8")

    def test_appended_to_the_preset_system_prompt(self):
        preset = replace(self.PRESET, system_prompt="Be terse.")
        _, ai = self._run(preset, context_file="glossary.md")
        self.assertEqual(len(ai.calls), 6)
        self.assertEqual(
            {c["system"] for c in ai.calls}, {"Be terse.\n\n=== REFERENCE CONTEXT ===\nGLOSSARY\n"}
        )

    def test_alone_when_the_preset_has_no_system_prompt(self):
        _, ai = self._run(self.PRESET, context_file="glossary.md")
        self.assertEqual({c["system"] for c in ai.calls}, {"=== REFERENCE CONTEXT ===\nGLOSSARY\n"})

    def test_absent_leaves_the_system_prompt_alone(self):
        _, ai = self._run(self.PRESET)
        self.assertEqual({c["system"] for c in ai.calls}, {""})

    def test_missing_file_stops_the_run(self):
        with self.assertRaises(FileNotFoundError):
            self._run(self.PRESET, context_file="absent.md")


if __name__ == "__main__":
    unittest.main()
