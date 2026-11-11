"""Progress file that makes an interrupted Iter run resumable.

Every completed AI call is appended to a JSON Lines file as soon as it returns.
A later run on the same input finds the file, restores what was done and
calls the AI only for the rest. The file is deleted once the run has written
all of its outputs.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import itertool.logger as logger
from itertool.config import AppConfig
from itertool.models import Corpus, Node, NodeResult, Preset

PROGRESS_SUFFIX = ".iter-progress.jsonl"
MISMATCH_POLICIES = ("ask", "continue", "restart")

_FORMAT_VERSION = 2

# Preset fields that only shape how saved results are assembled into the output.
# Changing them does not invalidate any AI call, so they stay out of the fingerprint.
_OUTPUT_ONLY_FIELDS = frozenset({
    "name",
    "show_similarity_separator",
    "output_include_preprocessed",
    "output_include_processed",
    "output_include_postprocessed",
    "save_preprocessed_separately",
    "save_postprocessed_separately",
    "interlace_input_output",
    "output_layout",
})

_FINGERPRINT_LABELS = {
    "input": "input document",
    "context": "reference context",
    "preset": "preset",
    "model": "model settings",
}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def text_key(text: str) -> str:
    """Key a Phase 1 result by the document it was computed on.

    A document edited between two runs then gets a new Phase 1, while an
    unchanged one keeps its saved result, whichever file it is in.
    """
    return _digest(text)[:16]


def progress_path(
    output_path: str | None,
    input_path: str | None,
    input_text: str | None,
) -> Path:
    """Decide where the progress file of a run lives.

    Next to the output when there is one (beside the folder, for a mirrored
    output), otherwise in the working directory, named after the input and a
    short hash of it so that two inputs with the same name do not share a file.

    Args:
        output_path: Main output path of the run (file or folder), if any.
        input_path: Input document or folder, used when there is no output.
        input_text: Raw input text, used when there is neither.

    Returns:
        The progress file path.
    """
    if output_path:
        out = Path(output_path)
        return out.with_name(out.name + PROGRESS_SUFFIX)
    if input_path:
        stem, seed = Path(input_path).stem, str(Path(input_path).resolve())
    else:
        stem, seed = "text", input_text or ""
    return Path.cwd() / f"{stem}-{_digest(seed)[:8]}{PROGRESS_SUFFIX}"


def fingerprint(corpus: Corpus, preset: Preset, config: AppConfig, context: str = "") -> dict:
    """Describe everything that determines the AI calls of a run.

    Args:
        corpus: The input: every file's name and text.
        preset: Active preset; output-only fields are left out.
        config: Application configuration; only the generation-relevant fields.
        context: The reference context added to the system prompt, if any.

    Returns:
        A JSON-serialisable dict, compared key by key on resume.
    """
    preset_fields = {k: v for k, v in asdict(preset).items() if k not in _OUTPUT_ONLY_FIELDS}
    files = [[source.name, source.text] for source in corpus.files]
    return {
        "version": _FORMAT_VERSION,
        "input": _digest(json.dumps(files, ensure_ascii=False)),
        "context": _digest(context),
        "preset": _digest(json.dumps(preset_fields, sort_keys=True)),
        "model": {
            "provider": config.provider,
            "model_name": config.model_name,
            "temperature": config.temperature,
            "thinking": config.thinking,
            "context_size": config.context_size,
            "embedding_provider": config.embedding_provider,
            "embedding_model": config.embedding_model,
        },
    }


def file_digest(path: Path) -> str:
    """Hash a file's bytes, for keying a node that sends the file to the AI."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def node_keys(nodes: list[Node], extra: dict[int, str] | None = None) -> list[str]:
    """Key each node by its content, so edits elsewhere in the input keep it valid.

    Identical contents are told apart by their occurrence number.

    Args:
        nodes: The node list as it is sent to the AI, after the pre-passes.
        extra: Further input a node sends to the AI besides its text, by
            Node.index: the digest of an attached image, so that a changed
            file invalidates the node even when its Markdown did not change.

    Returns:
        One key per node, in the same order.
    """
    extra = extra or {}
    seen: dict[str, int] = {}
    keys = []
    for node in nodes:
        digest = _digest(node.content + extra.get(node.index, ""))[:16]
        seen[digest] = seen.get(digest, 0) + 1
        keys.append(f"{digest}-{seen[digest]}")
    return keys


class CheckpointMismatchError(Exception):
    """A progress file exists but was written for a different run.

    Attributes:
        path: The progress file.
        differences: Human-readable names of what changed, e.g. ['preset'].
    """

    def __init__(self, path: Path, differences: list[str]):
        self.path = path
        self.differences = differences
        super().__init__(
            f"A progress file from an earlier run exists at {path}, "
            f"but the {', '.join(differences)} changed since."
        )


class Checkpoint:
    """The progress file of one run: what is already done, and a log of what gets done.

    Attributes:
        path: The progress file.
        preprocessed: Saved Phase 1 outputs, by text_key() of their document.
        postprocessed: Saved Phase 3 outputs, by source file name, or under
            None for the single corpus-wide call.
        nodes: Saved Phase 2 records, by node key.
    """

    def __init__(self, path: Path, run_fingerprint: dict, on_mismatch: str = "ask"):
        """Open the progress file, restoring what it holds.

        Args:
            path: The progress file; created if missing.
            run_fingerprint: The fingerprint of the run about to start.
            on_mismatch: What to do when the file belongs to a different run:
                'ask' raises CheckpointMismatchError so the caller can ask the
                user, 'continue' keeps the saved results, 'restart' drops them.

        Raises:
            ValueError: If on_mismatch is not one of MISMATCH_POLICIES.
            CheckpointMismatchError: On a mismatch with on_mismatch='ask'.
        """
        if on_mismatch not in MISMATCH_POLICIES:
            raise ValueError(f"on_mismatch must be one of {', '.join(MISMATCH_POLICIES)}; got {on_mismatch!r}")
        self.path = Path(path)
        self.preprocessed: dict[str, str] = {}
        self.postprocessed: dict[str | None, str] = {}
        self.nodes: dict[str, dict] = {}

        records, intact = self._read()
        if not records:
            self._rewrite(run_fingerprint, [])
            return

        header, body = records[0], records[1:]
        saved = header.get("fingerprint", {}) if header.get("type") == "header" else {}
        if saved.get("version") != run_fingerprint["version"]:
            # Records of another format cannot be read back: there is nothing to continue from.
            logger.log_warning(f"Starting over: {self.path} was written by a different version of Iter.")
            self._rewrite(run_fingerprint, [])
            return
        differences = [
            label for key, label in _FINGERPRINT_LABELS.items()
            if saved.get(key) != run_fingerprint[key]
        ]
        if differences:
            if on_mismatch == "ask":
                raise CheckpointMismatchError(self.path, differences)
            if on_mismatch == "restart":
                body = []
        # A damaged file is rewritten before anything is appended: a record added
        # after a truncated last line would land on that line and be lost.
        if differences or not intact:
            self._rewrite(run_fingerprint, body)
        self._restore(body)

    @property
    def resumed(self) -> bool:
        """Whether anything was restored from an earlier run."""
        return bool(self.nodes or self.preprocessed or self.postprocessed)

    def summary(self) -> str:
        """Describe what was restored, for the log."""
        parts = []
        if self.preprocessed:
            parts.append(f"phase 1: {len(self.preprocessed)} saved")
        if self.nodes:
            parts.append(f"{len(self.nodes)} nodes done")
        if self.postprocessed:
            parts.append(f"phase 3: {len(self.postprocessed)} saved")
        return ", ".join(parts)

    def record_preprocessed(self, key: str, output: str, elapsed: float) -> None:
        """Save a Phase 1 output, keyed by text_key() of its document."""
        self.preprocessed[key] = output
        self._append({"type": "pre", "key": key, "output": output, "elapsed": elapsed})

    def record_node(self, key: str, result: NodeResult) -> None:
        """Save the result of one node sent to the AI."""
        record = {
            "type": "node",
            "key": key,
            "source": result.node.source,
            "index": result.node.index,
            "output": result.output,
            "similarity": result.similarity,
            "input_chars": result.input_chars,
            "output_chars": result.output_chars,
            "elapsed": result.elapsed,
        }
        self.nodes[key] = record
        self._append(record)

    def record_postprocessed(self, source: str | None, output: str, elapsed: float) -> None:
        """Save a Phase 3 output: of one source file, or of the corpus for None."""
        self.postprocessed[source] = output
        self._append({"type": "post", "source": source, "output": output, "elapsed": elapsed})

    def delete(self) -> None:
        """Remove the progress file once the run is complete."""
        self.path.unlink(missing_ok=True)

    def _read(self) -> tuple[list[dict], bool]:
        """Read every record, skipping lines a crash left unreadable.

        Returns:
            The readable records, and whether the file was intact (every line
            readable and newline-terminated).
        """
        if not self.path.exists():
            return [], True
        text = self.path.read_text(encoding="utf-8")
        lines = text.splitlines()
        records = []
        intact = text.endswith("\n") or not text
        for number, line in enumerate(lines, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                intact = False
                # A truncated last line is the expected trace of an interruption;
                # anything else is worth telling the user about.
                if number != len(lines):
                    logger.log_warning(f"Skipping unreadable line {number} of {self.path}")
        return records, intact

    def _restore(self, records: list[dict]) -> None:
        for record in records:
            kind = record.get("type")
            if kind == "pre":
                self.preprocessed[record["key"]] = record["output"]
            elif kind == "post":
                self.postprocessed[record["source"]] = record["output"]
            elif kind == "node":
                self.nodes[record["key"]] = record

    def _rewrite(self, run_fingerprint: dict, records: list[dict]) -> None:
        """Replace the file with a fresh header and the given records, atomically."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = {"type": "header", "fingerprint": run_fingerprint}
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for record in [header, *records]:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def _append(self, record: dict) -> None:
        # One open per record keeps the file complete on disk after every call;
        # AI calls take seconds, so the cost is negligible.
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
