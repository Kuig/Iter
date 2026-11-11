# Iter - Iterative Node-Based Document Processor

## Overview

Iter applies a prompt to every paragraph or section node of a Markdown document, or of every
document in a folder. It supports three sequential processing phases:

1. **Preprocessing** - a single AI call on the full document (e.g., extracting context or a summary).
2. **Node Processing** - the same prompt is applied independently to each paragraph, section or file node.
3. **Postprocessing** - a single AI call on the collected results (e.g., global consolidation or quality check),
   per document or for the whole folder.

Each phase is optional. Cosine similarity between the original node and the processed result can be computed and shown as a separator between results.

---

## Installation

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows

# Install in editable mode (dev)
pip install -r requirements_dev.txt
```

---

## Usage

### GUI (recommended)

```bash
iter gui            # empty input
iter gui doc.md     # doc.md already set as the input file
iter gui chapters   # the chapters folder set as the input
```

The **Files** section takes a file or a folder as input (📁 / 🗂️), an output file, or an output
folder when the input is a folder in the mirror layout, and an optional reference context file.

### CLI

```bash
iter run --input document.md --output out.md --preset condensation
iter run --input document.md --output out.md --node-prompt "Summarize this section."
iter run --input chapters --output chapters_it --preset "Translate to it-IT" --context glossary.md
```

`--input` takes a file or a folder ([Processing a folder](#processing-a-folder)), `--context` a
[reference context](#reference-context) file. Without `--output`, the output is
`<name>_processed<ext>` next to a file; for a folder, `<folder>_processed` beside it in the mirror
layout, or `<folder>_processed.md` in the merged layout. `--node-mode` accepts `paragraph`,
`section` and `file`.

### MCP Server

```bash
iter mcp
```

Two tools: `iter_process_path` takes a file or a folder, `iter_process_text` a Markdown string.
Both take an optional `context_file`.

### Python Library

```python
from itertool import run_iter
from itertool.models import Preset
from itertool.config import AppConfig

preset = Preset(node_prompt="Summarize this section.", node_mode="section")
config = AppConfig(provider="ollama", model_name="gemma4:12b")
result = run_iter(input_path="doc.md", output_file="out.md", preset=preset, config=config)
print(result.results)
```

`Preset`, `AppConfig` and `ProcessingResult` are also re-exported from the package root, so
`from itertool import run_iter, Preset, AppConfig` works too.

#### API Reference

```python
run_iter(
    input_path: str | None = None,
    input_text: str | None = None,
    output_file: str | None = None,
    preset: Preset | None = None,
    config: AppConfig | None = None,
    context_file: str | None = None,
    on_mismatch: str = "ask",
) -> ProcessingResult
```

| Parameter | Meaning |
|---|---|
| `input_path` | The Markdown document to process, or a folder: see [Processing a folder](#processing-a-folder). |
| `input_text` | Raw Markdown, as an alternative to `input_path`. One of the two is required. |
| `output_file` | Where to write the composed result: a file, or a folder for a folder input in the mirror layout. Omit to only receive it in the return value. |
| `preset` | The task: prompts, node mode, handling rules, output composition. Defaults to `Preset()`. |
| `config` | Model and provider settings. Defaults to `AppConfig.load()`. |
| `context_file` | A text file appended to the system prompt of every AI call, as reference context. See [Reference context](#reference-context). |
| `on_mismatch` | What to do with a progress file from a different run: `"ask"`, `"continue"` or `"restart"`. See [Resuming an interrupted run](#resuming-an-interrupted-run). |

Returns a `ProcessingResult` with `preprocessed`, `node_results`, `results` and `postprocessed`,
plus `files`, the same results for each input file. For a folder, the four aggregate fields join
the files' results, each under a `# File: <name>` header.
Raises `ValueError` if neither `input_text` nor an existing `input_path` is given, a folder holds
no Markdown file, or its output is misplaced; `FileNotFoundError` if `context_file` is missing;
and `CheckpointMismatchError` (exported from `itertool`) when `on_mismatch` is `"ask"` and a
progress file from a different run is found.

`default_output_path(input_path, preset)` in `itertool.processor` gives the output name the CLI
uses when `--output` is omitted.

### Processing a folder

When the input is a folder, Iter processes the Markdown files (`.md`, `.markdown`) directly inside
it, in natural order of their names (`2.md` before `10.md`); subfolders are not read. No node or
section ever spans two files, and each file gets its own Phase 1. Two preset fields decide the
rest:

- `postprocessing_scope`: one Phase 3 call per file, or one for the whole folder.
- `output_layout`: one output file per input file in an output folder (`mirror`), or a single file
  with each input under a `# File: <name>` header (`merged`).

`node_mode="file"` sends each file whole, as one node: useful to extract the same information
from every document. The output may be neither the input folder nor directly inside it. Both
fields and the exact output composition are documented in [docs/presets.md](docs/presets.md#folder-input).

### Reference context

A context file, such as a glossary, is appended to the system prompt of every AI call of the run,
under `=== REFERENCE CONTEXT ===`. It is an input of the run, not of the preset, so the same preset
can be used with a different context for each corpus.

Two runs make a corpus-wide context that a single Phase 1 could not fit. For a translation with
consistent terminology across a folder:

1. A "glossary" run: `node_mode="file"` with a prompt that extracts the terms to keep consistent,
   `postprocessing_scope="corpus"` with only the node results attached and a prompt that merges
   them, `output_layout="merged"` with only Phase 3 in the output. It writes `glossary.md`.
2. The translation run on the same folder, with `glossary.md` as its reference context.

### Resuming an interrupted run

Every completed AI call is saved at once to a progress file, so a run that stops early (a
crash, a timeout, Ctrl+C) picks up where it left off when started again with the same input
and output. The file is deleted when the run completes.

- With an output file, it is `<output file>.iter-progress.jsonl`, next to the output.
- Without one, it goes in the working directory, named after the input:
  `<input name>-<hash>.iter-progress.jsonl`, or `text-<hash>.iter-progress.jsonl` for text input.

If the input, the reference context, the preset or the model settings changed since the file
was written, Iter asks whether to **continue**, keeping the saved results and redoing only the
nodes whose content changed (and Phase 1 of the documents that changed), or **restart** from
scratch. The CLI asks in the console, the GUI in a
dialog. The MCP tools reply with an explanation and take `on_mismatch="continue"` or
`"restart"` on the next call. Output-only preset settings (which phases go in the output
file, separators, interlacing, output layout) do not count as a change. A progress file written
by a version of Iter with a different format is discarded and the run starts over.

### Windows Context Menu Integration

You can integrate Iter into the Windows File Explorer context menu. Right-click any `.md` or `.markdown` file, or any folder, to get two entries:

- **Process with Iter...**: a submenu with one entry per saved preset. It runs the preset on the
  file or folder through `Windows Integration/run_preset.cmd`, from the project folder, and
  writes the CLI's default output: `<name>_processed<ext>` next to a file; for a folder,
  `<folder>_processed` beside it, or `<folder>_processed.md` when the preset's `output_layout`
  is `merged`. Running it again on the same input writes the same output, so an interrupted run
  resumes. The console window stays open only when the run fails.
- **Open with Iter**: opens the GUI with the file or folder already set as the input.

1. Open the `Windows Integration` directory.
2. Run the PowerShell script to scan your `Presets/` directory and generate the custom registry entries:
   ```powershell
   PowerShell -ExecutionPolicy Bypass -File .\generate_registry_files.ps1
   ```
3. Double-click the generated `register.reg` file to merge it into the Windows Registry.
4. To remove the context menu entries, double-click the `unregister.reg` file.

*Note: If you add, rename, or delete a preset in the `Presets/` folder, rerun the PowerShell script and merge the new `register.reg`: it replaces the whole submenu, so deleted presets disappear too. Explorer shows at most 16 entries in a submenu, so the script stops with an error above 16 presets.*

The entries call the `.venv` of the project folder, so they need the project installed there.

#### GUI launcher

`Windows Integration/launch_gui.bat [file]` starts the GUI from the project folder: it changes into
the project root, activates `.venv`, and runs `iter gui`, passing the file when given. Use it from a
desktop shortcut, so the GUI always starts where `Presets/`, `config.json` and `secrets.json` are.

---

## Configuration

Iter reads `config.json` and `secrets.json` from the folder you run it from. Copy the committed
templates to get started:

```bash
cp config.json.example config.json
cp secrets.json.example secrets.json
```

`config.json` holds model and provider settings; `secrets.json` holds API keys and is read by
UnifiedAiClient, not by Iter. Both are gitignored.

Every field is documented in **[docs/configuration.md](docs/configuration.md)**.

---

## Presets

Presets are `.json` files in the `Presets/` directory. Each one saves the prompt texts, the node
mode, and all processing options. Load them from the GUI sidebar or with the `--preset` CLI flag.

Every field is documented in **[docs/presets.md](docs/presets.md)**.

---

## Documentation

| Document | Contents |
|---|---|
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | How Iter is built: project layout, the four interfaces, node parsing, the three-phase pipeline, prompt composition, separator format, logging, GUI layout. |
| **[docs/configuration.md](docs/configuration.md)** | Every `AppConfig` field, both JSON files, and when settings take effect. |
| **[docs/presets.md](docs/presets.md)** | Every `Preset` field, and both handling value sets. |
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | Development install, running the tests, changelog policy. |
| **[CHANGELOG.md](CHANGELOG.md)** | Release history. |

---

## License

MIT. See [LICENSE](LICENSE).
