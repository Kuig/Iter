# Iter - Iterative Node-Based Document Processor

## Overview

Iter applies a prompt to every paragraph or section node of a Markdown document. It supports three sequential processing phases:

1. **Preprocessing** - a single AI call on the full document (e.g., extracting context or a summary).
2. **Node Processing** - the same prompt is applied independently to each paragraph or section node.
3. **Postprocessing** - a single AI call on the collected results (e.g., global consolidation or quality check).

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
iter gui
```

### CLI

```bash
iter run --input document.md --output out.md --preset condensation
iter run --input document.md --output out.md --node-prompt "Summarize this section."
```

### MCP Server

```bash
iter mcp
```

### Python Library

```python
from itertool import run_iter
from itertool.models import Preset
from itertool.config import AppConfig

preset = Preset(node_prompt="Summarize this section.", node_mode="section")
config = AppConfig(provider="ollama", model_name="gemma4:12b")
result = run_iter(input_file="doc.md", output_file="out.md", preset=preset, config=config)
print(result.results)
```

`Preset`, `AppConfig` and `ProcessingResult` are also re-exported from the package root, so
`from itertool import run_iter, Preset, AppConfig` works too.

#### API Reference

```python
run_iter(
    input_file: str | None = None,
    input_text: str | None = None,
    output_file: str | None = None,
    preset: Preset | None = None,
    config: AppConfig | None = None,
    on_mismatch: str = "ask",
) -> ProcessingResult
```

| Parameter | Meaning |
|---|---|
| `input_file` | Path to the Markdown document to process. |
| `input_text` | Raw Markdown, as an alternative to `input_file`. One of the two is required. |
| `output_file` | Where to write the composed result. Omit to only receive it in the return value. |
| `preset` | The task: prompts, node mode, handling rules, output composition. Defaults to `Preset()`. |
| `config` | Model and provider settings. Defaults to `AppConfig.load()`. |
| `on_mismatch` | What to do with a progress file from a different run: `"ask"`, `"continue"` or `"restart"`. See [Resuming an interrupted run](#resuming-an-interrupted-run). |

Returns a `ProcessingResult` with `preprocessed`, `node_results`, `results` and `postprocessed`.
Raises `ValueError` if neither `input_text` nor an existing `input_file` is given, and
`CheckpointMismatchError` (exported from `itertool`) when `on_mismatch` is `"ask"` and a
progress file from a different run is found.

### Resuming an interrupted run

Every completed AI call is saved at once to a progress file, so a run that stops early (a
crash, a timeout, Ctrl+C) picks up where it left off when started again with the same input
and output. The file is deleted when the run completes.

- With an output file, it is `<output file>.iter-progress.jsonl`, next to the output.
- Without one, it goes in the working directory, named after the input:
  `<input name>-<hash>.iter-progress.jsonl`, or `text-<hash>.iter-progress.jsonl` for text input.

If the input document, the preset or the model settings changed since the file was written,
Iter asks whether to **continue**, keeping the saved results and redoing only the nodes whose
content changed, or **restart** from scratch. The CLI asks in the console, the GUI in a
dialog. The MCP tools reply with an explanation and take `on_mismatch="continue"` or
`"restart"` on the next call. Output-only preset settings (which phases go in the output
file, separators, interlacing) do not count as a change.

### Windows Context Menu Integration

You can integrate Iter into the Windows File Explorer context menu. This lets you right-click any `.md` or `.markdown` file and process it directly with any of your saved presets in a single click.

1. Open the `Windows Integration` directory.
2. Run the PowerShell script to scan your `Presets/` directory and generate the custom registry entries:
   ```powershell
   PowerShell -ExecutionPolicy Bypass -File .\generate_registry_files.ps1
   ```
3. Double-click the generated `register.reg` file to merge it into the Windows Registry.
4. To remove the context menu entries, double-click the `unregister.reg` file.

*Note: If you add, rename, or delete a preset in the `Presets/` folder, simply rerun the PowerShell script and merge the new `register.reg` to keep the context menu in sync.*

#### GUI launcher

`Windows Integration/launch_gui.bat` starts the GUI from the project folder: it changes into the
project root, activates `.venv`, and runs `iter gui`. Use it from a desktop shortcut, so the GUI
always starts where `Presets/`, `config.json` and `secrets.json` are.

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
