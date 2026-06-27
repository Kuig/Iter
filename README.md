# Iter - Iterative Node-Based Document Processor

## Overview

Iter applies a prompt to every paragraph or section node of a Markdown document. It supports three sequential processing phases:

1. **Preprocessing** — a single AI call on the full document (e.g., extracting context or a summary).
2. **Node Processing** — the same prompt is applied independently to each paragraph or section node.
3. **Postprocessing** — a single AI call on the collected results (e.g., global consolidation or quality check).

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

---

## Configuration

### `config.json`

```json
{
  "provider": "ollama",
  "model_name": "gemma4:12b",
  "embedding_provider": "ollama",
  "embedding_model": "bge-m3",
  "temperature": 0.7,
  "ollama_url": "http://localhost:11434",
  "thinking": false
}
```

### `secrets.json` (do not commit)

```json
{
  "google_api_key": "your-key-here",
  "anthropic_api_key": "your-key-here"
}
```

---

## Presets

Presets are `.json` files stored in the `Presets/` directory. Each preset saves prompt texts, node mode, and all processing options. Load them in the GUI sidebar or via the `--preset` CLI flag.

---

## Supported Providers

`ollama`, `google`, `anthropic`, `openai`, `mistral`, `cohere`, `groq`, `xai`, `lmstudio`, `llamacpp`

---

## Architecture

### Project layout

```
Iter/
├── itertool/                  # Python package (installable)
│   ├── __init__.py            # UTF-8 fix + re-exports run_iter
│   ├── __main__.py            # CLI entry point  →  iter run / mcp / gui
│   ├── config.py              # AppConfig dataclass  (config.json loader)
│   ├── logger.py              # Dual-backend logger (console + Streamlit)
│   ├── markdown_utils.py      # Markdown parsing and block classification
│   ├── models.py              # Preset, Node, NodeResult, ProcessingResult, cosine_similarity
│   ├── mcp_tools.py           # MCP tool registrations
│   ├── processor.py           # Core 3-phase processing pipeline
│   └── gui/
│       └── app.py             # Streamlit web interface
├── Presets/                   # One JSON file per preset
│   ├── condensation.json
│   └── empty.json
├── config.json                # Default model settings (provider, model, temperature, …)
├── secrets.json               # API keys — never commit
├── pyproject.toml             # Package metadata; declares  iter  CLI entry point
├── requirements_dev.txt       # Editable installs for local dev
└── requirements_prod.txt      # Pinned Git installs for production
```

---

### Interfaces

Iter exposes **four interfaces** from the same business logic (`processor.py`):

| Interface | How to invoke | Entry point |
|---|---|---|
| **CLI** | `iter run --input f.md --output out.md` | `__main__.py` → `cmd_run()` |
| **MCP server** | `iter mcp` | `__main__.py` → `mcp_tools.py` → `register_tools()` |
| **Python library** | `from itertool import run_iter` | `processor.py` → `run_iter()` |
| **Streamlit GUI** | `iter gui` | `gui/app.py` |

All four ultimately call `run_iter()` in `processor.py`.

---

### Data models (`models.py`)

#### `AppConfig`
Loaded from `config.json`. Holds the **model settings** that are not saved in presets:

| Field | Default | Meaning |
|---|---|---|
| `provider` | `"ollama"` | AI provider name |
| `model_name` | `"gemma4:12b"` | Main generation model |
| `embedding_provider` | `"ollama"` | AI provider name for embeddings |
| `embedding_model` | `"bge-m3"` | Model used for cosine similarity |
| `temperature` | `0.7` | Sampling temperature |
| `ollama_url` | `"http://localhost:11434"` | Base URL for the Ollama server |
| `thinking` | `false` | Enable extended thinking / reasoning mode where the provider supports it |

#### `Preset`
Loaded from / saved to `Presets/<name>.json`. Holds the **task configuration**:

| Field | Type | Meaning |
|---|---|---|
| `name` | str | Preset display name |
| `system_prompt` | str | System instruction (overrides AppConfig default when non-empty) |
| `node_mode` | `"paragraph"` \| `"section"` | Granularity of document splitting |
| `preprocessing_prompt` | str | Phase 1 prompt (empty = skip) |
| `node_prompt` | str | Phase 2 prompt (empty = skip) |
| `postprocessing_prompt` | str | Phase 3 prompt (empty = skip) |
| `include_preprocessed_context` | bool | Inject Phase 1 result as prior assistant message in Phase 2 calls |
| `heading_handling` | see below | How to treat `#` heading blocks in Phase 2 |
| `non_text_handling` | see below | How to treat tables / code / images in Phase 2 |
| `compute_similarity` | bool | Compute cosine similarity per node |
| `show_similarity_separator` | bool | Insert `# §N — heading — Cosine similarity: score` between results |
| `postprocess_attach_input` | bool | Append original document to Phase 3 prompt |
| `postprocess_attach_preprocessed` | bool | Append Phase 1 result to Phase 3 prompt |
| `postprocess_attach_results` | bool | Append Phase 2 results to Phase 3 prompt |
| `output_include_preprocessed` | bool | Include Phase 1 result in the output file |
| `output_include_processed` | bool | Include Phase 2 node results in the output file *(default: true)* |
| `output_include_postprocessed` | bool | Include Phase 3 result in the output file |
| *(all three above)* | | Selected sections are appended in order, separated by `---` |
| `save_preprocessed_separately` | bool | Also write Phase 1 result to `<output>_preprocessed.md` |
| `save_postprocessed_separately` | bool | Also write Phase 3 result to `<output>_postprocessed.md` |
| `interlace_input_output` | bool | Save processed nodes interlaced with input nodes *(default: false)* |

**`heading_handling` values:**

| Value | Behaviour |
|---|---|
| `"report_unprocessed"` | Pass heading through unchanged *(default)* |
| `"discard"` | Drop the heading entirely |
| `"standalone_nodes"` | Send heading text to the AI as its own node |
| `"merge_next"` | Prepend heading to the next text block before sending to AI |

**`non_text_handling` values** (tables, code blocks, images):

| Value | Behaviour |
|---|---|
| `"report_unprocessed"` | Pass block through unchanged *(default)* |
| `"discard"` | Drop the block entirely |
| `"standalone_nodes"` | Send block to the AI as its own node |
| `"part_of_section"` | Absorb block into the surrounding text section *(section mode only)* |

#### `Node`
A single unit of text extracted from the document.

| Field | Meaning |
|---|---|
| `index` | 1-based sequential number |
| `node_type` | `"text"`, `"heading"`, `"code"`, `"table"`, `"image"` |
| `content` | Raw text of the block |
| `heading_prefix` | Most recent heading above this block (used to build `node_id`) |
| `node_id` *(property)* | `"3 - Introduction"` format used in separators |

#### `ProcessingResult`
Returned by `run_iter()`.

| Field | Meaning |
|---|---|
| `preprocessed` | Phase 1 AI output (empty string if skipped) |
| `node_results` | `list[NodeResult]` — one per node processed |
| `results` | Phase 2 assembled string (nodes joined with optional separators) |
| `postprocessed` | Phase 3 AI output (empty string if skipped) |

---

### Markdown parsing (`markdown_utils.py`)

The raw document is first split into **logical blocks** by blank lines, with fenced
code blocks kept intact (blank lines inside ` ``` ` fences are not split points).

#### Paragraph mode
Each block becomes exactly one `Node`. Block type is classified with:

- `is_header()` — starts with `#`
- `is_table()` — first two lines contain `|` and `-`
- `is_code_block()` — starts with ` ``` `
- `is_image_or_link_only()` — matches `!?[…](…)` alone on a line

#### Section mode
Structural blocks (headings, code, tables, images) flush the current text buffer
as a single `Node(node_type="text")`, then are emitted themselves as individual
structural nodes. The result is alternating text-section nodes and structural nodes.
This means a section containing multiple paragraphs is sent to the AI as one combined
string, achieving higher compression than paragraph mode.

---

### Processing pipeline (`processor.py`)

```
document_text
    │
    ▼
┌─────────────────────────────────┐
│  Phase 1 — Preprocessing        │  (skipped if preprocessing_prompt is empty)
│  call_ai(preprocessing_prompt   │
│           + full document)      │
│  → preprocessed: str            │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Phase 2 — Node Processing      │  (skipped if node_prompt is empty)
│                                 │
│  parse document into nodes      │
│  (paragraph or section mode)    │
│                                 │
│  pre-pass: apply merge_next     │
│            / part_of_section    │
│                                 │
│  for each node:                 │
│    apply heading/non-text rule  │
│    → discard / passthrough      │
│    → call_ai(node_prompt        │
│               + node.content,   │
│               messages=[preproc]│  (if include_preprocessed_context)
│              )                  │
│    → optional cosine similarity │
│    → append to results          │
│                                 │
│  → results: str                 │
│  → node_results: list[NodeResult│
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Phase 3 — Postprocessing       │  (skipped if postprocessing_prompt is empty)
│  prompt = postprocessing_prompt │
│         + [input doc]           │  (if postprocess_attach_input)
│         + [preprocessed]        │  (if postprocess_attach_preprocessed)
│         + [results]             │  (if postprocess_attach_results)
│  call_ai(prompt)                │
│  → postprocessed: str           │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Output composition             │
│  main output = one of:          │
│    preprocessed / results /     │
│    postprocessed                │
│  optionally write separate      │
│  files for preprocessed and     │
│  postprocessed                  │
└─────────────────────────────────┘
```

#### Prompt Composition

All prompts sent to the LLM are constructed using simple string concatenation. No placeholder replacement or templating engine is applied (e.g., `{node}` or `{document}` are treated as literal text).

- **Phase 1 (Preprocessing)**:
  ```
  [preprocessing_prompt]

  [full document text]
  ```
  *(concatenated as `f"{preprocessing_prompt}\n\n{document_text}"`)*

- **Phase 2 (Node Processing)**:
  ```
  [node_prompt]

  [node content text]
  ```
  *(concatenated as `f"{node_prompt}\n\n{node.content}"`)*

- **Phase 3 (Postprocessing)**:
  The prompt joins the `postprocessing_prompt` with the selected attachments in order, separated by newlines:
  ```
  [postprocessing_prompt]

  === ORIGINAL DOCUMENT ===
  [document_text]

  === PREPROCESSED ===
  [preprocessed]

  === NODE RESULTS ===
  [results]
  ```

#### Separator format (Phase 2)

When `show_similarity_separator` is `True`, each node result is preceded by:

```
# §{node_id} — Cosine similarity: {score:.3f} {emoji}
```

where `emoji` is `🟢` (≥ 0.85), `🟡` (≥ 0.70), or `🔴` (< 0.70).
The separator is omitted when `show_similarity_separator` is `False`.
The similarity part is omitted when `compute_similarity` is `False`.

#### Interlacing original and processed nodes

When `interlace_input_output` is `True` in the preset, each processed node outputs both its original text and the AI-generated response, formatted as follows:

```markdown
**Original:**
[original node content]

**Processed:**
[AI response output]
```

Nodes that were not sent to the AI (e.g. headings or code blocks passed through unchanged based on handling rules) are printed normally without the `**Original:**` / `**Processed:**` wrappers.

#### Context injection (Phase 2)

When `include_preprocessed_context` is `True` and Phase 1 ran, each node call
is made with:

```python
messages = [
    {"role": "user",      "content": preprocessing_prompt},
    {"role": "assistant", "content": preprocessed},
]
```

This injects the preprocessed result as a prior conversation turn so the model
has document-level context without repeating it verbatim in every node prompt.

---

### AI client integration

All AI calls go through `UnifiedAiClient.call_ai()` and `get_embedding()`.
No provider dispatch, retry logic, or HTTP calls exist in Iter itself.

```python
from unified_ai_client import call_ai, get_embedding

# Main generation call
response = call_ai(
    provider=config.provider,
    model=config.model_name,
    prompt=...,
    system_prompt=...,
    messages=...,        # optional context injection
    temperature=config.temperature,
)

# Embedding for cosine similarity
vector = get_embedding(
    provider=config.provider,
    model=config.embedding_model,
    text=node.content,
)
```

`UnifiedAiClient` lives at `../UnifiedAiClient` (editable install in dev,
pinned Git tag in prod). It handles retry logic, provider dispatch, and
Gemini file cleanup automatically.

---

### Logging (`logger.py`)

Iter uses a dual-backend logger that routes output to either the terminal
(CLI / MCP mode) or Streamlit HTML widgets (GUI mode).

```python
import itertool.logger as logger

logger.set_backend("streamlit")   # called once at GUI startup
logger.log_action("Processing node 3/12")
logger.log_ai("Response: 142 chars")
logger.log_metric("Similarity: 0.921 🟢")
logger.log_success("Phase 2 complete.")
```

The GUI calls `logger.set_backend("streamlit")` at startup so all business
logic in `processor.py` automatically routes its output to the Streamlit
sidebar log without any code changes.

---

### Streamlit GUI layout (`gui/app.py`)

```
┌────────────────────┬──────────────────────────────────────────────┐
│  SIDEBAR           │  MAIN AREA                                   │
│                    │                                              │
│  ⚙️ Model Settings  │  📄 Files                                    │
│    Provider        │    Input file  [path input] [📁]             │
│    Model           │    Output file [path input] [💾]             │
│    Embedding model │                                              │
│    Temperature     │  ▸ Phase 1 — Preprocessing (expander)        │
│    Ollama URL      │    [text area for preprocessing_prompt]      │
│    Thinking [✓]   │                                              │
│                    │  ▸ Phase 2 — Node Processing (expander)      │
│  📂 Presets        │    [text area for node_prompt]               │
│    System prompt   │    node mode: ◉ Paragraph  ○ Section         │
│    [dropdown]      │    ☐ Include preprocessed as context         │
│    [Load] [Save]   │    ☐ Compute cosine similarity               │
│    [name input]    │    ☐ Show §N separator                       │
│                    │    Heading handling:    [dropdown]           │
│                    │    Non-text handling:   [dropdown]           │
│                    │                                              │
│                    │  ▸ Phase 3 — Postprocessing (expander)       │
│                    │    [text area for postprocessing_prompt]     │
│                    │    Attach: ☐ input  ☐ preprocessed  ☐ results│
│                    │                                              │
│                    │  ▸ Output Settings (expander)                │
│                    │    Output file includes:                     │
│                    │      ☐ Phase 1  ☑ Phase 2  ☐ Phase 3        │
│                    │    ☐ Save Phase 1 to separate file           │
│                    │    ☐ Save Phase 3 to separate file           │
│                    │                                              │
│                    │  [▶ Run]                                     │
│                    │                                              │
│                    │  📊 Results (after run)                      │
│                    │    ▸ Phase 1 — Preprocessed                  │
│                    │    ▸ Phase 2 — Node Results  ← expanded      │
│                    │    ▸ Phase 3 — Postprocessed                  │
└────────────────────┴──────────────────────────────────────────────┘
```

Session state persists all widget values across Streamlit reruns.
The sidebar **Load** button reads a preset JSON and writes all its fields
into `st.session_state`, then calls `st.rerun()`. The **Save** button
reads `st.session_state` and writes a `Preset` JSON file to `Presets/`.
