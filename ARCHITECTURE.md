# Architecture

Iter applies one LLM prompt to every node of a Markdown document. The problem it solves is
that a whole document rarely fits a single useful prompt: a 200-page manual condensed in one
call loses everything specific, while the same prompt applied paragraph by paragraph keeps the
detail. So Iter splits, iterates, and reassembles.

Four interfaces (CLI, MCP, GUI, library import) fan into one function, `run_iter()` in
`processor.py`. Each builds a `Preset` and an `AppConfig` and calls it. Behaviour lives in the
processor; the interface layers only translate their own input into those two objects.

## Invariants

These hold across the whole codebase. Breaking one is a bug, not a style choice.

- **Business logic never imports `streamlit`.** The GUI is one of four frontends, and
  `processor.py` must stay usable without it.
- **Nothing reachable from `iter mcp` writes to stdout.** In stdio transport, stdout carries the
  JSON-RPC framing; a stray `print` corrupts the session. The `mcp` logger backend routes the
  identical output to stderr.
- **`AppConfig.apply_to_client()` is idempotent.** It may be called at interface startup and
  again before every run; unchanged settings register nothing.
- **A phase is skipped purely by its prompt being empty.** There is no enable flag anywhere.
- **`Node.index` is 1-based and global** across both node modes.
- **User files resolve against the working directory**, never against the package. Only the
  static assets Iter ships with itself are package-relative.

## Reference tables

The exhaustive field tables live beside this document:

- [docs/configuration.md](docs/configuration.md) - every `AppConfig` field and both JSON files.
- [docs/presets.md](docs/presets.md) - every `Preset` field and both handling value sets.

### Project layout

```
Iter/
├── itertool/                  # Python package (installable)
│   ├── __init__.py            # UTF-8 fix + re-exports run_iter
│   ├── __main__.py            # CLI entry point  →  iter run / mcp / gui
│   ├── checkpoint.py          # Progress file that makes runs resumable
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
├── secrets.json               # API keys - never commit
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

The two large field tables that used to sit here are in the reference documents linked
above. What follows are the models that are not user-configurable.

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
| `node_results` | `list[NodeResult]` - one per node processed |
| `results` | Phase 2 assembled string (nodes joined with optional separators) |
| `postprocessed` | Phase 3 AI output (empty string if skipped) |

---

### Markdown parsing (`markdown_utils.py`)

The raw document is first split into **logical blocks** by blank lines, with fenced
code blocks kept intact (blank lines inside ` ``` ` fences are not split points).

#### Paragraph mode
Each block becomes exactly one `Node`. Block type is classified with:

- `is_header()` - starts with `#`
- `is_table()` - first two lines contain `|` and `-`
- `is_code_block()` - starts with ` ``` `
- `is_image_or_link_only()` - matches `!?[…](…)` alone on a line

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
│  Phase 1 - Preprocessing        │  (skipped if preprocessing_prompt is empty)
│  call_ai(preprocessing_prompt   │
│           + full document)      │
│  → preprocessed: str            │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Phase 2 - Node Processing      │  (skipped if node_prompt is empty)
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
│  Phase 3 - Postprocessing       │  (skipped if postprocessing_prompt is empty)
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
# §{node_id} - Cosine similarity: {score:.3f} {emoji}
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

#### Resumable runs (`checkpoint.py`)

`run_iter()` opens a `Checkpoint` before Phase 1 and deletes it after the outputs are written.
The progress file is JSON Lines: a header holding the run's fingerprint, then one record per
completed AI call (`pre`, `node`, `post`), appended and flushed to disk as soon as the call
returns. Pass-through nodes are not recorded: they cost nothing to recompute.

- **Fingerprint.** Hashes of the input and of the preset, plus the generation-relevant model
  settings. Preset fields that only shape the output file are excluded, so changing them
  never invalidates saved work.
- **Node keys.** A node is keyed by the hash of its content (after the pre-passes) and its
  occurrence number, not by its index. When the user continues after editing the input, the
  unchanged nodes are still found and only the edited ones are redone.
- **Mismatch.** `on_mismatch="ask"` raises `CheckpointMismatchError` before any AI call; each
  interface catches it, asks the user, and calls `run_iter()` again with `"continue"` or
  `"restart"`. Both rewrite the header, so the next run does not ask again.
- **Phase 3 is redone when any node was.** It consolidates the node results, so a saved Phase 3
  result is dropped as soon as Phase 2 made a new call.
- **Damage.** A truncated last line, the trace of a hard kill, is skipped; the file is then
  rewritten before anything is appended, since a record added after a partial line would be
  lost with it.

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

Iter uses a four-backend logger. `console` writes to stdout (the CLI default),
`mcp` writes the identical format to stderr, `silent` discards everything, and
`streamlit` renders inline HTML widgets. The backend is selected once per
interface, never per call site.

```python
import itertool.logger as logger

logger.set_backend("streamlit")   # called once at GUI startup
logger.log_action("Processing node 3/12")
logger.log_ai("Response: 142 chars")
logger.log_metric("Similarity: 0.921 🟢")
logger.log_success("Phase 2 complete.")
```

The GUI calls `logger.set_backend("streamlit")` at startup, so all business
logic in `processor.py` routes its output to the Streamlit log without any code
changes. `iter mcp` calls `logger.set_backend("mcp")` before the server starts
serving: stdout carries the JSON-RPC framing, so nothing reachable from a tool
call may write there. Library callers that handle their own output can select
`silent`.

---

### Streamlit GUI layout (`gui/app.py`)

```
┌────────────────────┬──────────────────────────────────────────────┐
│  SIDEBAR           │  MAIN AREA                                   │
│                    │                                              │
│  ⚙️ Model Settings  │  📄 Files                                    │
│    Provider    [▾] │    Input file  [path input] [📁]             │
│    Model       [▾] │    Output file [path input] [💾]             │
│    Embedding   [▾] │                                              │
│     provider/model │  ▸ Phase 1 - Preprocessing (expander)        │
│    [🔄 Refresh]    │    [text area for preprocessing_prompt]      │
│    Temperature     │                                              │
│    Thinking [✓]   │                                              │
│    Timeout,        │                                              │
│    context size,   │                                              │
│    keep alive      │                                              │
│                    │  ▸ Phase 2 - Node Processing (expander)      │
│  📂 Presets        │    [text area for node_prompt]               │
│    System prompt   │    node mode: ◉ Paragraph  ○ Section         │
│    [dropdown]      │    ☐ Include preprocessed as context         │
│    [Load] [Save]   │    ☐ Compute cosine similarity               │
│    [name input]    │    ☐ Show §N separator                       │
│                    │    Heading handling:    [dropdown]           │
│                    │    Non-text handling:   [dropdown]           │
│                    │                                              │
│                    │  ▸ Phase 3 - Postprocessing (expander)       │
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
│                    │    ▸ Phase 1 - Preprocessed                  │
│                    │    ▸ Phase 2 - Node Results  ← expanded      │
│                    │    ▸ Phase 3 - Postprocessed                  │
└────────────────────┴──────────────────────────────────────────────┘
```

Session state persists all widget values across Streamlit reruns.
The sidebar **Load** button reads a preset JSON and writes all its fields
into `st.session_state`, then calls `st.rerun()`. The **Save** button
reads `st.session_state` and writes a `Preset` JSON file to `Presets/`.
