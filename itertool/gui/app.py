"""Streamlit GUI for Iter - Iterative Node-Based Document Processor."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable when launched via 'streamlit run gui/app.py'
_PKG_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PKG_ROOT))

import streamlit as st

from unified_ai_client import list_models, list_providers, silence_sdks

import itertool.logger as logger
from itertool.checkpoint import CheckpointMismatchError
from itertool.config import KEEP_ALIVE_FOREVER, AppConfig
from itertool.gui.file_dialog import ask_path
from itertool.models import Preset
from itertool.processor import run_iter

logger.set_backend("streamlit")
silence_sdks()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Iter - Node Processor",
    page_icon="🔁",
    layout="wide",
)

# ── Presets directory ─────────────────────────────────────────────────────────

# Always use the current working directory's Presets folder to remain
# consistent with the CLI and support non-editable production installs.
_PRESETS_DIR = Path.cwd() / "Presets"
_PRESETS_DIR.mkdir(exist_ok=True)

_DEFAULT_TEMPERATURE = 0.7

_FILETYPES = [("Markdown Files", "*.md"), ("Text Files", "*.txt"), ("All Files", "*.*")]

_HEADING_OPTIONS = {
    "report_unprocessed": "Pass through unchanged",
    "discard": "Discard",
    "standalone_nodes": "Treat as standalone nodes",
    "merge_next": "Merge with next node",
}

_NON_TEXT_OPTIONS = {
    "report_unprocessed": "Pass through unchanged",
    "discard": "Discard",
    "standalone_nodes": "Treat as standalone nodes",
    "part_of_section": "Include in section",
}

_IMAGE_OPTIONS = {
    "report_unprocessed": "Pass through unchanged",
    "discard": "Discard",
    "standalone_nodes": "Treat as standalone nodes",
    "part_of_section": "Include in section",
}

_NODE_MODE_LABELS = {
    "paragraph": "🔹 Paragraph",
    "section": "🔸 Section",
    "file": "📄 File",
}

_SCOPE_LABELS = {
    "file": "Per file",
    "corpus": "Whole folder",
}

_LAYOUT_LABELS = {
    "mirror": "Mirror (one output per input file)",
    "merged": "Merged (single file)",
}

_DIALOG_BUTTONS = {"open": "📁", "save": "💾", "folder": "🗂️"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _list_presets() -> list[str]:
    """Return sorted list of preset names (without .json extension)."""
    return sorted(p.stem for p in _PRESETS_DIR.glob("*.json"))


def _choice(labels: dict[str, str], label: str) -> str:
    """Return the value whose display label is label, in a value -> label mapping."""
    return next(value for value, shown in labels.items() if shown == label)


def _file_picker(label: str, key: str, dialogs: tuple[str, ...] = ("open",), help: str | None = None) -> str:
    """Render a text input with one button per native dialog.

    Args:
        label: Text input label.
        key: Unique Streamlit widget key prefix.
        dialogs: The dialogs offered, one button each: 'open' (📁, an existing
            file), 'save' (💾, a file to write) or 'folder' (🗂️).
        help: Tooltip of the text input.

    Returns:
        The selected path string (may be empty).
    """
    col_input, *col_buttons = st.columns([5, *([1] * len(dialogs))])
    with col_input:
        path_val = st.text_input(label, value=st.session_state.get(key, ""), key=f"{key}_input", help=help)
        st.session_state[key] = path_val
    for column, dialog in zip(col_buttons, dialogs):
        with column:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button(_DIALOG_BUTTONS[dialog], key=f"{key}_btn_{dialog}", use_container_width=True):
                selected = ask_path(
                    st.session_state.get(key, ""), _FILETYPES,
                    save=dialog == "save", default_ext=".md", directory=dialog == "folder",
                )
                if selected:
                    st.session_state[key] = selected
                    st.rerun()
    return st.session_state.get(key, "")


@st.cache_data(show_spinner=False)
def _available_providers(embeddings: bool) -> list[str]:
    """Names of the providers that can be used now: local servers that answer,
    cloud providers with a key.

    Args:
        embeddings: If True, only the providers that support embeddings.
    """
    return [p.name for p in list_providers() if p.supports_embeddings or not embeddings]


@st.cache_data(show_spinner=False)
def _available_models(provider: str, embeddings: bool) -> tuple[list[str], str]:
    """Model ids offered by a provider, and the error if they could not be listed.

    Args:
        provider: Provider name.
        embeddings: If True, leave out the models known not to embed. A model
            whose capability is unknown is kept.

    Returns:
        (model ids, '') on success, ([], error message) on failure.
    """
    try:
        models = list_models(provider)
    except Exception as exc:
        return [], str(exc) or type(exc).__name__
    return [m.id for m in models if m.embeddings is not False or not embeddings], ""


def _provider_picker(label: str, key: str, model_key: str, embeddings: bool) -> None:
    """Render a provider selectbox bound to st.session_state[key].

    A saved provider that is not available right now stays selectable, with a
    warning, so the menu never changes the saved value silently.

    Args:
        label: Widget label.
        key: Session state key of the provider.
        model_key: Session state key of the paired model, cleared on change so
            the model menu falls back to the new provider's first model.
        embeddings: If True, offer only the providers that support embeddings.
    """
    options = _available_providers(embeddings)
    current = st.session_state.get(key, "")
    if current and current not in options:
        options = [current, *options]
        st.caption(f"⚠️ '{current}' is not available now: server not running or no API key.")
    st.selectbox(
        label,
        options=options,
        key=key,
        on_change=st.session_state.update,
        kwargs={model_key: ""},
    )


def _model_picker(label: str, key: str, provider: str, embeddings: bool) -> None:
    """Render a model selectbox bound to st.session_state[key].

    Falls back to a text input when the provider cannot list its models (server
    down, script provider), so a model can always be entered.

    Args:
        label: Widget label.
        key: Session state key of the model.
        provider: Provider whose models are offered.
        embeddings: If True, offer only the models that can embed.
    """
    models, error = _available_models(provider, embeddings)
    if not models:
        st.text_input(label, key=key)
        reason = error.rstrip(".") or "the provider offers none"
        st.caption(f"⚠️ Models of '{provider}' could not be listed ({reason}). Type the model name.")
        return

    current = st.session_state.get(key, "")
    options = models
    if not current:
        st.session_state[key] = models[0]
    elif current not in models:
        options = [current, *models]
        st.caption(f"⚠️ '{current}' is not offered by '{provider}' now.")
    st.selectbox(label, options=options, key=key)


def _start_input() -> str:
    """The input given by ``iter gui <file>`` (``--input`` after ``--``), or ''.

    Other arguments, such as a test runner's, are ignored.
    """
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--input", default="")
    return parser.parse_known_args()[0].input


def _init_state() -> None:
    """Initialize all session state keys with defaults."""
    cfg = AppConfig.load()
    defaults: dict = {
        # File paths
        "input_path": _start_input(),
        "output_path": "",
        "context_path": "",
        # Model settings (bound directly to widget keys)
        "provider": cfg.provider,
        "model_name": cfg.model_name,
        "embedding_provider": cfg.embedding_provider,
        "embedding_model": cfg.embedding_model,
        # None leaves the temperature to the provider; the slider keeps a value
        # either way, so switching the toggle on restores the last one.
        "force_temperature": cfg.temperature is not None,
        "temperature": cfg.temperature if cfg.temperature is not None else _DEFAULT_TEMPERATURE,
        "thinking": cfg.thinking,
        "timeout": cfg.timeout,
        # 0 is the widget's way of saying "leave the model default" (None).
        "context_size": cfg.context_size or 0,
        "keep_alive_minutes": cfg.keep_alive_minutes,
        # Preset fields (bound directly to widget keys)
        "system_prompt": (
            "You are a professional editor. "
            "Respond only with the requested output, no greetings, no final questions."
        ),
        "preset_name": "",
        "node_mode": "paragraph",
        "preprocessing_prompt": "",
        "node_prompt": "",
        "postprocessing_prompt": "",
        "postprocessing_scope": "file",
        "include_preprocessed_context": False,
        "heading_handling": "report_unprocessed",
        "non_text_handling": "report_unprocessed",
        "image_handling": "report_unprocessed",
        "compute_similarity": False,
        "show_similarity_separator": True,
        "postprocess_attach_input": False,
        "postprocess_attach_preprocessed": False,
        "postprocess_attach_results": True,
        "output_include_preprocessed": False,
        "output_include_processed": True,
        "output_include_postprocessed": False,
        "save_preprocessed_separately": False,
        "save_postprocessed_separately": False,
        "interlace_input_output": False,
        "output_layout": "mirror",
        # Widget mapping helper keys (to avoid default value mismatch warnings)
        "radio_node_mode": _NODE_MODE_LABELS["paragraph"],
        "radio_postprocessing_scope": _SCOPE_LABELS["file"],
        "radio_output_layout": _LAYOUT_LABELS["mirror"],
        "sel_heading_handling": "Pass through unchanged",
        "sel_non_text_handling": "Pass through unchanged",
        "sel_image_handling": "Pass through unchanged",
        # Results
        "result_preprocessed": "",
        "result_processed": "",
        "result_postprocessed": "",
        "result_node_results": [],
        # Status messages
        "preset_load_success": "",
        "preset_load_error": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _load_preset_into_state(preset: Preset) -> None:
    """Write all Preset fields into st.session_state, syncing both properties and widget keys."""
    # System prompt
    st.session_state.system_prompt = preset.system_prompt or st.session_state.system_prompt

    # Node mode
    st.session_state.node_mode = preset.node_mode
    st.session_state.radio_node_mode = _NODE_MODE_LABELS.get(preset.node_mode, _NODE_MODE_LABELS["paragraph"])

    # Prompts
    st.session_state.preprocessing_prompt = preset.preprocessing_prompt
    st.session_state.node_prompt = preset.node_prompt
    st.session_state.postprocessing_prompt = preset.postprocessing_prompt

    # Phase 2 Options
    st.session_state.include_preprocessed_context = preset.include_preprocessed_context
    st.session_state.compute_similarity = preset.compute_similarity
    st.session_state.show_similarity_separator = preset.show_similarity_separator

    # Handlings
    st.session_state.heading_handling = preset.heading_handling
    st.session_state.sel_heading_handling = _HEADING_OPTIONS.get(
        preset.heading_handling, list(_HEADING_OPTIONS.values())[0]
    )

    st.session_state.non_text_handling = preset.non_text_handling
    st.session_state.sel_non_text_handling = _NON_TEXT_OPTIONS.get(
        preset.non_text_handling, list(_NON_TEXT_OPTIONS.values())[0]
    )

    st.session_state.image_handling = preset.image_handling
    st.session_state.sel_image_handling = _IMAGE_OPTIONS.get(
        preset.image_handling, list(_IMAGE_OPTIONS.values())[0]
    )

    # Phase 3 Options
    st.session_state.postprocessing_scope = preset.postprocessing_scope
    st.session_state.radio_postprocessing_scope = _SCOPE_LABELS.get(
        preset.postprocessing_scope, _SCOPE_LABELS["file"]
    )
    st.session_state.postprocess_attach_input = preset.postprocess_attach_input
    st.session_state.postprocess_attach_preprocessed = preset.postprocess_attach_preprocessed
    st.session_state.postprocess_attach_results = preset.postprocess_attach_results

    # Output Options
    st.session_state.output_include_preprocessed = preset.output_include_preprocessed
    st.session_state.output_include_processed = preset.output_include_processed
    st.session_state.output_include_postprocessed = preset.output_include_postprocessed
    st.session_state.save_preprocessed_separately = preset.save_preprocessed_separately
    st.session_state.save_postprocessed_separately = preset.save_postprocessed_separately
    st.session_state.interlace_input_output = preset.interlace_input_output
    st.session_state.output_layout = preset.output_layout
    st.session_state.radio_output_layout = _LAYOUT_LABELS.get(preset.output_layout, _LAYOUT_LABELS["mirror"])


def _state_to_preset() -> Preset:
    """Build a Preset from the current session state."""
    return Preset(
        name=st.session_state.preset_name,
        system_prompt=st.session_state.system_prompt,
        node_mode=st.session_state.node_mode,
        preprocessing_prompt=st.session_state.preprocessing_prompt,
        node_prompt=st.session_state.node_prompt,
        postprocessing_prompt=st.session_state.postprocessing_prompt,
        postprocessing_scope=st.session_state.postprocessing_scope,
        include_preprocessed_context=st.session_state.include_preprocessed_context,
        heading_handling=st.session_state.heading_handling,
        non_text_handling=st.session_state.non_text_handling,
        image_handling=st.session_state.image_handling,
        compute_similarity=st.session_state.compute_similarity,
        show_similarity_separator=st.session_state.show_similarity_separator,
        postprocess_attach_input=st.session_state.postprocess_attach_input,
        postprocess_attach_preprocessed=st.session_state.postprocess_attach_preprocessed,
        postprocess_attach_results=st.session_state.postprocess_attach_results,
        output_include_preprocessed=st.session_state.output_include_preprocessed,
        output_include_processed=st.session_state.output_include_processed,
        output_include_postprocessed=st.session_state.output_include_postprocessed,
        save_preprocessed_separately=st.session_state.save_preprocessed_separately,
        save_postprocessed_separately=st.session_state.save_postprocessed_separately,
        interlace_input_output=st.session_state.interlace_input_output,
        output_layout=st.session_state.output_layout,
    )


def _state_to_config() -> AppConfig:
    """Build an AppConfig from the current session state."""
    return AppConfig(
        provider=st.session_state.provider,
        model_name=st.session_state.model_name,
        embedding_provider=st.session_state.embedding_provider,
        embedding_model=st.session_state.embedding_model,
        temperature=st.session_state.temperature if st.session_state.force_temperature else None,
        thinking=st.session_state.thinking,
        timeout=st.session_state.timeout,
        context_size=st.session_state.context_size or None,
        keep_alive_minutes=st.session_state.keep_alive_minutes,
    )


# ── Init ──────────────────────────────────────────────────────────────────────

_init_state()


def _on_load_preset() -> None:
    """Callback executed before rendering to load a preset safely into session state."""
    selected = st.session_state.get("sb_preset_select")
    if selected and selected != "- Select -":
        try:
            preset = Preset.load(_PRESETS_DIR / f"{selected}.json")
            _load_preset_into_state(preset)
            st.session_state.preset_name = selected
            st.session_state.preset_load_success = f"Loaded '{selected}'"
            st.session_state.preset_load_error = ""
        except Exception as e:
            st.session_state.preset_load_error = f"Load failed: {e}"
            st.session_state.preset_load_success = ""

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🔁 Iter")
    st.caption("Iterative node-based document processor")

    st.divider()

    # ── Model Settings ────────────────────────────────────────────────────────
    st.header("⚙️ Model Settings")

    _provider_picker("Provider", "provider", "model_name", embeddings=False)
    _model_picker("Model", "model_name", st.session_state.provider, embeddings=False)
    _provider_picker("Embedding provider", "embedding_provider", "embedding_model", embeddings=True)
    _model_picker("Embedding model", "embedding_model", st.session_state.embedding_provider, embeddings=True)

    if st.button("🔄 Refresh providers and models", use_container_width=True, key="btn_refresh_models"):
        _available_providers.clear()
        _available_models.clear()
        st.rerun()

    st.toggle(
        "Force temperature",
        key="force_temperature",
        help="When off, the provider default applies (for Ollama, the Modelfile).",
    )

    st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        step=0.05,
        key="temperature",
        disabled=not st.session_state.force_temperature,
    )

    st.toggle(
        "Thinking / reasoning mode",
        key="thinking",
        help="Enable extended thinking where the provider supports it (e.g. Claude 3.7, Gemini 2.5).",
    )

    st.number_input(
        "Request timeout (s)",
        min_value=1,
        step=10,
        key="timeout",
        help="Per-attempt timeout for every AI call. Failed attempts are retried.",
    )

    st.number_input(
        "Context size (tokens)",
        min_value=0,
        step=1024,
        key="context_size",
        help="Ollama context window (num_ctx). 0 leaves the model default in place.",
    )

    st.number_input(
        "Keep alive (min)",
        min_value=KEEP_ALIVE_FOREVER,
        step=5,
        key="keep_alive_minutes",
        help="How long Ollama keeps the model loaded once idle. "
             "-1 keeps it loaded forever, 0 unloads it after each request.",
    )

    if st.button("💾 Save as default", use_container_width=True, key="btn_save_config"):
        try:
            target = Path.cwd() / "config.json"
            _state_to_config().save(target)
            st.success(f"Saved to {target}")
        except Exception as e:
            st.error(f"Save failed: {e}")

    st.divider()

    # ── Presets ───────────────────────────────────────────────────────────────
    st.header("📂 Presets")
    st.caption("System prompt and all processing options are saved per preset.")

    st.text_area(
        "System prompt",
        height=100,
        key="system_prompt",
        help="Saved with the preset. Leave empty to use no system prompt.",
    )

    preset_names = _list_presets()
    st.selectbox(
        "Load preset",
        options=["- Select -"] + preset_names,
        index=0,
        key="sb_preset_select",
    )

    col_load, col_save = st.columns(2)
    with col_load:
        st.button(
            "⬇ Load",
            use_container_width=True,
            key="btn_load_preset",
            on_click=_on_load_preset,
        )

    with col_save:
        if st.button("⬆ Save", use_container_width=True, key="btn_save_preset"):
            name = st.session_state.preset_name.strip()
            if name:
                try:
                    _state_to_preset().save(_PRESETS_DIR / f"{name}.json")
                    st.success(f"Saved '{name}'")
                except Exception as e:
                    st.error(f"Save failed: {e}")
            else:
                st.warning("Enter a preset name below first.")

    # Show load status messages if any
    if st.session_state.get("preset_load_success"):
        st.success(st.session_state.preset_load_success)
        st.session_state.preset_load_success = ""  # clear after showing
    if st.session_state.get("preset_load_error"):
        st.error(st.session_state.preset_load_error)
        st.session_state.preset_load_error = ""  # clear after showing

    st.session_state.preset_name = st.text_input(
        "Preset name (for saving)",
        value=st.session_state.preset_name,
        placeholder="my_preset",
        key="sb_preset_name",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    "<h1 style='margin-bottom:0'>🔁 Iter <span style='font-size:0.5em;color:gray'>Node Processor</span></h1>",
    unsafe_allow_html=True,
)
st.caption("Apply a prompt to every paragraph or section of a Markdown document, or of a folder of them.")

# ── File I/O ──────────────────────────────────────────────────────────────────

st.subheader("📄 Files")
col_in, col_out = st.columns(2)
with col_in:
    input_path = _file_picker(
        "Input file or folder", "input_path", ("open", "folder"),
        help="A folder contributes the .md and .markdown files directly inside it.",
    )
# A folder in the mirror layout writes one file per input file, into an output folder.
# The layout is read from its radio, which already holds this rerun's choice.
mirror_folder = (
    bool(input_path) and Path(input_path).is_dir()
    and _choice(_LAYOUT_LABELS, st.session_state.radio_output_layout) == "mirror"
)
with col_out:
    if mirror_folder:
        output_path = _file_picker("Output folder", "output_path", ("folder",))
    else:
        output_path = _file_picker("Output file", "output_path", ("save",))
col_context, _ = st.columns(2)
with col_context:
    context_path = _file_picker(
        "Reference context (optional)", "context_path", ("open",),
        help="A text file, such as a glossary, appended to the system prompt of every AI call.",
    )

st.divider()

# ── Phase 1: Preprocessing ────────────────────────────────────────────────────

with st.expander("**Phase 1 - Preprocessing** *(optional)*", expanded=bool(st.session_state.preprocessing_prompt)):
    st.caption("Single AI call on the entire document. Leave empty to skip.")
    st.text_area(
        "Preprocessing prompt",
        height=120,
        label_visibility="collapsed",
        placeholder="E.g.: Extract a structured outline of this document.",
        key="preprocessing_prompt",
    )

# ── Phase 2: Node Processing ──────────────────────────────────────────────────

with st.expander("**Phase 2 - Node Processing** *(core)*", expanded=True):
    st.caption("This prompt is applied independently to each node. Leave empty to skip.")
    st.text_area(
        "Node prompt",
        height=140,
        label_visibility="collapsed",
        placeholder="E.g.: Condense the following section without losing information.",
        key="node_prompt",
    )

    # Node mode radio
    mode_val = st.radio(
        "Node granularity",
        options=list(_NODE_MODE_LABELS.values()),
        horizontal=True,
        key="radio_node_mode",
        help="'File' sends each input file whole, as one node.",
    )
    st.session_state.node_mode = _choice(_NODE_MODE_LABELS, mode_val)

    st.markdown("**Options:**")
    opt_col1, opt_col2 = st.columns(2)

    with opt_col1:
        st.checkbox(
            "Include preprocessed as prior context",
            key="include_preprocessed_context",
            help="Inject the Phase 1 result as a prior assistant message when calling the node prompt.",
        )
        st.checkbox(
            "Compute cosine similarity",
            key="compute_similarity",
            help="Compute embedding cosine similarity between original node and AI result.",
        )
        st.checkbox(
            "Show §N separator between results",
            key="show_similarity_separator",
            help="Insert '# §N - Cosine similarity: score' before each result.",
        )

    with opt_col2:
        if st.session_state.node_mode == "file":
            st.caption("Each node is a whole file, so the heading, non-text and image rules do not apply.")
        else:
            # part_of_section absorbs a block into the section before it, so it is
            # meaningless in paragraph mode, where there are no sections.
            for field_name, all_options, label, help_text in (
                ("heading_handling", _HEADING_OPTIONS, "Heading handling", None),
                ("non_text_handling", _NON_TEXT_OPTIONS, "Non-text handling (tables, code)", None),
                (
                    "image_handling", _IMAGE_OPTIONS, "Image handling",
                    "'Treat as standalone nodes' sends the image file to the model, "
                    "resolved from the input file's folder or downloaded from its URL. "
                    "'Include in section' adds only the Markdown text to the section.",
                ),
            ):
                options = dict(all_options)
                if st.session_state.node_mode != "section":
                    options.pop("part_of_section", None)
                labels = list(options.values())
                keys = list(options.keys())
                sel_key = f"sel_{field_name}"
                # A selectbox hidden in file mode loses its widget state; rebuild it
                # from the field, falling back to the default when the field's value
                # is not offered in this mode.
                if st.session_state.get(sel_key) not in labels:
                    st.session_state[sel_key] = options.get(
                        st.session_state[field_name], options["report_unprocessed"]
                    )
                selected = st.selectbox(label, options=labels, key=sel_key, help=help_text)
                st.session_state[field_name] = keys[labels.index(selected)]

# ── Phase 3: Postprocessing ───────────────────────────────────────────────────

with st.expander("**Phase 3 - Postprocessing** *(optional)*", expanded=bool(st.session_state.postprocessing_prompt)):
    st.caption("Single AI call after all nodes are processed. Leave empty to skip.")
    st.text_area(
        "Postprocessing prompt",
        height=120,
        label_visibility="collapsed",
        placeholder="E.g.: Review the condensed sections for consistency and produce a final report.",
        key="postprocessing_prompt",
    )

    st.markdown("**Attach to the postprocessing prompt:**")
    pp_col1, pp_col2, pp_col3 = st.columns(3)
    with pp_col1:
        st.checkbox(
            "Original input document",
            key="postprocess_attach_input",
        )
    with pp_col2:
        st.checkbox(
            "Preprocessed result",
            key="postprocess_attach_preprocessed",
        )
    with pp_col3:
        st.checkbox(
            "Node results",
            key="postprocess_attach_results",
        )

    scope_val = st.radio(
        "Scope, for a folder input",
        options=list(_SCOPE_LABELS.values()),
        horizontal=True,
        key="radio_postprocessing_scope",
        help="One call per input file, or one call for the whole folder "
             "with every file's attachments joined under its header.",
    )
    st.session_state.postprocessing_scope = _choice(_SCOPE_LABELS, scope_val)
    if st.session_state.postprocessing_scope == "corpus" and st.session_state.postprocess_attach_input:
        st.warning("Attaching the original input of every file to a single call rarely fits the model's context.")

# ── Output Settings ───────────────────────────────────────────────────────────

with st.expander("**Output Settings**", expanded=True):
    st.caption(
        "Select which phase results to include in the output file. "
        "All checked sections are appended in order, separated by `---`."
    )
    out_col1, out_col2, out_col3 = st.columns(3)
    with out_col1:
        st.checkbox(
            "Phase 1 - Preprocessed",
            key="output_include_preprocessed",
        )
    with out_col2:
        st.checkbox(
            "Phase 2 - Node results",
            key="output_include_processed",
        )
    with out_col3:
        st.checkbox(
            "Phase 3 - Postprocessed",
            key="output_include_postprocessed",
        )

    sep_col1, sep_col2 = st.columns(2)
    with sep_col1:
        st.checkbox(
            "Save Phase 1 to separate file",
            key="save_preprocessed_separately",
        )
    with sep_col2:
        st.checkbox(
            "Save Phase 3 to separate file",
            key="save_postprocessed_separately",
        )

    st.checkbox(
        "Interlace original and processed nodes",
        key="interlace_input_output",
        help="For each processed node, output the original text followed by the AI response.",
    )

    layout_val = st.radio(
        "Layout, for a folder input",
        options=list(_LAYOUT_LABELS.values()),
        horizontal=True,
        key="radio_output_layout",
        help="Mirror writes one file per input file into the output folder; "
             "merged writes a single file, each input file under its own header.",
    )
    st.session_state.output_layout = _choice(_LAYOUT_LABELS, layout_val)

st.divider()

# ── Run Button ────────────────────────────────────────────────────────────────

@st.dialog("Saved progress found")
def _ask_mismatch_policy(message: str) -> None:
    """Let the user choose what to do with a progress file from a different run."""
    st.write(message)
    col_continue, col_restart = st.columns(2)
    # Callbacks rather than return values: they record the choice before any
    # rerun, whether Streamlit reruns just this dialog or the whole page.
    col_continue.button(
        "Continue", help="Keep the saved results, redo only what is missing.",
        use_container_width=True, on_click=st.session_state.update, kwargs={"resume_policy": "continue"},
    )
    col_restart.button(
        "Restart", help="Discard the saved results and start from scratch.",
        use_container_width=True, on_click=st.session_state.update, kwargs={"resume_policy": "restart"},
    )
    if st.session_state.get("resume_policy"):
        st.rerun()


run_col, _ = st.columns([1, 4])
with run_col:
    run_btn = st.button("▶ Run", type="primary", use_container_width=True, key="btn_run")

# Set by the dialog: the run that raised the mismatch is started again with it.
resume_policy = st.session_state.pop("resume_policy", None)

if run_btn or resume_policy:
    if not input_path:
        st.error("Please provide an input file or folder.")
    elif not st.session_state.node_prompt.strip() and not st.session_state.preprocessing_prompt.strip():
        st.warning("At least one of the preprocessing or node prompt must be non-empty.")
    else:
        preset = _state_to_preset()
        config = _state_to_config()
        mismatch: CheckpointMismatchError | None = None

        with st.spinner("Processing..."):
            try:
                # The user can change provider or model between runs, so settings
                # are applied here, where they become effective. No-op when unchanged.
                config.apply_to_client()
                result = run_iter(
                    input_path=input_path,
                    output_file=output_path or None,
                    preset=preset,
                    config=config,
                    context_file=context_path or None,
                    on_mismatch=resume_policy or "ask",
                )
                st.session_state.result_preprocessed = result.preprocessed
                st.session_state.result_processed = result.results
                st.session_state.result_postprocessed = result.postprocessed
                st.session_state.result_node_results = result.node_results
                st.success("✅ Processing complete!")
            except CheckpointMismatchError as exc:
                mismatch = exc
            except Exception as exc:
                st.error(f"❌ Error: {exc}")

        if mismatch:
            _ask_mismatch_policy(str(mismatch))

# ── Results ───────────────────────────────────────────────────────────────────

if st.session_state.result_preprocessed or st.session_state.result_processed or st.session_state.result_postprocessed:
    st.divider()
    st.subheader("📊 Results")

    if st.session_state.result_preprocessed:
        with st.expander("🔹 Phase 1 - Preprocessed"):
            st.markdown(st.session_state.result_preprocessed)

    if st.session_state.result_processed:
        with st.expander("🔸 Phase 2 - Node Results", expanded=True):
            # Show per-node similarity summary if computed
            node_results = st.session_state.result_node_results
            if node_results and any(nr.similarity is not None for nr in node_results):
                sims = [nr.similarity for nr in node_results if nr.similarity is not None]
                avg_sim = sum(sims) / len(sims)
                st.info(f"Average cosine similarity: **{avg_sim:.3f}**  ({len(sims)} nodes measured)")
            st.markdown(st.session_state.result_processed)

    if st.session_state.result_postprocessed:
        with st.expander("🔷 Phase 3 - Postprocessed"):
            st.markdown(st.session_state.result_postprocessed)
