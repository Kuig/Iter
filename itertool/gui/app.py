"""Streamlit GUI for Iter — Iterative Node-Based Document Processor."""
from __future__ import annotations

import sys
import json
from pathlib import Path

# Ensure project root is importable when launched via 'streamlit run gui/app.py'
_PKG_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PKG_ROOT))

import streamlit as st

import itertool.logger as logger
from itertool.config import AppConfig
from itertool.models import Preset
from itertool.processor import run_iter

logger.set_backend("streamlit")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Iter — Node Processor",
    page_icon="🔁",
    layout="wide",
)

# ── Presets directory ─────────────────────────────────────────────────────────

_PRESETS_DIR = _PKG_ROOT / "Presets"
_PRESETS_DIR.mkdir(exist_ok=True)

_PROVIDERS = [
    "ollama", "google", "anthropic", "openai",
    "mistral", "cohere", "groq", "xai", "lmstudio", "llamacpp",
]

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
    "part_of_section": "Include in section (section mode only)",
}

_OUTPUT_CONTENT_OPTIONS = {
    "preprocessed": "Preprocessed (Phase 1)",
    "processed": "Processed — node results (Phase 2)",
    "postprocessed": "Postprocessed (Phase 3)",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _list_presets() -> list[str]:
    """Return sorted list of preset names (without .json extension)."""
    return sorted(p.stem for p in _PRESETS_DIR.glob("*.json"))


def _file_picker(label: str, key: str, save: bool = False) -> str:
    """Render a text input + 📁 button for file picking via tkinter.

    Args:
        label: Text input label.
        key: Unique Streamlit widget key prefix.
        save: If True, open a save-as dialog instead of open dialog.

    Returns:
        The selected file path string (may be empty).
    """
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        path_val = st.text_input(label, value=st.session_state.get(key, ""), key=f"{key}_input")
        st.session_state[key] = path_val
    with col_btn:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("📁" if not save else "💾", key=f"{key}_btn", use_container_width=True):
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.wm_attributes("-topmost", 1)
            init_dir = str(Path(st.session_state.get(key, "")).parent) or str(Path.cwd())
            if not Path(init_dir).is_absolute():
                init_dir = str(Path.cwd() / init_dir)
            if save:
                selected = filedialog.asksaveasfilename(
                    initialdir=init_dir,
                    defaultextension=".md",
                    filetypes=[("Markdown Files", "*.md"), ("Text Files", "*.txt"), ("All Files", "*.*")],
                )
            else:
                selected = filedialog.askopenfilename(
                    initialdir=init_dir,
                    filetypes=[("Markdown Files", "*.md"), ("Text Files", "*.txt"), ("All Files", "*.*")],
                )
            if selected:
                try:
                    rel = Path(selected).relative_to(Path.cwd())
                    st.session_state[key] = rel.as_posix()
                except ValueError:
                    st.session_state[key] = Path(selected).as_posix()
                st.rerun()
    return st.session_state.get(key, "")


def _init_state() -> None:
    """Initialize all session state keys with defaults."""
    cfg = AppConfig.load()
    defaults: dict = {
        # File paths
        "input_path": "",
        "output_path": "",
        # Model settings
        "provider": cfg.provider,
        "model_name": cfg.model_name,
        "embedding_provider": cfg.embedding_provider,
        "embedding_model": cfg.embedding_model,
        "temperature": cfg.temperature,
        "ollama_url": cfg.ollama_url,
        "thinking": cfg.thinking,
        # Preset fields
        "system_prompt": (
            "You are a professional editor. "
            "Respond only with the requested output, no greetings, no final questions."
        ),
        # Preset
        "preset_name": "",
        # Node mode
        "node_mode": "paragraph",
        # Phase prompts
        "preprocessing_prompt": "",
        "node_prompt": "",
        "postprocessing_prompt": "",
        # Phase 2 options
        "include_preprocessed_context": False,
        "heading_handling": "report_unprocessed",
        "non_text_handling": "report_unprocessed",
        "compute_similarity": False,
        "show_similarity_separator": True,
        # Phase 3 options
        "postprocess_attach_input": False,
        "postprocess_attach_preprocessed": False,
        "postprocess_attach_results": True,
        # Output options
        "output_include_preprocessed": False,
        "output_include_processed": True,
        "output_include_postprocessed": False,
        "save_preprocessed_separately": False,
        "save_postprocessed_separately": False,
        "interlace_input_output": False,
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
    st.session_state.sb_system_prompt = st.session_state.system_prompt

    # Node mode
    st.session_state.node_mode = preset.node_mode
    st.session_state.radio_node_mode = "🔹 Paragraph" if preset.node_mode == "paragraph" else "🔸 Section"

    # Prompts
    st.session_state.preprocessing_prompt = preset.preprocessing_prompt
    st.session_state.ta_preprocessing_prompt = preset.preprocessing_prompt

    st.session_state.node_prompt = preset.node_prompt
    st.session_state.ta_node_prompt = preset.node_prompt

    st.session_state.postprocessing_prompt = preset.postprocessing_prompt
    st.session_state.ta_postprocessing_prompt = preset.postprocessing_prompt

    # Phase 2 Options
    st.session_state.include_preprocessed_context = preset.include_preprocessed_context
    st.session_state.cb_include_preprocessed = preset.include_preprocessed_context

    st.session_state.compute_similarity = preset.compute_similarity
    st.session_state.cb_compute_similarity = preset.compute_similarity

    st.session_state.show_similarity_separator = preset.show_similarity_separator
    st.session_state.cb_show_separator = preset.show_similarity_separator

    # Handlings
    st.session_state.heading_handling = preset.heading_handling
    st.session_state.sel_heading_handling = _HEADING_OPTIONS.get(
        preset.heading_handling, list(_HEADING_OPTIONS.values())[0]
    )

    st.session_state.non_text_handling = preset.non_text_handling
    st.session_state.sel_non_text_handling = _NON_TEXT_OPTIONS.get(
        preset.non_text_handling, list(_NON_TEXT_OPTIONS.values())[0]
    )

    # Phase 3 Options
    st.session_state.postprocess_attach_input = preset.postprocess_attach_input
    st.session_state.cb_pp_attach_input = preset.postprocess_attach_input

    st.session_state.postprocess_attach_preprocessed = preset.postprocess_attach_preprocessed
    st.session_state.cb_pp_attach_preprocessed = preset.postprocess_attach_preprocessed

    st.session_state.postprocess_attach_results = preset.postprocess_attach_results
    st.session_state.cb_pp_attach_results = preset.postprocess_attach_results

    # Output Options
    st.session_state.output_include_preprocessed = preset.output_include_preprocessed
    st.session_state.cb_out_preprocessed = preset.output_include_preprocessed

    st.session_state.output_include_processed = preset.output_include_processed
    st.session_state.cb_out_processed = preset.output_include_processed

    st.session_state.output_include_postprocessed = preset.output_include_postprocessed
    st.session_state.cb_out_postprocessed = preset.output_include_postprocessed

    st.session_state.save_preprocessed_separately = preset.save_preprocessed_separately
    st.session_state.cb_save_preprocessed_sep = preset.save_preprocessed_separately

    st.session_state.save_postprocessed_separately = preset.save_postprocessed_separately
    st.session_state.cb_save_postprocessed_sep = preset.save_postprocessed_separately

    st.session_state.interlace_input_output = preset.interlace_input_output
    st.session_state.cb_interlace = preset.interlace_input_output


def _state_to_preset() -> Preset:
    """Build a Preset from the current session state."""
    return Preset(
        name=st.session_state.preset_name,
        system_prompt=st.session_state.system_prompt,
        node_mode=st.session_state.node_mode,
        preprocessing_prompt=st.session_state.preprocessing_prompt,
        node_prompt=st.session_state.node_prompt,
        postprocessing_prompt=st.session_state.postprocessing_prompt,
        include_preprocessed_context=st.session_state.include_preprocessed_context,
        heading_handling=st.session_state.heading_handling,
        non_text_handling=st.session_state.non_text_handling,
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
    )


def _state_to_config() -> AppConfig:
    """Build an AppConfig from the current session state."""
    return AppConfig(
        provider=st.session_state.provider,
        model_name=st.session_state.model_name,
        embedding_provider=st.session_state.embedding_provider,
        embedding_model=st.session_state.embedding_model,
        temperature=st.session_state.temperature,
        ollama_url=st.session_state.ollama_url,
        thinking=st.session_state.thinking,
    )


# ── Init ──────────────────────────────────────────────────────────────────────

_init_state()


def _on_load_preset() -> None:
    """Callback executed before rendering to load a preset safely into session state."""
    selected = st.session_state.get("sb_preset_select")
    if selected and selected != "— Select —":
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

    st.session_state.provider = st.text_input(
        "Provider",
        value=st.session_state.provider,
        key="sb_provider",
    )

    st.session_state.model_name = st.text_input(
        "Model",
        value=st.session_state.model_name,
        key="sb_model_name",
    )

    st.session_state.embedding_provider = st.text_input(
        "Embedding provider",
        value=st.session_state.embedding_provider,
        key="sb_embedding_provider",
    )

    st.session_state.embedding_model = st.text_input(
        "Embedding model",
        value=st.session_state.embedding_model,
        key="sb_embedding_model",
    )

    st.session_state.temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=float(st.session_state.temperature),
        step=0.05,
        key="sb_temperature",
    )

    st.session_state.ollama_url = st.text_input(
        "Ollama URL",
        value=st.session_state.ollama_url,
        key="sb_ollama_url",
        help="Base URL for the Ollama server (used when provider is 'ollama').",
    )

    st.session_state.thinking = st.toggle(
        "Thinking / reasoning mode",
        value=st.session_state.thinking,
        key="sb_thinking",
        help="Enable extended thinking where the provider supports it (e.g. Claude 3.7, Gemini 2.5).",
    )

    st.divider()

    # ── Presets ───────────────────────────────────────────────────────────────
    st.header("📂 Presets")
    st.caption("System prompt and all processing options are saved per preset.")

    st.session_state.system_prompt = st.text_area(
        "System prompt",
        value=st.session_state.system_prompt,
        height=100,
        key="sb_system_prompt",
        help="Saved with the preset. Leave empty to use no system prompt.",
    )

    preset_names = _list_presets()
    selected_preset = st.selectbox(
        "Load preset",
        options=["— Select —"] + preset_names,
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
st.caption("Apply a prompt to every paragraph or section of a Markdown document.")

# ── File I/O ──────────────────────────────────────────────────────────────────

st.subheader("📄 Files")
col_in, col_out = st.columns(2)
with col_in:
    input_path = _file_picker("Input file", "input_path", save=False)
with col_out:
    output_path = _file_picker("Output file", "output_path", save=True)

st.divider()

# ── Phase 1: Preprocessing ────────────────────────────────────────────────────

with st.expander("**Phase 1 — Preprocessing** *(optional)*", expanded=bool(st.session_state.preprocessing_prompt)):
    st.caption("Single AI call on the entire document. Leave empty to skip.")
    st.session_state.preprocessing_prompt = st.text_area(
        "Preprocessing prompt",
        value=st.session_state.preprocessing_prompt,
        height=120,
        label_visibility="collapsed",
        placeholder="E.g.: Extract a structured outline of this document.",
        key="ta_preprocessing_prompt",
    )

# ── Phase 2: Node Processing ──────────────────────────────────────────────────

with st.expander("**Phase 2 — Node Processing** *(core)*", expanded=True):
    st.caption("This prompt is applied independently to each node. Leave empty to skip.")
    st.session_state.node_prompt = st.text_area(
        "Node prompt",
        value=st.session_state.node_prompt,
        height=140,
        label_visibility="collapsed",
        placeholder="E.g.: Condense the following section without losing information.",
        key="ta_node_prompt",
    )

    # Node mode radio
    mode_map = {"paragraph": 0, "section": 1}
    mode_labels = ["🔹 Paragraph", "🔸 Section"]
    mode_val = st.radio(
        "Node granularity",
        options=mode_labels,
        index=mode_map.get(st.session_state.node_mode, 0),
        horizontal=True,
        key="radio_node_mode",
    )
    st.session_state.node_mode = "paragraph" if "Paragraph" in mode_val else "section"

    st.markdown("**Options:**")
    opt_col1, opt_col2 = st.columns(2)

    with opt_col1:
        st.session_state.include_preprocessed_context = st.checkbox(
            "Include preprocessed as prior context",
            value=st.session_state.include_preprocessed_context,
            key="cb_include_preprocessed",
            help="Inject the Phase 1 result as a prior assistant message when calling the node prompt.",
        )
        st.session_state.compute_similarity = st.checkbox(
            "Compute cosine similarity",
            value=st.session_state.compute_similarity,
            key="cb_compute_similarity",
            help="Compute embedding cosine similarity between original node and AI result.",
        )
        st.session_state.show_similarity_separator = st.checkbox(
            "Show §N separator between results",
            value=st.session_state.show_similarity_separator,
            key="cb_show_separator",
            help="Insert '# §N — node_id — Cosine similarity: score' before each result.",
        )

    with opt_col2:
        heading_labels = list(_HEADING_OPTIONS.values())
        heading_keys = list(_HEADING_OPTIONS.keys())
        heading_idx = heading_keys.index(st.session_state.heading_handling) if st.session_state.heading_handling in heading_keys else 0
        selected_heading = st.selectbox(
            "Heading handling",
            options=heading_labels,
            index=heading_idx,
            key="sel_heading_handling",
        )
        st.session_state.heading_handling = heading_keys[heading_labels.index(selected_heading)]

        non_text_labels = list(_NON_TEXT_OPTIONS.values())
        non_text_keys = list(_NON_TEXT_OPTIONS.keys())
        non_text_idx = non_text_keys.index(st.session_state.non_text_handling) if st.session_state.non_text_handling in non_text_keys else 0
        selected_non_text = st.selectbox(
            "Non-text handling (tables, code, images)",
            options=non_text_labels,
            index=non_text_idx,
            key="sel_non_text_handling",
        )
        st.session_state.non_text_handling = non_text_keys[non_text_labels.index(selected_non_text)]

# ── Phase 3: Postprocessing ───────────────────────────────────────────────────

with st.expander("**Phase 3 — Postprocessing** *(optional)*", expanded=bool(st.session_state.postprocessing_prompt)):
    st.caption("Single AI call after all nodes are processed. Leave empty to skip.")
    st.session_state.postprocessing_prompt = st.text_area(
        "Postprocessing prompt",
        value=st.session_state.postprocessing_prompt,
        height=120,
        label_visibility="collapsed",
        placeholder="E.g.: Review the condensed sections for consistency and produce a final report.",
        key="ta_postprocessing_prompt",
    )

    st.markdown("**Attach to the postprocessing prompt:**")
    pp_col1, pp_col2, pp_col3 = st.columns(3)
    with pp_col1:
        st.session_state.postprocess_attach_input = st.checkbox(
            "Original input document",
            value=st.session_state.postprocess_attach_input,
            key="cb_pp_attach_input",
        )
    with pp_col2:
        st.session_state.postprocess_attach_preprocessed = st.checkbox(
            "Preprocessed result",
            value=st.session_state.postprocess_attach_preprocessed,
            key="cb_pp_attach_preprocessed",
        )
    with pp_col3:
        st.session_state.postprocess_attach_results = st.checkbox(
            "Node results",
            value=st.session_state.postprocess_attach_results,
            key="cb_pp_attach_results",
        )

# ── Output Settings ───────────────────────────────────────────────────────────

with st.expander("**Output Settings**", expanded=True):
    st.caption(
        "Select which phase results to include in the output file. "
        "All checked sections are appended in order, separated by `---`."
    )
    out_col1, out_col2, out_col3 = st.columns(3)
    with out_col1:
        st.session_state.output_include_preprocessed = st.checkbox(
            "Phase 1 — Preprocessed",
            value=st.session_state.output_include_preprocessed,
            key="cb_out_preprocessed",
        )
    with out_col2:
        st.session_state.output_include_processed = st.checkbox(
            "Phase 2 — Node results",
            value=st.session_state.output_include_processed,
            key="cb_out_processed",
        )
    with out_col3:
        st.session_state.output_include_postprocessed = st.checkbox(
            "Phase 3 — Postprocessed",
            value=st.session_state.output_include_postprocessed,
            key="cb_out_postprocessed",
        )

    sep_col1, sep_col2 = st.columns(2)
    with sep_col1:
        st.session_state.save_preprocessed_separately = st.checkbox(
            "Save Phase 1 to separate file",
            value=st.session_state.save_preprocessed_separately,
            key="cb_save_preprocessed_sep",
        )
    with sep_col2:
        st.session_state.save_postprocessed_separately = st.checkbox(
            "Save Phase 3 to separate file",
            value=st.session_state.save_postprocessed_separately,
            key="cb_save_postprocessed_sep",
        )

    st.session_state.interlace_input_output = st.checkbox(
        "Interlace original and processed nodes",
        value=st.session_state.interlace_input_output,
        key="cb_interlace",
        help="For each processed node, output the original text followed by the AI response.",
    )

st.divider()

# ── Run Button ────────────────────────────────────────────────────────────────

run_col, _ = st.columns([1, 4])
with run_col:
    run_btn = st.button("▶ Run", type="primary", use_container_width=True, key="btn_run")

if run_btn:
    if not input_path:
        st.error("Please provide an input file path.")
    elif not st.session_state.node_prompt.strip() and not st.session_state.preprocessing_prompt.strip():
        st.warning("At least one of the preprocessing or node prompt must be non-empty.")
    else:
        preset = _state_to_preset()
        config = _state_to_config()
        log_container = st.empty()

        with st.spinner("Processing..."):
            try:
                result = run_iter(
                    input_file=input_path,
                    output_file=output_path or None,
                    preset=preset,
                    config=config,
                )
                st.session_state.result_preprocessed = result.preprocessed
                st.session_state.result_processed = result.results
                st.session_state.result_postprocessed = result.postprocessed
                st.session_state.result_node_results = result.node_results
                st.success("✅ Processing complete!")
            except Exception as exc:
                st.error(f"❌ Error: {exc}")

# ── Results ───────────────────────────────────────────────────────────────────

if st.session_state.result_preprocessed or st.session_state.result_processed or st.session_state.result_postprocessed:
    st.divider()
    st.subheader("📊 Results")

    if st.session_state.result_preprocessed:
        with st.expander("🔹 Phase 1 — Preprocessed"):
            st.markdown(st.session_state.result_preprocessed)

    if st.session_state.result_processed:
        with st.expander("🔸 Phase 2 — Node Results", expanded=True):
            # Show per-node similarity summary if computed
            node_results = st.session_state.result_node_results
            if node_results and any(nr.similarity is not None for nr in node_results):
                sims = [nr.similarity for nr in node_results if nr.similarity is not None]
                avg_sim = sum(sims) / len(sims)
                st.info(f"Average cosine similarity: **{avg_sim:.3f}**  ({len(sims)} nodes measured)")
            st.markdown(st.session_state.result_processed)

    if st.session_state.result_postprocessed:
        with st.expander("🔷 Phase 3 — Postprocessed"):
            st.markdown(st.session_state.result_postprocessed)
