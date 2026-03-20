"""KazeFlow - Inference Tab."""

import json
import logging
import os
from pathlib import Path

import gradio as gr

logger = logging.getLogger("kazeflow.ui.inference")

# Lazy-loaded pipeline
_pipeline = None

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_NOW_DIR = Path(os.getcwd())
_MODELS_DIR = _NOW_DIR / "assets" / "checkpoints"
_INDEX_DIR = _NOW_DIR / "assets" / "index"
_LOGS_DIR = _NOW_DIR / "logs"

_SUPPORTED_AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac"}


def get_available_models() -> list:
    """List available model checkpoints from assets/checkpoints/."""
    results = []
    # Search assets/checkpoints/ for *.pt files
    for root, _, files in os.walk(_MODELS_DIR):
        for f in sorted(files):
            if f.endswith(".pt"):
                results.append(os.path.relpath(os.path.join(root, f), _NOW_DIR))
    # Also search logs/ for *.pt files (trained checkpoints saved there)
    for root, _, files in os.walk(_LOGS_DIR):
        for f in sorted(files):
            if f.endswith(".pt") and not f.startswith(("G_", "D_")):
                results.append(os.path.relpath(os.path.join(root, f), _NOW_DIR))
    return sorted(results)


def get_available_indexes() -> list:
    """List available FAISS index files from assets/index/ and logs/."""
    results = []
    for root, _, files in os.walk(_INDEX_DIR):
        for f in sorted(files):
            if f.endswith(".index"):
                results.append(os.path.relpath(os.path.join(root, f), _NOW_DIR))
    for root, _, files in os.walk(_LOGS_DIR):
        for f in sorted(files):
            if f.endswith(".index") and "trained" not in f:
                results.append(os.path.relpath(os.path.join(root, f), _NOW_DIR))
    return sorted(results)


def get_speakers_id(model_path: str) -> list:
    """
    Return a list of speaker IDs [0, 1, ..., N-1] by reading n_speakers
    from the model checkpoint or the accompanying model_info.json.
    """
    if not model_path:
        return [0]
    try:
        import torch
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        # Check metadata stored in checkpoint
        if isinstance(ckpt, dict):
            n = ckpt.get("n_speakers") or ckpt.get("speakers_id")
            if n and int(n) > 0:
                return list(range(int(n)))
            # Try reading from speaker_embed weight shape
            state = ckpt.get("generator") or ckpt.get("model") or ckpt
            if isinstance(state, dict):
                for key in ("speaker_embed.weight", "speaker_embedding.weight"):
                    if key in state:
                        return list(range(state[key].shape[0]))
    except Exception:
        pass
    # Fallback: look for model_info.json in the same dir or logs/<name>/
    model_p = Path(model_path)
    for info_path in [
        model_p.parent / "model_info.json",
        _LOGS_DIR / model_p.stem / "model_info.json",
    ]:
        if info_path.exists():
            try:
                with open(info_path) as f:
                    info = json.load(f)
                n = info.get("n_speakers") or info.get("speakers_id", 1)
                return list(range(int(n)))
            except Exception:
                pass
    return [0]


def match_index(model_path: str) -> str:
    """Try to auto-match an index file to the selected model."""
    if not model_path:
        return ""
    model_stem = Path(model_path).stem
    all_indexes = get_available_indexes()
    # Exact stem match
    for idx in all_indexes:
        if Path(idx).stem == model_stem:
            return idx
    # Partial stem match
    for idx in all_indexes:
        if model_stem in Path(idx).stem or Path(idx).stem in model_stem:
            return idx
    return ""


def refresh_all(model_path: str, index_path_current: str):
    """Refresh models, indexes, and speaker IDs; clear values if files no longer exist."""
    models = get_available_models()
    indexes = get_available_indexes()

    # Keep current model only if file still exists in the list
    valid_model = model_path if model_path in models else None

    # Keep current index only if file still exists; otherwise try to auto-match
    if index_path_current and index_path_current in indexes:
        valid_index = index_path_current
    else:
        valid_index = match_index(valid_model) if valid_model else None

    speakers = get_speakers_id(valid_model) if valid_model else [0]
    return (
        gr.update(choices=models, value=valid_model),
        gr.update(choices=indexes, value=valid_index),
        gr.update(choices=speakers, value=speakers[0] if speakers else 0),
    )


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------

def run_inference(
    audio_path: str,
    model_path: str,
    speaker_id: int,
    f0_shift: int,
    ode_steps: int,
    ode_method: str,
    f0_method: str,
    index_path: str,
    index_rate: float,
):
    """Run voice conversion inference."""
    global _pipeline

    if not audio_path:
        return None, "Please provide an audio file."
    if not model_path:
        return None, "Please select a model."

    try:
        from kazeflow.infer.pipeline import KazeFlowPipeline

        # Resolve index path
        idx_path = index_path.strip() if index_path and index_path.strip() else None
        cache_key = (model_path, idx_path)

        # Load pipeline (cache if same model + index)
        if _pipeline is None or getattr(_pipeline, "_cache_key", None) != cache_key:
            _pipeline = KazeFlowPipeline(
                checkpoint_path=model_path,
                index_path=idx_path,
                device="cuda" if __import__("torch").cuda.is_available() else "cpu",
            )
            _pipeline._cache_key = cache_key

        waveform = _pipeline.convert(
            source_audio_path=audio_path,
            speaker_id=int(speaker_id),
            f0_shift=f0_shift,
            ode_steps=ode_steps,
            ode_method=ode_method,
            f0_method=f0_method,
            index_rate=index_rate if idx_path else 0.0,
        )

        # Save output
        output_path = "output_kazeflow.wav"
        _pipeline.save_audio(waveform, output_path)

        return output_path, "Conversion complete!"

    except Exception as e:
        logger.exception("Inference failed")
        return None, f"Error: {e}"


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------

def create_inference_tab():
    """Build the inference Gradio tab."""
    with gr.Tab("Inference"):
        gr.Markdown("## 🎙 Voice Conversion")

        with gr.Row(equal_height=False):
            # ── Left: Input ───────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                audio_input = gr.Audio(
                    label="Source Audio",
                    type="filepath",
                )

                with gr.Group():
                    with gr.Row():
                        model_selector = gr.Dropdown(
                            label="Model",
                            choices=get_available_models(),
                            interactive=True,
                            allow_custom_value=True,
                            scale=4,
                        )
                        refresh_btn = gr.Button("🔄", size="sm", min_width=40, scale=0)

                    speaker_id = gr.Dropdown(
                        label="Speaker ID",
                        choices=[0],
                        value=0,
                        interactive=True,
                        allow_custom_value=False,
                        info="Auto-detected from model.",
                    )

                f0_shift = gr.Slider(
                    label="Pitch Shift (semitones)",
                    minimum=-24, maximum=24, step=1, value=0,
                )

            # ── Right: Settings & Run ────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                ode_steps = gr.Slider(
                    label="ODE Steps",
                    minimum=1, maximum=64, step=1, value=16,
                    info="More steps = higher quality, slower.",
                )

                with gr.Row():
                    ode_method = gr.Radio(
                        label="ODE Method",
                        choices=["euler", "midpoint"],
                        value="euler",
                    )
                    f0_method = gr.Radio(
                        label="F0 Method",
                        choices=["rmvpe"],
                        value="rmvpe",
                    )

                with gr.Accordion("FAISS Index", open=False):
                    index_path = gr.Dropdown(
                        label="Index File (.index)",
                        choices=get_available_indexes(),
                        interactive=True,
                        allow_custom_value=True,
                        info="FAISS index for speaker feature retrieval.",
                    )
                    index_rate = gr.Slider(
                        label="Index Rate",
                        minimum=0.0, maximum=1.0, step=0.05, value=0.0,
                        info="0 = off · 0.3–0.7 recommended · 1 = full retrieval",
                    )

                gr.Markdown("---")
                convert_btn = gr.Button("▶ Convert", variant="primary", size="lg")
                status_text = gr.Textbox(label="Status", interactive=False, lines=1)

        audio_output = gr.Audio(label="Output", type="filepath")

        # --- Events ---
        model_selector.change(
            fn=lambda m: (
                gr.update(value=match_index(m)),
                gr.update(choices=get_speakers_id(m),
                          value=get_speakers_id(m)[0] if get_speakers_id(m) else 0),
            ),
            inputs=[model_selector],
            outputs=[index_path, speaker_id],
        )
        refresh_btn.click(
            fn=refresh_all,
            inputs=[model_selector, index_path],
            outputs=[model_selector, index_path, speaker_id],
        )
        convert_btn.click(
            fn=run_inference,
            inputs=[audio_input, model_selector, speaker_id,
                    f0_shift, ode_steps, ode_method, f0_method,
                    index_path, index_rate],
            outputs=[audio_output, status_text],
        )

    return audio_input, audio_output
