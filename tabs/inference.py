"""KazeFlow - Inference Tab.

Loads a checkpoint and its accompanying config.json (if present) to
auto-populate inference parameters (ODE steps, method, guidance scale, etc.)
with the values used during training.
"""

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

# Defaults that mirror base.json ["inference"]
_INFER_DEFAULTS = {
    "ode_steps": 4,
    "ode_method": "midpoint",
    "guidance_scale": 1.5,
    "f0_method": "rmvpe",
    "f0_shift": 0,
    "index_rate": 0.0,
}


def get_available_models() -> list:
    """List available model checkpoints from assets/checkpoints/ and logs/."""
    results = []
    for root, _, files in os.walk(_MODELS_DIR):
        for f in sorted(files):
            if f.endswith(".pt"):
                results.append(os.path.relpath(os.path.join(root, f), _NOW_DIR))
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
        if isinstance(ckpt, dict):
            n = ckpt.get("n_speakers") or ckpt.get("speakers_id")
            if n and int(n) > 0:
                return list(range(int(n)))
            state = ckpt.get("generator") or ckpt.get("model") or ckpt
            if isinstance(state, dict):
                for key in ("speaker_embed.weight", "speaker_embedding.weight"):
                    if key in state:
                        return list(range(state[key].shape[0]))
    except Exception:
        pass
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


def _load_infer_config(model_path: str) -> dict:
    """
    Read the config.json next to the checkpoint and return the [inference]
    section merged with _INFER_DEFAULTS (checkpoint config takes priority).
    Falls back to _INFER_DEFAULTS if no config.json is found.
    """
    cfg = dict(_INFER_DEFAULTS)
    if not model_path:
        return cfg

    model_p = Path(model_path)
    # Candidates: same dir as checkpoint, or logs/<stem>/
    candidates = [
        model_p.parent / "config.json",
        _LOGS_DIR / model_p.stem / "config.json",
        _LOGS_DIR / model_p.parent.name / "config.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p) as f:
                    full_cfg = json.load(f)
                infer_cfg = full_cfg.get("inference", {})
                cfg.update({k: v for k, v in infer_cfg.items() if k in cfg})
                logger.info("Loaded inference config from %s", p)
                break
            except Exception as e:
                logger.warning("Could not read config %s: %s", p, e)
    return cfg


def match_index(model_path: str) -> str:
    """Try to auto-match an index file to the selected model."""
    if not model_path:
        return ""
    model_stem = Path(model_path).stem
    all_indexes = get_available_indexes()
    for idx in all_indexes:
        if Path(idx).stem == model_stem:
            return idx
    for idx in all_indexes:
        if model_stem in Path(idx).stem or Path(idx).stem in model_stem:
            return idx
    return ""


def refresh_all(model_path: str, index_path_current: str):
    """Refresh models, indexes, and speaker IDs; clear values if files no longer exist."""
    models = get_available_models()
    indexes = get_available_indexes()

    valid_model = model_path if model_path in models else None

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


def on_model_change(model_path: str):
    """
    Called when the user selects a different model.
    Returns updates for: index_path, speaker_id, ode_steps, ode_method,
    guidance_scale, f0_method.
    """
    cfg = _load_infer_config(model_path)
    speakers = get_speakers_id(model_path)
    auto_index = match_index(model_path)

    return (
        gr.update(value=auto_index),                                  # index_path
        gr.update(choices=speakers,
                  value=speakers[0] if speakers else 0),              # speaker_id
        gr.update(value=cfg["ode_steps"]),                            # ode_steps
        gr.update(value=cfg["ode_method"]),                           # ode_method
        gr.update(value=cfg["guidance_scale"]),                       # guidance_scale
        gr.update(value=cfg["f0_method"]),                            # f0_method
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
    guidance_scale: float = 1.5,
):
    """Run voice conversion inference."""
    global _pipeline

    if not audio_path:
        return None, "Please provide an audio file."
    if not model_path:
        return None, "Please select a model."

    try:
        from kazeflow.infer.pipeline import KazeFlowPipeline

        idx_path = index_path.strip() if index_path and index_path.strip() else None
        cache_key = (model_path, idx_path)

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
            guidance_scale=guidance_scale,
        )

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
        gr.Markdown("## Voice Conversion")

        with gr.Row(equal_height=False):
            # ── Left: Input ───────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                audio_input = gr.Audio(
                    label="Source Audio",
                    type="filepath",
                )

                with gr.Group():
                    with gr.Row(equal_height=True):
                        model_selector = gr.Dropdown(
                            label="Model",
                            choices=get_available_models(),
                            interactive=True,
                            allow_custom_value=True,
                            scale=6,
                        )
                        refresh_btn = gr.Button("🔄", size="sm", min_width=36, scale=0)

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
                gr.Markdown(
                    "### Settings\n"
                    "_Defaults are loaded from the model's `config.json` automatically._"
                )
                ode_steps = gr.Slider(
                    label="ODE Steps",
                    minimum=1, maximum=64, step=1,
                    value=_INFER_DEFAULTS["ode_steps"],
                    info="Steps for the RFM ODE solver. 4 is the default (fast & high-quality).",
                )

                with gr.Row():
                    ode_method = gr.Radio(
                        label="ODE Method",
                        choices=["euler", "midpoint", "rk4"],
                        value=_INFER_DEFAULTS["ode_method"],
                    )
                    f0_method = gr.Radio(
                        label="F0 Method",
                        choices=["rmvpe"],
                        value=_INFER_DEFAULTS["f0_method"],
                    )

                guidance_scale = gr.Slider(
                    label="Guidance Scale (CFG)",
                    minimum=1.0, maximum=5.0, step=0.1,
                    value=_INFER_DEFAULTS["guidance_scale"],
                    info="1.0 = off. Higher = stronger speaker conditioning (1.5 is the default).",
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
            fn=on_model_change,
            inputs=[model_selector],
            outputs=[index_path, speaker_id, ode_steps, ode_method,
                     guidance_scale, f0_method],
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
                    index_path, index_rate, guidance_scale],
            outputs=[audio_output, status_text],
        )

    return audio_input, audio_output
