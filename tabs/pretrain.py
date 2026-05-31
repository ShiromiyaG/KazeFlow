"""KazeFlow - Pretrain Tab (RFM only).

Stage 1 of 3: Pretrain the Rectified Flow Matching (RFM) model on a large
multi-speaker dataset. Only the RFM model is trained here — no vocoder, no GAN.

After pretraining, proceed to:
  Stage 2 → Vocoder tab: train ChouwaGAN on the frozen RFM
  Stage 3 → Train tab: joint fine-tune on a target speaker
"""

import json
import logging
import os
import shutil
import threading
from pathlib import Path

import gradio as gr

from tabs import _gpu_caps, get_precision_choices

logger = logging.getLogger("kazeflow.ui.pretrain")

_GPU_CAPS = _gpu_caps()

_pretrain_thread = None
_pretrain_status = "Idle"
_stop_event = threading.Event()

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_NOW_DIR = Path(os.getcwd())
_DATASETS_DIR = _NOW_DIR / "assets" / "datasets"
_LOGS_DIR = _NOW_DIR / "logs"

_SUPPORTED_AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac"}


def get_datasets_list() -> list:
    if not _DATASETS_DIR.exists():
        return []
    results = []
    for root, _, files in os.walk(_DATASETS_DIR):
        if any(Path(f).suffix.lower() in _SUPPORTED_AUDIO for f in files):
            results.append(os.path.relpath(root, _NOW_DIR))
    return sorted(results)


def get_models_list() -> list:
    if not _LOGS_DIR.exists():
        return []
    return sorted([
        d for d in os.listdir(_LOGS_DIR)
        if os.path.isdir(_LOGS_DIR / d)
        and d not in ("zips", "mute", "reference")
        and not d.startswith("mute")
    ])


def get_pretrain_checkpoints_list() -> list:
    """List pretrain_*.pt files under logs/ (RFM-only checkpoints)."""
    results = []
    for root, _, files in os.walk(_LOGS_DIR):
        for f in sorted(files):
            if f.startswith("pretrain_") and f.endswith(".pt"):
                results.append(os.path.relpath(os.path.join(root, f), _NOW_DIR))
    return sorted(results)


def refresh_pretrain_lists(current_resume: str):
    ckpts = get_pretrain_checkpoints_list()
    return (
        gr.update(choices=get_datasets_list()),
        gr.update(choices=get_models_list()),
        gr.update(
            choices=ckpts,
            value=current_resume if current_resume in ckpts else None,
        ),
    )


# ---------------------------------------------------------------------------
# Preprocessing: audio slicing + feature extraction (same as before)
# ---------------------------------------------------------------------------

def run_pretrain_audio_preprocess(
    dataset_dir: str,
    model_name: str,
    sample_rate: int,
    cut_preprocess: str,
    process_effects: bool,
    noise_reduction: bool,
    reduction_strength: float,
    chunk_len: float,
    overlap_len: float,
    normalization_mode: str,
    loading_resampling: str,
    use_smart_cutter: bool,
    num_processes: int,
):
    """Stage 1: Audio slicing / SmartCutter."""
    global _pretrain_status
    try:
        import multiprocessing
        from kazeflow.preprocess.audio import preprocess_training_set

        output_dir = Path("logs") / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        sliced_dir = output_dir / "sliced_audios"
        if sliced_dir.exists():
            _pretrain_status = "Deleting existing sliced audio to redo..."
            shutil.rmtree(sliced_dir)

        _pretrain_status = "Slicing audio..."
        n_proc = int(num_processes) if int(num_processes) > 0 else multiprocessing.cpu_count()

        preprocess_training_set(
            input_root=dataset_dir,
            sr=int(sample_rate),
            num_processes=n_proc,
            exp_dir=str(output_dir),
            cut_preprocess=cut_preprocess,
            process_effects=bool(process_effects),
            noise_reduction=bool(noise_reduction),
            reduction_strength=float(reduction_strength),
            chunk_len=float(chunk_len),
            overlap_len=float(overlap_len),
            normalization_mode=normalization_mode,
            loading_resampling=loading_resampling,
            use_smart_cutter=bool(use_smart_cutter),
        )

        n_files = len(list(sliced_dir.glob("*.wav"))) if sliced_dir.exists() else 0
        _pretrain_status = f"Audio preprocessing complete! {n_files} sliced files."
        return _pretrain_status

    except Exception as e:
        _pretrain_status = f"Audio preprocessing failed: {e}"
        logger.exception("Audio preprocessing failed")
        return _pretrain_status


def run_pretrain_feature_extraction(
    model_name: str,
    sample_rate: int,
    content_embedder: str,
):
    """Stage 2: Extract SPIN/RSPIN embeddings, F0, and mel spectrograms."""
    global _pretrain_status
    try:
        import torch
        from kazeflow.preprocess.pipeline import PreprocessPipeline
        from kazeflow.configs import load_config

        output_dir = Path("logs") / model_name
        sliced_dir = output_dir / "sliced_audios"

        if not sliced_dir.exists() or not any(sliced_dir.glob("*.wav")):
            _pretrain_status = "Error: No sliced audio found. Run 'Preprocess Audio' first."
            return _pretrain_status

        # Delete existing feature dirs to force re-extraction
        for feat_dir in ("spin", "f0", "mel"):
            feat_path = output_dir / feat_dir
            if feat_path.exists():
                _pretrain_status = f"Deleting existing {feat_dir}/ to redo..."
                shutil.rmtree(feat_path)

        _pretrain_status = "Extracting embeddings, pitch & mel..."

        config = load_config(sample_rate=int(sample_rate))
        config["preprocess"]["content_embedder"] = content_embedder
        config["model"]["architecture"] = "rfm"

        pipeline = PreprocessPipeline(
            config=config,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        pipeline.process_dataset(
            audio_dir=str(sliced_dir),
            output_dir=str(output_dir),
        )

        # Save config so training can pick it up later
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        info_path = output_dir / "model_info.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            total = info.get("total_samples", "?")
            _pretrain_status = (
                f"Feature extraction complete! {total} samples. "
                f"Output: {output_dir}"
            )
        else:
            _pretrain_status = "Feature extraction complete!"

        return _pretrain_status

    except Exception as e:
        _pretrain_status = f"Feature extraction failed: {e}"
        logger.exception("Feature extraction failed")
        return _pretrain_status


# ---------------------------------------------------------------------------
# Training: RFM only
# ---------------------------------------------------------------------------

_STANDARD_PROFILE = {"batch_size": 16, "save_every": 5, "epochs": 500}
_HIGH_THROUGHPUT_PROFILE = {"batch_size": 64, "save_every": 10, "epochs": 500, "num_workers": 8}


def apply_training_profile(profile: str, prec_default: str):
    if profile == "high_throughput":
        return (
            gr.update(value=64),    # batch_size
            gr.update(value=10),    # save_every
            gr.update(value=500),   # epochs
            gr.update(value="cosine_warmup_restarts"),  # lr_scheduler
            gr.update(value="tf32_bf16"),               # precision
            gr.update(value=True),                       # torch_compile
        )
    else:
        return (
            gr.update(value=16),
            gr.update(value=5),
            gr.update(value=500),
            gr.update(value="cosine"),
            gr.update(value=prec_default),
            gr.update(value=False),
        )


def stop_pretraining():
    global _pretrain_status
    _stop_event.set()
    _pretrain_status = "Stop requested — finishing current epoch..."
    return _pretrain_status


def start_rfm_pretraining(
    model_name: str,
    sample_rate: int,
    batch_size: int,
    save_every: int,
    epochs: int,
    resume_path: str,
    content_embedder: str,
    precision: str,
    torch_compile: bool,
    compile_mode: str,
    lr_scheduler: str,
    training_profile: str,
):
    """Launch RFM-only pretraining in a background thread."""
    global _pretrain_thread, _pretrain_status, _stop_event

    if not model_name or not model_name.strip():
        return "Error: Model Name is required."

    if _pretrain_thread is not None and _pretrain_thread.is_alive():
        return "Pretraining already in progress!"

    _stop_event.clear()

    def _train_fn():
        global _pretrain_status
        os.environ.setdefault(
            "TORCHINDUCTOR_CACHE_DIR",
            str(Path.home() / ".cache" / "torchinductor"),
        )
        try:
            _pretrain_status = "Loading config..."
            experiment_dir = Path("logs") / model_name
            existing_cfg = experiment_dir / "config.json"

            from kazeflow.configs import load_config
            if existing_cfg.exists():
                with open(existing_cfg, "r") as f:
                    config = json.load(f)
                logger.info(f"Loaded existing config from {existing_cfg}")
            else:
                config = load_config(sample_rate=sample_rate, preset="rfm")

            # Force RFM architecture
            config["model"]["architecture"] = "rfm"

            # UI overrides
            config["train"]["precision"] = precision
            config["train"]["torch_compile"] = torch_compile
            config["train"]["compile_mode"] = compile_mode
            config["train"]["lr_scheduler"] = lr_scheduler
            config["preprocess"]["content_embedder"] = content_embedder

            from kazeflow.models.embedder import EMBEDDER_DIMS
            config["model"]["flow_matching"]["cond_channels"] = EMBEDDER_DIMS[content_embedder]

            if batch_size > 0:
                config["train"]["batch_size"] = batch_size
            if save_every > 0:
                config["train"]["save_every"] = save_every
            if epochs > 0:
                config["train"]["epochs"] = epochs

            # High-Throughput profile LR scaling
            if training_profile == "high_throughput":
                import math
                base_bs = _STANDARD_PROFILE["batch_size"]
                actual_bs = config["train"].get("batch_size", base_bs)
                lr_scale = math.sqrt(actual_bs / base_bs)
                for lr_key in ("learning_rate_flow",):
                    old_lr = config["train"].get(lr_key, 0.0002)
                    config["train"][lr_key] = round(old_lr * lr_scale, 8)
                config["train"]["num_workers"] = _HIGH_THROUGHPUT_PROFILE["num_workers"]
                logger.info(
                    f"High-Throughput: num_workers={config['train']['num_workers']}, "
                    f"flow LR ×{lr_scale:.2f}"
                )

            filelist_path = experiment_dir / "filelist.txt"
            if not filelist_path.exists():
                _pretrain_status = "Error: filelist.txt not found. Run Preprocess first."
                return

            # Auto-detect n_speakers
            info_path = experiment_dir / "model_info.json"
            if info_path.exists():
                with open(info_path) as f:
                    info = json.load(f)
                config["model"]["n_speakers"] = info.get(
                    "n_speakers", config["model"].get("n_speakers", 1)
                )
            else:
                speaker_ids = set()
                with open(filelist_path, "r") as f:
                    for line in f:
                        parts = line.strip().split("|")
                        if len(parts) >= 2:
                            speaker_ids.add(parts[1].strip())
                config["model"]["n_speakers"] = max(len(speaker_ids), 1)

            import torch
            from kazeflow.train.pretrain import KazeFlowPretrainer

            _pretrain_status = (
                f"Initializing RFM pretrainer "
                f"({config['model']['n_speakers']} speakers)..."
            )
            output_dir = str(experiment_dir)

            pretrainer = KazeFlowPretrainer(
                config=config,
                output_dir=output_dir,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            resume = resume_path.strip() if resume_path and resume_path.strip() else None

            _pretrain_status = "Pretraining RFM..."
            pretrainer.train(
                filelist_path=str(filelist_path),
                dataset_root=output_dir,
                resume_path=resume,
                stop_event=_stop_event,
            )
            if _stop_event.is_set():
                _pretrain_status = "RFM pretraining stopped by user."
            else:
                _pretrain_status = "RFM pretraining complete! → Now train the Vocoder."

        except Exception as e:
            _pretrain_status = f"Pretraining failed: {e}"
            logger.exception("RFM pretraining failed")

    _pretrain_thread = threading.Thread(target=_train_fn, daemon=True)
    _pretrain_thread.start()
    return "RFM Pretraining started!"


def get_pretrain_status():
    return _pretrain_status


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------

def create_pretrain_tab():
    """Build the RFM Pretrain Gradio tab."""
    with gr.Tab("Pretrain (RFM)"):
        gr.Markdown(
            "## Stage 1 · Pretrain Rectified Flow Matching\n"
            "Train the **RFM mel model** on a large multi-speaker dataset. "
            "Only the RFM is trained here — no vocoder, no GAN, no warmup phases.\n\n"
            "> **Next steps after pretraining:** "
            "① Vocoder tab → train ChouwaGAN on frozen RFM mels · "
            "② Train tab → joint fine-tune on target speaker\n\n"
            "> **Dataset layout:** Subdirectories `0/`, `1/`, `2/`... for multi-speaker."
        )

        # ── Model Settings ─────────────────────────────────────────────
        with gr.Group():
            gr.Markdown("### Model Settings")
            with gr.Row(equal_height=True):
                model_name = gr.Dropdown(
                    label="Model Name",
                    choices=get_models_list(),
                    interactive=True,
                    allow_custom_value=True,
                    info="Output goes to logs/<name>/",
                    scale=6,
                )
                refresh_btn = gr.Button("🔄", size="sm", min_width=36, scale=0)
            sample_rate = gr.Radio(
                label="Sample Rate",
                choices=[32000, 40000, 44100, 48000],
                value=32000,
                info="Must match between Preprocess and Training.",
            )
            content_embedder = gr.Dropdown(
                label="Content Embedder",
                choices=["spin_v2", "rspin"],
                value="rspin",
                info="RSPIN (WavLM-based, recommended) or SPIN v2 (HuBERT-based).",
            )

        # ── 1. Preprocess Audio ──────────────────────────────────────────
        with gr.Accordion("1 · Preprocess Audio", open=True):
            dataset_dir = gr.Dropdown(
                label="Dataset Directory",
                choices=get_datasets_list(),
                interactive=True,
                allow_custom_value=True,
                info="Folder with audio (assets/datasets/). Subdirs 0/, 1/,... = multi-speaker.",
            )

            with gr.Row():
                cut_preprocess = gr.Radio(
                    label="Slicing Mode",
                    choices=["Automatic", "Simple", "Skip"],
                    value="Simple",
                    info="Automatic: silence-based · Simple: fixed chunks · Skip: no slicing",
                )
                normalization_mode = gr.Radio(
                    label="Normalization",
                    choices=["none", "post_peak"],
                    value="post_peak",
                )

            with gr.Row():
                chunk_len = gr.Number(label="Chunk Length (s)", value=3.0, minimum=0.1)
                overlap_len = gr.Number(label="Overlap Length (s)", value=0.3, minimum=0.0)

            with gr.Row():
                loading_resampling = gr.Radio(
                    label="Audio Loader", choices=["librosa", "ffmpeg"], value="librosa")
                num_processes = gr.Number(
                    label="Worker Processes", value=0, precision=0, minimum=0,
                    info="0 = auto (CPU count)",
                )

            with gr.Row():
                process_effects = gr.Checkbox(label="High-pass filter", value=True)
                noise_reduction = gr.Checkbox(label="Noise reduction", value=False)
                use_smart_cutter = gr.Checkbox(label="SmartCutter", value=True)

            reduction_strength = gr.Slider(
                label="Noise Reduction Strength",
                minimum=0.0, maximum=1.0, step=0.05, value=0.7,
                visible=False,
            )

            preprocess_btn = gr.Button("▶ Preprocess Audio", variant="primary")
            preprocess_status = gr.Textbox(label="Status", interactive=False, lines=1)

        # ── 2. Extract Features ────────────────────────────────────────
        with gr.Accordion("2 · Extract Embeddings & Pitch", open=True):
            gr.Markdown(
                "Extract content embeddings (RSPIN/SPIN), F0 pitch, and mel spectrograms. "
                "Re-running will delete existing features and redo extraction."
            )
            extract_btn = gr.Button("▶ Extract Features", variant="primary")
            extract_status = gr.Textbox(label="Status", interactive=False, lines=1)

        # ── 3. RFM Pretraining ───────────────────────────────────────────
        with gr.Accordion("3 · RFM Pretraining", open=True):
            gr.Markdown(
                "**Trains only the RFM model.** "
                "No vocoder, no GAN, no warmup phases. "
                "Saves `pretrain_<epoch>.pt` checkpoints that can be loaded in the Vocoder tab."
            )
            with gr.Group():
                gr.Markdown("#### Schedule")
                with gr.Row():
                    batch_size = gr.Number(
                        label="Batch Size", value=16, precision=0, minimum=0,
                        info="0 = config default",
                    )
                    save_every = gr.Number(
                        label="Save Every (epochs)", value=5, precision=0, minimum=0,
                        info="0 = config default",
                    )
                    total_epochs = gr.Number(
                        label="Total Epochs", value=500, precision=0, minimum=0,
                        info="0 = config default",
                    )

            with gr.Group():
                gr.Markdown("#### Resume")
                resume_ckpt = gr.Dropdown(
                    label="Resume Checkpoint (optional)",
                    choices=get_pretrain_checkpoints_list(),
                    value=None,
                    interactive=True, allow_custom_value=True,
                    info="Continue from a previous pretrain_*.pt run.",
                )

            _prec_choices, _prec_default = get_precision_choices(_GPU_CAPS)
            with gr.Accordion("⚙ Advanced Settings", open=False):
                with gr.Group():
                    gr.Markdown("#### Training Profile")
                    training_profile = gr.Radio(
                        label="Profile",
                        choices=[
                            ("Standard  —  any GPU, datasets < 10 h", "standard"),
                            ("High-Throughput  —  40 GB+ VRAM (A100/H100), datasets ≥ 50 h", "high_throughput"),
                        ],
                        value="standard",
                    )

                with gr.Group():
                    gr.Markdown("#### Precision & Compilation")
                    precision = gr.Radio(
                        label="Precision",
                        choices=_prec_choices,
                        value=_prec_default,
                    )
                    with gr.Row(visible=_GPU_CAPS["has_compile"]):
                        torch_compile = gr.Checkbox(
                            label="torch.compile", value=_GPU_CAPS["has_compile"])
                        compile_mode = gr.Dropdown(
                            label="Compile Mode",
                            choices=["default", "reduce-overhead",
                                     "max-autotune-no-cudagraphs", "max-autotune"],
                            value="max-autotune-no-cudagraphs",
                        )

                with gr.Group():
                    gr.Markdown("#### LR Scheduler")
                    lr_scheduler = gr.Dropdown(
                        label="LR Scheduler",
                        choices=["cosine", "cosine_warmup_restarts", "exponential"],
                        value="cosine",
                        info="cosine: recommended for RFM-only pretrain.",
                    )

            with gr.Row():
                train_btn = gr.Button("▶ Start RFM Pretraining", variant="primary")
                stop_btn = gr.Button("⏹ Stop", variant="stop")
                status_btn = gr.Button("↻ Status")
            train_status = gr.Textbox(
                label="Pretraining Status", interactive=False, lines=2)

        # ── Events ──────────────────────────────────────────────────────
        _clear_if_empty = lambda v: gr.update(value=None) if not v or not v.strip() else gr.update()
        resume_ckpt.change(fn=_clear_if_empty, inputs=[resume_ckpt], outputs=[resume_ckpt])

        _prec_default_captured = _prec_default
        training_profile.change(
            fn=lambda profile: apply_training_profile(profile, _prec_default_captured),
            inputs=[training_profile],
            outputs=[batch_size, save_every, total_epochs,
                     lr_scheduler, precision, torch_compile],
        )

        refresh_btn.click(
            fn=refresh_pretrain_lists,
            inputs=[resume_ckpt],
            outputs=[dataset_dir, model_name, resume_ckpt],
        )
        cut_preprocess.change(
            fn=lambda mode: (
                gr.update(visible=mode == "Simple"),
                gr.update(visible=mode == "Simple"),
            ),
            inputs=[cut_preprocess],
            outputs=[chunk_len, overlap_len],
        )
        noise_reduction.change(
            fn=lambda enabled: gr.update(visible=enabled),
            inputs=[noise_reduction],
            outputs=[reduction_strength],
        )

        preprocess_btn.click(
            fn=run_pretrain_audio_preprocess,
            inputs=[
                dataset_dir, model_name, sample_rate,
                cut_preprocess, process_effects, noise_reduction,
                reduction_strength, chunk_len, overlap_len,
                normalization_mode, loading_resampling,
                use_smart_cutter, num_processes,
            ],
            outputs=[preprocess_status],
        )
        extract_btn.click(
            fn=run_pretrain_feature_extraction,
            inputs=[model_name, sample_rate, content_embedder],
            outputs=[extract_status],
        )
        train_btn.click(
            fn=start_rfm_pretraining,
            inputs=[
                model_name, sample_rate,
                batch_size, save_every, total_epochs,
                resume_ckpt, content_embedder,
                precision, torch_compile, compile_mode,
                lr_scheduler, training_profile,
            ],
            outputs=[train_status],
        )
        stop_btn.click(fn=stop_pretraining, outputs=[train_status])
        status_btn.click(fn=get_pretrain_status, outputs=[train_status])
