"""KazeFlow - Training Tab."""

import json
import logging
import os
import shutil
import threading
from pathlib import Path

import gradio as gr

logger = logging.getLogger("kazeflow.ui.train")

_training_thread = None
_training_status = "Idle"
_stop_event = threading.Event()  # Set to request a graceful training stop

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_NOW_DIR = Path(os.getcwd())
_DATASETS_DIR = _NOW_DIR / "assets" / "datasets"
_LOGS_DIR = _NOW_DIR / "logs"

_SUPPORTED_AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac"}


def get_datasets_list() -> list:
    """List dataset directories that contain at least one audio file."""
    if not _DATASETS_DIR.exists():
        return []
    results = []
    for root, _, files in os.walk(_DATASETS_DIR):
        if any(Path(f).suffix.lower() in _SUPPORTED_AUDIO for f in files):
            results.append(os.path.relpath(root, _NOW_DIR))
    return sorted(results)


def get_models_list() -> list:
    """List experiment names in logs/ (subdirectory names)."""
    if not _LOGS_DIR.exists():
        return []
    return sorted([
        d for d in os.listdir(_LOGS_DIR)
        if os.path.isdir(_LOGS_DIR / d)
        and d not in ("zips", "mute", "reference")
        and not d.startswith("mute")
    ])


_CHECKPOINTS_DIR = _NOW_DIR / "assets" / "checkpoints"


def get_asset_checkpoints_list() -> list:
    """List .pt files in assets/checkpoints/ (pretrained base weights)."""
    if not _CHECKPOINTS_DIR.exists():
        return []
    return sorted(
        os.path.relpath(os.path.join(root, f), _NOW_DIR)
        for root, _, files in os.walk(_CHECKPOINTS_DIR)
        for f in files
        if f.endswith(".pt")
    )


def get_resume_checkpoints_list() -> list:
    """List checkpoint .pt files in logs/ (training run checkpoints)."""
    if not _LOGS_DIR.exists():
        return []
    return sorted(
        os.path.relpath(os.path.join(root, f), _NOW_DIR)
        for root, _, files in os.walk(_LOGS_DIR)
        for f in files
        if f.endswith(".pt") and not f.startswith(("G_", "D_"))
    )


def refresh_train_lists(current_pretrain: str, current_resume: str):
    """Refresh all dropdowns; clear value if the selected file no longer exists."""
    asset_ckpts = get_asset_checkpoints_list()
    resume_ckpts = get_resume_checkpoints_list()
    return (
        gr.update(choices=get_datasets_list()),
        gr.update(choices=get_models_list()),
        gr.update(
            choices=asset_ckpts,
            value=current_pretrain if current_pretrain in asset_ckpts else None,
        ),
        gr.update(
            choices=resume_ckpts,
            value=current_resume if current_resume in resume_ckpts else None,
        ),
    )


# ---------------------------------------------------------------------------
# Preprocessing (split: audio slicing + feature extraction)
# ---------------------------------------------------------------------------

def run_audio_preprocess(
    audio_dir: str,
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
    """Stage 1: Audio slicing / SmartCutter. Deletes existing sliced audio to redo."""
    global _training_status
    try:
        import multiprocessing
        from kazeflow.preprocess.audio import preprocess_training_set

        output_dir = Path("logs") / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        sliced_dir = output_dir / "sliced_audios"
        if sliced_dir.exists():
            _training_status = "Deleting existing sliced audio to redo..."
            shutil.rmtree(sliced_dir)

        _training_status = "Slicing audio..."
        n_proc = int(num_processes) if int(num_processes) > 0 else multiprocessing.cpu_count()

        preprocess_training_set(
            input_root=audio_dir,
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
        _training_status = f"Audio preprocessing complete! {n_files} sliced files."
        return _training_status

    except Exception as e:
        _training_status = f"Audio preprocessing failed: {e}"
        logger.exception("Audio preprocessing failed")
        return _training_status


def run_feature_extraction(
    model_name: str,
    sample_rate: int,
    content_embedder: str = "rspin",
):
    """Stage 2: Extract embeddings, pitch, mel. Deletes existing features to redo."""
    global _training_status
    try:
        import torch
        from kazeflow.preprocess.pipeline import PreprocessPipeline

        output_dir = Path("logs") / model_name
        sliced_dir = output_dir / "sliced_audios"

        if not sliced_dir.exists() or not any(sliced_dir.glob("*.wav")):
            _training_status = "Error: No sliced audio found. Run 'Preprocess Audio' first."
            return _training_status

        # Delete existing feature dirs to force re-extraction
        for feat_dir in ("spin", "f0", "mel"):
            feat_path = output_dir / feat_dir
            if feat_path.exists():
                _training_status = f"Deleting existing {feat_dir}/ to redo..."
                shutil.rmtree(feat_path)

        _training_status = "Extracting embeddings, pitch & mel..."

        from kazeflow.configs import load_config
        config = load_config(sample_rate=int(sample_rate))

        config["preprocess"]["content_embedder"] = content_embedder

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
            _training_status = (
                f"Feature extraction complete! {total} samples. "
                f"Output: {output_dir}"
            )
        else:
            _training_status = "Feature extraction complete!"

        return _training_status

    except Exception as e:
        _training_status = f"Feature extraction failed: {e}"
        logger.exception("Feature extraction failed")
        return _training_status


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def run_build_index(model_name: str, index_algorithm: str) -> str:
    """Build a FAISS index from the model's SPIN embeddings."""
    if not model_name or not model_name.strip():
        return "Error: Model Name is required."
    try:
        from kazeflow.train.build_index import build_index
        exp_dir = Path("logs") / model_name.strip()
        out = build_index(str(exp_dir), index_algorithm=index_algorithm)
        return f"Index saved → {out}"
    except Exception as e:
        logger.exception("Index building failed")
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def stop_training():
    """Signal the running training loop to stop after the current epoch."""
    global _training_status
    if _training_thread is not None and _training_thread.is_alive():
        _stop_event.set()
        _training_status = "Stop requested — finishing current epoch..."
        return _training_status
    return "No active training to stop."


def start_training(
    model_name: str,
    sample_rate: int,
    batch_size: int,
    save_every: int,
    epochs: int,
    pretrain_path: str,
    resume_path: str,
    content_embedder: str,
    architecture: str,
    vocoder_type: str,
    # Advanced
    precision: str,
    torch_compile: bool,
    compile_mode: str,
    lr_scheduler: str,
    gan_loss_type: str,
    gradient_balancer: bool,
    progressive_ode: bool,
    ode_ramp_epochs: int,
    # AFM (v2)
    afm_c_afm: float = 0.1,
    afm_adv_every: int = 2,
    afm_ramp_epochs: int = 30,
    afm_curriculum_epochs: int = 60,
):
    """Start training in a background thread."""
    global _training_thread, _training_status

    if _training_thread is not None and _training_thread.is_alive():
        return "Training already in progress!"

    _stop_event.clear()

    def _train_fn():
        global _training_status
        os.environ.setdefault(
            "TORCHINDUCTOR_CACHE_DIR",
            str(Path.home() / ".cache" / "torchinductor"),
        )
        try:
            _training_status = "Loading config..."

            experiment_dir = Path("logs") / model_name
            existing_cfg = experiment_dir / "config.json"

            if existing_cfg.exists():
                with open(existing_cfg, "r") as f:
                    config = json.load(f)
                logger.info(f"Loaded existing config from {existing_cfg}")
            else:
                from kazeflow.configs import load_config
                _vtype = vocoder_type if vocoder_type != "chouwa_gan" else None
                config = load_config(sample_rate=sample_rate, vocoder_type=_vtype)

            # Apply UI overrides
            config["train"]["precision"] = precision
            config["train"]["torch_compile"] = torch_compile
            config["train"]["compile_mode"] = compile_mode
            config["train"]["lr_scheduler"] = lr_scheduler
            config["train"]["gan_loss_type"] = gan_loss_type
            config["train"]["use_gradient_balancer"] = gradient_balancer
            config["preprocess"]["content_embedder"] = content_embedder

            from kazeflow.models.embedder import EMBEDDER_DIMS
            config["model"]["flow_matching"]["cond_channels"] = EMBEDDER_DIMS[content_embedder]

            if batch_size > 0:
                config["train"]["batch_size"] = batch_size
            if save_every > 0:
                config["train"]["save_every"] = save_every
            if epochs > 0:
                config["train"]["epochs"] = epochs

            config["train"]["progressive_ode"] = progressive_ode
            config["train"]["ode_ramp_epochs"] = int(ode_ramp_epochs)

            import torch

            is_v2 = architecture == "v2"

            # AFM config (v2 architecture)
            # If an afm block already exists in the saved config, preserve it.
            # Only write a new one if there's no existing afm block.
            if is_v2 and config["train"].get("afm") is None:
                config["train"]["afm"] = {
                    "c_afm": float(afm_c_afm),
                    "adv_every": int(afm_adv_every),
                    "ramp_epochs": int(afm_ramp_epochs),
                    "curriculum_epochs": int(afm_curriculum_epochs),
                }

            if is_v2:
                from kazeflow.train.trainer_v2 import KazeFlowV2Trainer
            else:
                from kazeflow.train.trainer import KazeFlowTrainer

            filelist_path = experiment_dir / "filelist.txt"

            if not filelist_path.exists():
                _training_status = (
                    "Error: filelist.txt not found in "
                    f"{experiment_dir}. Run Preprocess first."
                )
                return

            output_dir = str(experiment_dir)

            _training_status = "Initializing trainer..."
            if is_v2:
                trainer = KazeFlowV2Trainer(
                    config=config,
                    output_dir=output_dir,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
            else:
                trainer = KazeFlowTrainer(
                    config=config,
                    output_dir=output_dir,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )

            if pretrain_path and pretrain_path.strip():
                from kazeflow.train.pretrain import load_pretrain_for_finetune
                load_pretrain_for_finetune(pretrain_path.strip(), trainer)
                _training_status = "Loaded pretrain weights. Training..."

            resume = resume_path if resume_path and resume_path.strip() else None

            _training_status = "Training..."
            trainer.train(
                filelist_path=str(filelist_path),
                dataset_root=output_dir,
                resume_path=resume,
                stop_event=_stop_event,
            )
            if _stop_event.is_set():
                _training_status = "Training stopped by user."
            else:
                _training_status = "Training complete!"

        except Exception as e:
            _training_status = f"Training failed: {e}"
            logger.exception("Training failed")

    _training_thread = threading.Thread(target=_train_fn, daemon=True)
    _training_thread.start()
    return "Training started!"


def get_status():
    return _training_status


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------

def create_training_tab():
    """Build the training Gradio tab."""

    with gr.Tab("Training"):
        gr.Markdown("## Train a KazeFlow Model")

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
            architecture = gr.Radio(
                    label="Architecture",
                    choices=["v1", "v2"],
                    value="v2",
                    info="v1: Flow Matching. v2: Adversarial Flow Matching (adversarial signal to CFM + vocoder curriculum).",
                )
            vocoder_type = gr.Radio(
                    label="Vocoder",
                    choices=["chouwa_gan"],
                    value="chouwa_gan",
                    info="ChouwaGAN: HiFi-GAN backbone with SAN discriminator, anti-aliased activations and harmonic prior (8.7M params).",
                )

        # ── 1. Preprocess Audio ──────────────────────────────────────────
        with gr.Accordion("1 · Preprocess Audio", open=True):
            audio_dir = gr.Dropdown(
                label="Audio Directory",
                choices=get_datasets_list(),
                interactive=True,
                allow_custom_value=True,
                info="Folder with audio (assets/datasets/).",
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
                chunk_len = gr.Number(
                    label="Chunk Length (s)", value=3.0, minimum=0.1,
                    info="Used in Simple mode.",
                )
                overlap_len = gr.Number(
                    label="Overlap Length (s)", value=0.3, minimum=0.0,
                    info="Used in Simple mode.",
                )

            with gr.Row():
                loading_resampling = gr.Radio(
                    label="Audio Loader",
                    choices=["librosa", "ffmpeg"],
                    value="librosa",
                )
                num_processes = gr.Number(
                    label="Worker Processes", value=0, precision=0, minimum=0,
                    info="0 = auto (CPU count)",
                )

            with gr.Row():
                process_effects = gr.Checkbox(
                    label="High-pass filter", value=True,
                    info="48 Hz HPF to remove DC / sub-bass rumble.",
                )
                noise_reduction = gr.Checkbox(
                    label="Noise reduction", value=False,
                    info="Spectral noise reduction.",
                )
                use_smart_cutter = gr.Checkbox(
                    label="SmartCutter", value=True,
                    info="Silence removal before slicing (requires GPU).",
                )

            reduction_strength = gr.Slider(
                label="Noise Reduction Strength",
                minimum=0.0, maximum=1.0, step=0.05, value=0.7,
                visible=False,
            )

            preprocess_btn = gr.Button("▶ Preprocess Audio", variant="primary")
            preprocess_status = gr.Textbox(label="Status", interactive=False, lines=1)

        # ── 2. Extract Embeddings & Pitch ────────────────────────────────
        with gr.Accordion("2 · Extract Embeddings & Pitch", open=True):
            gr.Markdown(
                "Extract content embeddings (SPIN/RSPIN), F0 pitch, and mel spectrograms. "
                "Re-running will delete existing features and redo extraction."
            )
            content_embedder = gr.Dropdown(
                label="Content Embedder",
                choices=["spin_v2", "rspin"],
                value="rspin",
                info="SPIN v2: HuBERT-based. RSPIN: WavLM-based (recommended).",
            )
            extract_btn = gr.Button("▶ Extract Features", variant="primary")
            extract_status = gr.Textbox(label="Status", interactive=False, lines=1)

        # ── 3. Training ──────────────────────────────────────────────────
        with gr.Accordion("3 · Training", open=True):
            with gr.Group():
                gr.Markdown("#### Schedule")
                with gr.Row():
                    batch_size = gr.Number(
                        label="Batch Size", value=8, precision=0, minimum=0,
                        info="0 = config default",
                    )
                    save_every = gr.Number(
                        label="Save Every (epochs)", value=1, precision=0, minimum=0,
                        info="0 = config default",
                    )
                    training_epochs = gr.Number(
                        label="Total Epochs", value=500, precision=0, minimum=0,
                        info="0 = config default",
                    )

            with gr.Group():
                gr.Markdown("#### Checkpoints")
                with gr.Row():
                    pretrain_ckpt = gr.Dropdown(
                        label="Pretrain Checkpoint (optional)",
                        choices=get_asset_checkpoints_list(),
                        value=None,
                        interactive=True, allow_custom_value=True,
                        info="Load pretrained base weights (assets/checkpoints/).",
                    )
                    resume_ckpt = gr.Dropdown(
                        label="Resume Checkpoint (optional)",
                        choices=get_resume_checkpoints_list(),
                        value=None,
                        interactive=True, allow_custom_value=True,
                        info="Continue from a previous run (logs/).",
                    )

            # ── Advanced Settings ────────────────────────────────────
            with gr.Accordion("⚙ Advanced Settings", open=False):
                with gr.Group():
                    gr.Markdown("#### Precision & Compilation")
                    precision = gr.Radio(
                        label="Precision",
                        choices=["fp32", "fp32_fp16", "tf32", "tf32_fp16", "tf32_bf16"],
                        value="tf32_bf16",
                        info="tf32_bf16: recommended for Ampere+ GPUs.",
                    )
                    with gr.Row():
                        torch_compile = gr.Checkbox(
                            label="torch.compile", value=False,
                            info="~10–30% speedup (PyTorch 2.0+).",
                        )
                        compile_mode = gr.Dropdown(
                            label="Compile Mode",
                            choices=[
                                "default",
                                "reduce-overhead",
                                "max-autotune-no-cudagraphs",
                                "max-autotune",
                            ],
                            value="max-autotune-no-cudagraphs",
                        )

                with gr.Group():
                    gr.Markdown("#### LR & GAN")
                    with gr.Row():
                        lr_scheduler = gr.Dropdown(
                            label="LR Scheduler",
                            choices=["exponential", "cosine"],
                            value="cosine",
                            info="cosine: better convergence for long runs.",
                        )
                        gan_loss_type = gr.Dropdown(
                            label="GAN Loss Type",
                            choices=["hinge", "soft_hinge", "lsgan"],
                            value="soft_hinge",
                            info="soft_hinge: recommended (softplus disc for SAN).",
                        )
                    gradient_balancer = gr.Checkbox(
                        label="Gradient Balancer",
                        value=True,
                        info="Auto-balance mel/GAN gradient magnitudes (prevents GAN from dominating mel/STFT).",
                    )

                with gr.Group():
                    gr.Markdown("#### Warmup & Ramp")
                    with gr.Row():
                        progressive_ode = gr.Checkbox(
                            label="Progressive ODE", value=True,
                            info="Ramp ODE steps from min→max over ode_ramp_epochs.",
                        )
                        ode_ramp_epochs = gr.Number(
                            label="ODE Ramp (epochs)", value=200, precision=0,
                            minimum=0,
                            info="Epochs to ramp ODE steps.",
                        )

                with gr.Group(visible=True) as afm_group:
                    gr.Markdown("#### AFM (v2 only)")
                    gr.Markdown(
                        "Adversarial Flow Matching: discriminator signal "
                        "flows back to the flow model, teaching it to "
                        "produce mel that sounds real when vocoded.",
                    )
                    with gr.Row():
                        afm_c_afm = gr.Slider(
                            label="AFM Strength (c_afm)",
                            minimum=0.0, maximum=1.0, step=0.01, value=0.1,
                            interactive=True,
                            info="Weight of adversarial loss on flow. 0=off.",
                        )
                        afm_adv_every = gr.Number(
                            label="AFM Every N batches",
                            value=2, precision=0, minimum=1,
                            interactive=True,
                            info="Run adversarial path every N batches (saves VRAM).",
                        )
                    with gr.Row():
                        afm_ramp_epochs = gr.Number(
                            label="AFM Ramp (epochs)",
                            value=30, precision=0, minimum=0,
                            interactive=False,
                            info="Ramp c_afm from 0→full over N epochs.",
                        )
                        afm_curriculum_epochs = gr.Number(
                            label="Vocoder Curriculum (epochs)",
                            value=60, precision=0, minimum=0,
                            interactive=False,
                            info="Blend vocoder input: GT mel → CFM mel over N epochs.",
                        )

            with gr.Row():
                train_btn = gr.Button("▶ Start Training", variant="primary")
                stop_btn = gr.Button("⏹ Stop", variant="stop")
                status_btn = gr.Button("↻ Status")
            train_status = gr.Textbox(label="Training Status", interactive=False, lines=2)

        # ── 4. Build Index ───────────────────────────────────────────────
        with gr.Accordion("4 · Build FAISS Index", open=False):
            gr.Markdown("Build a FAISS index from SPIN embeddings for speaker retrieval.")
            index_algorithm = gr.Radio(
                label="Index Algorithm",
                choices=["Auto", "Faiss", "KMeans"],
                value="Auto",
            )
            index_btn = gr.Button("▶ Generate Index", variant="primary")
            index_status = gr.Textbox(label="Index Status", interactive=False, lines=1)

        # --- Events ---
        _clear_if_empty = lambda v: gr.update(value=None) if not v or not v.strip() else gr.update()
        pretrain_ckpt.change(fn=_clear_if_empty, inputs=[pretrain_ckpt], outputs=[pretrain_ckpt])
        resume_ckpt.change(fn=_clear_if_empty, inputs=[resume_ckpt], outputs=[resume_ckpt])

        refresh_btn.click(
            fn=refresh_train_lists,
            inputs=[pretrain_ckpt, resume_ckpt],
            outputs=[audio_dir, model_name, pretrain_ckpt, resume_ckpt],
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

        # Show/hide AFM group based on architecture
        architecture.change(
            fn=lambda arch: gr.update(visible=arch == "v2"),
            inputs=[architecture],
            outputs=[afm_group],
        )

        preprocess_btn.click(
            fn=run_audio_preprocess,
            inputs=[
                audio_dir, model_name, sample_rate,
                cut_preprocess, process_effects, noise_reduction,
                reduction_strength, chunk_len, overlap_len,
                normalization_mode, loading_resampling,
                use_smart_cutter, num_processes,
            ],
            outputs=[preprocess_status],
        )
        extract_btn.click(
            fn=run_feature_extraction,
            inputs=[model_name, sample_rate, content_embedder],
            outputs=[extract_status],
        )
        train_btn.click(
            fn=start_training,
            inputs=[model_name, sample_rate,
                    batch_size, save_every, training_epochs,
                    pretrain_ckpt, resume_ckpt, content_embedder,
                    architecture, vocoder_type,
                    # Advanced
                    precision, torch_compile, compile_mode,
                    lr_scheduler, gan_loss_type,
                    gradient_balancer,
                    progressive_ode, ode_ramp_epochs,
                    # AFM (v2)
                    afm_c_afm, afm_adv_every,
                    afm_ramp_epochs, afm_curriculum_epochs],
            outputs=[train_status],
        )
        stop_btn.click(fn=stop_training, outputs=[train_status])
        status_btn.click(fn=get_status, outputs=[train_status])
        index_btn.click(
            fn=run_build_index,
            inputs=[model_name, index_algorithm],
            outputs=[index_status],
        )
