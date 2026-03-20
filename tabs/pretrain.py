"""KazeFlow - Pretrain Tab."""

import json
import logging
import os
import threading
from pathlib import Path

import gradio as gr

logger = logging.getLogger("kazeflow.ui.pretrain")

_pretrain_thread = None
_pretrain_status = "Idle"
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


def get_pretrain_checkpoints_list() -> list:
    """List pretrain checkpoint .pt files under logs/."""
    results = []
    for root, _, files in os.walk(_LOGS_DIR):
        for f in sorted(files):
            if f.endswith(".pt") and not f.startswith(("G_", "D_")):
                results.append(os.path.relpath(os.path.join(root, f), _NOW_DIR))
    return sorted(results)


def refresh_pretrain_lists(current_resume: str):
    """Refresh all dropdowns; clear value if the selected file no longer exists."""
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
# Preprocessing
# ---------------------------------------------------------------------------

def run_pretrain_preprocess(
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
    """
    Run the full two-stage preprocessing pipeline for pretraining:
      1. Audio slicing / SmartCutter  (preprocess_training_set)
      2. Feature extraction           (PreprocessPipeline.process_dataset)
    """
    global _pretrain_status
    try:
        import multiprocessing
        import torch
        from kazeflow.preprocess.audio import preprocess_training_set
        from kazeflow.preprocess.pipeline import PreprocessPipeline

        output_dir = Path("logs") / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Stage 1: audio slicing ────────────────────────────────────────
        _pretrain_status = "Stage 1/2: Slicing audio..."
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

        # ── Stage 2: feature extraction ───────────────────────────────────
        _pretrain_status = "Stage 2/2: Extracting features..."

        sr_map = {32000: "32k", 40000: "40k", 44100: "44k", 48000: "48k"}
        cfg_dir = Path(__file__).parent.parent / "kazeflow" / "configs"
        cfg_name = sr_map.get(int(sample_rate), "48k")
        config_path = cfg_dir / f"{cfg_name}.json"

        with open(config_path, "r") as f:
            config = json.load(f)

        sliced_dir = output_dir / "sliced_audios"

        pipeline = PreprocessPipeline(
            config=config,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        pipeline.process_dataset(
            audio_dir=str(sliced_dir),
            output_dir=str(output_dir),
        )

        info_path = output_dir / "model_info.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            n_spk = info.get("n_speakers", "?")
            total = info.get("total_samples", "?")
            _pretrain_status = (
                f"Preprocessing complete! "
                f"{total} samples, {n_spk} speaker(s) detected."
            )
        else:
            _pretrain_status = "Preprocessing complete!"

        return _pretrain_status

    except Exception as e:
        _pretrain_status = f"Preprocessing failed: {e}"
        logger.exception("Preprocessing failed")
        return _pretrain_status


# ---------------------------------------------------------------------------
# Pretraining
# ---------------------------------------------------------------------------

def stop_pretraining():
    """Signal the running pretraining loop to stop after the current epoch."""
    global _pretrain_status
    if _pretrain_thread is not None and _pretrain_thread.is_alive():
        _stop_event.set()
        _pretrain_status = "Stop requested — finishing current epoch..."
        return _pretrain_status
    return "No active pretraining to stop."


def start_pretraining(
    model_name: str,
    sample_rate: int,
    precision: str,
    torch_compile: bool,
    compile_mode: str,
    gan_loss_type: str,
    batch_size: int,
    save_every: int,
    epochs: int,
    cfm_warmup_epochs: int,
    resume_path: str,
):
    """Start pretraining in a background thread."""
    global _pretrain_thread, _pretrain_status

    if _pretrain_thread is not None and _pretrain_thread.is_alive():
        return "Pretraining already in progress!"

    _stop_event.clear()  # Reset stop flag before starting

    def _train_fn():
        global _pretrain_status
        # Redirect inductor autotune cache to SSD — /tmp is a tmpfs and fills up fast
        os.environ.setdefault(
            "TORCHINDUCTOR_CACHE_DIR",
            str(Path.home() / ".cache" / "torchinductor"),
        )
        try:
            _pretrain_status = "Loading config..."

            sr_map = {32000: "32k", 40000: "40k", 44100: "44k", 48000: "48k"}
            cfg_dir = Path(__file__).parent.parent / "kazeflow" / "configs"
            cfg_name = sr_map.get(sample_rate, "48k")
            config_path = cfg_dir / f"{cfg_name}.json"

            experiment_dir = Path("logs") / model_name
            existing_cfg = experiment_dir / "config.json"

            # Prefer existing experiment config (preserves manual edits like
            # n_disc_steps, c_r1, reset_gan_optimizers, etc.) over the
            # template. Fall back to the template for new experiments.
            if existing_cfg.exists():
                with open(existing_cfg, "r") as f:
                    config = json.load(f)
                logger.info(f"Loaded existing config from {existing_cfg}")
            else:
                with open(config_path, "r") as f:
                    config = json.load(f)

            # Apply UI overrides (always take effect regardless of saved config)
            config["train"]["precision"] = precision
            config["train"]["torch_compile"] = torch_compile
            config["train"]["compile_mode"] = compile_mode
            config["train"]["gan_loss_type"] = gan_loss_type
            if batch_size > 0:
                config["train"]["batch_size"] = batch_size
            if save_every > 0:
                config["train"]["save_every"] = save_every
            if epochs > 0:
                config["train"]["epochs"] = epochs
            if cfm_warmup_epochs >= 0:
                config["train"]["cfm_warmup_epochs"] = cfm_warmup_epochs

            filelist_path = experiment_dir / "filelist.txt"

            if not filelist_path.exists():
                _pretrain_status = (
                    "Error: filelist.txt not found. "
                    "Run Preprocess first."
                )
                return

            # Auto-detect n_speakers from model_info or filelist
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
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split("|")
                        if len(parts) >= 2:
                            speaker_ids.add(parts[1].strip())
                config["model"]["n_speakers"] = max(len(speaker_ids), 1)

            import torch
            from kazeflow.train.pretrain import KazeFlowPretrainer

            _pretrain_status = (
                f"Initializing pretrainer "
                f"({config['model']['n_speakers']} speakers)..."
            )
            output_dir = str(experiment_dir)

            pretrainer = KazeFlowPretrainer(
                config=config,
                output_dir=output_dir,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            resume = resume_path if resume_path and resume_path.strip() else None

            _pretrain_status = "Pretraining..."
            pretrainer.train(
                filelist_path=str(filelist_path),
                dataset_root=output_dir,
                resume_path=resume,
                stop_event=_stop_event,
            )
            if _stop_event.is_set():
                _pretrain_status = "Pretraining stopped by user."
            else:
                _pretrain_status = "Pretraining complete!"

        except Exception as e:
            _pretrain_status = f"Pretraining failed: {e}"
            logger.exception("Pretraining failed")

    _pretrain_thread = threading.Thread(target=_train_fn, daemon=True)
    _pretrain_thread.start()
    return "Pretraining started!"


def get_pretrain_status():
    return _pretrain_status


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------

def create_pretrain_tab():
    """Build the pretraining Gradio tab."""
    with gr.Tab("Pretrain"):
        gr.Markdown(
            "## 🎓 Pretrain a Base Model\n"
            "Train a multi-speaker base that can be fine-tuned for any voice.\n\n"
            "> **Dataset layout:** Subdirectories `0/`, `1/`, `2/`... for multi-speaker — "
            "each folder holds one speaker's audio."
        )

        # ── Model Settings ─────────────────────────────────────────────
        with gr.Group():
            gr.Markdown("### Model Settings")
            with gr.Row():
                model_name = gr.Dropdown(
                    label="Model Name",
                    choices=get_models_list(),
                    interactive=True,
                    allow_custom_value=True,
                    info="Output goes to logs/<name>/",
                    scale=4,
                )
                refresh_btn = gr.Button("🔄 Refresh", size="sm", min_width=80, scale=0)
            sample_rate = gr.Radio(
                label="Sample Rate",
                choices=[32000, 40000, 44100, 48000],
                value=32000,
                info="Must match between Preprocess and Pretraining.",
            )

        # ── 1. Preprocess ────────────────────────────────────────────────
        with gr.Accordion("1 · Preprocess & Extract Features", open=True):
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
                    info="Peak-normalize each slice.",
                )

            with gr.Row():
                chunk_len = gr.Number(
                    label="Chunk Length (s)",
                    value=3.0, minimum=0.1,
                    info="Used in Simple mode.",
                    visible=True,
                )
                overlap_len = gr.Number(
                    label="Overlap Length (s)",
                    value=0.3, minimum=0.0,
                    info="Used in Simple mode.",
                    visible=True,
                )

            with gr.Row():
                loading_resampling = gr.Radio(
                    label="Audio Loader",
                    choices=["librosa", "ffmpeg"],
                    value="librosa",
                )
                num_processes = gr.Number(
                    label="Worker Processes",
                    value=0, precision=0, minimum=0,
                    info="0 = auto (CPU count)",
                )

            with gr.Row():
                process_effects = gr.Checkbox(
                    label="High-pass filter",
                    value=True,
                    info="48 Hz HPF to remove DC / sub-bass rumble.",
                )
                noise_reduction = gr.Checkbox(
                    label="Noise reduction",
                    value=False,
                    info="Spectral noise reduction (noisereduce).",
                )
                use_smart_cutter = gr.Checkbox(
                    label="SmartCutter",
                    value=True,
                    info="Silence removal before slicing (requires GPU).",
                )

            reduction_strength = gr.Slider(
                label="Noise Reduction Strength",
                minimum=0.0, maximum=1.0, step=0.05, value=0.7,
                visible=False,
            )

            preprocess_btn = gr.Button("▶ Preprocess & Extract Features", variant="primary")
            preprocess_status = gr.Textbox(label="Status", interactive=False, lines=2)

        # ── 2. Pretraining ───────────────────────────────────────────────
        with gr.Accordion("2 · Pretraining", open=True):
            with gr.Group():
                gr.Markdown("#### Precision & Compilation")
                precision = gr.Radio(
                    label="Precision",
                    choices=["fp32", "fp16", "bf16", "tf32", "tf32_fp16", "tf32_bf16"],
                    value="tf32_bf16",
                    info="bf16: BF16 AMP · tf32: TensorFloat-32 (Ampere+) · tf32_bf16: recommended (Ampere+)",
                )
                with gr.Row():
                    torch_compile = gr.Checkbox(
                        label="torch.compile",
                        value=False,
                        info="~10–30% speedup (PyTorch 2.0+).",
                    )
                    compile_mode = gr.Dropdown(
                        label="Compile Mode",
                        choices=["default", "max-autotune-no-cudagraphs"],
                        value="max-autotune-no-cudagraphs",
                        info="max-autotune-no-cudagraphs: kernel autotuning (slow first run).",
                    )
                with gr.Group():
                    gr.Markdown("#### GAN Loss")
                    gan_loss_type = gr.Dropdown(
                        label="GAN Loss Type",
                        choices=["hinge", "soft_hinge", "lsgan"],
                        value="soft_hinge",
                        info="Hinge: margin ±1, const grad −1 (BigVGAN) · Soft Hinge: disc=hinge, gen=softplus(-D) · LSGAN: quadratic targets 0/1.",
                    )

            with gr.Group():
                gr.Markdown("#### Schedule")
                with gr.Row():
                    batch_size = gr.Number(
                        label="Batch Size", value=16, precision=0, minimum=0,
                        info="0 = config default",
                    )
                    save_every = gr.Number(
                        label="Save Every (epochs)", value=1, precision=0, minimum=0,
                        info="0 = config default",
                    )
                    total_epochs = gr.Number(
                        label="Total Epochs", value=500, precision=0, minimum=0,
                        info="0 = config default",
                    )
                    cfm_warmup = gr.Number(
                        label="CFM Warmup Epochs", value=10, precision=0, minimum=-1,
                        info="-1 = config default · Phase 1: train flow only",
                    )

            with gr.Group():
                gr.Markdown("#### Resume")
                resume_ckpt = gr.Dropdown(
                    label="Resume Checkpoint (optional)",
                    choices=get_pretrain_checkpoints_list(),
                    interactive=True, allow_custom_value=True,
                    info="Continue from a previous run. Leave empty to start fresh.",
                )

            with gr.Row():
                train_btn = gr.Button("▶ Start Pretraining", variant="primary")
                stop_btn = gr.Button("⏹ Stop", variant="stop")
                status_btn = gr.Button("↻ Status")
            train_status = gr.Textbox(label="Pretraining Status", interactive=False, lines=2)

        # --- Events ---
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
            fn=run_pretrain_preprocess,
            inputs=[
                dataset_dir, model_name, sample_rate,
                cut_preprocess, process_effects, noise_reduction,
                reduction_strength, chunk_len, overlap_len,
                normalization_mode, loading_resampling,
                use_smart_cutter, num_processes,
            ],
            outputs=[preprocess_status],
        )
        train_btn.click(
            fn=start_pretraining,
            inputs=[model_name, sample_rate, precision, torch_compile,
                    compile_mode, gan_loss_type, batch_size, save_every,
                    total_epochs, cfm_warmup, resume_ckpt],
            outputs=[train_status],
        )
        stop_btn.click(fn=stop_pretraining, outputs=[train_status])
        status_btn.click(fn=get_pretrain_status, outputs=[train_status])
