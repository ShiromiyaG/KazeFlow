"""KazeFlow - Vocoder Training Tab.

Stage 2 of 3: Train the ChouwaGAN vocoder using a frozen RFM model for mel generation.

Requires:
  - A pretrain_*.pt checkpoint from Stage 1 (RFM pretrain)
  - The same preprocessed dataset used for pretraining

Saves vocoder_*.pt checkpoints (RFM + vocoder + discriminator)
compatible with Stage 3 (fine-tune).
"""

import json
import logging
import os
import threading
from pathlib import Path

import gradio as gr

from tabs import _gpu_caps, get_precision_choices

logger = logging.getLogger("kazeflow.ui.vocoder")

_GPU_CAPS = _gpu_caps()

_vocoder_thread = None
_vocoder_status = "Idle"
_stop_event = threading.Event()

_NOW_DIR = Path(os.getcwd())
_LOGS_DIR = _NOW_DIR / "logs"


def get_models_list() -> list:
    if not _LOGS_DIR.exists():
        return []
    return sorted([
        d for d in os.listdir(_LOGS_DIR)
        if os.path.isdir(_LOGS_DIR / d)
        and d not in ("zips", "mute", "reference")
        and not d.startswith("mute")
    ])


def get_rfm_checkpoints_list() -> list:
    """List pretrain_*.pt (RFM) checkpoints under logs/."""
    results = []
    for root, _, files in os.walk(_LOGS_DIR):
        for f in sorted(files):
            if f.startswith("pretrain_") and f.endswith(".pt"):
                results.append(os.path.relpath(os.path.join(root, f), _NOW_DIR))
    return sorted(results)


def get_vocoder_checkpoints_list() -> list:
    """List vocoder_*.pt checkpoints under logs/."""
    results = []
    for root, _, files in os.walk(_LOGS_DIR):
        for f in sorted(files):
            if f.startswith("vocoder_") and f.endswith(".pt"):
                results.append(os.path.relpath(os.path.join(root, f), _NOW_DIR))
    return sorted(results)


def refresh_vocoder_lists(rfm_val: str, resume_val: str):
    rfm_ckpts = get_rfm_checkpoints_list()
    voc_ckpts = get_vocoder_checkpoints_list()
    return (
        gr.update(choices=get_models_list()),
        gr.update(choices=rfm_ckpts,
                  value=rfm_val if rfm_val in rfm_ckpts else None),
        gr.update(choices=voc_ckpts,
                  value=resume_val if resume_val in voc_ckpts else None),
    )


def stop_vocoder_training():
    global _vocoder_status
    _stop_event.set()
    _vocoder_status = "Stop requested — finishing current epoch..."
    return _vocoder_status


def get_vocoder_status():
    return _vocoder_status


def start_vocoder_training(
    model_name: str,
    rfm_checkpoint: str,
    resume_checkpoint: str,
    batch_size: int,
    save_every: int,
    epochs: int,
    ode_steps: int,
    gt_mel_ratio: float,
    precision: str,
    torch_compile: bool,
    compile_mode: str,
    lr_scheduler: str,
    gan_loss_type: str,
):
    """Launch ChouwaGAN vocoder training in a background thread."""
    global _vocoder_thread, _vocoder_status, _stop_event

    if not model_name or not model_name.strip():
        return "Error: Model Name is required."

    if not rfm_checkpoint or not rfm_checkpoint.strip():
        return "Error: RFM Checkpoint is required (run Pretrain first)."

    rfm_path = Path(rfm_checkpoint.strip())
    if not rfm_path.exists():
        return f"Error: RFM checkpoint not found: {rfm_path}"

    if _vocoder_thread is not None and _vocoder_thread.is_alive():
        return "Vocoder training already in progress!"

    _stop_event.clear()

    def _train_fn():
        global _vocoder_status
        os.environ.setdefault(
            "TORCHINDUCTOR_CACHE_DIR",
            str(Path.home() / ".cache" / "torchinductor"),
        )
        try:
            _vocoder_status = "Loading config..."
            experiment_dir = Path("logs") / model_name
            existing_cfg = experiment_dir / "config.json"

            from kazeflow.configs import load_config
            if existing_cfg.exists():
                with open(existing_cfg, "r") as f:
                    config = json.load(f)
                logger.info(f"Loaded existing config from {existing_cfg}")
            else:
                config = load_config(preset="rfm")

            # Force RFM
            config["model"]["architecture"] = "rfm"

            # UI overrides
            config["train"]["precision"] = precision
            config["train"]["torch_compile"] = torch_compile
            config["train"]["compile_mode"] = compile_mode
            config["train"]["lr_scheduler"] = lr_scheduler
            config["train"]["gan_loss_type"] = gan_loss_type

            if batch_size > 0:
                config["train"]["batch_size"] = batch_size
            if save_every > 0:
                config["train"]["save_every"] = save_every
            if epochs > 0:
                config["train"]["epochs"] = epochs

            config["train"]["ode_steps_train_max"] = int(ode_steps)
            config["train"]["ode_steps_train_min"] = max(1, int(ode_steps) - 2)
            config["train"]["gt_mel_ratio"] = float(gt_mel_ratio)

            filelist_path = experiment_dir / "filelist.txt"
            if not filelist_path.exists():
                _vocoder_status = (
                    "Error: filelist.txt not found. "
                    "Run Preprocess in the Pretrain tab first."
                )
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
            from kazeflow.train.vocoder_trainer import KazeFlowVocoderTrainer

            _vocoder_status = (
                f"Initializing Vocoder trainer "
                f"({config['model']['n_speakers']} speakers)..."
            )
            output_dir = str(experiment_dir)

            trainer = KazeFlowVocoderTrainer(
                config=config,
                output_dir=output_dir,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            resume = resume_checkpoint.strip() \
                if resume_checkpoint and resume_checkpoint.strip() else None

            _vocoder_status = "Training ChouwaGAN vocoder (RFM frozen)..."
            trainer.train(
                filelist_path=str(filelist_path),
                dataset_root=output_dir,
                rfm_path=str(rfm_path),
                resume_path=resume,
                stop_event=_stop_event,
            )
            if _stop_event.is_set():
                _vocoder_status = "Vocoder training stopped by user."
            else:
                _vocoder_status = (
                    "Vocoder training complete! "
                    "→ Now use the saved vocoder_*.pt for fine-tuning (Train tab)."
                )

        except Exception as e:
            _vocoder_status = f"Vocoder training failed: {e}"
            logger.exception("Vocoder training failed")

    _vocoder_thread = threading.Thread(target=_train_fn, daemon=True)
    _vocoder_thread.start()
    return "Vocoder training started!"


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------

def create_vocoder_tab():
    """Build the Vocoder Training Gradio tab."""
    with gr.Tab("Vocoder"):
        gr.Markdown(
            "## Stage 2 · Train ChouwaGAN Vocoder\n"
            "Train the vocoder using a **frozen RFM model** to generate training mels. "
            "This isolates vocoder training from RFM training for faster, more stable convergence.\n\n"
            "> **Required:** A `pretrain_*.pt` checkpoint from Stage 1 (Pretrain tab)\n\n"
            "> **Output:** `vocoder_*.pt` — contains frozen RFM + trained vocoder + discriminator. "
            "Load this in the **Train tab** for fine-tuning."
        )

        with gr.Group():
            gr.Markdown("### Setup")
            with gr.Row(equal_height=True):
                model_name = gr.Dropdown(
                    label="Model Name",
                    choices=get_models_list(),
                    interactive=True,
                    allow_custom_value=True,
                    info="Same experiment as used in Pretrain. Output goes to logs/<name>/",
                    scale=6,
                )
                refresh_btn = gr.Button("🔄", size="sm", min_width=36, scale=0)

            rfm_checkpoint = gr.Dropdown(
                label="RFM Pretrain Checkpoint (pretrain_*.pt)",
                choices=get_rfm_checkpoints_list(),
                interactive=True,
                allow_custom_value=True,
                info="Required: the RFM model to use for mel generation (will be frozen).",
            )

        with gr.Accordion("Training Schedule", open=True):
            with gr.Row():
                batch_size = gr.Number(
                    label="Batch Size", value=8, precision=0, minimum=0,
                    info="Smaller than pretrain because GT waveform is needed.",
                )
                save_every = gr.Number(
                    label="Save Every (epochs)", value=5, precision=0, minimum=0,
                )
                total_epochs = gr.Number(
                    label="Total Epochs", value=200, precision=0, minimum=0,
                )

            with gr.Row():
                ode_steps = gr.Slider(
                    label="ODE Steps for Mel Generation",
                    minimum=1, maximum=8, step=1, value=4,
                    info="Steps used when RFM generates training mels. 4 = balanced quality/speed.",
                )
                gt_mel_ratio = gr.Slider(
                    label="GT Mel Ratio",
                    minimum=0.0, maximum=1.0, step=0.05, value=0.3,
                    info="Fraction of batches where GT mel (instead of RFM-generated mel) is used. "
                         "0.3 = 30% GT, 70% RFM-generated.",
                )

        with gr.Group():
            gr.Markdown("#### Resume")
            resume_ckpt = gr.Dropdown(
                label="Resume Checkpoint (vocoder_*.pt, optional)",
                choices=get_vocoder_checkpoints_list(),
                value=None,
                interactive=True, allow_custom_value=True,
                info="Continue from a previous vocoder training run.",
            )

        _prec_choices, _prec_default = get_precision_choices(_GPU_CAPS)
        with gr.Accordion("⚙ Advanced Settings", open=False):
            with gr.Group():
                gr.Markdown("#### Precision & Compilation")
                precision = gr.Radio(
                    label="Precision", choices=_prec_choices, value=_prec_default)
                with gr.Row(visible=_GPU_CAPS["has_compile"]):
                    torch_compile = gr.Checkbox(
                        label="torch.compile", value=False,
                        info="Compile vocoder/discriminator. Not recommended for short runs.",
                    )
                    compile_mode = gr.Dropdown(
                        label="Compile Mode",
                        choices=["default", "reduce-overhead",
                                 "max-autotune-no-cudagraphs", "max-autotune"],
                        value="default",
                    )

            with gr.Group():
                gr.Markdown("#### LR & GAN")
                with gr.Row():
                    lr_scheduler = gr.Dropdown(
                        label="LR Scheduler",
                        choices=["cosine", "cosine_warmup_restarts", "exponential"],
                        value="cosine",
                    )
                    gan_loss_type = gr.Dropdown(
                        label="GAN Loss Type",
                        choices=["hinge", "soft_hinge", "lsgan"],
                        value="soft_hinge",
                        info="soft_hinge: recommended for SAN discriminator.",
                    )

        with gr.Row():
            train_btn = gr.Button("▶ Start Vocoder Training", variant="primary")
            stop_btn = gr.Button("⏹ Stop", variant="stop")
            status_btn = gr.Button("↻ Status")
        train_status = gr.Textbox(
            label="Vocoder Training Status", interactive=False, lines=2)

        # ── Events ──────────────────────────────────────────────────────
        _clear_if_empty = lambda v: gr.update(value=None) if not v or not v.strip() else gr.update()
        resume_ckpt.change(fn=_clear_if_empty, inputs=[resume_ckpt], outputs=[resume_ckpt])

        refresh_btn.click(
            fn=refresh_vocoder_lists,
            inputs=[rfm_checkpoint, resume_ckpt],
            outputs=[model_name, rfm_checkpoint, resume_ckpt],
        )

        train_btn.click(
            fn=start_vocoder_training,
            inputs=[
                model_name, rfm_checkpoint, resume_ckpt,
                batch_size, save_every, total_epochs,
                ode_steps, gt_mel_ratio,
                precision, torch_compile, compile_mode,
                lr_scheduler, gan_loss_type,
            ],
            outputs=[train_status],
        )
        stop_btn.click(fn=stop_vocoder_training, outputs=[train_status])
        status_btn.click(fn=get_vocoder_status, outputs=[train_status])
