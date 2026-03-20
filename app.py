"""
KazeFlow — Flow Matching + ChouwaGAN Voice Conversion

A high-quality voice conversion system using Conditional Flow Matching
for content-to-mel generation and ChouwaGAN for mel-to-waveform synthesis.
"""

import os
import logging
import gradio as gr

from tabs.inference import create_inference_tab
from tabs.train import create_training_tab
from tabs.pretrain import create_pretrain_tab
from kazeflow.tools.prerequisites_download import check_and_download_prerequisites
from kazeflow.tools.generate_mutes import generate_mutes

# Reduce CUDA memory fragmentation — recommended by PyTorch for long training runs.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Ensure SPIN v2, RMVPE, and SmartCutter weights are present before launching.
check_and_download_prerequisites()

# Ensure mute reference files exist (no-op if already generated).
generate_mutes()


def create_app():
    with gr.Blocks(title="KazeFlow") as app:
        gr.Markdown(
            "# KazeFlow\n"
            "**Flow Matching + ChouwaGAN Voice Conversion**\n\n"
            "High-quality voice conversion using Conditional Flow Matching "
            "with SPIN v2 content features and ChouwaGAN vocoder."
        )

        create_inference_tab()
        create_training_tab()
        create_pretrain_tab()

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
