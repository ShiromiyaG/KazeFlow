"""
KazeFlow — Flow Matching + ChouwaGAN Voice Conversion

A high-quality voice conversion system using Conditional Flow Matching
for content-to-mel generation and ChouwaGAN for mel-to-waveform synthesis.
"""

import os
import logging
import gradio as gr

# Suppress Gradio telemetry HTTP noise (api.gradio.app, huggingface telemetry)
logging.getLogger("httpx").setLevel(logging.WARNING)
# Suppress Gradio's own startup banners logged through its internal logger
logging.getLogger("gradio").setLevel(logging.WARNING)
# Suppress inductor "Not enough SMs to use max_autotune_gemm mode" noise
logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)

from tabs.inference import create_inference_tab
from tabs.train import create_training_tab
from tabs.pretrain import create_pretrain_tab
from kazeflow.tools.prerequisites_download import check_and_download_prerequisites
from kazeflow.tools.generate_mutes import generate_mutes

# Reduce CUDA memory fragmentation — recommended by PyTorch for long training runs.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


class _KazeFormatter(logging.Formatter):
    """Custom formatter: replaces INFO/WARNING/ERROR with compact symbols."""

    _PREFIXES = {
        logging.DEBUG:    " · ",
        logging.INFO:     " › ",
        logging.WARNING:  " ⚠ ",
        logging.ERROR:    " ✗ ",
        logging.CRITICAL: " ✗ ",
    }

    def format(self, record: logging.LogRecord) -> str:
        # Section headers (start with ──) get no prefix for a clean look
        msg = record.msg if isinstance(record.msg, str) else str(record.msg)
        if msg.startswith("──"):
            record.sym = ""
        else:
            record.sym = self._PREFIXES.get(record.levelno, " • ")
        return super().format(record)


_handler = logging.StreamHandler()
_handler.setFormatter(_KazeFormatter("%(sym)s%(message)s"))

# Force our handler on the root logger — basicConfig is a no-op when
# any library has already configured logging before us.
_root = logging.getLogger()
_root.handlers.clear()
_root.addHandler(_handler)
_root.setLevel(logging.INFO)

# Ensure SPIN v2, RMVPE, and SmartCutter weights are present before launching.
check_and_download_prerequisites()

# Ensure mute reference files exist (no-op if already generated).
generate_mutes()


_LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "images", "logo.png")


def create_app():
    with gr.Blocks(title="KazeFlow") as app:
        gr.Markdown(
            "# KazeFlow\n"
            "High-quality voice conversion using Conditional Flow Matching"
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
        theme=gr.themes.Glass(),
        favicon_path=_LOGO_PATH,
    )
