"""
KazeFlow — Flow Matching + ChouwaGAN Voice Conversion

A high-quality voice conversion system using Conditional Flow Matching
for content-to-mel generation and ChouwaGAN for mel-to-waveform synthesis.
"""

import os
import sys
import logging
import gradio as gr

# Suppress Gradio telemetry HTTP noise (api.gradio.app, huggingface telemetry)
logging.getLogger("httpx").setLevel(logging.WARNING)
# Suppress Gradio's own startup banners logged through its internal logger
logging.getLogger("gradio").setLevel(logging.WARNING)
# Suppress inductor noise (utils warnings, autotune verbose output, select_algorithm)
logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)
logging.getLogger("torch._inductor.select_algorithm").setLevel(logging.ERROR)
logging.getLogger("torch._inductor.autotune_process").setLevel(logging.ERROR)
os.environ.setdefault("TORCHINDUCTOR_VERBOSE", "0")
os.environ.setdefault("AUTOTUNE_VERBOSE", "0")

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


_LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "images", "logo.png")


def _build_theme() -> gr.themes.Base:
    """Dark blue theme for KazeFlow."""
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#eef4ff",
            c100="#d9e5ff",
            c200="#b3ccff",
            c300="#809fff",
            c400="#4a7aff",
            c500="#3b6cff",
            c600="#2d5ae6",
            c700="#1e44bf",
            c800="#163399",
            c900="#0f2573",
            c950="#091a4d",
        ),
        secondary_hue=gr.themes.Color(
            c50="#e8ecf4",
            c100="#d0d6e6",
            c200="#a8b2cc",
            c300="#7d8aad",
            c400="#5a6a91",
            c500="#455578",
            c600="#374566",
            c700="#2b3652",
            c800="#1f2840",
            c900="#171e33",
            c950="#111726",
        ),
        neutral_hue=gr.themes.Color(
            c50="#e6e9f0",
            c100="#ccd1de",
            c200="#a3aac0",
            c300="#7a83a2",
            c400="#5e6888",
            c500="#4a5270",
            c600="#3b425a",
            c700="#2d3348",
            c800="#222838",
            c900="#1a1f2e",
            c950="#141824",
        ),
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ).set(
        # ── Page ──────────────────────────────────────────────────────
        body_background_fill="#141824",
        body_background_fill_dark="#141824",
        body_text_color="#d0d6e6",
        body_text_color_dark="#d0d6e6",
        body_text_color_subdued="#7a83a2",
        body_text_color_subdued_dark="#7a83a2",
        # ── Blocks / panels ───────────────────────────────────────────
        background_fill_primary="#1a1f2e",
        background_fill_primary_dark="#1a1f2e",
        background_fill_secondary="#1e2536",
        background_fill_secondary_dark="#1e2536",
        block_background_fill="#1e2536",
        block_background_fill_dark="#1e2536",
        block_border_color="#2d3348",
        block_border_color_dark="#2d3348",
        block_label_background_fill="#222838",
        block_label_background_fill_dark="#222838",
        block_label_text_color="#a3aac0",
        block_label_text_color_dark="#a3aac0",
        block_title_text_color="#d0d6e6",
        block_title_text_color_dark="#d0d6e6",
        # ── Inputs ────────────────────────────────────────────────────
        input_background_fill="#222838",
        input_background_fill_dark="#222838",
        input_border_color="#2d3348",
        input_border_color_dark="#2d3348",
        input_border_color_focus="#3b6cff",
        input_border_color_focus_dark="#3b6cff",
        input_placeholder_color="#5e6888",
        input_placeholder_color_dark="#5e6888",
        # ── Buttons ───────────────────────────────────────────────────
        button_primary_background_fill="#3b6cff",
        button_primary_background_fill_dark="#3b6cff",
        button_primary_background_fill_hover="#4a7aff",
        button_primary_background_fill_hover_dark="#4a7aff",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="#222838",
        button_secondary_background_fill_dark="#222838",
        button_secondary_background_fill_hover="#2d3348",
        button_secondary_background_fill_hover_dark="#2d3348",
        button_secondary_text_color="#d0d6e6",
        button_secondary_text_color_dark="#d0d6e6",
        button_cancel_background_fill="#4a2030",
        button_cancel_background_fill_dark="#4a2030",
        button_cancel_background_fill_hover="#5c2840",
        button_cancel_background_fill_hover_dark="#5c2840",
        button_cancel_text_color="#ff8a9e",
        button_cancel_text_color_dark="#ff8a9e",
        # ── Borders / shadows ─────────────────────────────────────────
        border_color_accent="#3b6cff",
        border_color_accent_dark="#3b6cff",
        border_color_primary="#2d3348",
        border_color_primary_dark="#2d3348",
        shadow_drop="0 2px 8px rgba(0,0,0,0.35)",
        shadow_drop_lg="0 4px 16px rgba(0,0,0,0.45)",
        # ── Accordion / panels ────────────────────────────────────────
        panel_background_fill="#1a1f2e",
        panel_background_fill_dark="#1a1f2e",
        panel_border_color="#2d3348",
        panel_border_color_dark="#2d3348",
        # ── Checkbox / radio ──────────────────────────────────────────
        checkbox_background_color="#222838",
        checkbox_background_color_dark="#222838",
        checkbox_border_color="#3b425a",
        checkbox_border_color_dark="#3b425a",
        checkbox_border_color_focus="#3b6cff",
        checkbox_border_color_focus_dark="#3b6cff",
        checkbox_label_background_fill="#1e2536",
        checkbox_label_background_fill_dark="#1e2536",
        # ── Slider ────────────────────────────────────────────────────
        slider_color="#3b6cff",
        slider_color_dark="#3b6cff",
        # ── Table ─────────────────────────────────────────────────────
        table_border_color="#2d3348",
        table_border_color_dark="#2d3348",
        table_even_background_fill="#1e2536",
        table_even_background_fill_dark="#1e2536",
        table_odd_background_fill="#1a1f2e",
        table_odd_background_fill_dark="#1a1f2e",
        table_row_focus="#222838",
        table_row_focus_dark="#222838",
        # ── Links ─────────────────────────────────────────────────────
        link_text_color="#4a7aff",
        link_text_color_dark="#4a7aff",
        link_text_color_hover="#809fff",
        link_text_color_hover_dark="#809fff",
        link_text_color_visited="#7a83a2",
        link_text_color_visited_dark="#7a83a2",
        link_text_color_active="#3b6cff",
        link_text_color_active_dark="#3b6cff",
    )


_HIDE_FOOTER_CSS = """
footer { display: none !important; }
"""


def create_app():
    with gr.Blocks(title="KazeFlow", theme=_build_theme(), css=_HIDE_FOOTER_CSS) as app:
        gr.Markdown(
            "# KazeFlow\n"
            "High-quality voice conversion using Conditional Flow Matching"
        )

        create_inference_tab()
        create_training_tab()
        create_pretrain_tab()

    return app


if __name__ == "__main__":
    # Ensure SPIN v2, RMVPE, and SmartCutter weights are present before launching.
    check_and_download_prerequisites()

    # Ensure mute reference files exist (no-op if already generated).
    generate_mutes()

    _share = "--share" in sys.argv
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=_share,
        inbrowser=True,
        favicon_path=_LOGO_PATH,
    )
