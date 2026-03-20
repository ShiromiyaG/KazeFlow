"""
KazeFlow Preparing Files — config copy and filelist generation.

Ported from Codename-fork-4/rvc/train/extract/preparing_files.py.

Key differences from the original:
- RVC-specific paths replaced with KazeFlow equivalents.
- Feature directories are spin/, f0/, mel/ (not extracted/, f0/, f0_voiced/).
- Filelist format: "stem|spk_id"  (matches KazeFlowDataset).
- Mute files are from logs/mute_spin_v2/ (SPIN v2 is the only embedder).
- generate_config copies from kazeflow/configs/{sr}.json instead of
  rvc/configs/{vocoder_arch}/{sr}.json.
- No global Config() instantiation.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from random import shuffle

logger = logging.getLogger("kazeflow.preprocess.preparing_files")

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent  # kazeflow/preprocess/ -> kazeflow/ -> root
_CONFIGS_DIR = _ROOT / "kazeflow" / "configs"
_LOGS_DIR = _ROOT / "logs"

# KazeFlow uses a single embedder (SPIN v2), so there is one mute folder.
_MUTE_DIR = _LOGS_DIR / "mute_spin_v2"


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_config(sample_rate: int, model_path: str) -> None:
    """
    Copy the KazeFlow JSON config for *sample_rate* into *model_path*/config.json.

    Args:
        sample_rate: One of 32000, 40000, 44100, 48000.
        model_path:  Experiment directory (e.g. logs/my_model).

    Raises:
        FileNotFoundError: If the source config does not exist.
    """
    sr_map = {32000: "32k", 40000: "40k", 44100: "44k", 48000: "48k"}
    cfg_name = sr_map.get(sample_rate, "48k")
    src = _CONFIGS_DIR / f"{cfg_name}.json"
    dst = Path(model_path) / "config.json"

    if not src.exists():
        raise FileNotFoundError(
            f"Config not found: {src}. "
            f"Expected one of: {[str(_CONFIGS_DIR / f'{v}.json') for v in sr_map.values()]}"
        )

    if dst.exists():
        logger.info("Config already exists at %s — skipping copy.", dst)
    else:
        shutil.copyfile(src, dst)
        logger.info("Config saved at %s", dst)


def generate_filelist(
    model_path: str,
    sample_rate: int,
    include_mutes: int = 2,
) -> None:
    """
    Build filelist.txt from the intersection of feature directories and optionally
    append mute entries for training stability.

    KazeFlow directory layout expected under *model_path*:

        sliced_audios/   *.wav    (GT audio)
        spin/            *.npy    (SPIN v2 embeddings, shape [T, 768])
        f0/              *.npy    (F0 contours, shape [T,] float64 Hz)
        mel/             *.npy    (log mel, shape [n_mels, T])

    Filelist line format::

        {stem}|{speaker_id}

    The stem encodes the speaker ID as a prefix: ``{sid}_{original_name}``.
    The ``|``-separated speaker ID is extracted from the stem prefix.

    Mute entries are appended *include_mutes* times per speaker and point to
    ``logs/mute_spin_v2/`` pre-generated mute files.

    Args:
        model_path:    Experiment directory (e.g. "logs/my_model").
        sample_rate:   Target sample rate (used to pick the mute wav).
        include_mutes: Number of mute entries per speaker (0 = disabled).

    Raises:
        FileNotFoundError: If *model_path* or required subdirs are missing.
    """
    model_path = Path(model_path)

    spin_dir = model_path / "spin"
    f0_dir = model_path / "f0"
    mel_dir = model_path / "mel"
    gt_dir = model_path / "sliced_audios"

    for d in (spin_dir, f0_dir, mel_dir, gt_dir):
        if not d.exists():
            raise FileNotFoundError(
                f"Required directory not found: {d}. "
                "Run the preprocessing pipeline before generating the filelist."
            )

    # ── Intersect stems across all feature dirs ───────────────────────────
    def _stems(directory: Path, suffix: str) -> set:
        return {p.stem for p in directory.glob(f"*{suffix}")}

    spin_stems = _stems(spin_dir, ".npy")
    f0_stems = _stems(f0_dir, ".npy")
    mel_stems = _stems(mel_dir, ".npy")
    gt_stems = _stems(gt_dir, ".wav")

    valid_stems = spin_stems & f0_stems & mel_stems & gt_stems

    logger.info(
        "Filelist intersection: spin=%d f0=%d mel=%d gt=%d valid=%d",
        len(spin_stems), len(f0_stems), len(mel_stems), len(gt_stems), len(valid_stems),
    )

    # ── Build options and collect speaker IDs ────────────────────────────
    options = []
    sids: list[str] = []

    for stem in sorted(valid_stems):
        # Stem format: "{sid}_{name}" — extract sid prefix
        sid = stem.split("_")[0]
        if sid not in sids:
            sids.append(sid)
        options.append(f"{stem}|{sid}")

    # ── Append mute entries ───────────────────────────────────────────────
    if include_mutes > 0:
        mute_wav = _MUTE_DIR / "sliced_audios" / f"mute{sample_rate}.wav"
        mute_spin = _MUTE_DIR / "spin" / "mute.npy"
        mute_f0 = _MUTE_DIR / "f0" / "mute.npy"
        mute_mel = _MUTE_DIR / "mel" / "mute.npy"

        missing_mutes = [
            p for p in (mute_wav, mute_spin, mute_f0, mute_mel) if not p.exists()
        ]
        if missing_mutes:
            logger.warning(
                "Mute files not found — skipping mute entries. Missing:\n  %s\n"
                "Run kazeflow/tools/generate_mutes.py to create them.",
                "\n  ".join(str(p) for p in missing_mutes),
            )
        else:
            for sid in sids * include_mutes:
                options.append(f"mute|{sid}")

    # ── Shuffle and write ────────────────────────────────────────────────
    shuffle(options)

    filelist_path = model_path / "filelist.txt"
    with open(filelist_path, "w", encoding="utf-8") as f:
        f.write("\n".join(options) + "\n")

    logger.info(
        "Filelist written: %d entries, %d speaker(s) → %s",
        len(options), len(sids), filelist_path,
    )

    # ── Update model_info.json ────────────────────────────────────────────
    info_path = model_path / "model_info.json"
    data: dict = {}
    if info_path.exists():
        with open(info_path, "r") as f:
            data = json.load(f)

    data["speakers_id"] = len(sids)
    data["n_speakers"] = len(sids)
    data["total_samples"] = len(options)
    data["sample_rate"] = sample_rate

    with open(info_path, "w") as f:
        json.dump(data, f, indent=2)
