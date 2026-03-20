"""
KazeFlow Audio Preprocessing — slicing, cutting, normalisation.

Ported from Codename-fork-4/rvc/train/preprocess/preprocess.py.

Key differences from the original:
- sys.exit() calls replaced with ValueError / RuntimeError raises.
- RVC-specific imports replaced with kazeflow equivalents.
- SmartCutter ckpt_dir points to kazeflow/models/pretrained/smartcutter/.
- load_audio / load_audio_ffmpeg defined locally (no rvc.lib.utils dependency).
"""

import json
import logging
import multiprocessing
import os
import shutil
import time
from fractions import Fraction
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import soxr
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm

logger = logging.getLogger("kazeflow.preprocess.audio")

# ── Constants ──────────────────────────────────────────────────────────────────
OVERLAP = 0.3
PERCENTAGE = 3.0
MAX_AMPLITUDE = 0.9
ALPHA = 0.75
HIGH_PASS_CUTOFF = 48
SAMPLE_RATE_16K = 16000
RES_TYPE = "soxr_vhq"

# SmartCutter checkpoint dir (relative to package root)
_SMARTCUTTER_CKPT_DIR = (
    Path(__file__).parent.parent / "models" / "pretrained" / "smartcutter"
)


# ── Audio loading helpers ──────────────────────────────────────────────────────

def load_audio(file: str, sample_rate: int) -> np.ndarray:
    """
    Load an audio file to a mono float32 numpy array at *sample_rate* Hz.
    Uses soundfile + librosa resampling (SoXr VHQ).
    """
    try:
        file = file.strip(' ').strip('"').strip('\n').strip('"').strip(' ')
        audio, sr = sf.read(file)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.T)
        if sr != sample_rate:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=sample_rate, res_type=RES_TYPE
            )
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}") from error
    return audio.flatten()


def load_audio_ffmpeg(
    source,
    sample_rate: int = 48000,
    source_sr: Optional[int] = None,
) -> np.ndarray:
    """
    Load / resample audio via FFmpeg.

    Args:
        source: File path (str) or in-memory numpy audio chunk.
        sample_rate: Target sample rate.
        source_sr: Source sample rate — required when *source* is a numpy array.

    Returns:
        Mono float32 numpy array at *sample_rate* Hz.
    """
    import ffmpeg

    if isinstance(source, str):
        source = source.strip(' ').strip('"').strip('\n').strip('"').strip(' ')
        if not os.path.exists(source):
            raise FileNotFoundError(
                f"The audio file was not found at the provided path: {source}"
            )
        try:
            out, _ = (
                ffmpeg.input(source, threads=0)
                .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sample_rate)
                .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(
                f"Failed to load audio file '{source}':\n{e.stderr.decode()}"
            ) from e

    elif isinstance(source, np.ndarray):
        if source_sr is None:
            raise ValueError("source_sr must be provided when passing a NumPy array.")
        if source.dtype != np.float32:
            source = source.astype(np.float32)
        if source.ndim > 1:
            source = np.mean(source, axis=1)
        try:
            import ffmpeg as _ffmpeg
            process = (
                _ffmpeg
                .input("pipe:0", format="f32le", acodec="pcm_f32le",
                       ar=source_sr, ac=1)
                .output("pipe:1", format="f32le", acodec="pcm_f32le", ar=sample_rate)
                .run_async(pipe_stdin=True, pipe_stdout=True, quiet=True)
            )
            out, _ = process.communicate(input=source.tobytes())
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred while processing audio chunk: {e}"
            ) from e
    else:
        raise ValueError(
            "Invalid source type. Must be a file path (str) or a NumPy array (np.ndarray)."
        )

    return np.frombuffer(out, np.float32).flatten()


# ── Helpers ────────────────────────────────────────────────────────────────────

def secs_to_samples(secs: float, sr: int) -> int:
    """Return an exact integer number of samples for *secs* at *sr* Hz."""
    frac = Fraction(str(secs)) * sr
    if frac.denominator != 1:
        raise ValueError(f"{secs}s × {sr}Hz is not an integer sample count")
    return frac.numerator


def format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"


def save_dataset_duration(file_path: str, dataset_duration: float) -> None:
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    data.update({
        "total_dataset_duration": format_duration(dataset_duration),
        "total_seconds": dataset_duration,
    })

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def cleanup_dirs(exp_dir: str) -> None:
    for sub in ("sliced_audios", "sliced_audios_16k"):
        d = os.path.join(exp_dir, sub)
        if os.path.exists(d):
            shutil.rmtree(d)
            logger.info("Deleted directory: %s", d)


# ── PreProcess class ───────────────────────────────────────────────────────────

class PreProcess:
    """
    Slices / chunks a single audio file and saves GT + 16 kHz versions.

    Output layout (inside *exp_dir*):
        sliced_audios/     — ground-truth at target sr
        sliced_audios_16k/ — downsampled copies at 16 kHz
    """

    def __init__(self, sr: int, exp_dir: str):
        from kazeflow.preprocess.slicer import Slicer

        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        self.b_high, self.a_high = signal.butter(
            N=5, Wn=HIGH_PASS_CUTOFF, btype="high", fs=sr
        )
        self.exp_dir = exp_dir
        self.gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
        self.wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _resample_16k(
        self, audio: np.ndarray, loading_resampling: str
    ) -> np.ndarray:
        if loading_resampling == "librosa":
            return librosa.resample(
                audio, orig_sr=self.sr, target_sr=SAMPLE_RATE_16K, res_type=RES_TYPE
            )
        return load_audio_ffmpeg(audio, sample_rate=SAMPLE_RATE_16K, source_sr=self.sr)

    def _save_segment(
        self,
        audio: np.ndarray,
        sid: int,
        idx0: int,
        idx1: int,
        loading_resampling: str,
    ) -> None:
        stem = f"{sid}_{idx0}_{idx1}"
        wavfile.write(
            os.path.join(self.gt_wavs_dir, f"{stem}.wav"),
            self.sr,
            audio.astype(np.float32),
        )
        chunk_16k = self._resample_16k(audio, loading_resampling)
        wavfile.write(
            os.path.join(self.wavs16k_dir, f"{stem}.wav"),
            SAMPLE_RATE_16K,
            chunk_16k.astype(np.float32),
        )

    # ── Cutting modes ──────────────────────────────────────────────────────

    def process_audio_segment(
        self,
        audio: np.ndarray,
        sid: int,
        idx0: int,
        idx1: int,
        loading_resampling: str,
    ) -> None:
        """Save a single pre-cut segment (used by 'Skip' and 'Automatic' modes)."""
        self._save_segment(audio, sid, idx0, idx1, loading_resampling)

    def simple_cut(
        self,
        audio: np.ndarray,
        sid: int,
        idx0: int,
        chunk_len: float,
        overlap_len: float,
        loading_resampling: str,
    ) -> None:
        """Split audio into fixed-length overlapping chunks ('Simple' mode)."""
        chunk_len_smpl = secs_to_samples(chunk_len, self.sr)
        stride = chunk_len_smpl - secs_to_samples(overlap_len, self.sr)

        slice_idx = 0
        i = 0
        while i < len(audio):
            chunk = audio[i : i + chunk_len_smpl]

            if len(chunk) < chunk_len_smpl:
                padding_needed = chunk_len_smpl - len(chunk)
                if len(chunk) > self.sr * 1.0:
                    chunk = np.concatenate(
                        (chunk, np.zeros(padding_needed, dtype=np.float32))
                    )
                    logger.info(
                        "Padded final slice %s_%s_%s with %d samples.",
                        sid, idx0, slice_idx, padding_needed,
                    )
                else:
                    break

            stem = f"{sid}_{idx0}_{slice_idx}"
            wavfile.write(
                os.path.join(self.gt_wavs_dir, f"{stem}.wav"),
                self.sr,
                chunk.astype(np.float32),
            )
            chunk_16k = self._resample_16k(chunk, loading_resampling)
            wavfile.write(
                os.path.join(self.wavs16k_dir, f"{stem}.wav"),
                SAMPLE_RATE_16K,
                chunk_16k.astype(np.float32),
            )

            slice_idx += 1
            i += stride

    def process_audio(
        self,
        path: str,
        idx0: int,
        sid: int,
        cut_preprocess: str,
        process_effects: bool,
        noise_reduction: bool,
        reduction_strength: float,
        chunk_len: float,
        overlap_len: float,
        loading_resampling: str,
    ) -> float:
        """
        Load, optionally filter, and slice a single audio file.

        Args:
            path: Source audio file.
            idx0: File index (used in output filenames).
            sid: Speaker ID.
            cut_preprocess: "Skip" | "Simple" | "Automatic"
            process_effects: Apply high-pass filter when True.
            noise_reduction: Apply noisereduce when True.
            reduction_strength: Noise reduction aggressiveness (0-1).
            chunk_len: Chunk length in seconds (Simple mode).
            overlap_len: Overlap length in seconds (Simple mode).
            loading_resampling: "librosa" | "ffmpeg"

        Returns:
            Duration of the processed audio in seconds.
        """
        audio_length = 0.0
        try:
            if loading_resampling == "librosa":
                audio = load_audio(path, self.sr)
            else:
                audio = load_audio_ffmpeg(path, self.sr)

            audio_length = librosa.get_duration(y=audio, sr=self.sr)

            if process_effects:
                audio = signal.lfilter(self.b_high, self.a_high, audio)
            if noise_reduction:
                import noisereduce as nr
                audio = nr.reduce_noise(
                    y=audio, sr=self.sr, prop_decrease=reduction_strength
                )

            if cut_preprocess == "Skip":
                self._save_segment(audio, sid, idx0, 0, loading_resampling)
            elif cut_preprocess == "Simple":
                self.simple_cut(audio, sid, idx0, chunk_len, overlap_len, loading_resampling)
            elif cut_preprocess == "Automatic":
                idx1 = 0
                for audio_segment in self.slicer.slice(audio):
                    i = 0
                    while True:
                        start = int(self.sr * (PERCENTAGE - OVERLAP) * i)
                        i += 1
                        if len(audio_segment[start:]) > (PERCENTAGE + OVERLAP) * self.sr:
                            tmp = audio_segment[start : start + int(PERCENTAGE * self.sr)]
                            self._save_segment(tmp, sid, idx0, idx1, loading_resampling)
                            idx1 += 1
                        else:
                            tmp = audio_segment[start:]
                            self._save_segment(tmp, sid, idx0, idx1, loading_resampling)
                            idx1 += 1
                            break
            else:
                raise ValueError(f"Unknown cut_preprocess mode: '{cut_preprocess}'")

        except Exception as e:
            logger.error("Error processing %s: %s", path, e)
            raise

        return audio_length


# ── Multiprocessing workers ────────────────────────────────────────────────────

def _process_audio_worker(args):
    (
        path, idx0, sid, sr, exp_dir,
        cut_preprocess, process_effects, noise_reduction, reduction_strength,
        chunk_len, overlap_len, loading_resampling,
    ) = args
    pp = PreProcess(sr, exp_dir)
    return pp.process_audio(
        path, idx0, sid,
        cut_preprocess, process_effects, noise_reduction, reduction_strength,
        chunk_len, overlap_len, loading_resampling,
    )


def _peak_normalize_worker(args):
    file_name, gt_wavs_dir, wavs16k_dir = args
    try:
        for folder in (gt_wavs_dir, wavs16k_dir):
            fpath = os.path.join(folder, file_name)
            audio, sr = sf.read(fpath)
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = (audio / peak) * 0.95
            wavfile.write(fpath, sr, audio.astype(np.float32))
    except Exception as e:
        logger.error("Error normalising %s: %s", file_name, e)
        raise


# ── Main preprocessing entry point ────────────────────────────────────────────

def preprocess_training_set(
    input_root: str,
    sr: int,
    num_processes: int,
    exp_dir: str,
    cut_preprocess: str = "Automatic",
    process_effects: bool = True,
    noise_reduction: bool = False,
    reduction_strength: float = 0.7,
    chunk_len: float = 3.0,
    overlap_len: float = 0.3,
    normalization_mode: str = "none",
    loading_resampling: str = "librosa",
    use_smart_cutter: bool = False,
) -> None:
    """
    Preprocess a raw audio dataset into sliced GT + 16 kHz pairs.

    Args:
        input_root: Root folder containing audio files or speaker subdirectories.
        sr: Target sample rate.
        num_processes: Number of parallel worker processes.
        exp_dir: Experiment directory where outputs are written.
        cut_preprocess: "Skip" | "Simple" | "Automatic"
        process_effects: Apply high-pass filter.
        noise_reduction: Apply noisereduce.
        reduction_strength: Noise reduction strength (0–1).
        chunk_len: Chunk length in seconds (Simple mode only).
        overlap_len: Overlap length in seconds (Simple mode only).
        normalization_mode: "none" | "post_peak"
        loading_resampling: "librosa" | "ffmpeg"
        use_smart_cutter: Run SmartCutter silence removal before slicing.

    Raises:
        ValueError: On invalid multi-speaker folder layout.
        FileNotFoundError: If SmartCutter weights are missing.
    """
    start_time = time.time()

    # ── Load SmartCutter (if requested) ───────────────────────────────────
    sc_engine = None
    if use_smart_cutter:
        from kazeflow.preprocess.smartcutter.inference import SmartCutterInterface
        sc_engine = SmartCutterInterface(sr, str(_SMARTCUTTER_CKPT_DIR))
        sc_engine.load_model()
        logger.info("SmartCutter model loaded.")

    # ── Build speaker map ─────────────────────────────────────────────────
    # speaker_map: { folder_path -> [audio_file_path, ...] }
    _AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".aac"}

    speaker_map: dict = {}

    root_files = [
        f for f in os.listdir(input_root)
        if Path(f).suffix.lower() in _AUDIO_EXTS
    ]
    if root_files:
        speaker_map[input_root] = [os.path.join(input_root, f) for f in root_files]

    for root, dirs, filenames in os.walk(input_root):
        if root == input_root:
            continue
        audio_files = [
            os.path.join(root, f) for f in filenames
            if Path(f).suffix.lower() in _AUDIO_EXTS
        ]
        if audio_files:
            speaker_map[root] = audio_files

    speaker_count = len(speaker_map)

    # ── Multi-speaker contiguity validation ───────────────────────────────
    if speaker_count > 1:
        detected_sids: set = set()
        for folder_path in speaker_map:
            if folder_path == input_root:
                detected_sids.add(0)
            else:
                folder_name = os.path.basename(folder_path)
                try:
                    sid = int(folder_name.split("_")[0])
                    detected_sids.add(sid)
                except (ValueError, IndexError):
                    raise ValueError(
                        f"Folder '{folder_name}' is invalid for multi-speaker. "
                        "Folders must start with an integer (e.g. '0_name' or '0')."
                    )

        expected_sids = set(range(speaker_count))
        if detected_sids != expected_sids:
            missing = sorted(expected_sids - detected_sids)
            raise ValueError(
                f"Speaker IDs are not contiguous. "
                f"Detected: {sorted(detected_sids)}. Missing: {missing}."
            )

        logger.info("Multi-speaker contiguity check passed.")

    logger.info("Found %d speaker(s) to process.", speaker_count)
    cleanup_dirs(exp_dir)

    # ── Process each speaker ──────────────────────────────────────────────
    total_audio_length = 0.0

    with multiprocessing.Pool(processes=num_processes) as pool:
        for speaker_dir, audio_paths in tqdm(
            speaker_map.items(), desc="Processing speakers"
        ):
            # Determine speaker ID
            if speaker_dir == input_root:
                sid = 0
            else:
                folder_name = os.path.basename(speaker_dir)
                try:
                    sid = int(folder_name.split("_")[0])
                except (ValueError, IndexError):
                    logger.warning(
                        "Folder '%s' has no integer prefix — using SID 0.",
                        folder_name,
                    )
                    sid = 0

            # Optionally run SmartCutter on this speaker's files first
            temp_speaker_dir = None
            current_batch_paths = audio_paths

            if use_smart_cutter and sc_engine is not None:
                temp_speaker_dir = os.path.join(
                    exp_dir, "smart_cut_temp", str(sid)
                )
                os.makedirs(temp_speaker_dir, exist_ok=True)
                current_batch_paths = []
                for file_path in audio_paths:
                    out_path = os.path.join(
                        temp_speaker_dir, os.path.basename(file_path)
                    )
                    sc_engine.process_file(file_path, out_path)
                    current_batch_paths.append(out_path)

            # Build worker arg list
            arg_list = [
                (
                    f_path, idx, sid, sr, exp_dir,
                    cut_preprocess, process_effects, noise_reduction,
                    reduction_strength, chunk_len, overlap_len, loading_resampling,
                )
                for idx, f_path in enumerate(current_batch_paths)
            ]

            for result in pool.imap_unordered(_process_audio_worker, arg_list):
                if result:
                    total_audio_length += result

            # Clean up SmartCutter temp folder for this speaker
            if temp_speaker_dir and os.path.exists(temp_speaker_dir):
                shutil.rmtree(temp_speaker_dir)

    # Clean up global SmartCutter temp dir
    main_temp_dir = os.path.join(exp_dir, "smart_cut_temp")
    if os.path.exists(main_temp_dir):
        shutil.rmtree(main_temp_dir)

    if sc_engine is not None:
        sc_engine.unload()

    # ── Save duration to model_info.json ──────────────────────────────────
    save_dataset_duration(
        os.path.join(exp_dir, "model_info.json"), total_audio_length
    )

    # ── Optional post-peak normalisation ─────────────────────────────────
    if normalization_mode == "post_peak":
        logger.info("Post-peak normalisation enabled.")
        gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
        wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")
        audio_files = sorted(
            f for f in os.listdir(gt_wavs_dir) if f.endswith(".wav")
        )
        arg_list = [(f, gt_wavs_dir, wavs16k_dir) for f in audio_files]
        with multiprocessing.Pool(processes=num_processes) as pool:
            list(tqdm(
                pool.imap_unordered(_peak_normalize_worker, arg_list),
                total=len(audio_files),
                desc="Peak normalisation",
            ))

    elapsed = time.time() - start_time
    logger.info(
        "Audio preprocessing done in %.1fs on %s of audio.",
        elapsed, format_duration(total_audio_length),
    )
