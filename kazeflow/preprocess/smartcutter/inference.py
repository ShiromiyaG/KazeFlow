"""
SmartCutter inference — ported from Codename-fork-4.

Changes from original:
- Import paths updated to kazeflow package structure.
- sys.exit() calls replaced with RuntimeError / ValueError raises.
- Standalone processing() entry-point retained but unused in library mode.
"""

import gc
import glob
import math
import os
import sys

import torch
import torchaudio
import torchaudio.functional as F_audio
import soundfile as sf
import numpy as np

from kazeflow.preprocess.smartcutter.model_v5 import CGA_ResUNet
from kazeflow.preprocess.smartcutter.model_v3 import DSCA_ResUNet_v3

# ── Inference / processing config ─────────────────────────────────────────────
MODEL_VERSION = "v3"
FORCE_CPU = False
MASK_MODE = "Soft"           # "Soft", "Hard", "PowerMean", "Hybrid"
DEBUG_MASK_PRED = False
SAVE_EXTENSION = "wave_16"   # "flac", "wave_16", "wave_32float"

# SmartCutter config (safe defaults)
SILENCE_TARGET_DURATION = 0.100   # seconds
MIN_SEGMENT_DURATION_MS = 100     # ms

# Prediction stabilisation
STABILITY_NOISE = False
STABILITY_DB_LEVEL = -75.0
STABILITY_FADE_MS = 1
ENABLE_BRIDGING = True

# Standalone paths (only used by processing())
IN_DIR = "infer_input"
OUT_DIR = "infer_output"
CKPT_DIR = "ckpts"

# Established safe params — do not tweak
SEARCH_WINDOW_MS = 25
FADE_DURATION_MS = 10
CUTTING_PROBABILITY = 0.5
SAFETY_BUFFER_MS = 5
SEGMENT_LEN = 8.0


# ── Low-level helpers ──────────────────────────────────────────────────────────

def get_cosine_fade(length, device):
    t = torch.linspace(0, math.pi, length, device=device)
    return 0.5 * (1 - torch.cos(t))


def apply_fade(waveform, fade_samples, mode="both"):
    if waveform.shape[1] < fade_samples * 2:
        return waveform
    fade_curve = get_cosine_fade(fade_samples, waveform.device)
    if mode in ("in", "both"):
        waveform[:, :fade_samples] *= fade_curve
    if mode in ("out", "both"):
        waveform[:, -fade_samples:] *= fade_curve.flip(0)
    return waveform


def inject_stability_noise(wav, sr, device):
    noise_amp = 10 ** (STABILITY_DB_LEVEL / 20.0)
    silence_mask = (wav.squeeze(0) == 0.0).float()
    diff = torch.diff(
        silence_mask,
        prepend=torch.tensor([0.0], device=device),
        append=torch.tensor([0.0], device=device),
    )
    starts = torch.where(diff == 1)[0]
    ends = torch.where(diff == -1)[0]
    if len(starts) == 0:
        return wav

    raw_noise = torch.randn_like(wav)
    alpha = 0.85
    neutral_noise = torchaudio.functional.lfilter(
        raw_noise,
        torch.tensor([1.0, 0.0], device=device),
        torch.tensor([1.0, -alpha], device=device),
    )
    neutral_noise *= noise_amp
    fade_samples = int(sr * (STABILITY_FADE_MS / 1000.0))

    for start, end in zip(starts, ends):
        length = end - start
        if length <= 0:
            continue
        noise_chunk = neutral_noise[:, start:end].clone()
        if length > fade_samples * 2:
            noise_chunk = apply_fade(noise_chunk, fade_samples, mode="both")
        else:
            window = torch.hann_window(length, device=device)
            noise_chunk *= window
        wav[:, start:end] = noise_chunk
    return wav


def find_cuts(mask, waveform, sr):
    if mask.device.type != "cpu":
        mask = mask.cpu()
    if waveform.device.type != "cpu":
        waveform = waveform.cpu()

    mask_binary = (mask > CUTTING_PROBABILITY).float()
    diff = torch.diff(
        mask_binary,
        prepend=torch.tensor([0]),
        append=torch.tensor([0]),
    )
    rough_starts = torch.where(diff == 1)[0]
    rough_ends = torch.where(diff == -1)[0]
    if len(rough_starts) == 0:
        return [], []

    min_samples = int(sr * (MIN_SEGMENT_DURATION_MS / 1000.0))
    durations = rough_ends - rough_starts
    valid_indices = torch.where(durations >= min_samples)[0]
    if len(valid_indices) == 0:
        return [], []

    rough_starts = rough_starts[valid_indices]
    rough_ends = rough_ends[valid_indices]

    buffer_samples = int(sr * (SAFETY_BUFFER_MS / 1000.0))
    rough_starts = rough_starts + buffer_samples
    rough_ends = rough_ends - buffer_samples

    valid_shrink = rough_ends > rough_starts
    rough_starts = rough_starts[valid_shrink]
    rough_ends = rough_ends[valid_shrink]

    wav_mono = waveform.mean(dim=0)
    zero_crossings = torch.diff(torch.sign(wav_mono))
    valid_zc_indices = torch.where(zero_crossings != 0)[0]
    if len(valid_zc_indices) == 0:
        return rough_starts, rough_ends

    def snap_to_nearest(targets, candidates):
        idx = torch.searchsorted(candidates, targets)
        idx = torch.clamp(idx, 0, len(candidates) - 1)
        prev_idx = torch.clamp(idx - 1, 0, len(candidates) - 1)
        dist_idx = torch.abs(targets - candidates[idx])
        dist_prev = torch.abs(targets - candidates[prev_idx])
        return torch.where(dist_prev < dist_idx, candidates[prev_idx], candidates[idx])

    search_win_samples = int(sr * (SEARCH_WINDOW_MS / 1000))
    safe_starts = snap_to_nearest(rough_starts, valid_zc_indices)
    safe_ends = snap_to_nearest(rough_ends, valid_zc_indices)

    start_diff = torch.abs(safe_starts - rough_starts)
    safe_starts = torch.where(start_diff < search_win_samples, safe_starts, rough_starts)
    end_diff = torch.abs(safe_ends - rough_ends)
    safe_ends = torch.where(end_diff < search_win_samples, safe_ends, rough_ends)

    return safe_starts, safe_ends


def SmartCutter(waveform, mask, sr=48000):
    waveform = waveform.cpu()
    mask = mask.cpu()

    if ENABLE_BRIDGING:
        bridge_frames = 5
        mask = mask.view(1, 1, -1)
        mask = torch.nn.functional.max_pool1d(mask, bridge_frames, 1, bridge_frames // 2)
        mask = -torch.nn.functional.max_pool1d(-mask, bridge_frames, 1, bridge_frames // 2)

    if mask.dim() == 1:
        mask = mask.view(1, 1, -1)
    elif mask.dim() == 2:
        mask = mask.unsqueeze(1)

    target_size = waveform.shape[1]
    mask_full = torch.nn.functional.interpolate(
        mask, size=target_size, mode="linear", align_corners=True
    ).squeeze()

    starts, ends = find_cuts(mask_full, waveform, sr)
    if len(starts) == 0:
        return torch.zeros_like(waveform), mask_full

    target_silence_samples = int(sr * SILENCE_TARGET_DURATION)
    fade_samples = int(sr * (FADE_DURATION_MS / 1000))

    pieces = []
    last_valid_idx = 0
    silence_tensor = torch.zeros((waveform.shape[0], target_silence_samples))

    for start_idx, end_idx in zip(starts.tolist(), ends.tolist()):
        start_idx = int(start_idx)
        end_idx = int(end_idx)
        if start_idx > last_valid_idx:
            speech_chunk = waveform[:, last_valid_idx:start_idx].clone()
            if last_valid_idx > 0:
                speech_chunk = apply_fade(speech_chunk, fade_samples, mode="in")
            speech_chunk = apply_fade(speech_chunk, fade_samples, mode="out")
            pieces.append(speech_chunk)
        pieces.append(silence_tensor)
        last_valid_idx = end_idx

    if last_valid_idx < target_size:
        tail_chunk = waveform[:, last_valid_idx:].clone()
        if last_valid_idx > 0:
            tail_chunk = apply_fade(tail_chunk, fade_samples, mode="in")
        pieces.append(tail_chunk)

    return torch.cat(pieces, dim=1), mask_full


def process_grid_aligned(model, transform, waveform, sr, hop_length, device, static_input_buffer):
    total_samples = waveform.shape[1]
    CHUNK_SEC = SEGMENT_LEN
    OVERLAP_SEC = CHUNK_SEC / 2
    chunk_samples = int(CHUNK_SEC * sr)
    overlap_samples = int(OVERLAP_SEC * sr)
    stride_samples = chunk_samples - overlap_samples

    dummy_input = torch.zeros(1, chunk_samples, device=device)
    dummy_mel = transform(dummy_input)
    frames_per_chunk = dummy_mel.shape[-1]
    total_frames = int(math.ceil(total_samples / hop_length)) + 100

    print(
        f"    -> WOLA chunking: Chunk={chunk_samples}, "
        f"Overlap={overlap_samples}, Total Frames={total_frames}"
    )

    mask_accumulator = torch.zeros((1, total_frames), dtype=torch.float32, device="cpu")
    weight_accumulator = torch.zeros((1, total_frames), dtype=torch.float32, device="cpu")
    window = torch.hann_window(frames_per_chunk, device=device).view(1, -1)
    window_cpu = window.cpu()

    current_sample = 0
    with torch.no_grad():
        while current_sample < total_samples:
            start = current_sample
            end = start + chunk_samples
            chunk_wav = waveform[:, start:end]

            original_len = chunk_wav.shape[1]
            if original_len < chunk_samples:
                pad_amt = chunk_samples - original_len
                chunk_wav = torch.nn.functional.pad(chunk_wav, (0, pad_amt))

            chunk_wav = chunk_wav.to(device)
            raw_mask = _run_inference(model, transform, chunk_wav, device, static_input_buffer)
            if raw_mask.dim() == 3:
                raw_mask = raw_mask.squeeze(1)

            start_frame = int(round(start / hop_length))
            if start_frame + frames_per_chunk > mask_accumulator.shape[1]:
                extra = (start_frame + frames_per_chunk) - mask_accumulator.shape[1]
                mask_accumulator = torch.nn.functional.pad(mask_accumulator, (0, extra))
                weight_accumulator = torch.nn.functional.pad(weight_accumulator, (0, extra))

            current_pred_cpu = raw_mask.cpu()
            mask_accumulator[:, start_frame : start_frame + frames_per_chunk] += (
                current_pred_cpu * window_cpu
            )
            weight_accumulator[:, start_frame : start_frame + frames_per_chunk] += window_cpu
            current_sample += stride_samples

    weight_accumulator[weight_accumulator < 1e-6] = 1.0
    final_mask = mask_accumulator / weight_accumulator
    actual_frames = int(total_samples / hop_length)
    final_mask = final_mask[:, :actual_frames]
    return final_mask


def _run_inference(model, mel_transform, wav_chunk, device, input_buffer):
    mel = mel_transform(wav_chunk).squeeze(0)
    mel = torchaudio.transforms.AmplitudeToDB()(mel)

    min_db, max_db = -80.0, 0.0
    mel = torch.clamp(mel, min=min_db, max=max_db)
    mel = (mel - min_db) / (max_db - min_db)

    delta = F_audio.compute_deltas(mel.unsqueeze(0)).squeeze(0)

    current_frames = mel.shape[-1]
    input_buffer[0, 0, :, :current_frames].copy_(mel)
    input_buffer[0, 1, :, :current_frames].copy_(delta)

    mask_2d = model(input_buffer[:, :, :, :current_frames])

    if MASK_MODE == "Soft":
        mask_pred = torch.mean(mask_2d, dim=2)
    elif MASK_MODE == "Hybrid":
        soft_mask = torch.mean(mask_2d, dim=2)
        hard_mask = torch.max(mask_2d, dim=2)[0]
        mask_pred = (0.7 * soft_mask) + (0.3 * hard_mask)
    elif MASK_MODE == "PowerMean":
        mask_pred = torch.sqrt(torch.mean(mask_2d ** 2, dim=2))
    elif MASK_MODE == "Hard":
        mask_pred = torch.max(mask_2d, dim=2)[0]
    else:
        raise ValueError(f"MASK_MODE '{MASK_MODE}' is unsupported.")

    return mask_pred


# ── SmartCutterInterface ───────────────────────────────────────────────────────

class SmartCutterInterface:
    def __init__(self, sr: int, ckpt_dir: str, device: str = "cuda"):
        self.sr = sr
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_version = MODEL_VERSION
        self.ckpt_dir = ckpt_dir

        self.model = None
        self.mel_transform = None
        self.static_buffer = None
        self.loaded = False

    def load_model(self):
        if self.loaded:
            return

        print(f"[SmartCutter] Loading model on {self.device}...")
        model_path = os.path.join(
            self.ckpt_dir, f"{self.model_version}_model_{self.sr}.pth"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[SmartCutter] Model not found: {model_path}. "
                "Run app.py to trigger the automatic prerequisites download."
            )

        if self.model_version == "v3":
            self.model = DSCA_ResUNet_v3(n_channels=2, n_classes=1).to(self.device)
        elif self.model_version == "v5":
            self.model = CGA_ResUNet(n_channels=2, n_classes=1).to(self.device)
        else:
            raise ValueError(f"Unknown model_version: '{self.model_version}'")

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        curr_hop = self.sr // 100
        if self.sr in [48000, 40000]:
            n_fft, n_mels = 2048, 160
        else:
            n_fft, n_mels = 1024, 128

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_mels=n_mels, n_fft=n_fft, hop_length=curr_hop
        ).to(self.device)

        dummy_frames = int(math.ceil((SEGMENT_LEN * self.sr) / curr_hop)) + 5
        self.static_buffer = torch.zeros(
            (1, 2, n_mels, dummy_frames), device=self.device
        )
        self.loaded = True

    def process_file(self, input_path: str, output_path: str) -> bool:
        """
        Read *input_path*, run SmartCutter, save result to *output_path*.
        Returns True on success, False on failure.
        """
        if not self.loaded:
            self.load_model()

        try:
            wav, load_sr = torchaudio.load(input_path)
            if load_sr != self.sr:
                resampler = torchaudio.transforms.Resample(load_sr, self.sr).to(wav.device)
                wav = resampler(wav)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            wav_gpu = wav.to(self.device)
            peak = torch.abs(wav_gpu).max()
            if peak > 0:
                wav_gpu = wav_gpu * (0.9 / peak)

            curr_hop = self.sr // 100
            mask = process_grid_aligned(
                self.model,
                self.mel_transform,
                wav_gpu,
                self.sr,
                curr_hop,
                self.device,
                self.static_buffer,
            )

            cleaned, _ = SmartCutter(wav, mask, sr=self.sr)
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            torchaudio.save(output_path, cleaned, self.sr)
            return True

        except Exception as e:
            print(f"[SmartCutter] Error on {input_path}: {e}")
            return False

    def unload(self):
        del self.model
        del self.mel_transform
        del self.static_buffer
        self.loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Standalone entry point (not used in library mode) ─────────────────────────

def processing():
    if FORCE_CPU:
        device = torch.device("cpu")
        print("FORCE_CPU is True. Using CPU.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = False
        print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")

    os.makedirs(IN_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    files = glob.glob(os.path.join(IN_DIR, "*.wav")) + glob.glob(
        os.path.join(IN_DIR, "*.flac")
    )
    print(f"Found {len(files)} files.")

    loaded_models = {}

    for f_path in files:
        try:
            fname = os.path.basename(f_path)
            print(f"Processing: {fname}...")

            wav, sr = torchaudio.load(f_path)
            if wav.shape[0] > 1:
                dc_offsets = torch.abs(wav.mean(dim=1))
                best_ch_idx = torch.argmin(dc_offsets)
                wav = wav[best_ch_idx].unsqueeze(0)

            wav_for_inference = wav.clone()
            input_peak = torch.abs(wav_for_inference).max()
            if input_peak > 0:
                wav_for_inference = wav_for_inference * (0.9 / input_peak)

            if STABILITY_NOISE:
                wav_for_inference = inject_stability_noise(
                    wav_for_inference, sr, wav.device
                )

            current_hop = sr // 100
            if sr not in loaded_models:
                if len(loaded_models) > 0:
                    print("Unloading previous model to free VRAM...")
                    loaded_models.clear()
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                model_path = os.path.join(
                    CKPT_DIR, f"{MODEL_VERSION}_model_{sr}.pth"
                )
                if not os.path.exists(model_path):
                    print(f"Skipping {fname}: No {MODEL_VERSION} model for {sr}Hz")
                    continue

                print(f"Loading {sr}Hz {MODEL_VERSION} model ...")

                if MODEL_VERSION == "v3":
                    model = DSCA_ResUNet_v3(n_channels=2, n_classes=1).to(device)
                elif MODEL_VERSION == "v5":
                    model = CGA_ResUNet(n_channels=2, n_classes=1).to(device)
                else:
                    raise ValueError(
                        f"'{MODEL_VERSION}' is not a valid model version choice."
                    )

                model.load_state_dict(
                    torch.load(model_path, map_location=device, weights_only=True)
                )
                model.eval()

                if sr in [48000, 40000]:
                    N_FFT, N_MELS = 2048, 160
                else:
                    N_FFT, N_MELS = 1024, 128

                mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=sr,
                    n_mels=N_MELS,
                    n_fft=N_FFT,
                    hop_length=current_hop,
                ).to(device)

                dummy_frames = (
                    int(math.ceil((SEGMENT_LEN * sr) / current_hop)) + 5
                )
                static_buffer = torch.zeros(
                    (1, 2, N_MELS, dummy_frames), device=device
                )
                loaded_models[sr] = (model, mel_transform, static_buffer)

            curr_model, curr_mel_transform, curr_buffer = loaded_models[sr]
            mel_mask = process_grid_aligned(
                curr_model,
                curr_mel_transform,
                wav_for_inference,
                sr,
                current_hop,
                device,
                curr_buffer,
            )

            if device.type == "cuda":
                torch.cuda.empty_cache()

            cleaned, binary_mask = SmartCutter(wav, mel_mask, sr=sr)

            peak = torch.abs(cleaned).max()
            if peak >= 0.95:
                cleaned = cleaned * (0.95 / peak.item())

            file_stem = os.path.splitext(fname)[0]
            if SAVE_EXTENSION == "flac":
                out_path = os.path.join(OUT_DIR, file_stem + ".flac")
                torchaudio.save(
                    out_path, cleaned, sr, format="flac", backend="soundfile"
                )
            elif SAVE_EXTENSION == "wave_16":
                out_path = os.path.join(OUT_DIR, file_stem + ".wav")
                torchaudio.save(
                    out_path, cleaned, sr, encoding="PCM_S", bits_per_sample=16
                )
            elif SAVE_EXTENSION == "wave_32float":
                out_path = os.path.join(OUT_DIR, file_stem + ".wav")
                torchaudio.save(
                    out_path, cleaned, sr, encoding="PCM_F", bits_per_sample=32
                )
            else:
                raise ValueError(
                    f"Specified saving extension: '{SAVE_EXTENSION}' is unsupported."
                )

            print(f"Saved: {out_path}")

        except Exception as e:
            print(f"Error processing {f_path}: {e}")
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue


if __name__ == "__main__":
    processing()
