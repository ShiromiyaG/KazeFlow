# KazeFlow

KazeFlow is a voice conversion system with a web interface. Feed it audio from any speaker and it converts it to the timbre of a trained model, preserving the original content and intonation.

Supported sample rates: 32kHz, 40kHz, 44.1kHz, and 48kHz.

## Requirements

- NVIDIA GPU with 8GB+ VRAM
- Windows or Linux

## Installation

**Linux:**
```bash
chmod +x install.sh start.sh
./install.sh
```

**Windows:**
```
install.bat
```

The installer automatically downloads Python, PyTorch with CUDA, and all dependencies into a local `env/` folder. It does not affect other Python environments on the system.

## Starting

```bash
# Linux
./start.sh

# Windows
start.bat
```

Opens the interface at `http://localhost:7860`.

---

## Tab: Inference

To convert audio using an already trained model.

1. Place the checkpoint (`.pt`) in `assets/checkpoints/`
2. Select the model in the interface
3. Upload the input audio (`.wav`, `.mp3`, `.flac`, `.ogg`, `.opus`, `.m4a`, `.aac`)
4. Adjust parameters as needed and click **Convert**

**Parameters:**

| Parameter | Description |
|---|---|
| Speaker ID | Which speaker to use (for multi-speaker models) |
| Pitch shift | Shift pitch in semitones (e.g. +12 raises one octave) |
| ODE steps | Solver iterations (default 16 — more steps = better quality, slower) |
| ODE method | `euler` is faster, `midpoint` is slightly more accurate |
| CFG scale | Speaker conditioning strength (1.0 = neutral, up to 3.0) |
| Index rate | How much to use the FAISS index to reinforce target speaker similarity (0.0–1.0) |

The FAISS index is optional. If no index is loaded, the parameter is ignored.

---

## Tab: Training (fine-tuning)

To train a model on a specific voice, starting from a pretrained base checkpoint.

### 1. Prepare the dataset

Place audio files in:
```
assets/datasets/<dataset_name>/
    audio1.wav
    audio2.mp3
    ...
```

There are no format or sample rate requirements — the system converts automatically.

**Recommended duration:** 10–30 minutes of clean audio, without background music or excessive noise.

### 2. Preprocess

In the Training tab:
- Enter the dataset name and model name
- Run **Slice** (cuts audio into segments)
- Run **Extract** (extracts features: content, F0, and mel)
- Optionally, **Build Index** (creates a FAISS index to improve speaker similarity at inference)

### 3. Train

- Select the sample rate matching your base checkpoint
- Point to the pretrain checkpoint
- Set batch size and epochs
- Click **Train**

Checkpoints are saved periodically to `logs/<model_name>/`.

---

## Tab: Pretrain

To create a base model from scratch using a large multi-speaker dataset.

The dataset must have speaker subfolders with a numeric prefix:
```
assets/datasets/<name>/
    0_Speaker1/
        audio1.wav
        ...
    1_Speaker2/
        audio1.wav
        ...
```

The number of speakers is detected automatically.

The process is the same as Training (Slice → Extract → Train), but trains from scratch without a base checkpoint. The result can be used as a base for fine-tuning.

---

## Monitoring Training

TensorBoard logs are saved to `logs/<model_name>/`. To view them:

```bash
env/bin/tensorboard --logdir logs/
```

Opens at `http://localhost:6006`.

## License

See LICENSE file for details.
