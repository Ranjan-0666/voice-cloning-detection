# WaveFake Dataset

## Overview
This directory contains (or links to) the WaveFake dataset used to train the voice cloning detection model.

## Expected Directory Structure

```
data/
└── WaveFake/
    ├── LJSpeech-1.1/
    │   └── wavs/                  ← Genuine LJSpeech recordings (.wav)
    │       ├── LJ001-0001.wav
    │       ├── LJ001-0002.wav
    │       └── ...
    └── generated_audio/           ← AI-synthesised speech
        ├── ljspeech_full_band_melgan/
        │   ├── LJ001-0001.wav
        │   └── ...
        ├── ljspeech_hifiGAN/
        │   └── ...
        ├── ljspeech_melgan/
        ├── ljspeech_melgan_large/
        ├── ljspeech_multi_band_melgan/
        ├── ljspeech_parallel_wavegan/
        └── ljspeech_waveglow/
```

## Download Instructions

1. Visit: https://github.com/RUB-SysSec/WaveFake
2. Download the archive from Zenodo (link is in the repository README)
3. Extract to this `data/WaveFake/` directory
4. Verify the structure above exists
5. Run the manifest builder:

```bash
# From the project root:
python -c "from src.dataset_loader import build_manifest; build_manifest(force=True)"
```

Or use the Streamlit app:
- Navigate to **📊 Dataset Analysis**
- Click **Build Manifest Now**

## Custom Path

If you extracted WaveFake to a different location, set the environment variable:

```bash
# Windows PowerShell
$env:WAVEFAKE_ROOT = "D:\datasets\WaveFake"

# Or edit config.py directly:
WAVEFAKE_ROOT = Path("D:/datasets/WaveFake")
```

## Dataset Citation

```
@inproceedings{frank2021wavefake,
  title     = {WaveFake: A Data Set to Facilitate Audio Deepfake Detection},
  author    = {Frank, Joel and Schönherr, Lea},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2021},
}
```

## Processed Files

After running the manifest builder, the `processed/` sub-directory will contain:

- `dataset_manifest.csv` — catalogue of all audio files with labels and split assignments
- `prediction_history.csv` — log of inference requests (metadata only, no audio retained)
