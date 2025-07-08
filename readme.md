# Project Name

This repository provides tools for running inference using pretrained models.

---

## Installation

Follow these steps to set up your environment:

### Step 1: Create and activate Conda environment

```bash
conda create -n env python=3.10 -y
conda activate env
```

### Step 2: Install dependencies

```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## Downloading Models

Make the script executable and run it to download required models:

```bash
chmod +x download_models
./download_models
```

---

## Run Inference

To run inference, execute `main.py`. Adjust the paths inside the script as necessary:

```bash
python main.py
```

- Modify `input_audio_path` with the path to your input audio file.
- Modify `output_save_path` to specify where the result should be saved.

---
