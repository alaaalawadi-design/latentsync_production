#!/bin/bash

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints

# Check if wget is installed, install if not
if ! command -v wget &> /dev/null; then
    echo "wget not found, installing..."
    sudo apt-get update && sudo apt-get install wget -y
fi

# Check if gdown is installed, install if not
if ! command -v gdown &> /dev/null; then
    echo "gdown not found, installing..."
    pip install gdown
fi

# Download the models into checkpoints directory
cd checkpoints

wget -c https://huggingface.co/ByteDance/LatentSync/resolve/main/config.json
wget -c https://huggingface.co/ByteDance/LatentSync/resolve/main/latentsync_syncnet.pt
wget -c https://huggingface.co/ByteDance/LatentSync/resolve/main/latentsync_unet.pt

gdown 1ypPNVRe4N9ne3K6Ba6GMQms__WhpfeEW 

# Create whisper subdirectory and download whisper model
mkdir -p whisper
cd whisper

wget -c https://huggingface.co/ByteDance/LatentSync/resolve/main/whisper/tiny.pt

# Go back to original directory
cd ../..

# Completion message
echo "All models downloaded successfully."