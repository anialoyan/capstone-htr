# Armenian Handwritten OCR Pipeline

This module implements the end-to-end inference pipeline for Armenian handwritten text recognition. It integrates a detection model (CRAFT) and two OCR backends (SimpleHTR and ClovaAI), providing a streamlined system for recognizing text from raw input images or freehand digital drawings.

## Features

- Word-level text detection using CRAFT with optional refinement
- Recognition via:
  - SimpleHTR (TensorFlow-based, supports Best Path, Beam Search, and Word Beam Search decoding)
  - ClovaAI (PyTorch-based CRNN with TPS + ResNet + BiLSTM + CTC)
- Interactive `board.py` canvas for freehand text input
- Inference script with live OCR feedback in terminal
- Modular `wrapper.py` interface to switch between models

## File Overview

- `main.ipynb` — Demonstrates usage of the full pipeline on custom images
- `run_board.py` — Launches the freehand input interface and runs recognition
- `board.py` — Implements the drawing board interface using OpenCV
- `wrapper.py` — Core logic for detection and recognition, supports both models

## Usage

Run the interactive tool:

```bash
cd pipeline
python run_board.py
