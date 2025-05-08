# Armenian Handwritten Text Recognition (HTR)
![Python](https://img.shields.io/badge/Python-3.8-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)


This repository presents a modular system for Armenian handwritten word recognition. The pipeline combines text detection and recognition methods and supports configurable decoding strategies.

## Repository Structure

```
capstone-htr/
├── pipeline/                       # Pipeline and UI logic
│   ├── wrapper.py                  # Detection and recognition wrapper
│   ├── board.py                    # OpenCV-based drawing interface
│   └── run_board.py                # CLI-based board launcher
│
├── SimpleHTR/                      # TensorFlow-based CRNN recognizer
│   ├── model_checkpoints_armo/     # Fine-tuned Armenian model: charList.txt, snapshots, wordCharList.txt
│   ├── model_arm/                  # Backup copy of the base model (not required for execution)
│   └── data/
│       ├── cropped_words/              # Real handwritten word images
│       ├── synthetic_words_inverted/  # Inverted synthetic images
│       ├── annotations_clean.txt      # Cleaned annotation file
│       └── corpus.txt                 # Corpus for Word Beam Search decoding
│
├── deep-text-recognition-benchmark/  # PyTorch-based ClovaAI OCR
│   ├── lmdb/                          # LMDB-format training data
│   ├── pretrained/                    # Pretrained MJ/ST models
│   └── fine_tuning.ipynb              # Model adaptation notebook
│
├── CRAFT-pytorch/                 # CRAFT text detector
│   ├── weights/                   # Pretrained detector weights (.pth)
│   └── inference.py              # Modified detection pipeline with orientation handling
│
├── CTCWordBeamSearch/            # Word Beam Search decoder
│   └── build/                     # Compiled C++ backend (.pyd/.so)
│
├── data_preparation/             # Dataset creation and refinement
│   ├── data_annotation.ipynb      # Creating and cleaning the real handwritten dataset
│   └── data_generation.ipynb      # Synthetic data generation and real data augmentation
│
├── requirements.txt              # Python dependency file
└── README.md
```
## Main Notebooks

This repository includes several key notebooks that support data preparation, model training, and end-to-end usage:

| Notebook                 | Location                                | Purpose                                                                 |
|--------------------------|-----------------------------------------|-------------------------------------------------------------------------|
| `data_annotation.ipynb` | `data_preparation/`                     | Creates, cleans, and filters annotated Armenian handwriting samples.              |
| `data_generation.ipynb` | `data_preparation/`                     | Generates and augments synthetic word-level handwritten images.         |
| `fine_tuning.ipynb`     | `deep-text-recognition-benchmark/`      | Performs transfer learning for the ClovaAI recognizer using Armenian data. |
| `fine_tuning.ipynb`     | `SimpleHTR/`                            | Performs transfer learning for the SimpleHTR recognizer using Armenian data. |
| `usage.ipynb`           | `CRAFT-pytorch/`                        | Demonstrates CRAFT text detection on sample images.                     |
| `main.ipynb`            | `pipeline/`                             | Interactive demo combining drawing, detection, and recognition.         |

## Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/anialoyan/capstone-htr.git
cd capstone-htr
```

### Step 2: Create virtual environment and install dependencies

Python 3.8.10 is recommended to ensure compatibility across modules.

```bash
python -m venv htr_env
source htr_env/bin/activate    # or htr_env\Scripts\activate on Windows
python -m pip install -r requirements.txt
```

### Step 3: Compile Word Beam Search decoder (Optional)

If using the `wbs` decoder in SimpleHTR, you must compile the decoder library from CTCWordBeamSearch. This requires a functional C++ compiler (e.g., MSVC on Windows or `g++` on Unix-based systems).

```bash
cd CTCWordBeamSearch
# Follow OS-specific build instructions
# Example (Linux): cmake . && make
```

If not using `wbs`, the default decoders `bestpath` and `beamsearch` require no external compilation.

## Required Downloads

Some files must be downloaded manually and extracted to the following locations. These files will be made available [here](https://www.dropbox.com/scl/fo/tonqqzvq3b0rp32jeqzd8/AA_LtzkcgiX1jxKYeU3Al3E?rlkey=179ssdgn07hby5cfawtt2evm9&st=yer8ftwj&dl=0) and should be placed in the correct subdirectories. Their empty folders live in the project. Additionally, some data that was used in the scope of this project also can be found following the same link.

### For a simple inference, you can download only the model weights

- CRAFT-pytorch/weights has the detector weights


- SimpleHTR/model has the English pretrained weights for SimpleHTR
- SimpleHTR/model_checkpoints_armo has the Armenian fine-tuned weights for SimpleHTR


- deep-text-recognition-benchmark/pretrained has the English pretrained weights for Clova
- deep-text-recognition-benchmark/saved_models has the fine-tuned Armenian weights for Clova

Pretraining data (both synthetic and real) can be downloaded as well. It lies in SimpleHTR/data 

## Running the Drawing Interface

Launch the real-time drawing interface and recognize text on the fly:

```bash
python pipeline/run_board.py
```

Controls:

* Press `s` to save the drawing and run recognition
* Press `ESC` to exit without saving

On launch, you will be prompted to select:

* OCR backend: `SimpleHTR` or `ClovaAI`
* Decoder type if SimpleHTR is selected (`wbs`, `beamsearch`, or `bestpath`)

## Decoding Options (SimpleHTR)

SimpleHTR supports three decoding mechanisms:

* `wbs` – Word Beam Search (requires compiled C++ backend and `corpus.txt`)
* `beamsearch` – Beam search decoding (no corpus required)
* `bestpath` – Greedy decoding

To use `wbs`, ensure the following:

* `CTCWordBeamSearch` is built and discoverable by the Python path
* `SimpleHTR/data/corpus.txt` is present

## License

This project is licensed under the MIT License.

### Third-Party Licenses

This repository incorporates code from the following open-source projects:

* CRAFT-pytorch (MIT License) – [https://github.com/clovaai/CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch)
* SimpleHTR (MIT License) – [https://github.com/githubharald/SimpleHTR](https://github.com/githubharald/SimpleHTR)
* Deep Text Recognition Benchmark (Apache 2.0) – [https://github.com/clovaai/deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)
* CTCWordBeamSearch (MIT License) – [https://github.com/githubharald/CTCWordBeamSearch](https://github.com/githubharald/CTCWordBeamSearch)

All third-party licenses are preserved in the `licenses/` directory.

## Citation

If you use this work or build upon it, please cite the original repositories listed in the Acknowledgements section.

## Contact

This project was developed by Ani Aloyan as a capstone project at the American University of Armenia.

For inquiries, contact: \[[ani_aloyan2@edu.aua.am](mailto:ani_aloyan2@edu.aua.am)]
