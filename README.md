# Armenian Handwritten Text Recognition (HTR)
![Python](https://img.shields.io/badge/Python-3.8-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)


This repository presents a modular system for Armenian handwritten word recognition. The pipeline combines text detection and recognition methods and supports configurable decoding strategies.

## Repository Structure

```
capstone-htr/
├── pipeline/                            # Integrated detection + recognition setup
│   ├── wrapper.py                       # Runs detection → recognition pipeline
│   ├── board.py                         # OpenCV-based handwriting board
│   ├── run_board.py                     # CLI launcher for board interface
│   ├── board_output.png                 # Example board image
│   ├── main.ipynb                       # Pipeline demo and testing notebook
│   └── README.md                        # Instructions for using the pipeline

├── SimpleHTR/                           # TensorFlow-based CRNN recognizer
│   ├── fine_tuning.ipynb                # Notebook for adapting model to Armenian
│   ├── armo_training_tuning_summary.json # Summary of training runs
│   ├── LICENSE.md, README.md
│   ├── model_checkpoints_armo/          # Final fine-tuned model weights
│   ├── model_arm/                       # Backup model snapshot (optional)
│   ├── model/                           # Unused checkpoint path
│   ├── data/                            # Armenian image + corpus
│   │   ├── corpus.txt                   # Armenian corpus for WBS
│   │   ├── line.png, word.png, test_picture.jpg
│   ├── doc/                             # Visuals used in explanations
│   │   ├── htr.png, decoder_comparison.png, graphics.svg
│   └── src/                             # Training + inference code
│       ├── dataloader_arm.py, model.py, train.py, etc.

├── deep-text-recognition-benchmark/     # PyTorch-based CRNN OCR (Clova)
│   ├── fine_tuning.ipynb                # Fine-tuning Clova OCR on Armenian words
│   ├── modules/                         # CRNN architecture
│   ├── pretrained/, saved_models/       # Model files
│   ├── demo_image/, figures/            # Output & evaluation visuals
│   └── *.py                             # Training, testing, utils

├── data_preparation/                    # Dataset construction tools
│   ├── data_annotation.ipynb            # Real handwritten word annotation and cleanup
│   ├── data_generation.ipynb            # Synthetic Armenian word image generation
│   ├── hye_wikipedia_2021_1M-words.txt  # Armenian word list (scraped)
│   ├── *.png                            # Sample generated/processed images
│   ├── sample_output/                   # Exported synthetic word images
│   └── README.md

├── CRAFT-pytorch/                       # CRAFT-based word detector
│   ├── inference.py                     # Modified inference with orientation handling
│   ├── craft.py, craft_utils.py, etc.
│   ├── basenet/                         # VGG16 backbone
│   ├── weights/                         # Placeholder for CRAFT weights
│   └── figures/                         # Demo visuals

├── CTCWordBeamSearch/                   # Word Beam Search CTC decoder
│   ├── cpp/                             # C++ decoder implementation
│   ├── extras/                          # Python and TF integration
│   ├── tests/                           # Unit tests
│   ├── setup.py                         # Package installer
│   └── README.md, LICENSE.md

├── licenses/                            # License files for third-party tools
│   ├── CRAFT, DeepTextRec, SimpleHTR, WBS

├── Capstone_Paper.pdf                   # Final Capstone Paper
├── requirements.txt                     # Python dependencies
└── README.md                            # Project description and usage
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
