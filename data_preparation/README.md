# Data Preparation

This folder contains all the preprocessing steps used to prepare data for training the Armenian handwritten text recognition models.

## Structure

The data preparation process consists of two main stages:

### 1. Synthetic Data Generation

Located in `data_generation.ipynb`

This step generates synthetic word-level images by stitching together individual character images. The characters are sourced from a labeled dataset where each character is represented as a separate grayscale image. Words are constructed by concatenating these characters based on real Armenian word lists.

This notebook also performs **data augmentation** to diversify the generated images and improve generalization. Augmentation techniques include brightness adjustment, contrast enhancement, and geometric transformations.

The output consists of:
- Synthetic word images saved into organized folders
- A corresponding TSV file with annotations mapping each image to its Armenian label

### 2. Manual Annotation of Cropped Real Data

Located in `data_annotation.ipynb`

This notebook is used to manually annotate word-level crops detected in real handwritten documents. Each cropped image is displayed one at a time for the annotator to enter the corresponding Armenian label. Unusable samples can be skipped by labeling them as "bad".

The results are appended to an `annotations.tsv` file, which can be reused and updated incrementally.

## Outputs

- `synthetic_words_refined/`: Folder of generated word images
- `annotations.tsv`: Label file mapping images to ground truth text
- `cropped_words/`: Manually cropped and labeled real word images

## Usage Notes (Optional)

- Run `data_generation.ipynb` before training to enrich the dataset with synthetic examples
- Use `data_annotation.ipynb` for building labeled samples from real handwritten material
