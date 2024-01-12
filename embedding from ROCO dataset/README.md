# CLIP Embedding for ROCO Dataset

This project utilizes the Contrastive Language–Image Pretraining (CLIP) model to generate embeddings for images and corresponding captions from the ROCO dataset. It is particularly focused on handling medical images and their descriptive text.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them:
- Python 3.x
- PyTorch
- clip
- PIL
- pandas
- matplotlib

### Installation

A step by step series of examples that tell you how to get a development environment running.


1. Clone the repository:
   ```bash
   git clone git@github.com:shirrashko/NLP-lab.git
    ```
   
2.  Install required packages
   ```bash
    pip install -r requirements.txt
   ```
The roco-dataset folder is achieved using the [roco-dataset repository](https://github.com/razorx89/roco-dataset).
## Usage
To run the script, navigate to the script's directory and use the following command:
   ```bash
    python script.py /path/to/roco-dataset
   ```

Replace /path/to/roco-dataset with the path to the ROCO dataset on your machine.

## Script Overview
- load_roco_dataset: Loads the ROCO dataset from the specified path.
- preprocess_images: Preprocesses images for the CLIP model.
- truncate_captions: Truncates captions to a maximum character length.
- generate_embeddings: Generates embeddings for images and captions using the CLIP model.
- test_clip_embeddings: Tests and visualizes the cosine similarity between image and text embeddings.

## Results
![img.png](img.png)

## Interpreting the results:
- The heatmap visualizes the similarity scores between each image and text embedding.
- Each cell in the heatmap corresponds to the similarity score between an image and a text.
- Ideally, higher similarity scores (brighter colors) should align along the diagonal, indicating that each image is most similar to its corresponding caption.
- Off-diagonal brighter cells indicate similarities between non-matching image-text pairs.
- We can see that the model is able to capture the similarity between images and text, but it is not perfect.
- The results can be improved by fine-tuning the model on the ROCO dataset.
- Such fine-tuning is not implemented in this project, although it is possible to do so using the [medclip repository](https://github.com/Kaushalya/medclip?tab=readme-ov-files).