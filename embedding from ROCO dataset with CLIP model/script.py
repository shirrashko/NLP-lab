import os
import clip
import torch
from PIL import Image
import pandas as pd
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import argparse

CLIP_MODEL_TEXT_MAX_LENGTH = 77

# Constants for ImageNet normalization
IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)

IMAGES_PATH = 'data/train/radiology/images'
CAPTIONS_PATH = 'data/train/radiology/captions.txt'


def load_roco_dataset(dataset_path):
    """
    Load the ROCO dataset from a given path.

    Parameters:
    dataset_path (str): Path to the dataset directory.

    Returns:
    list: A list of tuples, each containing the path to an image and its corresponding caption.
    """
    # Paths to images and captions
    images_path = os.path.join(dataset_path, IMAGES_PATH)
    captions_path = os.path.join(dataset_path, CAPTIONS_PATH)

    # Read captions without headers
    captions_df = pd.read_csv(captions_path, sep='\t', header=None, names=['image_name', 'caption'])

    # Combine image paths with captions
    data = []
    for _, row in captions_df.iterrows():
        image_file = f"{row['image_name']}.jpg"
        image_path = os.path.join(images_path, image_file)
        if os.path.exists(image_path):
            data.append((image_path, row['caption']))

    return data


def preprocess_images(image_paths, clip_model):
    """
    Preprocess images for CLIP model.

    Parameters:
    image_paths (list): List of image file paths.
    clip_model (CLIP model): Loaded CLIP model for getting preprocessing parameters.

    Returns:
    Tensor: Batch of preprocessed images.
    """

    # Purpose of Resize:
    # This transformation changes the size of each image to match the input resolution expected by the CLIP model.
    # clip_model.visual.input_resolution provides the necessary height and width, ensuring consistency across all images
    # fed into the model.

    # Purpose of ToTensor:
    # Converts the image, which is a PIL Image, into a PyTorch tensor. This conversion is essential because neural
    # networks, including CLIP, work with numerical tensors rather than traditional image formats.

    # Purpose of Normalization:
    # Normalizing the images based on the mean and standard deviation of the dataset on which the model was trained
    # (ImageNet, in many cases) is crucial for deep learning models. This process ensures that the model receives data
    # similar to what it saw during training, which can significantly impact the model's performance. If a model was
    # trained on normalized data, then the same normalization steps should be applied to any new data fed into the
    # model.

    # combines several image transformations into a single operation. The transformations specified within Compose are
    # applied in the order they are listed.
    preprocess = Compose([
        Resize((clip_model.visual.input_resolution, clip_model.visual.input_resolution)),
        ToTensor(),
        Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    # For each path, ensures the image is in RGB format, then the preprocess function declare previously is applied on
    # the image, and lastly unsqueeze(0) adds a new dimension at the beginning of the tensor to create a batch
    # dimension (as models typically expect inputs in batch format).
    images = [preprocess(Image.open(path).convert("RGB")).unsqueeze(0) for path in image_paths]
    return torch.cat(images)


def truncate_captions(captions, max_char_length):
    """
    Truncate captions to a maximum character length.

    Parameters:
    - captions (list of str): List of text captions.
    - max_char_length (int): Maximum number of characters allowed in a caption.

    Returns:
    - list of str: List of truncated captions.
    """
    truncated_captions = []
    for caption in captions:
        # Truncate each caption to the first 77 characters
        truncated_captions.append(caption[:max_char_length])
    return truncated_captions


def preprocess_captions(captions, max_char_length):
    """
      Preprocess captions by truncating them to a specified maximum character length.

      Parameters:
      - captions (list of str): List of text captions.
      - max_char_length (int): Maximum number of characters allowed in a caption.

      Returns:
      - list of str: List of preprocessed captions.
    """
    return truncate_captions(captions, max_char_length)


def generate_embeddings(images, captions, clip_model, device):
    """
    Generate embeddings for images and captions using CLIP model.

    Parameters:
    images (Tensor): Batch of preprocessed images.
    captions (list): List of text captions.
    clip_model (CLIP model): Loaded CLIP model.
    device (str): Computation device ('cuda' or 'cpu').

    Returns:
    tuple: Two tensors containing image and text embeddings.
    """
    image_embeddings = clip_model.encode_image(images).detach().cpu()

    # Tokenize captions directly (TODO: relying on CLIP's tokenizer to manage token length, so why not working
    #  without truncate?)
    tokenized_captions = clip.tokenize(captions).to(device)
    text_embeddings = clip_model.encode_text(tokenized_captions).detach().cpu()

    return image_embeddings, text_embeddings


def test_clip_embeddings(image_embeddings, text_embeddings):
    """
    Test and visualize the cosine similarity between image and text embeddings.

    Parameters:
    image_embeddings (Tensor): Embeddings of images.
    text_embeddings (Tensor): Embeddings of corresponding captions.
    """

    cosine_sim = cosine_similarity(image_embeddings, text_embeddings)

    # Visualizing the similarity matrix
    plt.figure(figsize=(10, 8))  # Set the figure size
    plt.imshow(cosine_sim, cmap='hot', interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title('Cosine Similarity between Image and Text Embeddings')
    plt.xlabel('Text Embeddings')
    plt.ylabel('Image Embeddings')
    plt.show()

    # Checking similarity for a specific pair
    pair_similarity = cosine_sim[0, 0]  # Similarity between the first image and its corresponding caption
    non_matching_similarity = cosine_sim[0, 1]  # Compare first image with second caption

    # Print explanation of results
    print(f"Similarity between the first image and its caption: {pair_similarity:.4f}")
    print(f"Similarity between the first image and a different caption: {non_matching_similarity:.4f}")
    print("\nInterpreting the results:")
    print("- The heatmap visualizes the similarity scores between each image and text embedding.")
    print("- Each cell in the heatmap corresponds to the similarity score between an image and a text.")
    print("- Ideally, higher similarity scores (brighter colors) should align along the diagonal, "
          "indicating that each image is most similar to its corresponding caption.")
    print("- Off-diagonal brighter cells indicate similarities between non-matching image-text pairs.")


def main(dataset_path):
    """
    Main function to load data, process images and captions from ROCO dataset, create embedding using clip model and
    test embeddings.

    Parameters:
    dataset_path (str): Path to the ROCO dataset.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    data = load_roco_dataset(dataset_path)
    image_paths, captions = zip(*data)

    images = preprocess_images(image_paths, clip_model).to(device)

    # Truncate captions to the first 77 characters
    captions = preprocess_captions(captions,  CLIP_MODEL_TEXT_MAX_LENGTH)  # TODO: think how to handle this better

    # Generate embeddings with truncated captions
    image_embeddings, text_embeddings = generate_embeddings(images, captions, clip_model, device)
    print("Image Embeddings:", image_embeddings)
    print("Text Embeddings:", text_embeddings)

    # Test similarity between image and caption embeddings using cosine similarity metric (higher is better)
    test_clip_embeddings(image_embeddings, text_embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract embedding of the ROCO dataset with a CLIP model.")
    parser.add_argument("dataset_path", type=str, help="Path to the ROCO dataset directory.")

    args = parser.parse_args()
    main(args.dataset_path)

