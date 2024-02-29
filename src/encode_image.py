import os
from typing import List, Tuple
import h5py
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from src.image_model import get_model_and_processor, get_image_embeddings

with open('config.yaml') as f:
    config = yaml.safe_load(f)
MODEL_NAME = config['CLIP_MODEL_STR']
model, processor = get_model_and_processor(MODEL_NAME)


def get_image_embedding(image: Image.Image, model, processor) -> torch.Tensor:
    """
    Generates an embedding for a given image using the specified model and processor.

    :param image: Image.Image, the image to generate the embedding for.
    :param model: The model used to generate embeddings.
    :param processor: The processor used to prepare the image for the model.
    :return: np.ndarray, the generated embedding for the image.
    """
    embeddings = get_image_embeddings(processor, model, image)
    return embeddings


def process_and_save_batch(batch_embeddings: torch.Tensor, batch_files: List[str], h5f: h5py.File) -> None:
    """
    Processes a batch of embeddings and file names to save them into an HDF5 file.

    :param batch_embeddings: np.ndarray, the embeddings of the batch images.
    :param batch_files: List[str], the file names of the batch images.
    :param h5f: h5py.File, the HDF5 file handle where data is saved.
    """
    embeddings_array = np.array(batch_embeddings.detach())

    current_size = h5f['embeddings'].shape[0]
    new_size = current_size + embeddings_array.shape[0]
    h5f['embeddings'].resize((new_size, embeddings_array.shape[1]))
    h5f['embeddings'][current_size:new_size, :] = embeddings_array

    h5f['image_names'].resize((new_size,))
    h5f['image_names'][current_size:new_size] = np.array(batch_files, dtype=h5py.special_dtype(vlen=str))


def embed_all(image_folder: str = 'test_images/val2017', batch_size: int = 32) -> None:
    """
    Processes all images in the specified folder to generate embeddings and saves them along with their names in an HDF5
    file.

    :param image_folder: str, the folder containing images to process.
    :param batch_size: int, the number of images to process in each batch.
    """
    all_image_files = os.listdir(image_folder)
    h5_file_path = 'image_embeddings.h5'

    # Check if the HDF5 file exists
    file_exists = os.path.isfile(h5_file_path)

    with h5py.File(h5_file_path, 'a') as h5f:  # Open file in append mode
        if not file_exists:
            # Create datasets if file does not exist
            h5f.create_dataset('embeddings', shape=(0, 512), maxshape=(None, 512), dtype=np.float32)
            h5f.create_dataset('image_names', shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str))

        embeddings_dataset = h5f['embeddings']
        names_dataset = h5f['image_names']

        for i in tqdm(range(0, len(all_image_files), batch_size)):
            batch_files = all_image_files[i:i + batch_size]
            batch_images = [Image.open(os.path.join(image_folder, img_file)) for img_file in batch_files]
            batch_embeddings = get_image_embedding(batch_images)

            # Get current dataset size
            current_size = embeddings_dataset.shape[0]
            new_size = current_size + len(batch_embeddings)

            # Resize datasets
            embeddings_dataset.resize(new_size, axis=0)
            names_dataset.resize(new_size, axis=0)

            # Append new data
            embeddings_dataset[current_size:new_size] = batch_embeddings
            names_dataset[current_size:new_size] = [image_folder + "/" + i for i in batch_files]


def load_embeddings() -> np.ndarray:
    """
    Loads and returns all image embeddings from the HDF5 file.

    :return: np.ndarray, all image embeddings stored in the HDF5 file.
    """
    with h5py.File('image_embeddings.h5', 'r') as h5f:
        loaded_embeddings = h5f['embeddings'][:]
        return loaded_embeddings


def load_names() -> List[str]:
    """
    Loads and returns all image names from the HDF5 file.

    :return: List[str], all image names stored in the HDF5 file, decoded to UTF-8.
    """
    with h5py.File('image_embeddings.h5', 'r') as h5f:
        loaded_embeddings = h5f['image_names'][:]
        loaded_embeddings = [name.decode('utf-8') for name in loaded_embeddings]
        return loaded_embeddings


if __name__ == "__main__":
    # embed_all("test_images/val2017")
    embed_all("test_images/images")
