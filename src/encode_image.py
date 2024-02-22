import os

import h5py
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm
from src.image_model import get_model_and_processor, get_image_embeddings

with open('config.yaml') as f:
    config = yaml.safe_load(f)
MODEL_NAME = config['CLIP_MODEL_STR']
model, processor = get_model_and_processor(MODEL_NAME)


def get_image_embedding(image, model=model, processor=processor):
    embeddings = get_image_embeddings(processor, model, image)
    return embeddings


def process_and_save_batch(batch_embeddings, batch_files, h5f):
    embeddings_array = np.array(batch_embeddings.detach())

    current_size = h5f['embeddings'].shape[0]
    new_size = current_size + embeddings_array.shape[0]
    h5f['embeddings'].resize((new_size, embeddings_array.shape[1]))
    h5f['embeddings'][current_size:new_size, :] = embeddings_array

    h5f['image_names'].resize((new_size,))
    h5f['image_names'][current_size:new_size] = np.array(batch_files, dtype=h5py.special_dtype(vlen=str))


def embed_all(image_folder='test_images/val2017', batch_size=32):
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


def load_embeddings():
    with h5py.File('image_embeddings.h5', 'r') as h5f:
        loaded_embeddings = h5f['embeddings'][:]
        return loaded_embeddings


def load_names():
    with h5py.File('image_embeddings.h5', 'r') as h5f:
        loaded_embeddings = h5f['image_names'][:]
        loaded_embeddings = [name.decode('utf-8') for name in loaded_embeddings]
        return loaded_embeddings


if __name__ == "__main__":
    # embed_all("test_images/val2017")
    embed_all("test_images/images")
