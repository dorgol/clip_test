import os

import h5py
import numpy as np
import streamlit as st
from PIL import Image


def load_all_images(dir_path):
    list_images = os.listdir(dir_path)
    images = []
    image_paths = []
    for image_path in list_images:
        image_path = dir_path + image_path
        image_paths.append(image_path)
        image = Image.open(image_path)
        images.append(image)
    return images, image_paths


def load_images(image_paths):
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        images.append(image)
    return images


def show_images(images):
    for image in images:
        image.show()


def get_embedding_for_image(image_name, h5f_path='image_embeddings_g.h5'):
    with h5py.File(h5f_path, 'r') as h5f:
        image_names = h5f['image_names'][:]
        embeddings = h5f['embeddings'][:]

        # Convert image_name to byte string
        image_name_bytes = image_name.encode('utf-8')

        # Find the index of the image
        idxs = np.where(image_names == image_name_bytes)[0]
        if len(idxs) == 0:
            raise ValueError(f"Image name '{image_name}' not found.")
        return embeddings[idxs[0]]


def display_thumbnail_in_column(column, image, thumbnail_size=(800, 200), tooltip_data=None):
    """Resize an image to thumbnail size and display it in a specified Streamlit column with a tooltip."""
    image.thumbnail(thumbnail_size)
    if tooltip_data.any():
        # Convert probability distribution to a string
        distribution_str = ", ".join([f"{prob:.2f}" for prob in tooltip_data])
        with column.container():
            st.image(image, use_column_width=True, caption=f"Dist: {distribution_str}")
    else:
        column.image(image)


def display_images_in_grid(path_prob_pairs, images_per_row=3):
    """Display images in a grid layout with tooltip data."""
    for i in range(0, len(path_prob_pairs), images_per_row):
        cols = st.columns(images_per_row)
        for col, (image_path, image_probs) in zip(cols, path_prob_pairs[i:i + images_per_row]):
            image = Image.open(image_path)
            display_thumbnail_in_column(col, image, tooltip_data=image_probs)


def display_images_in_grid_no_tooltip(paths, images_per_row=3):
    for i in range(0, len(paths), images_per_row):
        cols = st.columns(images_per_row)
        for col, image_path in zip(cols, paths[i:i + images_per_row]):
            image = Image.open(image_path)
            image.thumbnail((800,200))
            col.image(image)

