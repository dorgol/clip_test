from typing import List, Tuple
import os

import h5py
import numpy as np
import streamlit as st
from PIL import Image


def load_all_images(dir_path: str) -> Tuple[List[Image.Image], List[str]]:
    """
    Loads all images from a specified directory.

    :param dir_path: str, the path to the directory containing images.
    :return: A tuple containing a list of PIL Image objects and their corresponding file paths.
    """
    list_images = os.listdir(dir_path)
    images = []
    image_paths = []
    for image_path in list_images:
        image_path = dir_path + image_path
        image_paths.append(image_path)
        image = Image.open(image_path)
        images.append(image)
    return images, image_paths


def load_images(image_paths: List[str]) -> List[Image.Image]:
    """
    Loads images from a list of image paths.

    :param image_paths: List[str], a list containing paths to images.
    :return: List[Image.Image], a list of PIL Image objects.
    """
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        images.append(image)
    return images


def show_images(images: List[Image.Image]) -> None:
    """
    Displays images using the default image viewer.

    :param images: List[Image.Image], a list of PIL Image objects to be displayed.
    """
    for image in images:
        image.show()


def get_embedding_for_image(image_name: str, h5f_path: str = 'image_embeddings_g.h5') -> np.ndarray:
    """
    Retrieves the embedding for a specific image from an HDF5 file.

    :param image_name: str, the name of the image to retrieve the embedding for.
    :param h5f_path: str, the path to the HDF5 file containing embeddings.
    :return: np.ndarray, the embedding of the specified image.
    """
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


def display_thumbnail_in_column(column, image: Image.Image, thumbnail_size: Tuple[int, int] = (800, 200),
                                tooltip_data: np.ndarray = None) -> None:
    """
    Resizes an image to thumbnail size and displays it in a specified Streamlit column with optional tooltip data.

    :param column: Streamlit column object to display the image in.
    :param image: Image.Image, the image to be displayed.
    :param thumbnail_size: Tuple[int, int], the size to resize the image to.
    :param tooltip_data: np.ndarray, optional data to display as a tooltip for the image.
    """
    image.thumbnail(thumbnail_size)
    if tooltip_data.any():
        # Convert probability distribution to a string
        distribution_str = ", ".join([f"{prob:.2f}" for prob in tooltip_data])
        with column.container():
            st.image(image, use_column_width=True, caption=f"Dist: {distribution_str}")
    else:
        column.image(image)


def display_images_in_grid(path_prob_pairs: List[Tuple[str, np.ndarray]], images_per_row: int = 3) -> None:
    """
    Displays images in a grid layout with optional tooltip data for each image.

    :param path_prob_pairs: List[Tuple[str, np.ndarray]], a list of tuples containing image paths and corresponding tooltip data.
    :param images_per_row: int, the number of images to display per row.
    """
    for i in range(0, len(path_prob_pairs), images_per_row):
        cols = st.columns(images_per_row)
        for col, (image_path, image_probs) in zip(cols, path_prob_pairs[i:i + images_per_row]):
            image = Image.open(image_path)
            display_thumbnail_in_column(col, image, tooltip_data=image_probs)


def display_images_in_grid_no_tooltip(paths: List[str], images_per_row: int = 3) -> None:
    """
    Displays images in a grid layout without tooltip data.

    :param paths: List[str], a list of image paths.
    :param images_per_row: int, the number of images to display per row.
    """
    for i in range(0, len(paths), images_per_row):
        cols = st.columns(images_per_row)
        for col, image_path in zip(cols, paths[i:i + images_per_row]):
            image = Image.open(image_path)
            image.thumbnail((800,200))
            col.image(image)

