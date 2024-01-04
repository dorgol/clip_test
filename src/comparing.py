import numpy as np
import streamlit
import torch
from PIL import Image

from src.image_utils import show_images


def calculate_logits(batch_image_embeddings: torch.Tensor, prompts_embedding: torch.Tensor):
    image_embeddings_norm = batch_image_embeddings / batch_image_embeddings.norm(
        dim=-1, keepdim=True
    )
    prompt_embeddings_norm = prompts_embedding / prompts_embedding.norm(dim=-1, keepdim=True)
    logits = 100.0 * image_embeddings_norm @ prompt_embeddings_norm.T
    return logits


def calculate_probabilities(logits: torch.tensor) -> np.ndarray:
    return logits.softmax(dim=-1).cpu().numpy()


def get_top_k_images(probs, paths, k):
    partition_indices = np.argpartition(probs[:, 0], -k)[-k:]
    topk_indices = partition_indices[np.argsort(probs[partition_indices, 0])]
    selected_paths_probs = [(paths[idx], probs[idx]) for idx in topk_indices]
    return selected_paths_probs


def get_top_p_images(probs, paths, p, strict_filter=False):
    prob_path_pairs = zip(probs, paths)

    if strict_filter:
        # Include only pairs where the first element is the max in its row
        filtered_pairs = [(path, prob) for prob, path in prob_path_pairs if prob[0] >= p and prob[0] == max(prob)]
    else:
        # Include all pairs where the first element is greater than or equal to p
        filtered_pairs = [(path, prob) for prob, path in prob_path_pairs if prob[0] >= p]

    sorted_pairs = sorted(filtered_pairs, key=lambda x: x[1][0], reverse=True)
    return sorted_pairs


def get_relevant_images(probs, paths, k=3, show=True):
    top_k_images = get_top_k_images(probs, paths, k)
    images = []
    # Load and store each of the top k images
    for image in top_k_images:
        image = Image.open(image)
        images.append(image)

    if show:
        show_images(images)

    return images
