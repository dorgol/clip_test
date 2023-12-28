import numpy as np
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
    selected_paths = [paths[idx] for idx in topk_indices]
    return selected_paths


def get_top_p_images(probs, paths, p):
    # Pair each first element of the probability tensor with the corresponding path
    prob_path_pairs = zip([prob[0] for prob in probs], paths)

    # Filter pairs where the first element of the probability tensor is greater than or equal to p
    filtered_pairs = [(prob, path) for prob, path in prob_path_pairs if prob >= p]

    # Sort the pairs by the first element of the probability tensor in descending order
    sorted_pairs = sorted(filtered_pairs, key=lambda x: x[0], reverse=True)

    # Extract the paths from the sorted pairs
    selected_paths = [path for prob, path in sorted_pairs]

    return selected_paths




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
