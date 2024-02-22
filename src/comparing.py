import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score


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


def get_sampled_images(probs, paths, n, percentile):
    # Pair up paths and probabilities and sort by probabilities
    prob_path_pairs = sorted(zip(probs, paths), key=lambda x: x[0])

    # Find the index corresponding to the desired percentile
    percentile_index = int(len(prob_path_pairs) * (percentile / 100))

    # Calculate the number of elements to consider on each side of the percentile index
    half_n = n // 2

    # Determine the range to sample from, ensuring at least n elements are available
    range_start = max(0, percentile_index - half_n)
    range_end = min(len(prob_path_pairs), range_start + n)

    # Adjust range_start if range_end is at the end of the list
    range_start = max(0, range_end - n)

    # Randomly sample n items from the range
    sampled = random.sample(prob_path_pairs[range_start:range_end], n)

    return sampled


def get_results_df(probabilities_per_image, image_names, categories):
    results_df = pd.DataFrame(probabilities_per_image, columns=categories)
    results_df['Image Path'] = image_names
    return results_df


def extract_max_category(df):
    # Copy the 'Image Path' column
    # result_df = pd.DataFrame(df['Image Path'])

    # Find the category with the maximum value for each row
    df['Category'] = df.drop('Image Path', axis=1).idxmax(axis=1)

    return df


def compare_clip_and_gpt(clip_df, gpt_df):
    joined_df = pd.merge(clip_df, gpt_df, left_on='Image Path', right_on='Image Path', how='inner')
    return joined_df


def create_confusion_matrix(df):
    labels = sorted(set(df['Result'].unique()).union(df['Category'].unique()))

    # Calculating the confusion matrix
    cm = confusion_matrix(df['Result'], df['Category'], labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    return cm_df


def calculate_accuracy(df):
    """Calculate the accuracy of predictions."""
    return accuracy_score(df['Result'], df['Category'])


def get_classification_report(df):
    """Get precision, recall, and F1-score."""
    return classification_report(df['Result'], df['Category'])


def calculate_cohens_kappa(df):
    """Calculate Cohen's Kappa."""
    return cohen_kappa_score(df['Result'], df['Category'])


def get_classification_metrics(df):
    accuracy = calculate_accuracy(df)
    confusion = create_confusion_matrix(df)
    class_report = get_classification_report(df)
    cohens_kappa = calculate_cohens_kappa(df)
    return accuracy, class_report, cohens_kappa, confusion
