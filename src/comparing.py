import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score


def calculate_logits(batch_image_embeddings: torch.Tensor, prompts_embedding: torch.Tensor) -> torch.Tensor:
    """
    Calculates the cosine similarity logits between image embeddings and prompt embeddings.

    Args:
    - batch_image_embeddings (torch.Tensor): Tensor of image embeddings.
    - prompts_embedding (torch.Tensor): Tensor of prompt embeddings.

    Returns:
    - torch.Tensor: Tensor of logits representing the similarity scores.
    """
    image_embeddings_norm = batch_image_embeddings / batch_image_embeddings.norm(
        dim=-1, keepdim=True
    )
    prompt_embeddings_norm = prompts_embedding / prompts_embedding.norm(dim=-1, keepdim=True)
    logits = 100.0 * image_embeddings_norm @ prompt_embeddings_norm.T
    return logits


def calculate_probabilities(logits: torch.Tensor) -> np.ndarray:
    """
    Converts logits to probabilities using the softmax function.

    Args:
    - logits (torch.Tensor): Tensor of logits to be converted.

    Returns:
    - np.ndarray: Numpy array of probabilities obtained from the logits.
    """
    return logits.softmax(dim=-1).cpu().numpy()


def get_top_k_images(probs: np.ndarray, paths: list, k: int) -> list:
    """
    Selects the top k images based on probabilities for a specific class.

    Args:
    - probs (np.ndarray): Array of probabilities for each image.
    - paths (list): List of image paths.
    - k (int): Number of top images to select.

    Returns:
    - list: List of tuples containing the paths and probabilities of the top k images.
    """

    partition_indices = np.argpartition(probs[:, 0], -k)[-k:]
    topk_indices = partition_indices[np.argsort(probs[partition_indices, 0])]
    selected_paths_probs = [(paths[idx], probs[idx]) for idx in topk_indices]
    return selected_paths_probs


def get_top_p_images(probs: np.ndarray, paths: list, p: float, strict_filter: bool = False) -> list:
    """
    Filters images based on a probability threshold.

    Args:
    - probs (np.ndarray): Array of probabilities for each image.
    - paths (list): List of image paths.
    - p (float): Probability threshold.
    - strict_filter (bool): If True, only select images where the target class probability is the maximum and above p.

    Returns:
    - list: List of tuples containing paths and probabilities of selected images.
    """

    prob_path_pairs = zip(probs, paths)

    if strict_filter:
        # Include only pairs where the first element is the max in its row
        filtered_pairs = [(path, prob) for prob, path in prob_path_pairs if prob[0] >= p and prob[0] == max(prob)]
    else:
        # Include all pairs where the first element is greater than or equal to p
        filtered_pairs = [(path, prob) for prob, path in prob_path_pairs if prob[0] >= p]

    sorted_pairs = sorted(filtered_pairs, key=lambda x: x[1][0], reverse=True)
    return sorted_pairs


def get_sampled_images(probs: np.ndarray, paths: list, n: int, percentile: float) -> list:
    """
    Samples images around a specified percentile based on their probabilities.

    Args:
    - probs (np.ndarray): Array of probabilities for each image.
    - paths (list): List of image paths.
    - n (int): Number of images to sample.
    - percentile (float): Percentile to sample around.

    Returns:
    - list: List of sampled image paths and their probabilities.
    """

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


def get_results_df(probabilities_per_image: np.ndarray, image_names: list, categories: list) -> pd.DataFrame:
    """
    Creates a DataFrame from image probabilities and names, with categories as columns.

    Args:
    - probabilities_per_image (np.ndarray): Array of probabilities for each image.
    - image_names (list): List of image names.
    - categories (list): List of category names.

    Returns:
    - pd.DataFrame: DataFrame with probabilities, image names, and categories.
    """

    results_df = pd.DataFrame(probabilities_per_image, columns=categories)
    results_df['Image Path'] = image_names
    return results_df


def extract_max_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a column to the DataFrame indicating the category with the highest probability for each image.

    Args:
    - df (pd.DataFrame): DataFrame containing probabilities and image paths.

    Returns:
    - pd.DataFrame: Updated DataFrame with a new column for the maximum probability category.
    """

    df['Category'] = df.drop('Image Path', axis=1).idxmax(axis=1)

    return df


def compare_clip_and_gpt(clip_df: pd.DataFrame, gpt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges DataFrame containing CLIP model results with GPT model results based on image paths.

    Args:
    - clip_df (pd.DataFrame): DataFrame with CLIP model results.
    - gpt_df (pd.DataFrame): DataFrame with GPT model results.

    Returns:
    - pd.DataFrame: Merged DataFrame with results from both models.
    """

    joined_df = pd.merge(clip_df, gpt_df, left_on='Image Path', right_on='Image Path', how='inner')
    return joined_df


def create_confusion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a confusion matrix from a DataFrame containing actual and predicted labels.

    The function identifies all unique labels present in both 'Result' and 'Category' columns
    to include every category in the confusion matrix. This approach ensures a comprehensive
    comparison between actual and predicted classifications across all categories.

    Parameters:
    - df (pd.DataFrame): DataFrame with two columns:
        - 'Result': Actual labels of the dataset.
        - 'Category': Predicted labels by the model.

    Returns:
    - pd.DataFrame: A confusion matrix as a DataFrame, where rows represent actual labels,
      columns represent predicted labels, and cell values indicate the counts of predictions
      for each actual-predicted label pair.

    Example:
    >>> df = pd.DataFrame({'Result': ['cat', 'dog', 'cat'], 'Category': ['dog', 'dog', 'cat']})
    >>> create_confusion_matrix(df)
            cat  dog
        cat   1    1
        dog   0    1
    """
    labels = sorted(set(df['Result'].unique()).union(df['Category'].unique()))

    # Calculating the confusion matrix
    cm = confusion_matrix(df['Result'], df['Category'], labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    return cm_df


def calculate_accuracy(df):
    """Calculate the accuracy of predictions."""
    return accuracy_score(df['Result'], df['Category'])


def get_classification_report(df: pd.DataFrame) -> str:
    """
    Generates a classification report for the model predictions, including precision, recall, and F1-score for each class.

    Parameters:
    - df (pd.DataFrame): A DataFrame with 'Result' as actual labels and 'Category' as predicted labels.

    Returns:
    - str: A string representation of the classification report.
    """
    return classification_report(df['Result'], df['Category'])


def calculate_cohens_kappa(df: pd.DataFrame) -> float:
    """
    Calculates Cohen's Kappa coefficient to measure the agreement between two annotators on a classification problem.

    Parameters:
    - df (pd.DataFrame): A DataFrame with 'Result' as actual labels and 'Category' as predicted labels.

    Returns:
    - float: The Cohen's Kappa score, where 1 represents perfect agreement and values less than 0 indicate no agreement.
    """
    return cohen_kappa_score(df['Result'], df['Category'])


def get_classification_metrics(df: pd.DataFrame) -> tuple:
    """
    Aggregates various classification metrics, including accuracy, Cohen's Kappa, and a classification report, along with the confusion matrix.

    Parameters:
    - df (pd.DataFrame): A DataFrame with 'Result' as actual labels and 'Category' as predicted labels.

    Returns:
    - tuple: A tuple containing the accuracy score, classification report, Cohen's Kappa score, and confusion matrix DataFrame.
    """
    accuracy = calculate_accuracy(df)
    confusion = create_confusion_matrix(df)
    class_report = get_classification_report(df)
    cohens_kappa = calculate_cohens_kappa(df)
    return accuracy, class_report, cohens_kappa, confusion
