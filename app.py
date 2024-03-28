import os
from typing import List
from typing import Union, Dict

import pandas as pd
import streamlit as st
import torch
from pandas import DataFrame

# Import your module functions here
from src.comparing import (calculate_logits, calculate_probabilities, get_top_k_images,
                           get_top_p_images, get_results_df, extract_max_category,
                           compare_clip_and_gpt, get_classification_metrics, compare_to_tags)
from src.encode_image import load_embeddings, load_names
from src.enrich import run_enrichment
from src.image_tagging.tag_images import create_tagged_dataset
from src.image_utils import display_images_in_grid, display_images_in_grid_no_tooltip
from src.text_model import get_text_embeddings, get_model_and_processor, MODEL_NAME


def initialize_session_state() -> None:
    """Initializes session state variables."""
    if 'negative_inputs' not in st.session_state:
        st.session_state.negative_inputs = []
    if 'negative_tags' not in st.session_state:
        st.session_state.negative_tags = ['' for _ in st.session_state.negative_inputs]
    if 'input' not in st.session_state:
        st.session_state.input = []
    if 'input_tags' not in st.session_state:
        st.session_state.input_tags = ''


def main():
    initialize_session_state()
    manage_inputs()
    manage_dataset_selection()
    trigger_search_and_display_results()


def manage_inputs() -> None:
    """Manages user inputs including search term and negative prompts."""
    search_term_input()
    manage_negative_prompts()


def manage_dataset_selection() -> None:
    """Lets the user select a dataset and updates session state accordingly."""
    datasets = get_available_datasets()
    chosen_dataset = st.selectbox('Choose a dataset', datasets)

    # Dynamically update selected dataset in session state
    if 'selected_dataset' not in st.session_state or st.session_state.selected_dataset != chosen_dataset:
        st.session_state.selected_dataset = chosen_dataset


def trigger_search_and_display_results() -> None:
    """Handles the search functionality and displays results based on user input and selected dataset."""
    chosen_dataset_path = f'datasets/{st.session_state.selected_dataset}'
    search_and_display_results(chosen_dataset_path)


def add_negative():
    st.session_state.negative_inputs.append('')
    st.session_state.negative_tags.append('')


def remove_negative(index):
    del st.session_state.negative_inputs[index]
    del st.session_state.negative_tags[index]


def get_available_datasets() -> List[str]:
    """Returns a list of available dataset files."""
    directory = 'datasets/'
    h5_files = os.listdir(directory)
    return h5_files


def manage_negative_prompts():
    # Function to add a new negative prompt
    add_negative_prompt()

    # Loop over existing negative prompts
    for i in range(len(st.session_state.negative_inputs)):
        delete_negative_prompt(i)


def search_term_input() -> None:
    """
    Creates input fields for the user's search term and associated tag, updating
    the session state with the user's input.
    """
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        default_value = st.session_state.input[0] if st.session_state.input else 'A photo of a kid'
        search_term = st.text_input('Enter your search', value=default_value)

        if st.session_state.input:
            if st.session_state.input[0] != search_term:
                st.session_state.input[0] = search_term
        else:
            st.session_state.input.append(search_term)

    with col2:
        st.text_input('Tag', value=st.session_state.input_tags, key='input_tags')

    with col3:
        if st.button('Enrich Prompt'):
            if st.session_state.input:
                enriched_term = run_enrichment(st.session_state.input[0])
                st.session_state.input[0] = enriched_term['text']


def enrich_and_negative_buttons() -> None:
    """
    Provides buttons for enriching the primary search prompt and adding a new negative prompt.
    - The 'Enrich Prompt' button enriches the current main search term using an external enrichment function.
    - The 'Add negative prompt' button adds a new empty slot for a negative prompt.
    """
    if st.button('Enrich Prompt'):
        if st.session_state.input:
            enriched_term = run_enrichment(st.session_state.input[0])
            st.session_state.input[0] = enriched_term['text']

    if st.button('Add negative prompt'):
        add_negative()


def negative_prompts() -> None:
    """
    Displays input fields for each negative prompt and provides an interface
    for editing or enriching each negative prompt. Utilizes the `negative_prompt_buttons`
    function to display buttons for deletion and enrichment of negative prompts.
    """
    for i in range(len(st.session_state.negative_inputs)):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.session_state.negative_inputs[i] = st.text_input(
                f'Negative example {i + 1}',
                value=st.session_state.negative_inputs[i],
                key=f'negative_{i}')
        with col2:
            negative_prompt_buttons(i)


def add_negative_prompt() -> None:
    """
    Adds a new negative prompt entry to the session state. This is triggered by a button
    click in the Streamlit interface. The function adds an empty string to both the
    `negative_inputs` and `negative_tags` lists in the session state, effectively
    creating a new slot for a negative prompt and its corresponding tag.
    """
    if st.button('Add negative prompt'):
        add_negative()


def delete_negative_prompt(index: int) -> None:
    """
    Deletes a specified negative prompt and its associated tag from the session state.

    Parameters:
        index (int): The index of the negative prompt (and corresponding tag) to be removed.
    """
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        negative_prompt = st.text_input(
            f'Negative example {index + 1}',
            value=st.session_state.negative_inputs[index],
            key=f'negative_{index}')
        st.session_state.negative_inputs[index] = negative_prompt

    with col2:
        # Check if the index is within the bounds of negative_tags
        if index < len(st.session_state.negative_tags):
            tag_value = st.session_state.negative_tags[index]
        else:
            tag_value = ''
        st.session_state.negative_tags[index] = st.text_input(
            'Tag', value=tag_value, key=f'tag_{index}')
    with col3:
        if st.button('Delete', key=f'delete_{index}'):
            remove_negative(index)
        if st.button('Enrich', key=f'enrich_{index}'):
            enriched_term = run_enrichment(negative_prompt)
            st.session_state.negative_inputs[index] = enriched_term['text']


def negative_prompt_buttons(index: int) -> None:
    """
    Displays buttons for each negative prompt that allow the user to delete or enrich the prompt.
    This function should be called for each negative prompt in the session state.

    Parameters:
        index (int): The index of the negative prompt to which these buttons apply.
    """
    if st.button('Delete', key=f'delete_{index}'):
        remove_negative(index)
    if st.button('Enrich', key=f'enrich_{index}'):
        enriched_term = run_enrichment(st.session_state.negative_inputs[index])
        st.session_state.negative_inputs[index] = enriched_term['text']


def search_and_display_results(dataset: str) -> bool:
    """
    Handles the search operation against the selected dataset and prepares the results
    for display. This function orchestrates the search process, including loading the
    dataset, executing the search, and setting up the results in the session state.

    Parameters:
        dataset (str): The file path to the selected dataset.

    Returns:
        bool: True if the search was executed and results are ready to be displayed,
              False otherwise.
    """
    search_terms = [st.session_state.input[0]] + st.session_state.negative_inputs + ["some general random stuff"]
    df = pd.DataFrame(search_terms, columns=["Strings"])
    st.table(df)
    images_features = torch.tensor(load_embeddings(dataset))
    image_names = load_names(dataset)

    if st.button('Search'):
        execute_search(search_terms, images_features, image_names)

    if 'search_results' in st.session_state:
        display_search_results(dataset)

    return True


def execute_search(search_terms: List[str], images_features: torch.Tensor, image_names: List[str]) -> None:
    """
    Executes the search operation by calculating the embeddings for the given search terms,
    computing logits and probabilities for images based on these embeddings, and updating
    the session state with the search results.

    Parameters:
        search_terms (List[str]): A list of search terms.
        images_features (torch.Tensor): A tensor containing the features of the images.
        image_names (List[str]): A list of image names corresponding to the image features.
    """
    model, processor = get_model_and_processor(MODEL_NAME)
    text_emb = get_text_embeddings(text=search_terms, model=model, processor=processor)
    logits_per_image = calculate_logits(images_features, text_emb)
    probabilities_per_image = calculate_probabilities(logits_per_image)
    st.session_state.search_results = (probabilities_per_image, image_names)


def load_and_display_tag_comparisons(dataset_name: str, probabilities_per_image: torch.Tensor, image_names: List[str]) \
        -> None:
    """
    Loads the appropriate dataset for tag comparisons based on the dataset name and
    displays the tag comparison in a Streamlit tab.

    Parameters:
        dataset_name (str): The name of the dataset.
        probabilities_per_image (torch.Tensor): The probabilities associated with each image.
        image_names (List[str]): The names of the images.
    """
    if 'fashion' in dataset_name:
        fashion_dataset = pd.read_csv('aux_data/list_attr_clothes.csv')
        id_col = 'file_name'
        dataset = fashion_dataset.copy()
    elif 'celeb' in dataset_name:
        celebs_dataset = pd.read_csv('aux_data/list_attr_celeba.csv')
        id_col = 'file_name'
        if 'full' in dataset_name:
            celebs_dataset[id_col] = 'test_images/img_celeba/' + celebs_dataset[id_col]
        else:
            celebs_dataset[id_col] = 'test_images/img_align_celeba/' + celebs_dataset[id_col]
        dataset = celebs_dataset.copy()
    tag_col = st.selectbox('tag column', dataset.columns[2:])
    get_confusion_tags(probabilities_per_image, image_names,
                       dataset, id_col, tag_col)


def display_search_results(dataset_name: str) -> None:
    """
    Displays the search results in various tabs including top K images, top P images,
    comparison with ChatGPT, and comparison with tags.

    Parameters:
        dataset_name (str): The name of the dataset used for search to determine specific
                            actions for different datasets (e.g., 'fashion', 'celeb').
    """
    tab1, tab2, tab3, tab4 = st.tabs(["Top K", "Top P", "Compare to ChatGPT", "Compare to Tags"])
    probabilities_per_image, image_names = st.session_state.search_results
    with tab1:
        display_top_k_images(probabilities_per_image, image_names)
    with tab2:
        display_top_p_images(probabilities_per_image, image_names)
    with tab3:
        on = st.toggle('Compare to ChatGPT')
        if on:
            display_confusion_matrix(probabilities_per_image, image_names)
    with tab4:
        load_and_display_tag_comparisons(dataset_name, probabilities_per_image, image_names)


def display_confusion_matrix(probabilities_per_image: torch.Tensor, image_names: List[str]) -> None:
    """
    Displays a confusion matrix comparing the classification results with another set of
    results, such as those from ChatGPT, based on the probabilities associated with each image.

    Parameters:
        probabilities_per_image (torch.Tensor): The probabilities associated with each image.
        image_names (List[str]): The names of the images.
    """
    categories = [st.session_state.input_tags] + st.session_state.negative_tags + ["general"]
    df = get_results_df(probabilities_per_image, image_names, categories)
    df = extract_max_category(df)
    gpt_df = get_ground_truth(probabilities_per_image, image_names)
    joined = compare_clip_and_gpt(clip_df=df, gpt_df=gpt_df)
    st.dataframe(joined)
    accuracy, class_report, cohens_kappa, confusion = get_classification_metrics(joined)

    # Display the results using the utility function
    display_metrics(accuracy, confusion, class_report, cohens_kappa)


def display_top_k_images(probabilities_per_image: torch.Tensor, image_names: List[str]) -> None:
    """
    Displays the top K images based on the probabilities of matching the search criteria.

    Parameters:
        probabilities_per_image (torch.Tensor): A tensor containing the probabilities
                                                for each image.
        image_names (List[str]): A list of image names corresponding to each probability.
    """
    st.markdown("### Top K Images")
    k = st.number_input('Insert a number for Top K', min_value=1, value=5, step=1)
    if k:
        # Example call
        path_prob_pairs = get_top_k_images(probabilities_per_image, image_names, k)
        display_images_in_grid(path_prob_pairs)


def display_top_p_images(probabilities_per_image: torch.Tensor, image_names: List[str]) -> None:
    """
    Displays images with a probability above a specified threshold, allowing the user
    to define what constitutes a "top" image based on probability.

    Parameters:
        probabilities_per_image (torch.Tensor): A tensor containing the probabilities
                                                for each image.
        image_names (List[str]): A list of image names corresponding to each probability.
    """
    st.markdown("#### Top P")
    on = st.toggle('max probability')
    p = st.number_input('Insert a probability for Top P', min_value=0.01, value=0.99, step=0.01)
    if p:
        images_paths = get_top_p_images(probabilities_per_image, image_names, p, on)
        num_images = len(images_paths)
        st.write(f"There are {num_images} images with probability above {p}")
        slider_range = st.slider("Range to show images", 0, num_images, (0, num_images))
        pairs = images_paths[slider_range[0]:slider_range[1]]
        display_images_in_grid(pairs)


def get_ground_truth(probabilities_per_image: torch.Tensor, image_names: List[str]) -> pd.DataFrame:
    """
    Generates a dataset of images sampled based on the probabilities, serving as a
    ground truth for comparison.

    Parameters:
        probabilities_per_image (torch.Tensor): A tensor containing the probabilities
                                                for each image.
        image_names (List[str]): A list of image names.

    Returns:
        pd.DataFrame: A dataframe containing paths to sampled images and their categories.
    """
    st.markdown("### Sampled Images")
    n = st.number_input('Insert a number for sampling', min_value=1, value=10, step=1)
    if n:
        categories = st.session_state.negative_tags + [st.session_state.input_tags]
        df_truth = create_tagged_dataset(probabilities_per_image[:, 0], image_names, n, categories)

        display_images_in_grid_no_tooltip(df_truth['Image Path'].values)
        return df_truth


def get_confusion_tags(probs: torch.Tensor, paths: List[str], dataset: pd.DataFrame, id_col: str, tag_col: str) -> None:
    """
    Compares the probabilities against a threshold to determine classification,
    and merges this classification with existing tags from a dataset for further analysis.

    Parameters:
        probs (torch.Tensor): The probabilities associated with each image.
        paths (List[str]): The paths or identifiers for each image.
        dataset (pd.DataFrame): The dataset containing additional tags or information.
        id_col (str): The column in `dataset` that corresponds to the image paths/identifiers.
        tag_col (str): The column in `dataset` that contains the tags for comparison.
    """
    threshold = st.slider('score threshold', 0.0, 1.0, 0.5)
    df = compare_to_tags(probs, paths, dataset, id_col, tag_col, threshold)
    st.dataframe(df)
    accuracy, class_report, cohens_kappa, confusion = get_classification_metrics(df)

    # Display the results using the utility function
    display_metrics(accuracy, confusion, class_report, cohens_kappa)


def display_metrics(
    accuracy: float,
    confusion_matrix: DataFrame,
    classification_report: Union[str, Dict],
    cohens_kappa: float
) -> None:
    """
    Displays classification metrics including accuracy, confusion matrix,
    classification report, and Cohen's Kappa using Streamlit.

    Parameters:
    - accuracy (float): The accuracy score of the model, represented as a fraction between 0 and 1.
    - confusion_matrix (DataFrame): A pandas DataFrame representing the confusion matrix of the model's predictions.
    - classification_report (Union[str, Dict]): The classification report as a string or a dictionary
      that summarises the precision, recall, F1-score for each class.
    - cohens_kappa (float): Cohen's kappa statistic as a float, measuring the inter-annotator agreement.

    Returns:
    - None: This function does not return any value, it only uses Streamlit to display metrics.

    Note:
    - This function is designed to be used within a Streamlit application.
    - Ensure that `st` (Streamlit) has been imported before using this function.
    """
    st.write("### Accuracy")
    st.write(accuracy)

    st.write("### Confusion Matrix")
    st.dataframe(confusion_matrix)

    st.write("### Classification Report")
    st.table(classification_report)

    st.write("### Cohen's Kappa")
    st.write(cohens_kappa)


if __name__ == "__main__":
    main()
