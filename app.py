import os

import pandas as pd
import streamlit as st
import torch

from src.comparing import calculate_logits, calculate_probabilities, get_top_k_images, get_top_p_images, \
    get_results_df, extract_max_category, compare_clip_and_gpt, get_classification_metrics
from src.encode_image import load_embeddings, load_names
from src.enrich import run_enrichment
from src.image_tagging.tag_images import create_tagged_dataset
from src.image_utils import display_images_in_grid, display_images_in_grid_no_tooltip
from src.text_model import get_text_embeddings, get_model_and_processor, MODEL_NAME


def initialize_session_state():
    if 'negative_inputs' not in st.session_state:
        st.session_state.negative_inputs = []
    if 'negative_tags' not in st.session_state:
        st.session_state.negative_tags = ['' for _ in st.session_state.negative_inputs]
    if 'input' not in st.session_state:
        st.session_state.input = []
    if 'input_tags' not in st.session_state:
        st.session_state.input_tags = ''


def add_negative():
    st.session_state.negative_inputs.append('')
    st.session_state.negative_tags.append('')


def remove_negative(index):
    del st.session_state.negative_inputs[index]
    del st.session_state.negative_tags[index]


def get_available_datasets():
    directory = 'datasets/'
    h5_files = os.listdir(directory)
    return h5_files


def main():
    initialize_session_state()

    # Main search term input and enrich button
    search_term_input()

    # Function to add a new negative prompt and manage existing ones
    manage_negative_prompts()
    datasets = get_available_datasets()
    chosen_dataset = st.selectbox('choose a dataset', datasets)
    chosen_dataset = 'datasets/' + chosen_dataset
    # Combine search terms and search
    search_and_display_results(chosen_dataset)


def manage_negative_prompts():
    # Function to add a new negative prompt
    add_negative_prompt()

    # Loop over existing negative prompts
    for i in range(len(st.session_state.negative_inputs)):
        delete_negative_prompt(i)


def search_term_input():
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


def enrich_and_negative_buttons():
    if st.button('Enrich Prompt'):
        if st.session_state.input:
            enriched_term = run_enrichment(st.session_state.input[0])
            st.session_state.input[0] = enriched_term['text']

    if st.button('Add negative prompt'):
        add_negative()


def negative_prompts():
    for i in range(len(st.session_state.negative_inputs)):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.session_state.negative_inputs[i] = st.text_input(
                f'Negative example {i + 1}',
                value=st.session_state.negative_inputs[i],
                key=f'negative_{i}')
        with col2:
            negative_prompt_buttons(i)


def add_negative_prompt():
    if st.button('Add negative prompt'):
        add_negative()


def delete_negative_prompt(index):
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


def negative_prompt_buttons(index):
    if st.button('Delete', key=f'delete_{index}'):
        remove_negative(index)
    if st.button('Enrich', key=f'enrich_{index}'):
        enriched_term = run_enrichment(st.session_state.negative_inputs[index])
        st.session_state.negative_inputs[index] = enriched_term['text']


def search_and_display_results(dataset):
    search_terms = [st.session_state.input[0]] + st.session_state.negative_inputs + ["some general random stuff"]
    df = pd.DataFrame(search_terms, columns=["Strings"])
    st.table(df)
    images_features = torch.tensor(load_embeddings(dataset))
    image_names = load_names(dataset)

    if st.button('Search'):
        execute_search(search_terms, images_features, image_names)

    if 'search_results' in st.session_state:
        display_search_results()


def execute_search(search_terms, images_features, image_names):
    model, processor = get_model_and_processor(MODEL_NAME)
    text_emb = get_text_embeddings(text=search_terms, model=model, processor=processor)
    logits_per_image = calculate_logits(images_features, text_emb)
    probabilities_per_image = calculate_probabilities(logits_per_image)
    st.session_state.search_results = (probabilities_per_image, image_names)


def display_search_results():
    probabilities_per_image, image_names = st.session_state.search_results
    display_top_k_images(probabilities_per_image, image_names)
    display_top_p_images(probabilities_per_image, image_names)
    on = st.toggle('compare to chatgpt')
    if on:
        display_confusion_matrix(probabilities_per_image, image_names)


def display_confusion_matrix(probabilities_per_image, image_names):
    categories = [st.session_state.input_tags] + st.session_state.negative_tags + ["general"]
    df = get_results_df(probabilities_per_image, image_names, categories)
    df = extract_max_category(df)
    gpt_df = get_ground_truth(probabilities_per_image, image_names)
    joined = compare_clip_and_gpt(clip_df=df, gpt_df=gpt_df)
    st.dataframe(joined)
    accuracy, class_report, cohens_kappa, confusion = get_classification_metrics(joined)
    # Display the results in Streamlit
    st.write("### Accuracy")
    st.write(accuracy)

    st.write("### Confusion Matrix")
    st.dataframe(confusion)

    st.write("### Classification Report")
    st.write(class_report)

    st.write("### Cohen's Kappa")
    st.write(cohens_kappa)


def display_top_k_images(probabilities_per_image, image_names):
    st.markdown("### Top K Images")
    k = st.number_input('Insert a number for Top K', min_value=1, value=5, step=1)
    if k:
        # Example call
        path_prob_pairs = get_top_k_images(probabilities_per_image, image_names, k)
        display_images_in_grid(path_prob_pairs)


def display_top_p_images(probabilities_per_image, image_names):
    st.markdown("#### Top P")
    on = st.toggle('max probability')
    p = st.number_input('Insert a probability for Top P', min_value=0.01, value=0.9, step=0.01)
    if p:
        images_paths = get_top_p_images(probabilities_per_image, image_names, p, on)
        num_images = len(images_paths)
        st.write(f"There are {num_images} images with probability above {p}")
        slider_range = st.slider("Range to show images", 0, num_images, (0, num_images))
        pairs = images_paths[slider_range[0]:slider_range[1]]
        display_images_in_grid(pairs)


def get_ground_truth(probabilities_per_image, image_names):
    st.markdown("### Sampled Images")
    n = st.number_input('Insert a number for sampling', min_value=1, value=10, step=1)
    if n:
        categories = st.session_state.negative_tags + [st.session_state.input_tags]
        df_truth = create_tagged_dataset(probabilities_per_image[:, 0], image_names, n, categories)

        display_images_in_grid_no_tooltip(df_truth['Image Path'].values)
        return df_truth


if __name__ == "__main__":
    main()
