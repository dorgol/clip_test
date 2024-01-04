import pandas as pd
import streamlit as st
import torch

from src.comparing import calculate_logits, calculate_probabilities, get_top_k_images, get_top_p_images
from src.encode_image import load_embeddings, load_names
from src.enrich import run_enrichment
from src.image_utils import display_images_in_grid
from src.text_model import get_text_embeddings, get_model_and_processor, MODEL_NAME


def initialize_session_state():
    if 'negative_inputs' not in st.session_state:
        st.session_state.negative_inputs = []
    if 'input' not in st.session_state:
        st.session_state.input = []


def add_negative():
    st.session_state.negative_inputs.append('')


def remove_negative(index):
    del st.session_state.negative_inputs[index]


def main():
    initialize_session_state()

    # Fetch image embeddings
    images_features = torch.tensor(load_embeddings())
    image_names = load_names()

    # Main search term input and enrich button
    search_term_input()

    # Function to add a new negative prompt and manage existing ones
    manage_negative_prompts()

    # Combine search terms and search
    search_and_display_results()


def manage_negative_prompts():
    # Function to add a new negative prompt
    add_negative_prompt()

    # Loop over existing negative prompts
    for i in range(len(st.session_state.negative_inputs)):
        delete_negative_prompt(i)


def search_term_input():
    col1, col2 = st.columns([4, 1])
    with col1:
        # Default value is the current state or a placeholder if empty
        default_value = st.session_state.input[0] if st.session_state.input else 'A photo of a kid'
        search_term = st.text_input('Enter your search', value=default_value)

        # Update only if there's a change
        if st.session_state.input:
            if st.session_state.input[0] != search_term:
                st.session_state.input[0] = search_term
        else:
            st.session_state.input.append(search_term)

    with col2:
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
    col1, col2 = st.columns([4, 1])
    with col1:
        negative_prompt = st.text_input(
            f'Negative example {index + 1}',
            value=st.session_state.negative_inputs[index],
            key=f'negative_{index}')
        st.session_state.negative_inputs[index] = negative_prompt
    with col2:
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


def search_and_display_results():
    search_terms = [st.session_state.input[0]] + st.session_state.negative_inputs + ["some general random stuff"]
    df = pd.DataFrame(search_terms, columns=["Strings"])
    st.table(df)
    images_features = torch.tensor(load_embeddings())
    image_names = load_names()

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


if __name__ == "__main__":
    main()
