import ast

import pandas as pd
import requests
import streamlit as st

api_key = st.secrets["OPENAI_API_KEY"]


def get_super_category(attribute_list, super_category):
    # Convert lists to string representations to include in the f-string
    attribute_list_str = ", ".join(attribute_list)
    super_category_str = ", ".join(super_category)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    text_message = {
        "type": "text",
        "text": f"""Given a comprehensive list of attributes related to images, including clothing colors, clothing 
        attributes, facial features, lighting conditions, and other relevant details, create a simplified mapping 
        that categorizes these detailed attributes into broader, predefined categories. Specifically, 
        for each category I provide (e.g., 'Yellow', 'Bright Lighting', 'Casual Wear'), associate all relevant terms 
        from the long list that fit within these categories (such as 'mustard', 'sunflower', 'beige' for 'Yellow', 
        etc.).
        
        The output should be formatted as a dictionary in Python, where each key represents one of my predefined 
        categories, and the associated value is a list of terms from the long list that belong to that category. For 
        example, if the categories are based on colors, clothing types, and lighting conditions, the structure should 
        be as follows:
        
        {{'Yellow': ['yellow', 'mustard', 'lemon', 'gold'], 'Bright Lighting': ['sunlit', 'brightly lit', 'glare'], 
        'Casual Wear': ['t-shirt', 'jeans', 'sneakers', 'hoodie']}}
        
        This is just an example. Please don't stick to this. Only use categories from the given list.
        Please ensure that all terms are accurately 
        grouped under the correct categories, and if any terms could belong to multiple categories, include them in 
        each relevant category. Also, if there are terms that do not fit into any of the predefined categories, 
        please drop them. Please don't add anything but the json itself.
        
        Long List of Attributes:
        {attribute_list_str}
        
        Predefined Categories:
        {super_category_str}
        """

    }

    payload = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": [text_message]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    result = response.json()['choices'][0]['message']['content']
    return result


def extract_json_from_string(string_data):
    """
    Extracts a JSON object from a string representation of a Python dictionary.

    :param string_data: str, the string representation of a Python dictionary.
    :return: str, a JSON-formatted string.
    """

    try:
        # Safely evaluate the string as a Python dictionary
        dict_data = ast.literal_eval(string_data)

        return dict_data
    except (ValueError, SyntaxError) as e:
        print(f"Error converting string to dict: {e}")
        return None


def validate_categorization(results, expected_categories, expected_attributes):
    """
    Validates the categorization of attributes against expected categories.

    :param results: dict, the categorization output to validate.
                    Format: {'Category': ['attribute1', 'attribute2', ...]}
    :param expected_categories: list, the expected categories.
                                Format: ['category1', 'category1', ...]}
    :param expected_attributes: list, the expected attributes.
                                Format: ['attribute1', 'attribute2', ...]}
    :return: bool, True if the categorization is correct, False otherwise.
    """
    print(results)
    # Check if all categories in results exist in expected categories
    for category in results:
        if category not in expected_categories:
            print(f"Unexpected category: {category}")
            return False

    all_attributes = [value for sublist in results.values() for value in sublist]

    # Check if all attributes in each category match the expected ones
    for attribute in all_attributes:
        if attribute not in expected_attributes:
            print(f"Unexpected attribute '{attribute}'")
            return False

    # Optionally, check if there are categories or attributes in expected categories
    # that are missing in the results
    for category in expected_categories:
        if category not in results:
            print(f"Missing category: {category}")
            return False

    print("Validation successful: All categories and attributes match expected values.")
    return True


def create_super_df(original_df, mapping_dict):
    df2 = pd.DataFrame(index=original_df.index, columns=mapping_dict.keys())

    # Populate df2 based on the maximum values from df as per the mapping_dict
    for new_col, old_cols in mapping_dict.items():
        df2[new_col] = original_df[old_cols].max(axis=1)
    return df2


def super_process(df, categories):
    attribute_list = df.columns
    mapping_dict = get_super_category(attribute_list, categories)
    mapping_dict = extract_json_from_string(mapping_dict)
    is_valid = validate_categorization(mapping_dict, categories, attribute_list)
    if is_valid:
        new_df = create_super_df(df, mapping_dict)
        return new_df
    else:
        print("Dictionary schema is not valid")


def process_in_batches(df, categories, batch_size=30):
    """
    Process attributes in batches and unify the results into a single mapping dictionary.

    :param df: DataFrame, the source DataFrame with attributes.
    :param categories: list, predefined categories to classify attributes into.
    :param batch_size: int, the number of attributes to process in each batch.
    :return: DataFrame, the new DataFrame based on unified categorization.
    """
    attribute_list = df.columns
    total_attributes = len(attribute_list)
    unified_mapping_dict = {}

    for start in range(0, total_attributes, batch_size):
        end = min(start + batch_size, total_attributes)
        batch_attributes = attribute_list[start:end]
        batch_mapping_dict = get_super_category(batch_attributes, categories)
        batch_mapping_dict = extract_json_from_string(batch_mapping_dict)

        # Validate and unify the batch result
        is_valid = validate_categorization(batch_mapping_dict, categories, batch_attributes)
        if is_valid:
            # Merge batch_mapping_dict into unified_mapping_dict
            for key, values in batch_mapping_dict.items():
                if key in unified_mapping_dict:
                    unified_mapping_dict[key].extend(values)
                else:
                    unified_mapping_dict[key] = values
        else:
            print(f"Batch {start} to {end} processing failed or is invalid.")

    # Remove duplicates in values list
    for key in unified_mapping_dict:
        unified_mapping_dict[key] = list(set(unified_mapping_dict[key]))

    # Now use the unified mapping to create a new DataFrame
    new_df = create_super_df(df, unified_mapping_dict)
    return new_df


if __name__ == '__main__':
    df = pd.read_csv('aux_data/list_attr_clothes.csv')
    categories = ['Yellow', 'Red']
    a = process_in_batches(df, categories, batch_size=10)
    print(a)
