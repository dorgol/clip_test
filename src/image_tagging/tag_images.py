import base64
import requests
import streamlit as st
import pandas as pd

from src.comparing import get_sampled_images

api_key = st.secrets["OPENAI_API_KEY"]


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def encode_images(images_paths):
    image_encoded = []
    for i in images_paths:
        encoding = encode_image(i)
        image_encoded.append(encoding)
    return image_encoded


def create_image_dicts(base64_images):
    image_dicts = []
    for base64_image in base64_images:
        image_dict = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "low"
            }
        }
        image_dicts.append(image_dict)
    return image_dicts


def get_responses(base64_images, categories):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    text_message = {
        "type": "text",
        "text": f"I will define some categories and you need to return the real category only. "
                f"Don't add any other word, since I want the reply to have well defined format "
                f"Answer for each image separately. Use commas between answers."
                f"If none is true or the question seem unreasonable given the image return general."
                f"categories: {categories}"

    }

    image_messages = create_image_dicts(base64_images)

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [text_message] + image_messages
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    result = response.json()['choices'][0]['message']['content']
    return result


def validate_answer_format(answer, categories, num_images):
    # Check if the answer is a string
    if not isinstance(answer, str):
        return False

    # Split the answer by commas and strip whitespace
    responses = [response.strip() for response in answer.split(',')]

    # Check if the number of responses matches the number of images
    if len(responses) != num_images:
        return False

    # Validate each response
    for response in responses:
        if response not in categories and response.lower() != 'general':
            return False

    return True


def create_dataframe(image_paths, results):
    results = [result.strip() for result in results.split(',')]
    # Check if the lengths of image_paths and results are equal
    if len(image_paths) != len(results):
        raise ValueError("The number of image paths and results must be the same")
    # image_paths = [path.decode('utf-8') if isinstance(path, bytes) else path for path in image_paths]

    # Create a DataFrame with two columns
    df = pd.DataFrame({
        'Image Path': image_paths,
        'Result': results
    })

    return df


def create_ground_truth(paths, categories):
    base64_images = encode_images(paths)
    result = get_responses(base64_images, categories)
    is_valid = validate_answer_format(answer=result, categories=categories, num_images=len(paths))
    if is_valid:
        paths = [str(path) for path in paths]
        paths = [path.replace('b', '', 1) for path in paths]
        paths = [path.replace("'", "", 2) for path in paths]
        df = create_dataframe(paths, result)
        return df
    else:
        st.write("response is not valid")


def create_tagged_dataset(probs, paths, n, categories):
    """
    Create a dataset by sampling images at different percentiles and tagging them with ground truth.

    :param probs: List of probabilities associated with each path.
    :param paths: List of image paths.
    :param n: Number of images to sample at each percentile.
    :param categories: The categories to be used for tagging.
    :return: A concatenated DataFrame of tagged data for all percentiles.
    """
    # Define percentiles
    percentiles = [90, 75]
    data_frames = []

    for percentile in percentiles:
        # Sample images at the current percentile
        sampled = get_sampled_images(probs, paths, n, percentile)

        # Extract paths from the sampled images
        sampled_paths = [path for _, path in sampled]

        # Create ground truth data for the sampled paths
        data = create_ground_truth(sampled_paths, categories)

        # Append the data to the list
        data_frames.append(data)

    # Concatenate all data frames
    combined_data = pd.concat(data_frames, axis=0)
    return combined_data


if __name__ == '__main__':
    # Path to your image
    categories = ["blue eyes", "green eyes", "brown eyes"]
    image_path1 = "test_images/images/image_117.jpg"
    image_path2 = "test_images/images/image_111.jpg"
    image_path3 = "test_images/images/image_544.jpg"
    paths = [image_path1, image_path2, image_path3]
    base64_images = encode_images(paths)
    result = get_responses(base64_images, categories)
    print(result)
    is_valid = validate_answer_format(answer=result, categories=categories, num_images=len(paths))
    print(f"Is the answer format valid? {is_valid}")
    if is_valid:
        df = create_dataframe(paths, result)
        print(df)
