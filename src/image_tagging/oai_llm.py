from typing import List
import streamlit as st
import requests
from src.image_tagging.baseLLm import BaseLLMClient
from src.image_tagging.category_manager import category_manager

api_key = st.secrets["OPENAI_API_KEY"]


class OpenAIClient(BaseLLMClient):
    def __init__(self, image_paths: List[str], api_key: str, context: str, categories: dict) -> None:
        """
        Initializes the OpenAIClient with image paths, an API key, context, and categories.

        :param image_paths: List[str], paths to the images to be processed.
        :param api_key: str, the API key for authenticating requests to OpenAI.
        :param context: str, contextual information to guide the generation.
        :param categories: dict, categories with options for validating the response.
        """
        super().__init__(image_paths, context, categories)
        self.api_key = api_key

    def create_image_dicts(self) -> List[dict]:
        """
        Converts base64 encoded images to dictionaries expected by the OpenAI API, marking each as a low-detail image URL.

        :return: List[dict], a list of dictionaries each representing an image for the API request.
        """
        image_dicts = []
        for base64_image in self.base64_images:
            image_dict = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low"
                }
            }
            image_dicts.append(image_dict)
        return image_dicts

    def generate_response(self) -> str:
        """
        Generates a response from OpenAI's API using the provided images and context.

        :return: str, the generated response from OpenAI.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        text_message = {
            "type": "text",
            "text": self.context
        }

        image_messages = self.create_image_dicts()

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


if __name__ == "__main__":
    openai_context = category_manager.generate_context()
    image_path1 = "test_images/images/image_117.jpg"
    paths = [image_path1]
    categories = dict(category_manager.load_categories())
    oai = OpenAIClient(api_key=api_key, context=openai_context, categories=categories, image_paths=[paths[0]])
    a = oai.get_responses(max_retries=3)
