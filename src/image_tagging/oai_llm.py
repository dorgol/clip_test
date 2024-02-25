import streamlit as st
import requests
from src.image_tagging.baseLLm import BaseLLMClient
from src.image_tagging.category_manager import category_manager

api_key = st.secrets["OPENAI_API_KEY"]


class OpenAIClient(BaseLLMClient):
    def __init__(self, image_paths, api_key, context, categories):
        super().__init__(image_paths, context, categories)
        self.api_key = api_key

    def create_image_dicts(self):
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

    def generate_response(self):
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
