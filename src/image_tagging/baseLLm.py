import base64
import pandas as pd
import requests
import streamlit as st

api_key = st.secrets["OPENAI_API_KEY"]
openai_context = """I will define some categories and you need to return the real category only. "
                 "Don't add any other word, since I want the reply to have well defined format "
                 "Answer for each image separately. Use commas between answers."
                 "If none is true or the question seem unreasonable given the image return general."
                 "categories: 
                 """

categories = "blue eyes, green eyes, brown eyes"
image_path1 = "test_images/images/image_117.jpg"
image_path2 = "test_images/images/image_111.jpg"
image_path3 = "test_images/images/image_544.jpg"
paths = [image_path1, image_path2, image_path3]


class BaseLLMClient:
    def __init__(self, image_paths, context, categories):
        self.image_paths = image_paths
        self.categories = categories
        self.context = context

    def get_responses(self, base64_images, categories):
        raise NotImplementedError("This method should be implemented by subclasses.")

    @staticmethod
    def validate_answer_format(answer, categories, num_images):
        if not isinstance(answer, str):
            return False
        responses = [response.strip() for response in answer.split(',')]
        if len(responses) != num_images:
            return False
        for response in responses:
            if response not in categories and response.lower() != 'general':
                return False
        return True

    @staticmethod
    def create_dataframe(image_paths, results):
        results = [result.strip() for result in results.split(',')]
        if len(image_paths) != len(results):
            raise ValueError("The number of image paths and results must be the same")
        return pd.DataFrame({
            'Image Path': image_paths,
            'Result': results
        })


class OpenAIClient(BaseLLMClient):
    def __init__(self, image_paths, api_key, context, categories):
        super().__init__(image_paths, context, categories)
        self.api_key = api_key
        self.base64_images = self.encode_images()

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def encode_images(self):
        image_encoded = []
        for i in self.images_paths:
            encoding = self.encode_image(i)
            image_encoded.append(encoding)
        return image_encoded

    def create_image_dicts(self, base64_images):
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

    def get_responses(self, categories):

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        text_message = {
            "type": "text",
            "text": self.context + self.categories

        }

        image_messages = self.create_image_dicts(self.base64_images)

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


oai = OpenAIClient(api_key=api_key, context=openai_context, categories=categories, image_paths=paths)


class DatasetCreator:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def create_ground_truth(self, paths, categories):
        base64_images = self.llm_client.encode_images(paths)
        result = self.llm_client.get_responses(base64_images, categories)
        is_valid = BaseLLMClient.validate_answer_format(answer=result, categories=categories, num_images=len(paths))
        if is_valid:
            df = BaseLLMClient.create_dataframe(paths, result)
            return df
        else:
            print("Response is not valid")
            return None

    def create_tagged_dataset(self, probs, paths, n, categories):
        # Similar to your original `create_tagged_dataset` function but using `self.llm_client`
        pass
