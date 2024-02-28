from typing import List
import ollama
from src.image_tagging.baseLLm import BaseLLMClient
from src.image_tagging.category_manager import category_manager


class LlavaClient(BaseLLMClient):
    def __init__(self, model: str, image_paths: List[str], context: str, categories: dict) -> None:
        """
        Initializes the LlavaClient with the model name, image paths, context for generation, and categories for response validation.

        :param model: str, the name of the model to use with Ollama.
        :param image_paths: List[str], paths to the images to be processed.
        :param context: str, contextual information to be included in the prompt for the model.
        :param categories: dict, a dictionary defining categories and their options for validating responses.
        """
        super().__init__(image_paths, context, categories)
        self.model = model
        self.pull_model()

    def pull_model(self) -> None:
        """
        Checks if the specified model is available in Ollama and pulls it if not. This ensures the model is ready for use when generating responses.
        """
        models_list = ollama.list()
        models_list = [i['name'] for i in models_list['models']]
        is_exist = self.model in models_list
        if not is_exist:
            ollama.pull(self.model)

    def generate_response(self) -> dict:
        """
        Generates a response for the given images and context using the specified Ollama model. Handles potential errors by attempting to pull the model again if not found.

        :return: dict, the response from Ollama for the images and context provided.
        """
        try:
            # Initial call to generate to handle specific error handling, can be removed if not necessary
            ollama.generate(self.model)
        except ollama.ResponseError as e:
            print('Error:', e.error)
            if e.status_code == 404:
                ollama.pull(self.model)

        stream = ollama.generate(
            model=self.model,
            prompt=self.context,
            images=self.base64_images,
            stream=False,
        )
        response = stream['response']

        return response


if __name__ == "__main__":
    llava_context = category_manager.generate_context()
    image_path1 = "test_images/tmp/woman2.jpeg"
    paths = [image_path1]
    categories = dict(category_manager.load_categories())
    llava = LlavaClient(context=llava_context, image_paths=paths, categories=categories, model='llava:34b-v1.6')
    b = llava.get_responses()
    print(b)
