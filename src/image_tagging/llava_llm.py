import ollama
from src.image_tagging.baseLLm import BaseLLMClient
from src.image_tagging.category_manager import category_manager


class LlavaClient(BaseLLMClient):
    def __init__(self, model, image_paths, context, categories):
        super().__init__(image_paths, context, categories)
        self.model = model
        self.pull_model()

    def pull_model(self):
        models_list = ollama.list()
        models_list = [i['name'] for i in models_list['models']]
        is_exist = self.model in models_list
        if not is_exist:
            ollama.pull(self.model)

    def generate_response(self):
        try:
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
        response = self.extract_and_parse_json(response)
        # print(response)

        return response


if __name__ == "__main__":
    llava_context = category_manager.generate_context()
    image_path1 = "test_images/tmp/woman2.jpeg"
    paths = [image_path1]
    categories = dict(category_manager.load_categories())
    llava = LlavaClient(context=llava_context, image_paths=paths, categories=categories, model='llava:34b-v1.6')
    b = llava.get_responses()
    print(b)
