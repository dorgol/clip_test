from typing import Type, List, Any, Dict
import csv
from src.image_tagging.category_manager import category_manager
from src.image_tagging.llava_llm import LlavaClient
from src.image_tagging.oai_llm import OpenAIClient, api_key
import json
import os
import datetime


class ClientFactory:
    def __init__(self, client_class: Type[Any], image_paths: List[str], model_name: str, **kwargs) -> None:
        """
        Initializes the ClientFactory with a client class, a list of image paths, the model name, and any additional keyword arguments to be passed to the client class.

        :param client_class: Type[Any], the class of the client to be used for generating responses.
        :param image_paths: List[str], a list containing the paths to the images to be processed.
        :param model_name: str, a string indicating the name of the model to be used in the responses.
        :param kwargs: Additional keyword arguments to be passed to the client class.
        """
        self.client_class = client_class
        self.kwargs = kwargs
        self.model_name = model_name
        self.responses = []  # To store responses for each image
        self.image_paths = image_paths

    def create_clients_and_get_responses(self) -> None:
        """
        Creates client instances for each image path using the specified client class and collects their responses.
        """
        for path in self.image_paths:
            client_instance = self.client_class(image_paths=[path], **self.kwargs)
            response = client_instance.get_responses()
            print(response)
            self.responses.append(response)

    def write_responses_to_csv(self, csv_file='responses.csv'):
        """
        Writes the stored responses to a CSV file, ensuring the header is written only if the file is new or empty.

        :param csv_file: The path to the CSV file to write.
        """
        # Check if the file exists and has content by checking its size
        file_exists = os.path.isfile(csv_file) and os.path.getsize(csv_file) > 0

        with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # Write the header only if the file didn't exist or was empty
            if not file_exists:
                writer.writerow(['Image Path', 'Model', 'Person', 'Category', 'Selected Value',
                                 'Timestamp'])  # Add 'Timestamp' column

            responses = [item for item in self.responses if item is not None]
            for path, response in zip(self.image_paths, responses):
                if not isinstance(response, dict):
                    response = json.loads(response)
                for person, attributes in response.items():
                    person_num = person.split()[-1]
                    for category, value in attributes.items():
                        # Get current timestamp
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        writer.writerow([path, self.model_name, person_num, category, value,
                                         timestamp])  # Include timestamp in each row

    def generate_and_save(self) -> None:
        """
        Generates responses for all provided image paths and saves them to a CSV file. This method combines the process of creating clients, getting responses, and writing these responses to a CSV file.
        """
        self.create_clients_and_get_responses()
        self.write_responses_to_csv()


if __name__ == '__main__':
    context = category_manager.generate_context()

    file_names = os.listdir('test_images/tmp/')
    paths = ['test_images/tmp/' + i for i in file_names[2:4]]
    categories = dict(category_manager.load_categories())

    # For OpenAIClient
    # factory_openai = ClientFactory(
    #     client_class=OpenAIClient,
    #     api_key=api_key,
    #     context=context,
    #     categories=categories,
    #     model_name='gpt4-v',
    #     image_paths=paths
    # )
    # factory_openai.generate_and_save()

    # For LlavaClient
    factory_llava = ClientFactory(
        LlavaClient,
        context=context,
        categories=categories,
        model='llava:34b-v1.6',
        model_name='llava_34',
        image_paths=paths
    )
    factory_llava.generate_and_save()
