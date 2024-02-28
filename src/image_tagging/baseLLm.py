import base64
import json
import pandas as pd


class BaseLLMClient:
    def __init__(self, image_paths, context, categories=None):
        self.image_paths = image_paths
        self.categories = categories
        self.context = context
        self.base64_images = self.encode_images()

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def encode_images(self):
        image_encoded = []
        for i in self.image_paths:
            encoding = self.encode_image(i)
            image_encoded.append(encoding)
        return image_encoded

    def get_responses(self, max_retries=3):
        """
        Attempts to get a valid response, retrying up to max_retries times.
        Keeps the current answer if retries don't yield a valid response.

        :param max_retries: int, maximum number of retries if the response is not valid.
        :return: the best obtained response, which could be an invalid response if no valid response is obtained after retries.
        """
        response = None  # Initialize response to None
        for attempt in range(1, max_retries + 1):
            try:
                temp_response = self.generate_response()  # Attempt to generate the response.
                temp_response = self.extract_and_parse_json(temp_response)
                # Only update the main response if the new response is valid
                if self.is_valid_response(temp_response):
                    response = temp_response
                    return response  # Return immediately if a valid response is obtained
            except Exception as e:
                print(f'Attempt {attempt}: An error occurred - {e}')
                # Optionally, update response with an error message or partial response
                # response = temp_response or some error indication if needed

            # If reaching this point, the attempt failed
            print(f'Retry {attempt}/{max_retries} failed.')

        if response is None or not self.is_valid_response(response):
            print("Maximum retries reached. No valid response obtained or keeping the last known response.")
        return response

    def generate_response(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    @staticmethod
    def extract_and_parse_json(s):
        """
        Attempts to extract a JSON object from a string and parse it,
        adapting for single quotes.

        :param s: str, the string containing the JSON object.
        :return: The parsed JSON object if successful, or None if parsing fails.
        """
        try:
            # Attempt to normalize single quotes to double quotes
            normalized_str = s.replace("'", '"')

            # Find the indices of the first opening brace and the last closing brace
            start_index = normalized_str.find('{')
            end_index = normalized_str.rfind('}') + 1  # Add 1 to include the closing brace

            if start_index == -1 or end_index == -1:
                print("No JSON object found in the string.")
                return None

            # Extract the substring that potentially contains the JSON object
            json_str = normalized_str[start_index:end_index]

            # Attempt to parse the substring as JSON
            return json.loads(json_str)

        except (ValueError, json.JSONDecodeError) as e:
            print(f"Error parsing JSON: {e}")
            return None

    def is_valid_response(self, response):
        """
        Validates a nested response structure against the defined schema.

        :param response: dict, nested dictionary representing people and their attributes.
        :return: bool, True if the response structure and values match the schema, False otherwise.
        """
        # Check if the response is not a dict and try to convert it
        if not isinstance(response, dict):
            try:
                response = json.loads(response)  # Attempt to convert from JSON string to dict
            except (TypeError, ValueError) as e:
                print(f"Error converting response to dict: {e}")
                return False
        # Check each person in the response
        for person, attributes in response.items():
            # Check each category for the person
            for category, value in attributes.items():
                # Validate the category exists in the schema
                if category not in self.categories:
                    print(f"Invalid category '{category}' for {person}.")
                    return False
                valid_options = set(self.categories[category]) | {'Unknown'}
                # Validate the value is a valid option for the category
                if value not in (list(valid_options)):
                    print(f"Invalid option '{value}' for category '{category}' in {person}.")
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
