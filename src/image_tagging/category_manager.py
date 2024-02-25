import json


class CategoryManager:
    def __init__(self, json_path):
        self.json_path = json_path
        self.categories = self.load_categories()

    def load_categories(self):
        with open(self.json_path, 'r') as file:
            return json.load(file)

    def save_categories(self):
        with open(self.json_path, 'w') as file:
            json.dump(self.categories, file, indent=4)

    def add_update_category(self, category, options):
        self.categories[category] = options
        self.save_categories()

    def remove_category(self, category):
        if category in self.categories:
            del self.categories[category]
            self.save_categories()
        else:
            print(f"Category '{category}' does not exist.")

    def generate_context(self):
        context_lines = ["Please answer with the following format:",
                         "json with the following answers about the image:"]
        for category, options in self.categories.items():
            options_text = "/".join(options)
            context_lines.append(f"{category}: {options_text}")
        context_lines.append("\nanswer with one sub json for each person in the image.\nHere's an example:")
        context_lines.append(
            "{'person 1': \n{\n" + "\n".join([f'"{category}": {options[0]},'
                                              for category, options in self.categories.items()])[
                    :-1] + "\n}")
        return "\n".join(context_lines)


# Example usage
json_path = 'src/image_tagging/categories.json'
category_manager = CategoryManager(json_path)


if __name__ == '__main__':
    category_manager.generate_context()
