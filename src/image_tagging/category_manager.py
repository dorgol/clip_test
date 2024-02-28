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
        context_lines = ["Focus only on the main people in the image",
                         "Please answer with the following format:",
                         "json with the following answers about the image:"]
        for category, options in self.categories.items():
            options_text = "/".join(options)
            context_lines.append(f"{category}: {options_text}")
        context_lines.append("\nanswer with one sub json for each person in the image.\nHere's an example:")
        context_lines.append(
            "{'person 1': \n{\n" + "\n".join([f'"{category}": f'"{options[0]},"''
                                              for category, options in self.categories.items()])[
                                   :-1] + "\n}")
        context_lines.append("\nStick to the options exactly as stated.")
        context_lines.append("\nIf you don't know the answer or it can be determined from the image return 'Unknown'.")
        return "\n".join(context_lines)


json_path = 'src/image_tagging/categories.json'
category_manager = CategoryManager(json_path)

if __name__ == '__main__':
    # Example usage
    # category_manager.add_update_category(category='hair style', options=['pixie cut', 'buzz cut', 'bob cut',
    #                                                                      'shag cut', 'layered cut', 'curtain bangs',
    #                                                                      'straight cut', 'feathered cut', ''])
    category_manager.add_update_category(category='hair type', options=['straight hair', 'wavy hair', 'curly hair',
                                                                        'kinky hair'])
    category_manager.generate_context()
