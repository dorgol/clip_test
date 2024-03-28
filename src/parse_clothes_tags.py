import pandas as pd
import os

# Load the attribute names
attr_names_path = 'aux_data/list_attr_cloth.txt'
attr_names = pd.read_csv(attr_names_path, skiprows=2, names=['attribute_name'])
names = ['item_names'] + attr_names['attribute_name'].tolist()

attr_items_path = 'aux_data/list_attr_items.txt'
attr_items = pd.read_csv(attr_items_path, delim_whitespace=True, names=names)

colors_path = 'aux_data/list_color_cloth.txt'
with open(colors_path, 'r') as file:
    lines = file.readlines()
    colors_data = []
    for line in lines[1:]:
        # Assuming the last word of each line is the color, and the rest is the image name
        parts = line.strip().split()
        image_name = ' '.join(parts[:-1])  # Join everything except the last part
        clothes_color = parts[-1]  # Last part is the color
        colors_data.append([image_name, clothes_color])

colors = pd.DataFrame(colors_data, columns=['image_name', 'clothes_color'])
colors['item_id'] = colors['image_name'].str.extract('(id_[0-9]+)')
colors['Indicator'] = 1
df_wide = colors.pivot_table(index='item_id', columns='clothes_color', values='Indicator', fill_value=-1, aggfunc='max')

df = pd.merge(df_wide, attr_items, left_on='item_id', right_on='item_names', how='outer')
df.set_index('item_names', inplace=True)
df.reset_index(inplace=True)

files = [f"test_images/fashion/{file}" for file in os.listdir("test_images/fashion")]


# Function to find a matching file name
def find_matching_file(item_name, file_list):
    for file in file_list:
        if item_name in file:
            return file
    return None  # Return None or an appropriate value if no match is found


df['file_name'] = df['item_names'].apply(lambda x: find_matching_file(x, files))

df.to_csv('aux_data/list_attr_clothes.csv')
