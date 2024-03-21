import pandas as pd

# Assuming the file path is 'aux_data/list_attr_celeba.txt'
file_path = 'aux_data/list_attr_celeba.txt'

# Loading the data into a DataFrame
# You might need to adjust parameters based on the exact file structure
df = pd.read_csv(file_path, delim_whitespace=True, header=1)
df.to_csv('aux_data/list_attr_celeba.csv')
