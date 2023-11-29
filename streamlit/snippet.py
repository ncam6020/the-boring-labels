from datasets import load_dataset

# Path to your dataset
dataset_path = r"C:\dev\the-boring-labels\streamlit\training\training.csv"

# Load the dataset
raw_dataset = load_dataset('csv', data_files=dataset_path, encoding='utf-8')

# Counting unique labels in the dataset
unique_labels = set(raw_dataset['train']['Selected Boring Name'])
print("Number of unique labels in the dataset:", len(unique_labels))
