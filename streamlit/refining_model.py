##############################env\Scripts\Activate.ps1#######################
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset, DatasetDict
import torch
import os

# Define the base directory and model save path
base_dir = os.path.dirname(os.path.abspath(__file__))
model_save_path = r'C:\dev\the-boring-labels\streamlit\training\finetunedmodel'

# Load the dataset using a relative path
dataset_path = os.path.join(base_dir, 'training', 'training.csv')

# Specify latin1 encoding for the CSV file
csv_file_encoding = 'latin1'

# Load the dataset with the specified encoding
raw_dataset = load_dataset('csv', data_files=dataset_path, encoding=csv_file_encoding)

# Filter out rows with None values in 'Selected Boring Name'
filtered_dataset = raw_dataset.filter(lambda example: example['Selected Boring Name'] is not None)

# Convert string labels to numerical labels
unique_labels = set(label for label in filtered_dataset['train']['Selected Boring Name'] if label is not None)
label_dict = {label: idx for idx, label in enumerate(sorted(unique_labels))}

# Load the tokenizer from the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_save_path)

# Load the pre-trained model with the updated number of labels
model = AutoModelForSequenceClassification.from_pretrained(model_save_path, num_labels=len(unique_labels))

def preprocess_function(examples):
    return tokenizer(examples['Original Room Name'], truncation=True, padding='max_length', max_length=128)

tokenized_dataset = filtered_dataset.map(preprocess_function, batched=True)

# Add labels to the tokenized dataset
def add_labels(examples):
    return {'labels': [label_dict.get(label, -1) for label in examples['Selected Boring Name']]}

tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)
tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Split the dataset
train_test_split = tokenized_dataset['train'].train_test_split(test_size=0.2)
final_dataset = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch"
)

# Initialize the Trainer with a data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=final_dataset['train'],
    eval_dataset=final_dataset['test'],
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save the updated model and tokenizer
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)


























