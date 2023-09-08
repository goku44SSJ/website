import json
import tempfile
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load your JSON dataset
with open("E:/JURIMIND/v0.0.1/ipc.json", 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Initialize the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Add a special token for padding
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Filter examples with "train" set to true
filtered_data = [example for example in data if example.get("train") == True]

# Format your data
formatted_data = ""
for item in filtered_data:
    formatted_data += f"Chapter {item['chapter']}: {item['chapter_title']}\n"
    formatted_data += f"Section {item['Section']}: {item['section_title']}\n"
    formatted_data += f"Description: {item['section_desc']}\n\n"

# Tokenize the formatted data
inputs = tokenizer(formatted_data, return_tensors="pt", max_length=512, truncation=True, padding=True)

# Create a temporary file to hold the text data
with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as temp_file:
    temp_file.write(formatted_data)

# Create a PyTorch dataset
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=temp_file.name,  # Use the temporary file path
    block_size=128  # Adjust the block size as needed
)

# Create a data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",  # Directory to save the fine-tuned model
    overwrite_output_dir=True,
    num_train_epochs=1000,  # Set the number of training epochs to 1000
    per_device_train_batch_size=32,  # Batch size of 32
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=10_000,
    logging_dir="./logs",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model locally
model.save_pretrained("./fine_tuned_model")

# Save the tokenizer locally
tokenizer.save_pretrained("./fine_tuned_model")
