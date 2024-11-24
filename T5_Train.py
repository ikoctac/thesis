from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load the dataset (TSV format with tab delimiter).
dataset = load_dataset(
    'csv',  # Use 'csv' as the loader supports custom delimiters.
    data_files=r"C:\Users\kostas\Desktop\thesis\dataset\Data-huggingface\wiki.full.aner.ori.train.95.tsv",
    delimiter='\t'
)

# Split dataset into training and testing sets (90% train, 10% test).
dataset = dataset['train'].train_test_split(test_size=0.1)

# Load tokenizer and model.
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Preprocess the dataset.
def preprocess_function(examples):
    inputs = examples['Normal']
    targets = examples['Simple']
    
    # Tokenize input and target text.
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=512, truncation=True, padding='max_length')

    # Add labels to the model inputs.
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Apply preprocessing to the dataset.
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)

# Define training arguments.
training_args = TrainingArguments(
    per_device_train_batch_size=8,  # Adjust based on your GPU VRAM (e.g., 8GB for RTX 4060).
    per_device_eval_batch_size=8,
    output_dir=r"C:\Users\kostas\Desktop\thesis\model",  # Save model here.
    overwrite_output_dir=True,  # Overwrite if the directory exists.
    num_train_epochs=4,  # Increase if needed.
    logging_dir='./logs',  # Directory for logs.
    logging_steps=100,  # Log every 100 steps.
    save_steps=500,  # Save checkpoint every 500 steps.
    evaluation_strategy="epoch",  # Evaluate after each epoch.
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=1000,
    fp16=True,  # Enable mixed precision (requires compatible GPU and environment).
    save_total_limit=2,  # Keep only the last 2 checkpoints.
)

# Initialize the Trainer.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
)

# Train the model.
trainer.train()

# Save the trained model and tokenizer.
model.save_pretrained(r"C:\Users\kostas\Desktop\thesis\model-T5")
tokenizer.save_pretrained(r"C:\Users\kostas\thesis\model-T5")

# Evaluate the model.
evaluation_results = trainer.evaluate()
print("Evaluation results:", evaluation_results)
