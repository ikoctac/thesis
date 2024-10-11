from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
import torch

print(torch.cuda.is_available())  # This should return True if GPU is available

# loads my tsv(tab seperated values.)
dataset = load_dataset('csv', data_files=r'C:\Users\kostas\Desktop\thesis\converted_svo_dataset.tsv', delimiter='\t')

# uses 90% of my dataset as training data and the rest 10% as testing data.
dataset = dataset['train'].train_test_split(test_size=0.1)

# load the token and model used.
token = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# preprocess the dataset 
def preprocess_function(examples):
    inputs = examples['Normal']
    targets = examples['Simple']
    model_inputs = token(inputs, max_length=512, truncation=True, padding='max_length')
    
    # preparing the data from dataset by converting text to tokens to be efficiently processed by the model.
    with token.as_target_tokenizer():
        labels = token(targets, max_length=512, truncation=True, padding='max_length')

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# use the preprocess function to the dataset and prepare the model for trainning(transforms the example to include the created tokens.)
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# define training arguments for my model(based on my computer power and having an efficient training, make the process faster.) It took≈3.45 hours to train Bart in this dataset
training_args = TrainingArguments(
    output_dir='./results',                  # Directory for saving the model
    num_train_epochs=5,                      # Number of training epochs
    per_device_train_batch_size=16,          # Batch size for training
    per_device_eval_batch_size=16,           # Batch size for evaluation
    logging_dir='./logs',                    # Directory for storing logs
    logging_steps=500,                       # Log every 500 steps (reduced for more frequent logging)
    learning_rate=5e-5,                      # Learning rate
    weight_decay=0.01,                       # Weight decay for regularization
    warmup_steps=1000,                       # Number of warmup steps
    fp16=True,                               # Use mixed precision training if supported
    save_total_limit=1,                      # Limit the total number of saved checkpoints
    evaluation_strategy='steps',              # Evaluate every X steps
    eval_steps=500,                          # Number of steps between evaluations
    save_steps=500,                          # Save the model every X steps
    load_best_model_at_end=True,             # Load the best model when finished training
    metric_for_best_model='loss',             # Specify the metric to use for the best model
    greater_is_better=False,                  # Specify whether a higher or lower metric is better
    report_to='tensorboard',                  # Report results to TensorBoard
    gradient_accumulation_steps=2,           # Accumulate gradients over multiple steps
    fp16_opt_level='O1',                      # Mixed precision level for FP16
)


# initialize the trainer( which model is used, training config, train dataset and test dataset.)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=token,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Stop if no improvement for 2 evaluations
)


# train the model(Bart)
trainer.train()

# save the model 
model.save_pretrained(r"C:\Users\kostas\Desktop\DIplomatic Project\model-bart2")
token.save_pretrained(r"C:\Users\kostas\Desktop\DIplomatic Project\model-bart2")

# evaluate the model based on the dataset.
evaluation_results = trainer.evaluate()
print("Evaluation results:", evaluation_results)

# Evaluation results: {'eval_loss': 0.054918255656957626, 'eval_model_preparation_time': 0.002, 'eval_runtime': 159.2898, 'eval_samples_per_second': 93.446, 'eval_steps_per_second': 18.689}
