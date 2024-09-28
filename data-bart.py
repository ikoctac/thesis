from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# loads my tsv(tab seperated values.)
dataset = load_dataset('csv', data_files=r'C:\Users\kostas\Desktop\DIplomatic Project\dataset\Data-huggingface\wiki.full.aner.ori.train.95.tsv', delimiter='\t')

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
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    output_dir='./results',
    num_train_epochs=2,
    logging_dir='./logs',
    logging_steps=50000,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=1000,
    fp16=True,  
)

# initialize the trainer( which model is used, training config, train dataset and test dataset.)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=token
)

# train the model(Bart)
trainer.train()

# save the model 
model.save_pretrained(r"C:\Users\kostas\Desktop\DIplomatic Project\model-bart")
token.save_pretrained(r"C:\Users\kostas\Desktop\DIplomatic Project\model-bart")

# evaluate the model based on the dataset.
evaluation_results = trainer.evaluate()
print("Evaluation results:", evaluation_results)

# Evaluation results: {'eval_loss': 0.054918255656957626, 'eval_model_preparation_time': 0.002, 'eval_runtime': 159.2898, 'eval_samples_per_second': 93.446, 'eval_steps_per_second': 18.689}
