from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# loads my tsv(tab seperated values.)
dataset = load_dataset('csv', data_files=r"C:\Users\kostas\Desktop\DIplomatic Project\dataset\Data-huggingface\wiki.full.aner.ori.train.95.tsv", delimiter='\t')

# uses 90% of my dataset as training data and the rest 10% as testing data.
dataset = dataset['train'].train_test_split(test_size=0.1)

# load the token and model used.
token = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# preprocess the dataset 
def preprocess(examples):
    inputs = examples['Normal']
    targets = examples['Simple']
    model_inputs = token(inputs, max_length=512, truncation=True, padding='max_length')
    
    # preparing the data from dataset by converting text to tokens to be efficiently processed by the model.
    with token.as_target_tokenizer():
        labels = token(targets, max_length=512, truncation=True, padding='max_length')

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# use the preprocess function to the dataset and prepare the model for trainning(transforms the example to include the created tokens.)
token_datasets = dataset.map(preprocess, batched=True)

# define training arguments for my model(based on my computer power and having an efficient training, make the process faster.) It took≈3.25 hours to train t5-small in this dataset.
training_args = TrainingArguments(
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    output_dir='./results',
    num_train_epochs=2,
    logging_dir='./logs',
    logging_steps=1000,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=1000,
    fp16=True,  
)
# initialize the trainer( which model is used, training config, train dataset and test dataset.)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=token_datasets['train'],
    eval_dataset=token_datasets['test'],
)

# train the model(t5-small.)
trainer.train()

# save the model 
model.save_pretrained(r"C:\Users\kostas\Desktop\DIplomatic Project\model")
token.save_pretrained(r"C:\Users\kostas\Desktop\DIplomatic Project\model")

# evaluate the model based on the dataset.
evaluation_results = trainer.evaluate()
print("Evaluation results:", evaluation_results)

# Evaluation results: {'eval_loss': 0.07657283544540405, 'eval_model_preparation_time': 0.001, 'eval_runtime': 171.0166, 'eval_samples_per_second': 87.038, 'eval_steps_per_second': 17.408}