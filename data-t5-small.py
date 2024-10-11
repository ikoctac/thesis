from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset
import torch
import os

torch.cuda.is_available()

# Load your dataset
dataset = load_dataset('csv', data_files=r"C:\Users\kostas\Desktop\DIplomatic Project\dataset\Data-huggingface\wiki.full.aner.ori.train.95.tsv", delimiter='\t')
dataset = dataset['train'].train_test_split(test_size=0.1)

# Load the tokenizer and model
model_path = r"C:\Users\kostas\Desktop\DIplomatic Project\t5\model-t5"
token = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Preprocess the dataset
def preprocess(examples):
    inputs = examples['Normal']
    targets = examples['Simple']
    model_inputs = token(inputs, max_length=512, truncation=True, padding='max_length')
    
    with token.as_target_tokenizer():
        labels = token(targets, max_length=512, truncation=True, padding='max_length')

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

token_datasets = dataset.map(preprocess, batched=True)

# Define a custom callback for saving the model every 2 epochs
class SaveEveryNEpochsCallback(TrainerCallback):
    def __init__(self, save_dir, every_n_epochs=2):
        self.save_dir = save_dir
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch % self.every_n_epochs == 0:
            save_path = os.path.join(self.save_dir, f"model_epoch_{int(state.epoch)}")
            kwargs['model'].save_pretrained(save_path)
            print(f"Model saved to {save_path}")

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    output_dir='./results',
    num_train_epochs=6,  # Set total epochs
    logging_dir='./logs',
    logging_steps=1000,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=1000,
    fp16=True,  
)

# Initialize the trainer with the custom callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=token_datasets['train'],
    eval_dataset=token_datasets['test'],
    callbacks=[SaveEveryNEpochsCallback(r"C:\Users\kostas\Desktop\DIplomatic Project\t5", every_n_epochs=2)]  # Save every 2 epochs in T5 folder
)


# Train the model
trainer.train()

# Save the final model
model.save_pretrained(r"C:\Users\kostas\Desktop\DIplomatic Project\t5")
token.save_pretrained(r"C:\Users\kostas\Desktop\DIplomatic Project\t5")

# Evaluate the model
evaluation_results = trainer.evaluate()
print("Evaluation results:", evaluation_results)
