import os
import pandas as pd
import spacy
from transformers import BartForConditionalGeneration, BartTokenizer, T5ForConditionalGeneration, T5Tokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from evaluate import load  # Hugging Face evaluate library
from nltk.translate.meteor_score import meteor_score  # NLTK METEOR
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Load test dataset
test_data = pd.read_csv(os.getenv('TEST_DATA_PATH'), sep='\t')

# Load your models and their tokenizers for different epochs
MODEL_T5_2EPOCH_PATH = os.getenv('MODEL_T5_2EPOCH_PATH')
MODEL_T5_4EPOCH_PATH = os.getenv('MODEL_T5_4EPOCH_PATH')
MODEL_T5_6EPOCH_PATH = os.getenv('MODEL_T5_6EPOCH_PATH')
MODEL_T5_8EPOCH_PATH = os.getenv('MODEL_T5_8EPOCH_PATH')

t5_2_model = T5ForConditionalGeneration.from_pretrained(MODEL_T5_2EPOCH_PATH)
t5_2_tokenizer = T5Tokenizer.from_pretrained(MODEL_T5_2EPOCH_PATH)

t5_4_model = T5ForConditionalGeneration.from_pretrained(MODEL_T5_4EPOCH_PATH)
t5_4_tokenizer = T5Tokenizer.from_pretrained(MODEL_T5_4EPOCH_PATH)

t5_6_model = T5ForConditionalGeneration.from_pretrained(MODEL_T5_6EPOCH_PATH)
t5_6_tokenizer = T5Tokenizer.from_pretrained(MODEL_T5_6EPOCH_PATH)

t5_8_model = T5ForConditionalGeneration.from_pretrained(MODEL_T5_8EPOCH_PATH)
t5_8_tokenizer = T5Tokenizer.from_pretrained(MODEL_T5_8EPOCH_PATH)

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")  # Make sure you have SpaCy installed and the model downloaded

# Load Rouge scorer using Hugging Face's evaluate
rouge = load("rouge")

# Define a function to generate simplified text
def generate_simplified_text(model, tokenizer, input_text, max_length=50):
    inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=max_length, min_length=5, num_beams=4, length_penalty=2.0, early_stopping=True)
    simplified_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return simplified_text

# Define a function to calculate BLEU score with smoothing
def calculate_bleu_score(reference, hypothesis):
    reference = [reference.split()]  # BLEU expects a list of references (each a list of words)
    hypothesis = hypothesis.split()
    smoothie = SmoothingFunction()  # Initialize smoothing function
    return sentence_bleu(reference, hypothesis, smoothing_function=smoothie.method1)  # Use method1 for smoothing

# Define a function to evaluate models using multiple metrics
def evaluate_models(test_data):
    metrics = {
        '2 Epochs': {'bleu': [], 'rouge': [], 'meteor': []},
        '4 Epochs': {'bleu': [], 'rouge': [], 'meteor': []},
        '6 Epochs': {'bleu': [], 'rouge': [], 'meteor': []},
        '8 Epochs': {'bleu': [], 'rouge': [], 'meteor': []}
    }

    models = [
        ('2 Epochs', t5_2_model, t5_2_tokenizer),
        ('4 Epochs', t5_4_model, t5_4_tokenizer),
        ('6 Epochs', t5_6_model, t5_6_tokenizer),
        ('8 Epochs', t5_8_model, t5_8_tokenizer)
    ]

    for index, row in test_data.iterrows():
        original_text = row['Normal']
        target_text = row['Simple']
        
        # Skip if the original or target text is empty
        if pd.isna(original_text) or pd.isna(target_text):
            continue
        
        for model_name, model, tokenizer in models:
            # Generate simplified text
            simplified_text = generate_simplified_text(model, tokenizer, original_text)
            
            # Calculate BLEU score
            bleu_score = calculate_bleu_score(target_text, simplified_text)
            metrics[model_name]['bleu'].append(bleu_score)
            
            # Calculate ROUGE score using Hugging Face's evaluate
            rouge_score = rouge.compute(predictions=[simplified_text], references=[target_text])['rougeL']
            metrics[model_name]['rouge'].append(rouge_score)
            
            # Calculate METEOR score
            meteor = meteor_score([target_text], simplified_text)
            metrics[model_name]['meteor'].append(meteor)

            # Print sample outputs for comparison
            if index < 5:  # Change this to print more examples if needed
                print(f"Epoch: {model_name}")
                print(f"Original: {original_text}")
                print(f"Simplified: {simplified_text}")
                print(f"Target: {target_text}")
                print(f"BLEU: {bleu_score:.4f}, ROUGE-L: {rouge_score:.4f}, METEOR: {meteor:.4f}\n")
    
    # Calculate average scores for each metric
    for model_name in metrics:
        avg_bleu = sum(metrics[model_name]['bleu']) / len(metrics[model_name]['bleu']) if metrics[model_name]['bleu'] else 0
        avg_rouge = sum(metrics[model_name]['rouge']) / len(metrics[model_name]['rouge']) if metrics[model_name]['rouge'] else 0
        avg_meteor = sum(metrics[model_name]['meteor']) / len(metrics[model_name]['meteor']) if metrics[model_name]['meteor'] else 0
        
        print(f"Average BLEU for {model_name}: {avg_bleu:.4f}")
        print(f"Average ROUGE-L for {model_name}: {avg_rouge:.4f}")
        print(f"Average METEOR for {model_name}: {avg_meteor:.4f}")

# Run evaluation
evaluate_models(test_data)
