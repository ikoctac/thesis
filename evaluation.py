import os
import pandas as pd
import spacy
from transformers import BartForConditionalGeneration, BartTokenizer, T5ForConditionalGeneration, T5Tokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Load test dataset
test_data = pd.read_csv(os.getenv('TEST_DATA_PATH'), sep='\t')

# Paths for different epoch models
MODEL_T5_2EPOCH_PATH = os.getenv('MODEL_T5_2EPOCH_PATH', 'path_to_t5_2epoch_model')
MODEL_T5_4EPOCH_PATH = os.getenv('MODEL_T5_4EPOCH_PATH', 'path_to_t5_4epoch_model')
MODEL_T5_6EPOCH_PATH = os.getenv('MODEL_T5_6EPOCH_PATH', 'path_to_t5_6epoch_model')
MODEL_T5_8EPOCH_PATH = os.getenv('MODEL_T5_8EPOCH_PATH', 'path_to_t5_8epoch_model')

# Load the four models and their tokenizers
def load_model_and_tokenizer(model_path):
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    return model, tokenizer

t5_2epoch_model, t5_2epoch_tokenizer = load_model_and_tokenizer(MODEL_T5_2EPOCH_PATH)
t5_4epoch_model, t5_4epoch_tokenizer = load_model_and_tokenizer(MODEL_T5_4EPOCH_PATH)
t5_6epoch_model, t5_6epoch_tokenizer = load_model_and_tokenizer(MODEL_T5_6EPOCH_PATH)
t5_8epoch_model, t5_8epoch_tokenizer = load_model_and_tokenizer(MODEL_T5_8EPOCH_PATH)

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

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

# Evaluate models on the test dataset
def evaluate_model_on_dataset(model, tokenizer, test_data):
    scores = []
    spacy_scores = []
    
    for index, row in test_data.iterrows():
        original_text = row['Normal']
        target_text = row['Simple']
        
        # Skip if the original or target text is empty
        if pd.isna(original_text) or pd.isna(target_text):
            continue
        
        # Generate simplified text
        simplified = generate_simplified_text(model, tokenizer, original_text)

        # Calculate BLEU score
        bleu_score = calculate_bleu_score(target_text, simplified)
        scores.append(bleu_score)
        
        # Process simplified text with SpaCy
        spacy_doc = nlp(simplified)
        spacy_simplified = " ".join([token.text for token in spacy_doc])

        # Calculate BLEU score for SpaCy-processed sentences
        spacy_bleu_score = calculate_bleu_score(target_text, spacy_simplified)
        spacy_scores.append(spacy_bleu_score)

    avg_bleu = sum(scores) / len(scores) if scores else 0
    avg_spacy_bleu = sum(spacy_scores) / len(spacy_scores) if spacy_scores else 0
    return avg_bleu, avg_spacy_bleu

# Compare models
def compare_models():
    models = [
        ('T5 2 Epochs', t5_2epoch_model, t5_2epoch_tokenizer),
        ('T5 4 Epochs', t5_4epoch_model, t5_4epoch_tokenizer),
        ('T5 6 Epochs', t5_6epoch_model, t5_6epoch_tokenizer),
        ('T5 8 Epochs', t5_8epoch_model, t5_8epoch_tokenizer)
    ]
    
    results = []
    
    for name, model, tokenizer in models:
        print(f"Evaluating {name}...")
        avg_bleu, avg_spacy_bleu = evaluate_model_on_dataset(model, tokenizer, test_data)
        print(f"{name} BLEU: {avg_bleu:.4f} | SpaCy BLEU: {avg_spacy_bleu:.4f}\n")
        results.append((name, avg_bleu, avg_spacy_bleu))

    # Plotting the results
    model_names = [result[0] for result in results]
    bleu_scores = [result[1] for result in results]
    spacy_bleu_scores = [result[2] for result in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(model_names, bleu_scores, marker='o', label='BLEU')
    plt.plot(model_names, spacy_bleu_scores, marker='x', label='SpaCy BLEU')
    plt.title("Model Performance Across Epochs")
    plt.xlabel("Model")
    plt.ylabel("BLEU Score")
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the comparison
compare_models()
