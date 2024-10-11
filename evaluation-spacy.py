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

# Load your models and their tokenizers
MODEL_BART3_PATH = os.getenv('MODEL_BART3_PATH', 'path_to_bart3_model')
MODEL_T5_PATH = os.getenv('MODEL_T5_PATH', 'path_to_t5_model')

bart3_model = BartForConditionalGeneration.from_pretrained(MODEL_BART3_PATH)
bart3_tokenizer = BartTokenizer.from_pretrained(MODEL_BART3_PATH)

t5_model = T5ForConditionalGeneration.from_pretrained(MODEL_T5_PATH)
t5_tokenizer = T5Tokenizer.from_pretrained(MODEL_T5_PATH)

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")  # Make sure you have SpaCy installed and the model downloaded

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
def evaluate_models(test_data):
    bart3_scores = []
    t5_scores = []
    bart3_spacy_scores = []
    t5_spacy_scores = []
    
    for index, row in test_data.iterrows():
        original_text = row['Normal']
        target_text = row['Simple']
        
        # Skip if the original or target text is empty
        if pd.isna(original_text) or pd.isna(target_text):
            continue
        
        # Generate simplified text using BART3 and T5 models
        bart3_simplified = generate_simplified_text(bart3_model, bart3_tokenizer, original_text)
        t5_simplified = generate_simplified_text(t5_model, t5_tokenizer, original_text)

        # Calculate BLEU scores
        bart3_bleu = calculate_bleu_score(target_text, bart3_simplified)
        t5_bleu = calculate_bleu_score(target_text, t5_simplified)
        
        bart3_scores.append(bart3_bleu)
        t5_scores.append(t5_bleu)
        
        # Process simplified text with SpaCy
        bart3_spacy_doc = nlp(bart3_simplified)
        t5_spacy_doc = nlp(t5_simplified)
        
        # Extract processed text (you can customize this step as needed)
        bart3_spacy_simplified = " ".join([token.text for token in bart3_spacy_doc])
        t5_spacy_simplified = " ".join([token.text for token in t5_spacy_doc])

        # Calculate BLEU scores for SpaCy processed sentences
        bart3_spacy_bleu = calculate_bleu_score(target_text, bart3_spacy_simplified)
        t5_spacy_bleu = calculate_bleu_score(target_text, t5_spacy_simplified)
        
        bart3_spacy_scores.append(bart3_spacy_bleu)
        t5_spacy_scores.append(t5_spacy_bleu)

        # Print a sample comparison
        if index < 10:  # Change this to print more examples if needed
            print(f"Original Text: {original_text}")
            print(f"Target Simplified: {target_text}")
            print(f"BART3 Simplified: {bart3_simplified} | BLEU: {bart3_bleu:.4f}")
            print(f"T5 Simplified: {t5_simplified} | BLEU: {t5_bleu:.4f}")
            print(f"BART3 SpaCy Simplified: {bart3_spacy_simplified} | BLEU: {bart3_spacy_bleu:.4f}")
            print(f"T5 SpaCy Simplified: {t5_spacy_simplified} | BLEU: {t5_spacy_bleu:.4f}\n")
    
    # Calculate average BLEU score for all models
    bart3_avg_bleu = sum(bart3_scores) / len(bart3_scores) if bart3_scores else 0
    t5_avg_bleu = sum(t5_scores) / len(t5_scores) if t5_scores else 0
    bart3_spacy_avg_bleu = sum(bart3_spacy_scores) / len(bart3_spacy_scores) if bart3_spacy_scores else 0
    t5_spacy_avg_bleu = sum(t5_spacy_scores) / len(t5_spacy_scores) if t5_spacy_scores else 0
    
    print(f"Average BLEU Score for BART3: {bart3_avg_bleu:.4f}")
    print(f"Average BLEU Score for T5: {t5_avg_bleu:.4f}")
    print(f"Average BLEU Score for BART3 with SpaCy: {bart3_spacy_avg_bleu:.4f}")
    print(f"Average BLEU Score for T5 with SpaCy: {t5_spacy_avg_bleu:.4f}")

# Run evaluation
evaluate_models(test_data)
