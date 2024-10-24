import os
import pandas as pd
from dotenv import load_dotenv
from transformers import BartTokenizer, BartForConditionalGeneration
import evaluate
from tabulate import tabulate 

# load .env paths
load_dotenv()

# load path for bart and token 
model_path = os.getenv('MODEL_BART') 
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

# limits max by the models arch
max_length = model.config.max_position_embeddings

# load data
test_data_path = os.getenv('TEST_DATA_PATH')
test_data = pd.read_csv(test_data_path, delimiter='\t') 

inputs = test_data['Normal'].tolist()  # normal on the left is used as input
expected_outputs = test_data['Simple'].tolist()  # simple on the right is used as expected output

# load evaluate metrics 
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

# Evaluate model
predictions = []
for input_text in inputs:
    input_ids = tokenizer.encode(input_text, return_tensors='pt', padding=True, truncation=True, max_length=max_length)  # Set max_length for truncation
    output_ids = model.generate(input_ids, max_length=150)  # Generate output using BART model
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)  # Decode generated output
    predictions.append(output_text)

# Calculate evaluation metrics
rouge_result = rouge.compute(predictions=predictions, references=expected_outputs)
bleu_result = bleu.compute(predictions=predictions, references=[[output] for output in expected_outputs])

# BLEU score
bleu_score = bleu_result['bleu']

# Handling ROUGE result based on its type (dictionary or scalar)
rouge1 = rouge_result['rouge1'].get('fmeasure', rouge_result['rouge1']) if isinstance(rouge_result['rouge1'], dict) else rouge_result['rouge1']
rouge2 = rouge_result['rouge2'].get('fmeasure', rouge_result['rouge2']) if isinstance(rouge_result['rouge2'], dict) else rouge_result['rouge2']
rougeL = rouge_result['rougeL'].get('fmeasure', rouge_result['rougeL']) if isinstance(rouge_result['rougeL'], dict) else rouge_result['rougeL']

# Prepare performance metrics for display
performance_metrics = {
    'Metric': ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU'],
    'Score': [
        f"{rouge1:.4f}",  # Access ROUGE-1 score
        f"{rouge2:.4f}",  # Access ROUGE-2 score
        f"{rougeL:.4f}",  # Access ROUGE-L score
        f"{bleu_score:.4f}"  # Only showing the main BLEU score
    ]
}

# Create a DataFrame for the performance metrics
performance_df = pd.DataFrame(performance_metrics)

# Display the performance table using 'tabulate' for a better format
print("\nPerformance Metrics:")
print(tabulate(performance_df, headers='keys', tablefmt='fancy_grid', showindex=False))

# Save the performance metrics to a CSV file
performance_df.to_csv('performance_metrics.csv', index=False)

# Optionally, save predictions and expected outputs for further analysis
output_df = pd.DataFrame({'Input': inputs, 'Expected': expected_outputs, 'Predicted': predictions})
output_df.to_csv('evaluation_results.csv', index=False)