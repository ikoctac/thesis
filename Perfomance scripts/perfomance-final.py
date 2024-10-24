import os
import pandas as pd
from dotenv import load_dotenv
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
import evaluate

# Load environment values
load_dotenv()

# Load T5 model and tokenizer
t5_model_path = os.getenv('MODEL_T5')  # Update to your T5 model path
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_path)

# Load BART model and tokenizer
bart_model_path = os.getenv('MODEL_BART')  # Update to your BART model path
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_path)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_path)

# Load dataset
test_data_path = os.getenv('TEST_DATA_PATH')
test_data = pd.read_csv(test_data_path, delimiter='\t') 

inputs = test_data['Normal'].tolist()  # Use 'Normal' as input text
expected_outputs = test_data['Simple'].tolist()  # 'Simple' is used as expected output

# Initialize lists for predictions and metrics
t5_predictions = []
bart_predictions = []
t5_metrics = []
bart_metrics = []

# Initialize metric calculators
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

# Evaluate both models
for input_text in inputs:
    # T5 Prediction
    t5_input_ids = t5_tokenizer.encode(input_text, return_tensors='pt', padding=True, truncation=True)
    t5_output_ids = t5_model.generate(t5_input_ids, max_length=150)
    t5_output_text = t5_tokenizer.decode(t5_output_ids[0], skip_special_tokens=True)
    t5_predictions.append(t5_output_text)

    # BART Prediction
    bart_input_ids = bart_tokenizer.encode(input_text, return_tensors='pt', padding=True, truncation=True)
    bart_output_ids = bart_model.generate(bart_input_ids, max_length=150)
    bart_output_text = bart_tokenizer.decode(bart_output_ids[0], skip_special_tokens=True)
    bart_predictions.append(bart_output_text)

# Calculate metrics for each prediction and store them
for i in range(len(inputs)):
    # Calculate ROUGE and BLEU for T5
    t5_rouge_result = rouge.compute(predictions=[t5_predictions[i]], references=[expected_outputs[i]])
    t5_bleu_result = bleu.compute(predictions=[t5_predictions[i]], references=[[expected_outputs[i]]])
    
    # Calculate ROUGE and BLEU for BART
    bart_rouge_result = rouge.compute(predictions=[bart_predictions[i]], references=[expected_outputs[i]])
    bart_bleu_result = bleu.compute(predictions=[bart_predictions[i]], references=[[expected_outputs[i]]])

    # Store metrics in a tuple (ROUGE-1, ROUGE-2, ROUGE-L, BLEU)
    t5_metrics.append((t5_rouge_result['rouge1'], t5_rouge_result['rouge2'], 
                       t5_rouge_result['rougeL'], t5_bleu_result['bleu']))
    
    bart_metrics.append((bart_rouge_result['rouge1'], bart_rouge_result['rouge2'],
                         bart_rouge_result['rougeL'], bart_bleu_result['bleu']))

# Convert metrics to DataFrame for easier processing
metrics_df = pd.DataFrame({
    'Model': ['T5'] * len(inputs) + ['BART'] * len(inputs),
    'ROUGE-1': [m[0] for m in t5_metrics] + [m[0] for m in bart_metrics],
    'ROUGE-2': [m[1] for m in t5_metrics] + [m[1] for m in bart_metrics],
    'ROUGE-L': [m[2] for m in t5_metrics] + [m[2] for m in bart_metrics],
    'BLEU': [m[3] for m in t5_metrics] + [m[3] for m in bart_metrics],
})

# Normalize metrics (simple min-max normalization)
for metric in ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU']:
    metrics_df[metric] = (metrics_df[metric] - metrics_df[metric].min()) / (metrics_df[metric].max() - metrics_df[metric].min())

# Assign weights to each metric (customize as needed)
weights = {
    'ROUGE-1': 0.3,
    'ROUGE-2': 0.3,
    'ROUGE-L': 0.3,
    'BLEU': 0.1
}

# Calculate composite score for each model based on the average of the metrics
metrics_summary = metrics_df.groupby('Model').mean().reset_index()
metrics_summary['Composite Score'] = (
    metrics_summary['ROUGE-1'] * weights['ROUGE-1'] +
    metrics_summary['ROUGE-2'] * weights['ROUGE-2'] +
    metrics_summary['ROUGE-L'] * weights['ROUGE-L'] +
    metrics_summary['BLEU'] * weights['BLEU']
)

# Select the best model based on the composite score
best_model_row = metrics_summary.loc[metrics_summary['Composite Score'].idxmax()]

print("\nBest Model Selection:")
print(best_model_row)

# Optionally save predictions and metrics to a CSV file for further analysis
results_data = {
    'Input': inputs,
    'Expected': expected_outputs,
    'T5 Predicted': t5_predictions,
    'BART Predicted': bart_predictions,
}

results_df = pd.DataFrame(results_data)
results_df.to_csv('evaluation_results.csv', index=False)

print("\nEvaluation results saved to 'evaluation_results.csv'.")