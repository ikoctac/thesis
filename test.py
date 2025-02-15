from transformers import BertTokenizer, BertModel
import torch
import pandas as pd

# Load pre-trained multilingual model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Load pre-trained multilingual model (weights)
model = BertModel.from_pretrained('bert-base-multilingual-cased')

# Input sentence in Greek
sentence = "Computer science is a widely used technology field globally."

# Tokenize input
inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)

# Get output
with torch.no_grad():
    outputs = model(**inputs)

# The embeddings are in the last hidden-state of the first batch
embeddings = outputs.last_hidden_state[0]

# Convert embeddings to a numpy array
embedding_array = embeddings.numpy()

# Get token names
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# Create a DataFrame for better visualization
df = pd.DataFrame(embedding_array.T, columns=tokens)

# Insert a row at the beginning with token names
df_with_tokens = pd.concat([pd.DataFrame([tokens], columns=df.columns), df], ignore_index=True)

# Print the DataFrame
print(df_with_tokens)