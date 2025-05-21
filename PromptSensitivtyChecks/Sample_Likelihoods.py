import os
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
import csv

# Set paths
base_dir = "./groupedSpeeches/speeches_097"  # Update this path if needed
model_path = "./local_llama3b"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
model.eval()

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Define max length for truncation
max_length = 4000  # Adjust if needed

files = [f for f in os.listdir(base_dir) if f.endswith(".txt")]
# Randomly select 2500 chunks

# Function to calculate log likelihood
def calculate_log_likelihood(text, tokenizer, model):
    """Calculate log likelihood, number of tokens, and perplexity for the given text."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=4000)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Create labels but ignore prompt tokens
    labels = input_ids.clone()

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            log_likelihood = outputs.loss.item()
            num_tokens = input_ids.shape[1]
            perplexity = torch.exp(outputs.loss).item()

    return log_likelihood, num_tokens, perplexity

def get_last_chunk_csv(csv_file_path):
    with open(csv_file_path, mode='r', newline='', encoding='latin-1') as file:
        reader = csv.reader(file)
        rows = list(reader)
    if len(rows) == 0:
        return 0
    return int(rows[-1][0])

csv_file_path = 'log_likelihoods_no_prompt_97.csv'
csv_temp_path = 'log_likelihoods_no_prompt_97_temp.csv'
line_item_added = 0
if os.path.exists(csv_file_path):
    os.remove(csv_file_path)
if os.path.exists(csv_temp_path):
    line_item_added = get_last_chunk_csv(csv_temp_path)

with open(csv_temp_path, mode='w', newline='', encoding='latin-1') as file:  # Use latin-1 encoding
    writer = csv.writer(file)
    if line_item_added == 0:
        writer.writerow(['Sample', 'File', 'Log Likelihood', '# of Tokens', 'Perplexity'])

    # Process each selected file
    for i, file_name in enumerate(files[line_item_added:], start=line_item_added + 1):
        file_path = os.path.join(base_dir, file_name)
        
        with open(file_path, "r", encoding="latin-1") as f:  # Use latin-1 encoding
            text = f.read().strip()

        log_likelihood, num_tokens, perplexity = calculate_log_likelihood(text, tokenizer, model)

        writer.writerow([i, file_name, log_likelihood, num_tokens, perplexity])
        file.flush()  # Ensure the file is flushed to disk after each write

os.rename(csv_temp_path, csv_file_path)
print(f"Log likelihoods have been written to {csv_file_path}")