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

files = [f for f in os.listdir(base_dir) if f.endswith(".txt")]

# Define good prompts
good_prompts = [
     "Write a speech from the 97th Congress (1981-1983) where a senator discusses routine administrative matters, such as the scheduling of bills, committee assignments, or acknowledgments of colleagues. Maintain the formal tone and procedural structure typical of Senate floor discussions.",

    "Pretend to be a Republican senator during the 97th Congress(1981-1983) and give a speech that maintains the formal tone and procedural structure typical of Senate floor discussions",
    "Pretend to be a Democratic senator during the 97th Congress(1981-1983) and give a speech that maintains the formal tone and procedural structure typical of Senate floor discussions",

    "Give me a speech that could've been spoken by a senator in the 97th Congress(1981-1983). The senator could be explaining a procedural motion, invoking cloture, referring a bill to committee, or debating amendments. The speech should be neutral and focused on Senate rules and processes.",
    "Generate a speech from the 97th Congress in which a senator delivers a commemorative address, such as honoring a retiring colleague, recognizing a national holiday, or acknowledging a significant historical event. The tone should be respectful and unifying."
]
# Start with super generic promopt and then branch into more specific prompts, agglomerative clustiering switching to top down clustering
# Function to calculate metrics
# When is the right time to use xyz prompt?
def calculate_metrics(text, prompt, tokenizer, model):
    """Calculate log likelihood, number of tokens, and perplexity for the given text."""
    inputs = tokenizer(prompt, text, return_tensors="pt", padding=True, truncation=True, max_length=4000)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Create labels but ignore prompt tokens
    labels = input_ids.clone()
    prompt_length = len(tokenizer(prompt)["input_ids"])
    labels[:, :len(tokenizer(prompt)["input_ids"])] = -100  # Ignore loss for prompt tokens

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            log_likelihood = outputs.loss.item()
            num_tokens = input_ids.shape[1] - prompt_length
            perplexity = torch.exp(outputs.loss).item()

    return log_likelihood, num_tokens, perplexity


def get_last_chunk_csv(csv_file_path):
    with open(csv_file_path, mode='r', newline='', encoding='latin-1') as file:
        reader = csv.reader(file)
        rows = list(reader)
    if len(rows) == 0:
        return 0
    return rows[-1][0]


for idx, prompt in enumerate(good_prompts):
    csv_file_path = f'log_likelihoods_good_prompt_97_{idx+1}.csv'
    csv_temp_path = f'log_likelihoods_good_prompt_97_{idx+1}_temp.csv'
    line_item_added = 0
    if os.path.exists(csv_file_path):
        continue
    if os.path.exists(csv_temp_path):
        line_item_added = int(get_last_chunk_csv(csv_temp_path))

    with open(csv_temp_path, mode='w', newline='', encoding='latin-1') as file:  # Use latin-1 encoding
        writer = csv.writer(file)
        if line_item_added == 0:
            writer.writerow(['Sample', 'File', 'Log Likelihood', '# of Tokens', 'Perplexity'])

        # Process each selected file
        for i, file in enumerate(files[line_item_added+1:], start=1):
            file_path = os.path.join(base_dir, file)
            
            with open(file_path, "r", encoding="latin-1") as f:  # Use latin-1 encoding
                text = f.read().strip()

            log_likelihood, num_tokens, perplexity = calculate_metrics(text, prompt, tokenizer, model)

            writer.writerow([i, file, log_likelihood, num_tokens, perplexity])
    os.rename(csv_temp_path, csv_file_path)
    print(f"Log likelihoods for prompt {idx+1} have been written to {csv_file_path}")