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
def calculate_log_likelihood(text, prompt):
    text_with_prompt = prompt + text
    inputs = tokenizer(text_with_prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        log_likelihood = outputs.loss.item()  # Loss is the negative log likelihood
    return log_likelihood

# Open CSV file for writing
csv_file_path = 'log_likelihoods.csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Sample', 'File', 'Prompt Type', 'Log Likelihood'])

    # Process each selected file
    for i, file in enumerate(files, start=1):
        file_path = os.path.join(base_dir, file)
        
        with open(file_path, "r", encoding="latin1") as f:
            text = f.read().strip()

        good_log_likelihood = calculate_log_likelihood(text, good_prompt)
        bad_log_likelihood = calculate_log_likelihood(text, bad_prompt)
        
        writer.writerow([i, file, 'Good', good_log_likelihood])
        writer.writerow([i, file, 'Bad', bad_log_likelihood])

print(f"Log likelihoods have been written to {csv_file_path}")

# Read the log likelihoods back from the CSV file for visualization
log_likelihoods = {'Good': [], 'Bad': []}
with open(csv_file_path, mode='r', newline='') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        log_likelihoods[row[2]].append(float(row[3]))

# Randomly sample a subset of log likelihoods for plotting
sample_size = 1000  # Adjust the sample size as needed
sampled_good_likelihoods = random.sample(log_likelihoods['Good'], sample_size)
sampled_bad_likelihoods = random.sample(log_likelihoods['Bad'], sample_size)

# Set Seaborn style
sns.set(style="whitegrid")

# Plot histogram of log likelihoods for good prompts
plt.figure(figsize=(10, 5))
sns.histplot(sampled_good_likelihoods, bins=15, kde=True, color='blue', label='Good Prompt')
sns.histplot(sampled_bad_likelihoods, bins=15, kde=True, color='red', label='Bad Prompt')
plt.xlabel("Log Likelihood")
plt.ylabel("Frequency")
plt.title("Distribution of Log Likelihoods for Good and Bad Prompts")
plt.legend()
plt.show()

# Plot box plot
plt.figure(figsize=(8, 5))
sns.boxplot(data=[sampled_good_likelihoods, sampled_bad_likelihoods], palette=['blue', 'red'])
plt.xticks([0, 1], ['Good Prompt', 'Bad Prompt'])
plt.ylabel("Log Likelihood")
plt.title("Box Plot of Log Likelihoods for Good and Bad Prompts")
plt.show()