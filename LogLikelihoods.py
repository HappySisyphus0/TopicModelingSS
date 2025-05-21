import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer (adjust the model path or name accordingly)
model_path ="./local_llama3b"
model_name = "llama-models/models/llama3_2"  # Update with the actual model name or path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

# Move the model to GPU if available for faster inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the output directory where chunks are saved
output_dir = "grouped_speeches/speeches_097"  # Update with your output path

# DataFrame to store results
log_likelihood_data = []

# Function to calculate the log likelihood of a chunk
def calculate_log_likelihood(text_chunk):
    # Tokenize the chunk
    inputs = tokenizer(text_chunk, return_tensors="pt", truncation=True, padding=True)

    # Move tensors to the same device as the model (GPU or CPU)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Get model outputs (logits and attention scores)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    # Calculate log likelihood by comparing logits and actual labels
    logits = outputs.logits
    labels = inputs["input_ids"]

    # Use cross entropy to calculate log likelihood (negative log probability)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Calculate the log likelihood for each token in the sequence
    log_likelihood = torch.gather(shift_logits, dim=-1, index=shift_labels.unsqueeze(-1))
    log_likelihood = log_likelihood.squeeze(-1)

    # Sum the log likelihoods (or average depending on your preference)
    total_log_likelihood = log_likelihood.sum().item()  # Total log likelihood
    return total_log_likelihood

# Function to read and process the chunks from files
def process_chunks():
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".txt"):  # Process only the text files
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="latin1") as f:
                    text_chunk = f.read()

                # Calculate log likelihood for the chunk
                log_likelihood = calculate_log_likelihood(text_chunk)

                # Append the result to the DataFrame list
                log_likelihood_data.append({
                    "File": file,
                    "Log Likelihood": log_likelihood
                })
                print(f"Processed: {file}")

    # Convert the list to a DataFrame
    df = pd.DataFrame(log_likelihood_data)

    # Save the DataFrame to a CSV file
    df.to_csv("no_prompt_log_likelihood_results.csv", index=False)

    # Display the DataFrame (optional)
    #print(df)

# Run the processing function
process_chunks()