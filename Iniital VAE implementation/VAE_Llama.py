import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from torch import nn
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from datasets import Dataset
import os

# Load the preprocessed BoW representation
counts = sp.load_npz("counts.npz")

# Load the vocabulary from a .txt file
with open("vocabulary.txt", "r") as vocab_file:
    vocabulary = [line.strip() for line in vocab_file]

# Reconstruct text samples for initial inspection
def reconstruct_text(bow_matrix, vocab):
    docs = []
    for row in bow_matrix:
        doc = " ".join([vocab[idx] for idx in row.indices for _ in range(row.data[idx])])
        docs.append(doc)
    return docs

documents = reconstruct_text(counts, vocabulary)

# Create a Dataset object from documents
dataset = Dataset.from_dict({"text": documents})

# Dirichlet-based Encoder
class DirichletEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(DirichletEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2_alpha = nn.Linear(128, latent_dim)  # Outputs concentration parameters

    def forward(self, x):
        hidden = torch.relu(self.fc1(x))
        alpha = torch.softplus(self.fc2_alpha(hidden)) + 1e-3  # Ensure positivity
        return alpha

# Define the model components
input_dim = counts.shape[1]  # Vocabulary size
latent_dim = 8  # Number of topics

encoder = DirichletEncoder(input_dim, latent_dim)

# Decoder - Generating Prompts
class PromptDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(PromptDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, latent_dim)

    def forward(self, z):
        hidden = torch.relu(self.fc1(z))
        prompt = self.fc2(hidden)
        return prompt

decoder = PromptDecoder(latent_dim)

# Load the Llama 3-8B model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-3-8b")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3-8b")

# Training Loop Skeleton
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
loss_fn = nn.MSELoss()  # Placeholder loss function (reconstruction loss to be defined)

# Directory to save intermediate results
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

for epoch in range(10):
    epoch_loss = 0
    for i in range(len(documents)):
        bow_vector = torch.tensor(counts[i].toarray(), dtype=torch.float32)  # Single document BoW

        # Forward pass
        alpha = encoder(bow_vector)
        topic_dist = torch.distributions.Dirichlet(alpha).rsample()
        prompt = decoder(topic_dist)

        # Generate text via Llama 3-8B
        prompt_text = "Generate a list of keywords about topics: " + " ".join(
            [vocabulary[j] for j, val in enumerate(topic_dist) if val > 0.1]
        )
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
        output = model.generate(input_ids, max_length=50)

        # BoW reconstruction
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_bow = np.zeros(input_dim)
        for word in generated_text.split():
            if word in vocabulary:
                generated_bow[vocabulary.index(word)] += 1
        #Change to full text not BOW and create data instances (speech or senate session)
        #Use A40s for training -measure speed; 
        #For the llama- get likelhiood of the documents, finetune it (what clusters work best?)
        #Likelihood of individual documents^*
        #1. get likelihood of documents in collection
        #2. finetune the llama model w/ LoRA


        #DATA:
        #full speech vs. 


        reconstruction_loss = loss_fn(torch.tensor(generated_bow, dtype=torch.float32), bow_vector)
        epoch_loss += reconstruction_loss.item()
        optimizer.zero_grad()
        reconstruction_loss.backward()
        optimizer.step()

    # Save intermediate results and model states after each epoch
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(documents):.4f}")
    torch.save(encoder.state_dict(), os.path.join(results_dir, f"encoder_epoch_{epoch + 1}.pt"))
    torch.save(decoder.state_dict(), os.path.join(results_dir, f"decoder_epoch_{epoch + 1}.pt"))

# Save final components
torch.save(encoder.state_dict(), "encoder.pt")
torch.save(decoder.state_dict(), "decoder.pt")
