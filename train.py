# train.py
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from models.tokenizer import Tokenizer
from models.simplifier import TransformerSimplifier
from dataset import SimplificationDataset

# Hyperparameters
BATCH_SIZE = 100
NUM_EPOCHS = 10
MAX_LEN = 500
LEARNING_RATE = 1e-3

# Build tokenizer from dataset
df = pd.read_csv(os.path.join('data', 'processed', 'cleaned_simplification_dataset.csv'))
tokenizer = Tokenizer()
tokenizer.build_vocab(df['Normal'].tolist() + df['Simple'].tolist())

# Create dataset and loader
dataset = SimplificationDataset(os.path.join('data', 'processed', 'cleaned_simplification_dataset.csv'), tokenizer, max_len=MAX_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model, optimizer, and loss
vocab_size = len(tokenizer.token_to_id)
model = TransformerSimplifier(vocab_size, max_len=MAX_LEN+2)  # +2 for <s> and </s>
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id['<pad>'])

model.train()
for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    for src, tgt_in, tgt_out in loader:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        optimizer.zero_grad()
        logits = model(src, tgt_in)
        loss = loss_fn(logits.view(-1, vocab_size), tgt_out.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}")

# Save the trained model and tokenizer (as a simple pickle)
torch.save(model.state_dict(), "simplifier.pt")
import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
