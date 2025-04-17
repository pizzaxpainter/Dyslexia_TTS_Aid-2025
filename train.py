import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from dataset import SimplificationDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import GradScaler, autocast  # For mixed precision
from torch.nn.utils import clip_grad_norm_  # Gradient clipping import
import pickle

# Paths
data_dir = "data/processed"
input_csv = os.path.join(data_dir, "cleaned_simplification_dataset.csv")
train_csv = os.path.join(data_dir, "train.csv")
val_csv = os.path.join(data_dir, "val.csv")

# Only split if not already split
if not (os.path.exists(train_csv) and os.path.exists(val_csv)):
    print("Splitting cleaned dataset into train/val CSVs...")
    df = pd.read_csv(input_csv)
    df.columns = ['Normal', 'Simple']  # Rename just in case
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    print(f"Train size: {len(train_df)} | Val size: {len(val_df)}")
else:
    print("Train/val CSVs already exist. Skipping split.")

def collate_fn(batch):
    input_texts = [str(item['input']) for item in batch]
    target_texts = [str(item['target']) for item in batch]

    input_encodings = tokenizer(input_texts, padding=False, truncation=True, return_tensors="pt")
    target_encodings = tokenizer(target_texts, padding=False, truncation=True, return_tensors="pt")

    input_encodings = tokenizer.pad(input_encodings, padding=True, return_tensors="pt")
    target_encodings = tokenizer.pad(target_encodings, padding=True, return_tensors="pt")

    return {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids'],
    }

def train(model, dataloader, optimizer, loss_fn, device, tokenizer, scaler, accumulation_steps=1):
    model.train()
    running_loss = 0.0
    total_batches = len(dataloader)

    for batch_idx, batch in tqdm(enumerate(dataloader), total=total_batches, desc="Training", unit="batch"):
        src, tgt = batch['input_ids'].to(device), batch['labels'].to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(input_ids=src, labels=tgt)
            loss = outputs.loss

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()

        running_loss += loss.item()
        torch.cuda.empty_cache()

    return running_loss / len(dataloader)

def main():
    global tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    train_dataset = SimplificationDataset(train_csv, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    pad_id = tokenizer.pad_token_id
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    scaler = GradScaler()
    epochs = 5

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}")
        train_loss = train(model, train_loader, optimizer, loss_fn, device, tokenizer, scaler, accumulation_steps=4)
        print(f"Train Loss: {train_loss:.4f}")

    # Save tokenizer and model
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    torch.save(model.state_dict(), "simplifier.pt")

if __name__ == "__main__":
    main()
