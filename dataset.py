# dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd

class SimplificationDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=50):
        df = pd.read_csv(csv_path)
        self.pairs = []
        self.max_len = max_len
        self.tokenizer = tokenizer
        for _, row in df.iterrows():
            complex_ids = tokenizer.encode(row['Normal'])[:max_len]
            simple_ids = tokenizer.encode(row['Simple'])[:max_len]
            # Add start (<s>) and end (</s>) tokens for target sequence
            simple_ids = [tokenizer.token_to_id['<s>']] + simple_ids + [tokenizer.token_to_id['</s>']]
            self.pairs.append((complex_ids, simple_ids))
        self.pad_id = tokenizer.token_to_id['<pad>']

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        # Pad source
        src = src + [self.pad_id] * (self.max_len - len(src))
        # Pad target: assume tgt length <= max_len + 2
        tgt = tgt + [self.pad_id] * ((self.max_len + 2) - len(tgt))
        # For teacher forcing, input is all but last token, output is all but first
        tgt_input = torch.tensor(tgt[:-1])
        tgt_output = torch.tensor(tgt[1:])
        return torch.tensor(src), tgt_input, tgt_output
