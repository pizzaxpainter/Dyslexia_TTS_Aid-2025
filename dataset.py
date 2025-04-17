from torch.utils.data import Dataset
import pandas as pd
from transformers import T5Tokenizer

class SimplificationDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source_text = str(self.data.iloc[idx, 0])
        target_text = str(self.data.iloc[idx, 1])

        # Tokenizing the input and target
        source_encoding = self.tokenizer.encode_plus(
            source_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer.encode_plus(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt"
        )

        # Ensure that we're returning tensors
        input_ids = source_encoding['input_ids'].squeeze(0)  # Convert to 1D tensor
        target_ids = target_encoding['input_ids'].squeeze(0)
        
        return {
            'input': input_ids,
            'target': target_ids
        }
