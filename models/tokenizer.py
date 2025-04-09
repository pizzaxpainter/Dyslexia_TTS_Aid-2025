# models/tokenizer.py
import re

class Tokenizer:
    def __init__(self):
        # Special tokens: <pad>, <s>, </s>, <unk>
        self.token_to_id = {'<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def build_vocab(self, texts, min_freq=1):
        freq = {}
        for text in texts:
            for token in self.tokenize(text):
                freq[token] = freq.get(token, 0) + 1
        for token, count in freq.items():
            if count >= min_freq and token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

    def tokenize(self, text):
        # Split text into words and punctuation
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def encode(self, text):
        return [self.token_to_id.get(tok, self.token_to_id['<unk>']) for tok in self.tokenize(text)]

    def decode(self, ids):
        return " ".join([self.id_to_token.get(i, '<unk>') for i in ids])
