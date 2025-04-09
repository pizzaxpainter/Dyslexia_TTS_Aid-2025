# models/simplifier.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x

class TransformerSimplifier(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=3, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        # src, tgt: (batch_size, seq_len)
        src_emb = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)
        memory = self.encoder(src_emb.transpose(0,1))
        out = self.decoder(tgt_emb.transpose(0,1), memory)
        out = self.output_proj(out.transpose(0,1))
        return out  # (batch_size, tgt_seq_len, vocab_size)

def greedy_decode(model, src, tokenizer, max_len=50, start_token="<s>", end_token="</s>"):
    model.eval()
    device = next(model.parameters()).device
    src_ids = torch.tensor([tokenizer.encode(src)]).to(device)
    tgt_ids = [tokenizer.token_to_id[start_token]]
    for _ in range(max_len):
        tgt_tensor = torch.tensor([tgt_ids]).to(device)
        output = model(src_ids, tgt_tensor)
        next_token = output[0, -1].argmax().item()
        tgt_ids.append(next_token)
        if next_token == tokenizer.token_to_id[end_token]:
            break
    # Decode, excluding the start and end tokens
    return tokenizer.decode(tgt_ids[1:-1])
