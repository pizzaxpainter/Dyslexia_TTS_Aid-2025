from transformers import T5Tokenizer

class Tokenizer:
    def __init__(self, model_name="t5-small"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    def build_vocab(self, sentences):
        # T5 already has a built-in vocab, so this may not be needed
        pass

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=True)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def token_to_id(self):
        return self.tokenizer.get_vocab()
