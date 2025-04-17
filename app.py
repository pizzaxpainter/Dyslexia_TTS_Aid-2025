# app.py (Streamlit UI)

import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config


import torch

class TransformerSimplifier(T5ForConditionalGeneration):
    def __init__(self, config):
        super(TransformerSimplifier, self).__init__(config)
        self.tokenizer = None

    def set_tokenizer(self, tokenizer: T5Tokenizer):
        self.tokenizer = tokenizer

    def generate_simplified_text(self, input_text: str, max_length=512):
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not set.")

        # Prefix input with task type for better processing
        input_text = "simplify: " + input_text.strip()

        # Tokenize the input with appropriate max length, truncating any overly long inputs
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )

        # Generate with parameters aimed for simplification:
        output_ids = self.generate(
            inputs,
            max_length=max_length,
            num_beams=4,                 # Beam search for better output quality
            temperature=0.7,             # Creativity in the generated text
            top_p=0.9,                   # Nucleus sampling (helps with less common words)
            repetition_penalty=1.2,      # Avoid repetition of phrases or words
            no_repeat_ngram_size=2,      # Prevent repeating phrases of size 2
            early_stopping=True          # Stop generation when answer seems complete
        )

        # Decode the generated ids and return the simplified text
        simplified_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return simplified_text

# Streamlit UI
import streamlit as st

# Function to load the model and tokenizer
def load_model_and_tokenizer():
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    config = T5Config.from_pretrained('t5-small')
    model = TransformerSimplifier(config=config)
    model.load_state_dict(torch.load("simplifier.pt", map_location=torch.device('cpu')))
    model.eval()
    return model, tokenizer

# Streamlit app
def main():
    st.title("Dyslexia Text-to-Speech Aid")

    model, tokenizer = load_model_and_tokenizer()
    model.set_tokenizer(tokenizer)  # Set the tokenizer in the model

    text_input = st.text_area("Enter text to simplify:", "")

    if st.button("Simplify"):
        if text_input:
            # Simplify the input text using the model
            simplified_text = model.generate_simplified_text(text_input)

            # Display the simplified text
            st.write("Simplified Text:")
            st.write(simplified_text)
        else:
            st.write("Please enter some text to simplify.")

if __name__ == "__main__":
    main()
