# app.py
import streamlit as st
import torch
import pickle
from models.tokenizer import Tokenizer
from models.simplifier import TransformerSimplifier, greedy_decode
from utils.pdf_reader import extract_text
from utils.expressive_tts import expressive_speak

# Load model and tokenizer
@st.cache(allow_output_mutation=True)
def load_model():
    # Load tokenizer
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    vocab_size = len(tokenizer.token_to_id)
    model = TransformerSimplifier(vocab_size, max_len=52)  # max_len must match training (MAX_LEN+2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("simplifier.pt", map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

st.title("Dyslexia Book Reader + Expressive TTS")
uploaded_pdf = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_pdf:
    # Save the uploaded PDF temporarily
    with open("data/temp_book.pdf", "wb") as f:
        f.write(uploaded_pdf.read())
    raw_text = extract_text("data/temp_book.pdf")
    st.subheader("Extracted Text (first 500 chars)")
    st.write(raw_text[:500] + "...")
    
    # Split text into sentences (for demonstration)
    import nltk
    nltk.download('punkt')
    sentences = nltk.sent_tokenize(raw_text)
    
    # Simplify sentences using the transformer (inference)
    st.subheader("Simplified Text with Highlighting")
    simplified_text = []
    for i, sent in enumerate(sentences):
        with st.spinner(f"Simplifying sentence {i+1}/{len(sentences)}..."):
            simple = greedy_decode(model, sent, tokenizer)
            simplified_text.append(simple)
            # Highlight each sentence (Streamlit markdown)
            st.markdown(f"<mark>{simple}</mark>", unsafe_allow_html=True)
            # Expressive TTS for each sentence (triggered button for demo)
            if st.button(f"Speak Sentence {i+1}", key=f"tts_{i}"):
                expressive_speak(simple)
    
    st.subheader("Full Simplified Text")
    st.write("\n".join(simplified_text))
