# app.py (Streamlit UI)

# import streamlit as st
# from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config


# import torch

# class TransformerSimplifier(T5ForConditionalGeneration):
#     def __init__(self, config):
#         super(TransformerSimplifier, self).__init__(config)
#         self.tokenizer = None

#     def set_tokenizer(self, tokenizer: T5Tokenizer):
#         self.tokenizer = tokenizer

#     def generate_simplified_text(self, input_text: str, max_length=512):
#         if self.tokenizer is None:
#             raise ValueError("Tokenizer is not set.")

#         # Prefix input with task type for better processing
#         input_text = "simplify: " + input_text.strip()

#         # Tokenize the input with appropriate max length, truncating any overly long inputs
#         inputs = self.tokenizer.encode(
#             input_text,
#             return_tensors="pt",
#             max_length=max_length,
#             truncation=True,
#             padding="max_length"
#         )

#         # Generate with parameters aimed for simplification:
#         output_ids = self.generate(
#             inputs,
#             max_length=max_length,
#             num_beams=4,                 # Beam search for better output quality
#             temperature=0.7,             # Creativity in the generated text
#             top_p=0.9,                   # Nucleus sampling (helps with less common words)
#             repetition_penalty=1.2,      # Avoid repetition of phrases or words
#             no_repeat_ngram_size=2,      # Prevent repeating phrases of size 2
#             early_stopping=True          # Stop generation when answer seems complete
#         )

#         # Decode the generated ids and return the simplified text
#         simplified_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
#         return simplified_text

# # Streamlit UI
# import streamlit as st

# # Function to load the model and tokenizer
# def load_model_and_tokenizer():
#     tokenizer = T5Tokenizer.from_pretrained('t5-small')
#     config = T5Config.from_pretrained('t5-small')
#     model = TransformerSimplifier(config=config)
#     model.load_state_dict(torch.load("./models/simplifier.pt", map_location=torch.device('cpu')))
#     model.eval()
#     return model, tokenizer

# # Streamlit app
# def main():
#     st.title("Dyslexia Text-to-Speech Aid")

#     model, tokenizer = load_model_and_tokenizer()
#     model.set_tokenizer(tokenizer)  # Set the tokenizer in the model

#     text_input = st.text_area("Enter text to simplify:", "")

#     if st.button("Simplify"):
#         if text_input:
#             # Simplify the input text using the model
#             simplified_text = model.generate_simplified_text(text_input)

#             # Display the simplified text
#             st.write("Simplified Text:")
#             st.write(simplified_text)
#         else:
#             st.write("Please enter some text to simplify.")

# if __name__ == "__main__":
#     main()

# app.py (Dyslexia Text-to-Speech Aid with Expressive TTS)
import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
import pyttsx3
import torch
import time
import re
import queue
import threading

# Thread-safe singleton for pyttsx3 engine
class TTSEngineSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(TTSEngineSingleton, cls).__new__(cls)
                    cls._instance.engine = pyttsx3.init()
        return cls._instance.engine

def get_tts_engine():
    return TTSEngineSingleton()

# UI update lock
ui_lock = threading.Lock()

# Simplifier class
class TransformerSimplifier(T5ForConditionalGeneration):
    def __init__(self, config):
        super(TransformerSimplifier, self).__init__(config)
        self.tokenizer = None

    def set_tokenizer(self, tokenizer: T5Tokenizer):
        self.tokenizer = tokenizer

    def generate_simplified_text(self, input_text: str, max_length=512):
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not set.")

        input_text = "simplify: " + input_text.strip()
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )

        output_ids = self.generate(
            inputs,
            max_length=max_length,
            num_beams=4,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Load model/tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    config = T5Config.from_pretrained('t5-small')
    model = TransformerSimplifier(config=config)
    model.load_state_dict(torch.load("./models/simplifier.pt", map_location=torch.device('cpu')))
    model.set_tokenizer(tokenizer)
    model.eval()
    return model, tokenizer

# Apply accessibility CSS
def apply_accessibility_settings(font_size, use_dyslexic_font, high_contrast, color_blind_mode):
    css = "<style>"
    if use_dyslexic_font:
        css += """
        @font-face {
            font-family: 'OpenDyslexic';
            src: url('https://raw.githubusercontent.com/antijingoist/open-dyslexic/master/otf/OpenDyslexic-Regular.otf') format('opentype');
        }
        * {
            font-family: 'OpenDyslexic', sans-serif !important;
        }
        """
    if high_contrast:
        css += """
        body {
            background-color: #000000 !important;
            color: #FFFFFF !important;
        }
        .stTextArea>div>div>textarea, .stButton>button {
            background-color: #000000 !important;
            color: #FFFF00 !important;
            border: 2px solid #FFFF00 !important;
        }
        """
    if color_blind_mode:
        css += """
        body, .stTextArea, .stButton>button {
            background-color: #f5f5f5 !important;
            color: #000000 !important;
        }
        .stButton>button {
            background-color: #004488 !important;
            color: #ffffff !important;
            border: 2px solid #ffaa00 !important;
        }
        """
    css += f"""
    .main .block-container {{
        font-size: {font_size}px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Thread-safe TTS function
def threaded_speak(text, reading_speed, speech_pitch, word_queue, stop_event, placeholder, tokens, font_size):
    try:
        engine = get_tts_engine()
        engine.setProperty('rate', int(150 * reading_speed))
        engine.startLoop(False)  # Start the loop manually

        words = re.findall(r"[\w']+|[.,!?;]", text)

        for word in words:
            if stop_event.is_set():
                break
            word_queue.put(word)
            engine.say(word)
            time.sleep(0.1)

            # Update UI with lock
            with ui_lock:
                highlighted = []
                for token in tokens:
                    if token == word:
                        highlighted.append(f"<span style='color: #FF6D00; font-size: {font_size}px;'><b>{token}</b></span>")
                    else:
                        highlighted.append(f"<span style='font-size: {font_size}px;'>{token}</span>")
                placeholder.markdown(" ".join(highlighted), unsafe_allow_html=True)

        engine.endLoop()
        word_queue.put(None)
    except Exception as e:
        word_queue.put(f"ERROR::{str(e)}")
    finally:
        engine.stop()

# Main Streamlit app
def main():
    st.set_page_config(page_title="Dyslexia TTS Aid", layout="wide")

    # Sidebar - Accessibility
    with st.sidebar:
        st.title("Accessibility Settings")
        font_size = st.slider("Font Size", 16, 36, 20)
        use_dyslexic_font = st.checkbox("Use Dyslexia Font", True)
        high_contrast = st.checkbox("High Contrast", False)
        color_blind_mode = st.checkbox("Color Blind Mode", False)
        reading_speed = st.slider("Reading Speed", 0.5, 2.0, 1.0, 0.1)
        speech_pitch = st.slider("Speech Pitch (visual only)", 50, 100, 70)

    apply_accessibility_settings(font_size, use_dyslexic_font, high_contrast, color_blind_mode)

    st.title("🧠 Dyslexia Text-to-Speech Aid")
    st.markdown("Enter complex text, simplify it, and hear it read aloud with word highlighting.")

    model, tokenizer = load_model_and_tokenizer()

    # Session state
    if "simplified_text" not in st.session_state:
        st.session_state.simplified_text = ""

    # Input + Simplify
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Text")
        text_input = st.text_area("Enter text to simplify", height=200, label_visibility="collapsed")

        if st.button("✨ Simplify Text", use_container_width=True):
            if text_input.strip():
                with st.spinner("Simplifying..."):
                    simplified = model.generate_simplified_text(text_input)
                    st.session_state.simplified_text = simplified
            else:
                st.warning("Please enter some text.")

    with col2:
        st.subheader("Simplified Text")
        if st.session_state.simplified_text:
            st.markdown(f"<div style='font-size:{font_size}px; line-height:1.6;'>{st.session_state.simplified_text}</div>", unsafe_allow_html=True)

            if st.button("🔊 Read Aloud", key="read_aloud", use_container_width=True):
                full_text = st.session_state.simplified_text
                tokens = re.findall(r"[\w']+|[.,!?;]", full_text)
                placeholder = st.empty()
                word_queue = queue.Queue()
                stop_event = threading.Event()

                tts_thread = threading.Thread(
                    target=threaded_speak,
                    args=(full_text, reading_speed, speech_pitch, word_queue, stop_event, placeholder, tokens, font_size),
                    daemon=True
                )
                tts_thread.start()

                while True:
                    try:
                        current_word = word_queue.get(timeout=1.0)

                        if isinstance(current_word, str) and current_word.startswith("ERROR::"):
                            st.error("TTS Error: " + current_word[7:])
                            break

                        if current_word is None:
                            break

                    except queue.Empty:
                        continue

if __name__ == "__main__":
    main()
