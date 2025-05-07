import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.scriptrunner import add_script_run_ctx
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
import pyttsx3
import torch
import time
import re
import queue
import threading
import os
from gtts import gTTS
import base64

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

    def generate_simplified_text(self, input_text: str, max_length=4096):
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
    model.load_state_dict(torch.load(os.path.join("models", "simplifier.pt"), map_location=torch.device('gpu' if torch.cuda.is_available() else 'cpu')))
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
    if color_blind_mode == "Protanopia":
        css += """
        .stButton>button {
            background-color: #0000cc !important;
            color: #ffffff !important;
            border: 2px solid #ffff00 !important;
        }
        span.highlight {
            background-color: #ffcc00 !important;
            color: #000000 !important;
        }
        """
    elif color_blind_mode == "Deuteranopia":
        css += """
        .stButton>button {
            background-color: #003366 !important;
            color: #ffffff !important;
            border: 2px solid #ff9900 !important;
        }
        span.highlight {
            background-color: #66ccff !important;
            color: #000000 !important;
        }
        """
    elif color_blind_mode == "Tritanopia":
        css += """
        .stButton>button {
            background-color: #cc0000 !important;
            color: #ffffff !important;
            border: 2px solid #00ffcc !important;
        }
        span.highlight {
            background-color: #00ffcc !important;
            color: #000000 !important;
        }
        """

    css += f"""
    .main .block-container {{
        font-size: {font_size}px;
    }}
    """
    st.markdown(css, unsafe_allow_html=True)

# Thread-safe TTS function
def threaded_speak(text, reading_speed, word_queue, stop_event, placeholder, tokens, font_size, highlighting_mode=None):
    try:
        engine = get_tts_engine()
        # Make the engine speak slower by reducing the rate more significantly
        engine.setProperty('rate', int(50 * reading_speed))
        engine.startLoop(False)
        
        # Split text for different highlighting modes
        words = re.findall(r"[\w']+|[.,!?;]", text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if highlighting_mode == "sentence":
            # Sentence-level highlighting
            for sentence in sentences:
                if stop_event.is_set():
                    break
                
                word_queue.put(sentence)
                engine.say(sentence)
                time.sleep(2.5)  # Longer pause between sentences
                
                with ui_lock:
                    highlighted_text = ""
                    for s in sentences:
                        if s == sentence:
                            highlighted_text += f"<span style='color: #FF6D00; font-size: {font_size}px;'><b>{s}</b></span> "
                        else:
                            highlighted_text += f"<span style='font-size: {font_size}px;'>{s}</span> "
                    placeholder.markdown(highlighted_text, unsafe_allow_html=True)
        else:
            # Default: Word-level highlighting
            for word in words:
                if stop_event.is_set():
                    break
                
                word_queue.put(word)
                engine.say(word)
                # Longer pause between words for slower reading
                time.sleep(0.3 / reading_speed)  

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
        color_blind_mode = st.selectbox(
        "Color Blind Mode",
        ("None", "Protanopia", "Deuteranopia", "Tritanopia")
    )
        reading_speed = st.slider("Highlighting Speed", 0.5, 2.0, 1.0, 0.1)
        # Toggle for word vs sentence highlighting
        word_highlighting = st.radio("Highlighting Mode", ("Word", "Sentence"), index=0)


    apply_accessibility_settings(font_size, use_dyslexic_font, high_contrast, color_blind_mode)

    st.title("👁️ Dyslexia Text-to-Speech Aid 👁️")
    st.markdown("Enter complex text, simplify it, and hear it read aloud with word highlighting.")

    model, tokenizer = load_model_and_tokenizer()

    # Session state
    if "simplified_text" not in st.session_state:
        st.session_state.simplified_text = ""

    # Input + Simplify
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Text")
        text_input = st.text_area("Enter text to simplify", height=200, label_visibility="collapsed", key="manual_input")

        # Upload document
        uploaded_file = st.file_uploader(
            "Upload a document (PDF, DOCX, or TXT)", 
            type=["pdf", "docx", "txt"]
        )

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == "txt":
            text_input = uploaded_file.read().decode("utf-8")
        elif file_extension == "docx":
            import docx
            doc = docx.Document(uploaded_file)
            text_input = "\n".join([para.text for para in doc.paragraphs])
        elif file_extension == "pdf":
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text_input = "\n".join([page.extract_text() or '' for page in pdf_reader.pages])

        # Display uploaded text (optional)
        st.text_area("Uploaded Text", text_input, height=200, label_visibility="collapsed", key="uploaded_text")

    
    with col2:
        st.subheader("Simplified Text")
        
        if st.button("✨ Simplify Text", use_container_width=True):
            if text_input.strip():
                with st.spinner("Simplifying..."):
                    simplified = model.generate_simplified_text(text_input)
                    st.session_state.simplified_text = simplified
            else:
                st.warning("Please enter some text or upload a file.")

   
        if st.session_state.simplified_text:
            st.markdown(f"<div style='font-size:{font_size}px; line-height:1.6;'>{st.session_state.simplified_text}</div>", unsafe_allow_html=True)

            if st.button("🔊 Read Aloud", key="read_aloud", use_container_width=True):
                full_text = st.session_state.simplified_text
                tokens = re.findall(r"[\w']+|[.,!?;]", full_text)
                placeholder = st.empty()

                # Generat e gTTS audio
                tts = gTTS(text=full_text, lang='en', slow=False)
                audio_file = "temp_audio.mp3"
                tts.save(audio_file)
                
                with open(audio_file, "rb") as f:
                    audio_bytes = f.read()
                os.remove(audio_file)

                
                
                # Display audio player with autoplay
                # st.audio(audio_bytes, format="audio/mp3", start_time=0, autoplay=True)
                import base64
                
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                audio_html = f"""
                    <audio autoplay="true" controls>
                        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)

                # Start highlighting thread
                word_queue = queue.Queue()
                stop_event = threading.Event()

                tts_thread = threading.Thread(
                    target=threaded_speak,
                    args=(full_text, reading_speed, word_queue, stop_event, placeholder, tokens, font_size, word_highlighting.lower()),
                    daemon=True
                )

                ctx = get_script_run_ctx()
                if ctx:
                    add_script_run_ctx(tts_thread, ctx)

                tts_thread.start()

                # Process queue (required for error handling)
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