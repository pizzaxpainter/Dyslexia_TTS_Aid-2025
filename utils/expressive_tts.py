# utils/expressive_tts.py
import pyttsx3
import threading

def expressive_speak(text, rate=150, pitch=70, volume=1.0):
    def run():
        engine = pyttsx3.init()
        engine.setProperty('rate', rate)
        engine.setProperty('volume', volume)
        # Note: pyttsx3 does not support pitch control directly on all platforms.
        # This is a placeholder to indicate expressiveness.
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run).start()
