import pyttsx3
from abc import ABC, abstractmethod
from speech_provider import SpeechProvider

class WindowsTTSProvider(SpeechProvider):
    def __init__(self):
        # Initialize the TTS engine
        self.engine = pyttsx3.init()

        # Optionally, you can set properties such as rate, volume, and voice
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.engine.setProperty('volume', 1)  # Volume (0.0 to 1.0)

    def generate_speech(self, text: str):
        # Use the engine to say the text
        self.engine.say(text)
        # Blocks while processing all currently queued commands
        self.engine.runAndWait()
