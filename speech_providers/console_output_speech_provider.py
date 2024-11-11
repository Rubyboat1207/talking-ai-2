from abc import ABCMeta
from speech_provider import SpeechProvider


class ConsoleOutputSpeechProvider(SpeechProvider, metaclass=ABCMeta):
    def generate_speech(self, text: str):
        print(text)