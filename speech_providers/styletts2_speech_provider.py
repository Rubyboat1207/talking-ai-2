from abc import ABCMeta
from styletts2 import tts
from speech_provider import SpeechProvider
import sounddevice as sd


class StyleTTS2SpeechProvider(SpeechProvider, metaclass=ABCMeta):
    def __init__(self):
        self.tts = tts.StyleTTS2()

    def generate_speech(self, text: str):
        print(text)
        wav = self.tts.inference(text)

        sd.play(wav, 24000)

        sd.wait()