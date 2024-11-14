from abc import ABCMeta
from styletts2 import tts
from speech_provider import SpeechProvider
import sounddevice as sd


class StyleTTS2SpeechProvider(SpeechProvider, metaclass=ABCMeta):
    def __init__(self):
        self.tts = tts.StyleTTS2()

    def generate_speech(self, text: str):
        wav = self.tts.inference(text, target_voice_path='./speech_providers/sample/voice.mp3')

        sd.play(wav, 24000)

        sd.wait()