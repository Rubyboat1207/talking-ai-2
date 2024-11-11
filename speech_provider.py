import abc


class SpeechProvider(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def generate_speech(self, text: str):
        pass
