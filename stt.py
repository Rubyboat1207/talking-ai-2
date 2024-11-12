import asyncio


async def stt(start_key='f8', end_key='f9'):
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, blocking_stt_function, start_key, end_key)
    return text


def blocking_stt_function(start_key, end_key):
    import keyboard
    import sounddevice as sd
    import numpy as np
    import threading
    import speech_recognition as sr

    # Wait for the user to press the start key
    print(f"Press {start_key.upper()} to start recording...")
    keyboard.wait(start_key)
    print(f"Recording started. Press {end_key.upper()} to stop recording.")

    # Audio recording parameters
    samplerate = 16000  # Sample rate required by Whisper
    channels = 1
    dtype = 'int16'

    # Prepare to record
    audio_frames = []

    # Event to stop recording
    stop_event = threading.Event()

    # Callback function to collect audio data
    def callback(indata, frames, time, status):
        audio_frames.append(indata.copy())

    # Function to wait for end_key
    def check_stop():
        keyboard.wait(end_key)
        stop_event.set()

    threading.Thread(target=check_stop, daemon=True).start()

    # Start recording
    with sd.InputStream(samplerate=samplerate, channels=channels, dtype=dtype, callback=callback):
        while not stop_event.is_set():
            sd.sleep(100)

    print("Recording stopped.")

    # Concatenate audio frames
    if len(audio_frames) == 0:
        print("No audio data was recorded.")
        return ""

    audio_data = np.concatenate(audio_frames, axis=0)
    audio_data = audio_data.flatten()

    # Convert numpy array to audio data
    audio_data = audio_data.tobytes()

    # Use speech_recognition to transcribe
    recognizer = sr.Recognizer()
    audio = sr.AudioData(audio_data, samplerate, 2)  # sample_width=2 bytes (int16)

    try:
        text = recognizer.recognize_whisper(audio)
        return text
    except sr.UnknownValueError:
        print("Whisper could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Whisper service; {e}")
        return ""
