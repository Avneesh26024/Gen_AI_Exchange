import sounddevice as sd
import numpy as np
import whisper
import tempfile
import scipy.io.wavfile as wavfile

# Load Whisper model once
model = whisper.load_model("base")

def record_and_transcribe(duration=5, fs=16000):
    """
    Records audio from the mic and transcribes it instantly.
    duration: seconds to record
    fs: sampling rate
    """
    print(f"🎤 Recording for {duration} seconds... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("✅ Recording complete.")

    # Save to temp WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wavfile.write(tmp.name, fs, recording)
        tmp_path = tmp.name

    # Transcribe
    result = model.transcribe(tmp_path)
    text = result['text']
    print(f"📝 Transcribed text: {text}")
    return text

# Example usage
if __name__ == "__main__":
    text = record_and_transcribe(duration=7)
    # Now you can feed `text` to your agent
    print("Send this to agent:", text)
