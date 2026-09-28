from faster_whisper import WhisperModel
import time

AUDIO_FILE = "audio/test_audi_1.ogg"

print("Loading Whisper Large-v3...")

model = WhisperModel(
    "large-v3",
    device="cpu",
    compute_type="int8",
)

print("Model loaded.")

start_time = time.time()

segments, info = model.transcribe(
    AUDIO_FILE,
    language="km",
    task="transcribe",
)

transcription = ""

for segment in segments:
    transcription += segment.text

elapsed_time = time.time() - start_time

print("\n===== RESULT =====")
print("Detected language:", info.language)
print("Language probability:", info.language_probability)
print("Transcription:", transcription.strip())
print("Processing time:", round(elapsed_time, 2), "seconds")