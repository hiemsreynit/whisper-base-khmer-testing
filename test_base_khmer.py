import time
import os
import json
from transformers import pipeline

AUDIO_FILE = "audio/test_audi_4.wav"
MODEL_NAME = "sengtha/whisper-base-khmer"

print("Loading Whisper Base Khmer...")

pipe = pipeline(
    "automatic-speech-recognition",
    model=MODEL_NAME,
    device=-1,
)

print("Model loaded.")

start_time = time.time()

result = pipe(
    AUDIO_FILE,
    generate_kwargs={
        "language": "khmer",
        "task": "transcribe",
    },
)

elapsed_time = time.time() - start_time

transcription = result["text"].strip()

print("\n===== RESULT =====")
print("Transcription:", transcription)
print("Processing time:", round(elapsed_time, 2), "seconds")


os.makedirs("results", exist_ok=True)

output = {
    "model": MODEL_NAME,
    "audio_file": AUDIO_FILE,
    "transcription": transcription,
    "processing_time_seconds": round(elapsed_time, 2)
}

with open("results/base-khmer.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)

print("\nResult saved to: results/base-khmer.json")