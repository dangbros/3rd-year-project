from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import torch
import json
import librosa  # <-- 1. IMPORT LIBROSA

# --- SCRIPT SETUP ---
# NOTE: You will need to have an audio file named 'audio.wav' in the same directory.
# This audio file should be mono (single channel).
# You can install the necessary libraries with:
# pip install transformers torch soundfile accelerate librosa

# 1. Load the processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

# 2. Load and prepare the audio file
try:
    speech, sample_rate = sf.read("audio.wav")
    
    # <-- 2. RESAMPLE THE AUDIO
    # If the sample rate is not 16kHz, resample it
    if sample_rate != 16000:
        speech = librosa.resample(y=speech, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000 # Update the sample rate variable

    input_features = processor(speech, sampling_rate=sample_rate, return_tensors="pt").input_features

    # 3. Generate token IDs
    predicted_ids = model.generate(input_features)

    # 4. Decode the token IDs to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    # --- FINAL OUTPUT ---
    # 5. Format the output as a JSON object as per your workflow
    output_data = {
        "text": transcription[0],
        "traits": {
            "openness": None,
            "conscientiousness": None,
            "extraversion": None,
            "agreeableness": None,
            "neuroticism": None
        }
    }

    # Print the final JSON output
    print(json.dumps(output_data, indent=2))

except FileNotFoundError:
    print("Error: 'audio.wav' not found. Please add a mono audio file to the directory.")
except Exception as e:
    print(f"An error occurred: {e}")
