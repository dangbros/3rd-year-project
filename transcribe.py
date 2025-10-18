from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import soundfile as sf
import torch
import json
import librosa  


# NOTE: You will need to have an audio file named 'audio.wav' in the same directory.
# This audio file should be mono (single channel).

# 1. Load the processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# 2. Load and prepare the audio file
try:
    speech, sample_rate = sf.read("audio.wav")
    
    # <-- 2. RESAMPLE THE AUDIO
    # If the sample rate is not 16kHz, resample it
    if sample_rate != 16000:
        speech = librosa.resample(y=speech, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000 # Update the sample rate variable

    #process for whisper
    input_features = processor(speech, sampling_rate=sample_rate, return_tensors="pt").input_features
    # 3. Generate token IDs
    predicted_ids = model.generate(input_features)
    # 4. Decode the token IDs to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)


    #process for wav2vec2
    audio_inputs = audio_processor(speech, sampling_rate=sample_rate, return_tensors="pt")
    #this gets the audio features from wav2vec2
    with torch.no_grad():
        outputs = audio_model(**audio_inputs)
    #averages the feature to get a single vector representation
    audio_features = outputs.last_hidden_state.mean(dim=1)
    
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
    print("\n---- Audio Features ----")
    print(f"Shape of the feature vector:{audio_features.shape}")
    #print(audio_features) <- to get the all the numbers
    
except FileNotFoundError:
    print("Error: 'audio.wav' not found. Please add a mono audio file to the directory.")
except Exception as e:
    print(f"An error occurred: {e}")
