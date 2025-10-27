#main transcription and personality prediction script
import torch
import torch.nn as nn
import librosa
import soundfile as sf
import json
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Wav2Vec2Processor, Wav2Vec2Model, BertTokenizer, BertModel


# --- 1. Define the Personality Model Architecture ---

# This MUST match the class definition used in train.py
class PersonalityModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(PersonalityModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(128, output_size)

    def forward(self, combined_features):
        x = self.layer1(combined_features)
        x = self.activation(x)
        output = self.output_layer(x)
        # Apply sigmoid to ensure output is between 0 and 1
        return torch.sigmoid(output)

# --- 2. Load ALL Models ---
print("Loading all models...")
# Transcription models
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

# Audio feature models
audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# Text feature models
text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text_model = BertModel.from_pretrained("bert-base-uncased")

# --- Load Your Trained Personality Model ---
INPUT_SIZE = 1536  # 768 from text + 768 from audio
OUTPUT_SIZE = 5    
personality_model = PersonalityModel(INPUT_SIZE, OUTPUT_SIZE)

# Load the saved weights
try:
    personality_model.load_state_dict(torch.load("personality_model_final.pth"))
    personality_model.eval() # Set to evaluation mode
    print("Loaded trained personality model weights.")
except FileNotFoundError:
    print("ERROR: personality_model_final.pth not found. Please train the model first.")
    exit()
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit()

print("All models loaded.")

# --- 3. Main Processing Block ---
try:
    # Load and resample the audio file
    speech, sample_rate = sf.read("audio.wav")
    if sample_rate != 16000:
        speech = librosa.resample(y=speech, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    # Put models on CPU (since preprocessing was done on CPU)
    device = "cpu"
    audio_model.to(device)
    text_model.to(device)
    personality_model.to(device)

    # --- Part A: Transcription using Whisper ---
    input_features = processor(speech, sampling_rate=sample_rate, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    transcription_text = transcription[0]

    # --- Part B: Text Feature Extraction using BERT ---
    text_inputs = text_tokenizer(transcription_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        text_outputs = text_model(**text_inputs)
    text_features = text_outputs.last_hidden_state.mean(dim=1)

    # --- Part C: Audio Feature Extraction using wav2vec2 ---
    audio_inputs = audio_processor(speech, sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        audio_outputs = audio_model(**audio_inputs)
    audio_features = audio_outputs.last_hidden_state.mean(dim=1)

    # --- Part D: Personality Prediction ---
    # Combine features
    combined_features = torch.cat((text_features, audio_features), dim=1)
    
    # Get predictions from your trained model
    with torch.no_grad():
        predicted_scores = personality_model(combined_features)
        
    # Squeeze to remove unnecessary batch dimension and convert to list
    scores = predicted_scores.squeeze().tolist()

    # --- FINAL OUTPUT ---
    print("\n--- Prediction Results ---")
    output_data = {
        "text": transcription_text,
        "traits": {
            "openness": round(scores[0], 4),
            "conscientiousness": round(scores[1], 4),
            "extraversion": round(scores[2], 4),
            "agreeableness": round(scores[3], 4),
            "neuroticism": round(scores[4], 4)
        }
    }
    print(json.dumps(output_data, indent=2))

except FileNotFoundError:
    print("Error: 'audio.wav' not found. Please add a mono audio file to the directory.")
except Exception as e:
    print(f"An error occurred: {e}")