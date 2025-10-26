from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import BertTokenizer, BertModel
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import soundfile as sf
import torch
import json
import librosa  


# NOTE: You will need to have an audio file named 'audio.wav' in the same directory.
# This audio file should be mono (single channel).

# 1. Loading the processors and models
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text_model = BertModel.from_pretrained("bert-base-uncased")

# 2. Main processing block
try:
    speech, sample_rate = sf.read("audio.wav")
    
    # RESAMPLING THE AUDIO
    if sample_rate != 16000:
        speech = librosa.resample(y=speech, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000 

    # PROCESS FOR TRANSCRIPTION(whisper-model)
    input_features = processor(speech, sampling_rate=sample_rate, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    transcription_text = transcription[0]

    # PROCESS FOR TEXT FEATURE(BERT-model)
    text_inputs = text_tokenizer(transcription_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_outputs = text_model(**text_inputs)
    text_features = text_outputs.last_hidden_state.mean(dim=1)

    # PROCESS FOR AUDIO FEATURE(wav2vec2-model)
    audio_inputs = audio_processor(speech, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        outputs = audio_model(**audio_inputs)
    audio_features = outputs.last_hidden_state.mean(dim=1)
    
    
   # --- FINAL OUTPUT ---
    print("\n--- Transcription ---")
    output_data = { "text": transcription_text }
    print(json.dumps(output_data, indent=2))

    print("\n--- Audio Features ---")
    print(f"Shape of the audio feature vector: {audio_features.shape}")

    print("\n--- Text Features ---")
    print(f"Shape of the text feature vector: {text_features.shape}")
    
except FileNotFoundError:
    print("Error: 'audio.wav' not found. Please add a mono audio file to the directory.")
except Exception as e:
    print(f"An error occurred: {e}")
