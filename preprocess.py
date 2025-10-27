# SCRIPT 1: preprocess.py 

import pickle
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model, BertTokenizer, BertModel
from moviepy import VideoFileClip 
import os
import time
import numpy

DATA_FOLDER = 'datasets'

LABEL_FILE_PATH = os.path.join(DATA_FOLDER, 'train-annotation/annotation_training.pkl')
TRANS_FILE_PATH = os.path.join(DATA_FOLDER, 'train-transcription/transcription_training.pkl')

# --- Helper Function to Extract Audio ---
def extract_audio(video_path, output_audio_path="temp_audio.wav"):
    try:
        video_clip = VideoFileClip(video_path)
        # Use fps=16000, remove audio_codec
        video_clip.audio.write_audiofile(output_audio_path, fps=16000, nbytes=2, logger=None)
        video_clip.close()
        return output_audio_path
    except Exception as e:
        print(f"  [Audio Error] Could not extract audio from {video_path}: {e}")
        return None

# --- Helper Function to Find Video Files ---
# --- Helper Function to Find Video Files ---
def find_video_file(root_search_path, video_filename):
    """
    Searches all subdirectories of 'root_search_path' for 'video_filename'.
    Relies on os.walk starting from the correct root path (e.g., 'datasets').
    """
    for dirpath, _, filenames in os.walk(root_search_path):
        if video_filename in filenames:
            # Found the file, return its full path
            return os.path.join(dirpath, video_filename)
    
    # If the loop finishes without finding the file, return None
    return None

# --- Main Preprocessing Function ---
def preprocess_and_save(device):

    # --- 1. Load Feature Models ---
    print("Loading feature extraction models (BERT, Wav2Vec2)...")
    audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text_model = BertModel.from_pretrained("bert-base-uncased").to(device)
    print("Models loaded and moved to device.")

    # --- 2. Load Label & Transcription Files ---
    print("Loading .pkl files for labels and transcriptions...")
    try:
        # Load label file with 'latin1' encoding
        with open(LABEL_FILE_PATH, 'rb') as f:
            labels_dict = pickle.load(f, encoding='latin1')

        # Load transcription file
        with open(TRANS_FILE_PATH, 'rb') as f:
            trans_dict = pickle.load(f)

        print(f"Successfully loaded {len(labels_dict)} label categories and {len(trans_dict)} transcriptions.")
    except FileNotFoundError:
        print("CRITICAL ERROR: Could not find .pkl files.")
        print(f"Make sure you manually downloaded and unzipped the Kaggle dataset into the '{DATA_FOLDER}' subfolder.")
        print(f"Looking for: {LABEL_FILE_PATH}")
        return

    all_features = []
    all_labels = []

    video_names = list(trans_dict.keys())
    total_videos = len(video_names)

    print(f"--- Starting FULL feature extraction for {total_videos} files ---")
    start_time = time.time()

    # --- 3. Loop, Extract, and Save ---
    files_processed = 0
    for index, video_name in enumerate(video_names):

        print(f"Processing file {index+1}/{total_videos}: {video_name}")
        try:
            # --- Get Text & Labels from Dictionaries ---
            text = trans_dict.get(video_name)

            if video_name not in labels_dict['openness']:
                print(f"Skipping {video_name}: Labels not found in annotation file.")
                continue

            labels_list = [
                labels_dict['openness'][video_name],
                labels_dict['conscientiousness'][video_name],
                labels_dict['extraversion'][video_name],
                labels_dict['agreeableness'][video_name],
                labels_dict['neuroticism'][video_name]
            ]

            if text is None:
                print(f"Skipping {video_name}: Missing transcription.")
                continue

            # --- Find the video file ---
            # Start search inside the DATA_FOLDER ('database')
            video_path = find_video_file(DATA_FOLDER, video_name)
            if video_path is None:
                print(f"Skipping {video_name}: Video file not found. Ensure data is unzipped correctly.")
                continue

            # --- Get Text Features ---
            text_inputs = text_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                text_outputs = text_model(**text_inputs)
            text_feat = text_outputs.last_hidden_state.mean(dim=1)

            # --- Get Audio Features ---
            audio_file_path = extract_audio(video_path)
            if audio_file_path is None:
                continue

            speech, sample_rate = librosa.load(audio_file_path, sr=16000, mono=True)
            os.remove(audio_file_path) # Clean up temp file

            audio_inputs = audio_processor(speech, sampling_rate=16000, return_tensors="pt").to(device)
            with torch.no_grad():
                audio_outputs = audio_model(**audio_inputs)
            audio_feat = audio_outputs.last_hidden_state.mean(dim=1)

            # --- Get Labels ---
            labels = torch.tensor(labels_list, dtype=torch.float32)

            # --- Concatenate & Store (on CPU) ---
            combined_feat = torch.cat((text_feat.cpu(), audio_feat.cpu()), dim=1)
            all_features.append(combined_feat)
            all_labels.append(labels.unsqueeze(0))

            files_processed += 1

            if (index + 1) % 50 == 0: # Print progress every 50 files
                 print(f"  Completed {index + 1} / {total_videos}...")


        except Exception as e:
            print(f"Skipping file {video_name} due to unexpected error: {e}")

    # --- 4. Save to File ---
    if not all_features:
        print("No features were extracted. Exiting.")
        return

    final_features = torch.cat(all_features, dim=0)
    final_labels = torch.cat(all_labels, dim=0)

    torch.save(final_features, "train_features.pt")
    torch.save(final_labels, "train_labels.pt")

    end_time = time.time()
    print("\n--- 🚀 FULL feature extraction complete! ---")
    print(f"Processed {files_processed} files.")
    print(f"Total time: {(end_time - start_time) / 3600:.2f} hours") # Display time in hours
    print(f"Saved features to train_features.pt (Shape: {final_features.shape})")
    print(f"Saved labels to train_labels.pt (Shape: {final_labels.shape})")


# --- Main execution ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: No GPU detected, using CPU. This will be extremely slow.")

    preprocess_and_save(device)