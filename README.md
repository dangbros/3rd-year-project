# Speech-to-Text + Personality Prediction Model

This project uses AI models (Whisper, BERT, wav2vec2) to extract features from speech and trains a model to predict the Big Five personality traits (OCEAN).

## Project Structure

* `transcribe.py`: Transcribes a single audio file and predicts personality traits using a pre-trained model.
* `preprocess.py`: Extracts text/audio features from the manually downloaded dataset (computationally intensive, GPU recommended). Assumes data is in a `datasets` subfolder.
* `train.py`: Trains the final personality prediction model using the preprocessed features (runs quickly on CPU).
* `requirements.txt`: Lists all necessary Python libraries.
* `datasets/`: Folder containing the unzipped Kaggle data (created manually during training setup).
* `personality_model_final.pth`: The pre-trained model weights (created after training).

---

## Running the Pre-trained Model (Inference)

These instructions explain how to use the already trained model (`personality_model_final.pth`) to predict personality traits from a new audio file.

### 1. Setup

* **Clone the repository:**
    ```bash
    git clone [https://github.com/dangbros/3rd-year-project.git](https://github.com/dangbros/3rd-year-project.git)
    cd 3rd-year-project
    ```
* **Create and activate a virtual environment:**
    * **Linux/macOS**:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    * **Windows**:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
* **Install required packages:**
    ```bash
    pip install -r requirements.txt
    ```
* **Ensure you have the trained model:** Make sure the file `personality_model_final.pth` is present in the main project folder. If you haven't trained it yourself, you might need to download it (e.g., using `git pull` if a collaborator trained it and pushed the file).

### 2. Prepare Audio File

* Place your audio file in the main project folder.
* Rename it to `audio.wav`.
* The audio file must be **mono** (single channel).

### 3. Run Prediction

* Make sure your virtual environment is active.
* Execute the script:
    ```bash
    python transcribe.py
    ```
* The script will print the transcription and the predicted OCEAN traits in JSON format.

---

## Training the Model (for project team members)

These instructions are for training the model from scratch using the Kaggle dataset. This involves a **very long preprocessing step** (Part 1) that is best done on a computer with a **powerful GPU**.

### Part 1: Preprocessing (GPU Recommended)

This step downloads the large dataset (18GB) and extracts features. **This takes many hours.**

1.  **General Setup:** Ensure you've completed the "Getting Started (General Setup for Both Users)" steps from above (cloning, virtual environment, installing requirements).
2.  **Download and Unzip Data Manually:**
    * Go to the Kaggle dataset page: [https://www.kaggle.com/datasets/animesh2cool/first-impressions-v2-cvpr17-training](https://www.kaggle.com/datasets/animesh2cool/first-impressions-v2-cvpr17-training)
    * Click the **"Download (18 GB)"** button.
    * Once downloaded, **create a folder named `datasets`** inside the main project folder (`3rd-year-project`).
    * Move the downloaded `.zip` file into the new `datasets` folder.
    * **Unzip** the file *inside* the `datasets` folder. This will create folders like `train-annotation`, `train-transcription`, `train-1`, etc., within the `datasets` directory.
3.  **Run the Preprocessing Script:**
    * Make sure your virtual environment is active.
    * Execute the script from the main project folder:
        ```bash
        python preprocess.py
        ```
    * This script will look inside the `datasets` folder, find the unzipped data, and start processing all 6000 training files. It will create two files in the main project folder: `train_features.pt` and `train_labels.pt`.

### Part 2: Training (CPU is Fine)

Once the preprocessed files (`train_features.pt` and `train_labels.pt`) are available, you can train the final model.

1.  **Run the Training Script:**
    * Make sure your virtual environment is active.
    * Execute the script:
        ```bash
        python train.py
        ```
    * This script runs quickly, loads the `.pt` files, trains the model, and saves the final weights as `personality_model_final.pth`.

### Part 3: Uploading Results (If Collaborating)

If one person runs the preprocessing (Part 1), they need to share the results:

1.  **Upload the `.pt` files to GitHub:**
    ```bash
    git add train_features.pt train_labels.pt
    git commit -m "Add processed features and labels"
    git push origin main
    ```
2.  **Upload the final model:** After running Part 2, upload