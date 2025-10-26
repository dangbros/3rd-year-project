# Speech-to-Text + Personality Prediction Model

This project uses AI models (Whisper, BERT, wav2vec2) to extract features from speech and trains a model to predict the Big Five personality traits (OCEAN).

## Project Structure

* `preprocess.py`: Extracts text/audio features from the manually downloaded dataset (computationally intensive, GPU recommended). Assumes data is in a `datasets` subfolder.
* `train.py`: Trains the final personality prediction model using the preprocessed features (runs quickly on CPU).
* `transcribe.py`: Transcribes a single audio file and predicts personality traits using the trained model.
* `requirements.txt`: Lists all necessary Python libraries.
* `datasets/`: Folder containing the unzipped Kaggle data (created manually).

## Getting Started (General Setup for Both Users)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/dangbros/3rd-year-project.git
    ```
2.  **Navigate into the project directory:**
    ```bash
    cd 3rd-year-project
    ```
3.  **Create and activate a virtual environment:**
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
4.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Part 1: Preprocessing (GPU Recommended - For Collaborator)

This step uses the downloaded dataset files to extract features. **This takes many hours and requires a good GPU for reasonable speed.**

1.  **Download and Unzip Data Manually:**
    * Go to the Kaggle dataset page: [https://www.kaggle.com/datasets/animesh2cool/first-impressions-v2-cvpr17-training](https://www.kaggle.com/datasets/animesh2cool/first-impressions-v2-cvpr17-training)
    * Click the **"Download (18 GB)"** button.
    * Once downloaded, **create a folder named `datasets`** inside the main project folder (`3rd-year-project`).
    * Move the downloaded `.zip` file into the new `datasets` folder.
    * **Unzip** the file *inside* the `datasets` folder. This will create folders like `train-annotation`, `train-transcription`, `train-1`, etc., within the `datasets` directory.

2.  **Run the Preprocessing Script:**
    * Make sure your virtual environment is active.
    * Execute the script from the main project folder:
        ```bash
        python preprocess.py
        ```
    * This script will look inside the `datasets` folder, find the unzipped data, and start processing. Expect this to run for a **long time** (hours, potentially). It will create two files in the main project folder: `train_features.pt` and `train_labels.pt`.

3.  **Upload the Results to GitHub:**
    * After `preprocess.py` finishes, add the two new `.pt` files to Git:
        ```bash
        git add train_features.pt train_labels.pt
        ```
    * Commit the files:
        ```bash
        git commit -m "Add processed features and labels"
        ```
    * Push them to the repository:
        ```bash
        git push origin main
        ```

---

## Part 2: Training the Model (CPU - For Project Owner)

Once the preprocessed files (`train_features.pt` and `train_labels.pt`) are available (either from Part 1 or downloaded from GitHub using `git pull`), you can train the final model.

1.  **Make sure you have the latest files:**
    ```bash
    git pull origin main
    ```
2.  **Run the Training Script:**
    * Make sure your virtual environment is active.
    * Execute the script:
        ```bash
        python train.py
        ```
    * This script runs quickly, loads the `.pt` files, trains the model, and saves the final weights as `personality_model_final.pth`.

---

## Part 3: Running Inference

To transcribe a new audio file and predict its personality traits using the trained model:

1.  Make sure `personality_model_final.pth` exists (created by `train.py`).
2.  Place your audio file in the main project folder and name it `audio.wav` (it should be mono).
3.  Run the script:
    ```bash
    python transcribe.py
    ```
4.  The script will print the transcription and the predicted OCEAN traits in JSON format.