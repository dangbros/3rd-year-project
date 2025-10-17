# Speech-to-Text Transcription Model

This project uses the OpenAI Whisper model to transcribe speech from an audio file into text.

## Getting Started

Follow these instructions to get a copy of the project running on your local machine.

### Prerequisites

* Python 3
* The `venv` module for Python

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/dangbros/3rd-year-project.git
    ```

2.  **Navigate into the project directory:**
    ```bash
    cd 3rd-year-project
    ```

3.  **Create and activate a virtual environment:**
    * For **Linux/macOS**:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    * For **Windows**:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

4.  **Install the required packages:**
    ```bash
    pip install transformers torch soundfile accelerate librosa
    ```

## How to Use

1.  Place your audio file in the main project folder and make sure it is named `audio.wav`. The audio file should be in a mono format.

2.  Run the script from your terminal:
    ```bash
    python transcribe.py
    ```

3.  The script will print the transcribed text in a JSON format to the terminal.
