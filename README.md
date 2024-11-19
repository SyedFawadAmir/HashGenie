# HashGenie

HashGenie is an AI-powered tool that transcribes audio/video files, analyzes content for sentiment, extracts keywords, categorizes topics, and generates context-based hashtags.

## Features
- **Transcription**: Converts audio/video files into text.
- **Sentiment Analysis**: Determines the emotional tone of the content.
- **Keyword Extraction**: Identifies key phrases from the content.
- **Topic Categorization**: Categorizes content into topics.
- **Hashtag Generation**: Creates hashtags based on the context.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/HashGenie.git
    cd HashGenie
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate     # On Windows
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Replace `'YOUR_OPENAI_API_KEY'` in `app.py` with your own OpenAI API key.
2. Run the app:
    ```bash
    streamlit run app.py
    ```

3. Upload an audio/video file and let HashGenie generate context-based hashtags, analyze sentiment, and more.

## Notes
- Ensure `ffmpeg` is installed and its path is added to the system environment.
- Replace `'YOUR_OPENAI_API_KEY'` in `app.py` with your OpenAI API key.

## License
[MIT License](LICENSE)
