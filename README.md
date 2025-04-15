# Multilingual Audio Emotion Analyzer

This application analyzes audio files to detect emotions in both English and Marathi speech. It combines speech recognition, language detection, and emotion analysis into a single web interface.

## Features

- **Audio Input**: Upload audio files (WAV, MP3, OGG, FLAC) or record directly from the browser
- **Text Input**: Directly input text for emotion analysis without audio
- **Language Detection**: Automatically detects whether the speech is in English or Marathi
- **Speech-to-Text**: Transcribes audio content to text
- **Emotion Analysis**: Analyzes the text to identify emotions using a trained multilingual model
- **Visual Results**: Displays detected emotions with confidence scores and visual bars

## Architecture

1. **Web Interface**: Built with Flask, HTML, CSS, and JavaScript
2. **Audio Processing**: Uses SpeechRecognition library to convert audio to text
3. **Language Detection**: Identifies the language using langdetect
4. **Emotion Detection**: Utilizes a pre-trained TensorFlow model for emotion detection
5. **Results Display**: Shows transcription and emotion analysis results in the browser

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd multilingual-audio-emotion-analyzer
```

2. Create a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Make sure you have the required model files:
   - `multilingual_model.keras` (or similar model file)
   - `multilingual_tokenizer.pickle`
   - `multilingual_label_mappings.pickle`

## Usage

1. Start the Flask application:
```
python app.py
```

2. Open a web browser and navigate to:
```
http://127.0.0.1:5000/
```

3. Use the application:
   - Upload an audio file or record directly in the browser
   - Alternatively, input text directly for analysis
   - View the results showing language, transcription, and detected emotions

## System Requirements

- **Python**: 3.8 or higher
- **OS**: Windows, macOS, or Linux
- **Memory**: At least 4GB RAM recommended
- **Disk Space**: At least 500MB for model files and dependencies

## Technologies Used

- **Backend**: Flask, Python
- **ML/AI**: TensorFlow, SpeechRecognition, langdetect
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Audio**: Web Audio API, recorder.js

## Model Information

The emotion detection model is a multilingual model trained on both English and Marathi text data. It uses:

- GloVe word embeddings (trainable)
- LSTM neural networks
- Dropout regularization
- Class weighting for balanced predictions

The model can detect the following emotions:
- Anger/Angry
- Happiness/Happy
- Sadness/Sad
- Surprise
- Fear
- Neutral
- Love
- And more depending on the training data

## Limitations

- Speech recognition accuracy may vary based on accent, voice clarity, and background noise
- Language detection works best for clearly spoken English or Marathi
- Audio recording in the browser requires microphone permissions
- Model accuracy depends on the quality and diversity of the training data

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow team for the machine learning framework
- SpeechRecognition library for audio processing capabilities
- Bootstrap team for the frontend framework
- The open-source community for various tools and libraries used 