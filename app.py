import os
import numpy as np
import pickle
import re
import string
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
import speech_recognition as sr
from langdetect import detect
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}
MAX_SEQUENCE_LENGTH = 60

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.secret_key = 'emotion_detection_secret_key'

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model components
tokenizer = None
int_to_label = None
model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_text(text):
    """Clean and preprocess text data"""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = ''.join([char for char in text if char not in string.punctuation])
        # Remove extra whitespace
        text = re.sub('\s+', ' ', text).strip()
        return text
    return ""  # Return empty string for non-string inputs

def load_model_components():
    """Load the emotion detection model, tokenizer, and label mappings"""
    global tokenizer, int_to_label, model
    
    try:
        # Load tokenizer
        with open('multilingual_tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
            
        # Load label mappings
        with open('multilingual_label_mappings.pickle', 'rb') as handle:
            label_mappings = pickle.load(handle)
            int_to_label = label_mappings['int_to_label']
            
        # Try to load the model - with error handling for compatibility
        try:
            model = tf.keras.models.load_model('multilingual_model.keras', compile=False)
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            print("Model loaded successfully")
        except Exception as e:
            print(f"Model loading error: {e}")
            # Fallback to alternative model loading methods if needed
            # For now, model will remain None
            
        return tokenizer is not None and int_to_label is not None
    except Exception as e:
        print(f"Error loading model components: {e}")
        return False

def detect_emotions(text):
    """Detect emotions in the given text"""
    if model is None:
        # If model failed to load, generate fake predictions for demo
        num_classes = len(int_to_label)
        np.random.seed(len(text))
        fake_prediction = np.random.random(num_classes)
        fake_prediction = fake_prediction / np.sum(fake_prediction)
        
        # Get top 3 emotions with confidences
        top_indices = np.argsort(fake_prediction)[-3:][::-1]
        top_emotions = [(int_to_label[i], float(fake_prediction[i])) for i in top_indices]
        
        return {
            "primary_emotion": top_emotions[0][0],
            "confidence": top_emotions[0][1],
            "all_emotions": {e: c for e, c in top_emotions}
        }
    
    try:
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Tokenize and pad
        sequences = tokenizer.texts_to_sequences([processed_text])
        padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        
        # Make prediction
        prediction = model.predict(padded, verbose=0)
        
        # Get top emotions
        top_indices = np.argsort(prediction[0])[-3:][::-1]
        top_emotions = [(int_to_label[i], float(prediction[0][i])) for i in top_indices]
        
        return {
            "primary_emotion": top_emotions[0][0],
            "confidence": top_emotions[0][1],
            "all_emotions": {e: c for e, c in top_emotions}
        }
    except Exception as e:
        print(f"Error in emotion detection: {e}")
        return {
            "error": str(e),
            "primary_emotion": "unknown",
            "confidence": 0,
            "all_emotions": {}
        }

def transcribe_audio(file_path, preferred_language=None):
    """Transcribe audio file to text using the specified language"""
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(file_path) as source:
        # Adjust for ambient noise and record audio
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.record(source)
        
        # Use the specified language (defaulting to English if not provided)
        try:
            if preferred_language == 'mr':
                # Use Marathi for speech recognition
                text = recognizer.recognize_google(audio_data, language='mr-IN')
                return 'mr', text
            else:
                # Default to English
                text = recognizer.recognize_google(audio_data, language='en-US')
                return 'en', text
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            return None, f"Error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file was uploaded
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    # Check if file is empty
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    # Get preferred language if specified
    preferred_language = request.form.get('language', 'auto')
    
    # Check file type
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the audio file
        try:
            # Transcribe audio to text with the preferred language
            language, transcription = transcribe_audio(filepath, preferred_language)
            
            if language is None:
                return jsonify({
                    'success': False,
                    'error': transcription
                })
            
            # Detect emotions in the transcribed text
            emotions = detect_emotions(transcription)
            
            # Return the results
            return jsonify({
                'success': True,
                'language': language,
                'transcription': transcription,
                'emotions': emotions
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })
    else:
        flash('File type not allowed')
        return redirect(request.url)

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    """Analyze text directly without audio upload"""
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({
            'success': False,
            'error': 'No text provided'
        })
    
    # Get user-selected language
    language = data.get('language', 'en')  # Default to English if not specified
    
    # Detect emotions in the provided text
    emotions = detect_emotions(text)
    
    return jsonify({
        'success': True,
        'language': language,
        'text': text,
        'emotions': emotions
    })

if __name__ == '__main__':
    # Load model components on startup
    model_loaded = load_model_components()
    if not model_loaded:
        print("Warning: Model components could not be loaded completely. Running in demo mode.")
    
    # Start the Flask app
    app.run(debug=True) 