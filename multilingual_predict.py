#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import pickle
import re
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Text preprocessing function
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

def load_model_and_artifacts():
    """Load the trained model, tokenizer, and label mappings"""
    try:
        # Load the model
        print("Loading model...")
        model = tf.keras.models.load_model('multilingual_model.keras')
        
        # Load the tokenizer
        print("Loading tokenizer...")
        with open('multilingual_tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        # Load the label mappings
        print("Loading label mappings...")
        with open('multilingual_label_mappings.pickle', 'rb') as handle:
            label_mappings = pickle.load(handle)
            int_to_label = label_mappings['int_to_label']
            label_to_int = label_mappings['label_to_int']
        
        return model, tokenizer, int_to_label, label_to_int
    except Exception as e:
        print(f"Error loading model and artifacts: {e}")
        return None, None, None, None

def predict_emotion(text, model, tokenizer, int_to_label, max_sequence_length=60):
    """Predict the emotion of a given text"""
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Tokenize the text
    sequences = tokenizer.texts_to_sequences([processed_text])
    
    # Pad the sequence
    padded = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    
    # Make prediction
    prediction = model.predict(padded, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Get confidence score
    confidence = prediction[0][predicted_class]
    
    # Get top 3 emotions with confidences
    top_indices = np.argsort(prediction[0])[-3:][::-1]
    top_emotions = [(int_to_label[i], float(prediction[0][i])) for i in top_indices]
    
    # Return the emotion label, confidence, and top 3 emotions
    return int_to_label[predicted_class], confidence, top_emotions

def main():
    print("Multilingual Emotion Detection - Prediction Demo")
    print("===============================================")
    
    # Load model and artifacts
    print("\nLoading model and artifacts...")
    model, tokenizer, int_to_label, label_to_int = load_model_and_artifacts()
    
    if model is None:
        print("Failed to load model. Please make sure you've trained the model first.")
        return
    
    print("\nModel loaded successfully!")
    print(f"Available emotion classes: {sorted(list(int_to_label.values()))}")
    
    # Example texts in English and Marathi
    example_texts = [
        "I am so happy today!",
        "I'm feeling sad and depressed",
        "I'm angry about the situation",
        "The weather is nice today",
        "I'm surprised by the outcome of the match",
        "मला आज खूप आनंद झाला आहे",  # Marathi: I'm very happy today
        "मला खूप राग आला आहे",  # Marathi: I'm very angry
        "मी आज खूप दुःखी आहे",  # Marathi: I'm very sad today
        "मला आश्चर्य वाटलं",  # Marathi: I'm surprised
    ]
    
    # Predict emotions for example texts
    print("\nPredicting emotions for example texts:")
    for i, text in enumerate(example_texts):
        try:
            emotion, confidence, top_emotions = predict_emotion(text, model, tokenizer, int_to_label)
            print(f"\nText {i+1}: {text}")
            print(f"Predicted emotion: {emotion}")
            print(f"Confidence: {confidence:.4f}")
            print("Top 3 emotions:")
            for emotion, conf in top_emotions:
                print(f"  - {emotion}: {conf:.4f}")
        except Exception as e:
            print(f"\nError predicting for text {i+1}: {text}")
            print(f"Error: {e}")
    
    # Interactive mode
    print("\n\nInteractive Mode:")
    print("Enter text to predict emotion (type 'exit' to quit)")
    
    while True:
        user_input = input("\nEnter text: ")
        if user_input.lower() == 'exit':
            break
        
        try:
            emotion, confidence, top_emotions = predict_emotion(user_input, model, tokenizer, int_to_label)
            print(f"Predicted emotion: {emotion}")
            print(f"Confidence: {confidence:.4f}")
            print("Other possible emotions:")
            for emotion, conf in top_emotions[1:]:  # Skip the top one which we already showed
                print(f"  - {emotion}: {conf:.4f}")
        except Exception as e:
            print(f"Error predicting emotion: {e}")
    
    print("\nThank you for using the Multilingual Emotion Detection Model!")

if __name__ == "__main__":
    main() 