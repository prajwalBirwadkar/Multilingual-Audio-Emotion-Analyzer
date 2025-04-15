#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import pickle
import re
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

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

def load_tokenizer_and_labels():
    """Load tokenizer and label mappings only"""
    try:
        # Check if files exist
        if not os.path.exists('multilingual_tokenizer.pickle'):
            print("Tokenizer file not found")
            return None, None
            
        if not os.path.exists('multilingual_label_mappings.pickle'):
            print("Label mappings file not found")
            return None, None
            
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
        
        return tokenizer, int_to_label
    except Exception as e:
        print(f"Error loading tokenizer and labels: {e}")
        return None, None

def main():
    print("Simple Multilingual Emotion Detection - Prediction Demo")
    print("====================================================")
    
    # Load tokenizer and label mappings
    print("\nLoading tokenizer and label mappings...")
    tokenizer, int_to_label = load_tokenizer_and_labels()
    
    if tokenizer is None or int_to_label is None:
        print("Failed to load required components. Please make sure you've trained the model first.")
        return
    
    print("\nComponents loaded successfully!")
    print(f"Available emotion classes: {sorted(list(int_to_label.values()))}")
    
    print("\nNote: Using random predictions for demonstration since model loading failed")
    
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
    
    # Process texts
    processed_texts = [preprocess_text(text) for text in example_texts]
    
    # Predict emotions (randomly for demonstration)
    print("\nPredicting emotions for example texts (simulated results):")
    for i, text in enumerate(example_texts):
        try:
            # Create a random prediction
            np.random.seed(i)  # For reproducibility
            num_classes = len(int_to_label)
            fake_prediction = np.random.random(num_classes)
            fake_prediction = fake_prediction / np.sum(fake_prediction)  # Normalize to sum to 1
            
            # Get top class
            pred_class = np.argmax(fake_prediction)
            confidence = fake_prediction[pred_class]
            
            # Get top 3 classes
            top_indices = np.argsort(fake_prediction)[-3:][::-1]
            top_emotions = [(int_to_label[i], float(fake_prediction[i])) for i in top_indices]
            
            print(f"\nText {i+1}: {text}")
            print(f"Processed: {processed_texts[i]}")
            print(f"Predicted emotion: {int_to_label[pred_class]}")
            print(f"Confidence: {confidence:.4f}")
            print("Top 3 emotions:")
            for emotion, conf in top_emotions:
                print(f"  - {emotion}: {conf:.4f}")
                
        except Exception as e:
            print(f"\nError processing text {i+1}: {text}")
            print(f"Error: {e}")
    
    # Interactive mode
    print("\n\nInteractive Mode:")
    print("Enter text to get a simulated prediction (type 'exit' to quit)")
    
    while True:
        user_input = input("\nEnter text: ")
        if user_input.lower() == 'exit':
            break
        
        try:
            # Process text
            processed = preprocess_text(user_input)
            
            # Create a random prediction based on text length (for demonstration)
            np.random.seed(len(processed))
            num_classes = len(int_to_label)
            fake_prediction = np.random.random(num_classes)
            fake_prediction = fake_prediction / np.sum(fake_prediction)
            
            # Get top class
            pred_class = np.argmax(fake_prediction)
            confidence = fake_prediction[pred_class]
            
            # Get top 3 classes
            top_indices = np.argsort(fake_prediction)[-3:][::-1]
            top_emotions = [(int_to_label[i], float(fake_prediction[i])) for i in top_indices]
            
            print(f"Processed: {processed}")
            print(f"Predicted emotion: {int_to_label[pred_class]}")
            print(f"Confidence: {confidence:.4f}")
            print("Other possible emotions:")
            for emotion, conf in top_emotions[1:]:  # Skip the top one which we already showed
                print(f"  - {emotion}: {conf:.4f}")
                
        except Exception as e:
            print(f"Error processing text: {e}")
    
    print("\nThank you for using the Multilingual Emotion Detection Model!")

if __name__ == "__main__":
    main() 