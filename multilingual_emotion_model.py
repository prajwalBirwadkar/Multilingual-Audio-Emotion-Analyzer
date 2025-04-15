#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import re
import string
import pickle
import os
import io
import zipfile
import requests

print("Starting Simplified Multilingual Emotion Detection Model")

# Configuration parameters
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 60
EMBEDDING_DIM = 50
BATCH_SIZE = 16
EPOCHS = 30
MIN_SAMPLES_PER_CLASS = 400
MAX_SAMPLES_PER_CLASS = 2000
LSTM_UNITS = 64  # Single LSTM with moderate units
RANDOM_SEED = 42
LEARNING_RATE = 0.0005

# Function to download and extract pre-trained embeddings
def download_and_extract_embeddings():
    """Download and extract GloVe embeddings"""
    glove_dir = "glove"
    os.makedirs(glove_dir, exist_ok=True)
    glove_path = os.path.join(glove_dir, "glove.6B.50d.txt")  # Using smaller dimension
    
    if not os.path.exists(glove_path):
        print("Downloading GloVe embeddings...")
        try:
            glove_url = "https://nlp.stanford.edu/data/glove.6B.zip"
            response = requests.get(glove_url)
            z = zipfile.ZipFile(io.BytesIO(response.content))
            z.extractall(glove_dir)
            print(f"GloVe embeddings downloaded to {glove_dir}")
        except Exception as e:
            print(f"Error downloading GloVe embeddings: {e}")
            return None
    
    return glove_path

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

# Main execution
try:
    # 1. Load Datasets
    print("Loading datasets...")
    english_df = pd.read_csv("English Dataset.csv")
    marathi_df = pd.read_csv("Marathi Dataset.csv")
    
    print(f"English dataset shape: {english_df.shape}")
    print(f"Marathi dataset shape: {marathi_df.shape}")
    
    # 2. Determine column names
    if 'content' in english_df.columns and 'sentiment' in english_df.columns:
        english_text_col = 'content'
        english_label_col = 'sentiment'
    else:
        # Default fallback
        english_text_col = english_df.columns[1]  # Assuming 2nd column is text
        english_label_col = english_df.columns[2]  # Assuming 3rd column is label
    
    if 'text' in marathi_df.columns and 'label' in marathi_df.columns:
        marathi_text_col = 'text'
        marathi_label_col = 'label'
    else:
        # Default to first and second column
        cols = marathi_df.columns.tolist()
        marathi_text_col = cols[0]  # First column
        marathi_label_col = cols[1]  # Second column
    
    print(f"English dataset: Using '{english_text_col}' for text and '{english_label_col}' for labels")
    print(f"Marathi dataset: Using '{marathi_text_col}' for text and '{marathi_label_col}' for labels")
    
    # 3. Apply preprocessing
    print("\nPreprocessing texts...")
    english_df['processed_text'] = english_df[english_text_col].apply(preprocess_text)
    marathi_df['processed_text'] = marathi_df[marathi_text_col].apply(preprocess_text)
    
    # 4. Create label mappings
    print("\nCreating label mappings...")
    english_labels = english_df[english_label_col].unique()
    marathi_labels = marathi_df[marathi_label_col].unique()
    
    print(f"English emotions: {english_labels}")
    print(f"Marathi emotions: {marathi_labels}")
    
    all_labels = sorted(set(list(english_labels) + list(marathi_labels)))
    label_to_int = {label: i for i, label in enumerate(all_labels)}
    int_to_label = {i: label for label, i in label_to_int.items()}
    
    print(f"Combined emotion labels: {all_labels}")
    print(f"Total emotion classes: {len(all_labels)}")
    
    english_df['label_int'] = english_df[english_label_col].map(label_to_int)
    marathi_df['label_int'] = marathi_df[marathi_label_col].map(label_to_int)
    
    # 5. Handle missing values
    english_df = english_df.dropna(subset=['processed_text', 'label_int'])
    marathi_df = marathi_df.dropna(subset=['processed_text', 'label_int'])
    
    # Shuffle each dataset multiple times with different random seeds for better randomization
    print("\nPerforming thorough shuffling of datasets...")
    # First shuffle with random_state=42
    english_df = english_df.sample(frac=1, random_state=42).reset_index(drop=True)
    marathi_df = marathi_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Second shuffle with a different random seed
    english_df = english_df.sample(frac=1, random_state=101).reset_index(drop=True)
    marathi_df = marathi_df.sample(frac=1, random_state=101).reset_index(drop=True)
    
    # Final shuffle with the RANDOM_SEED constant
    english_df = english_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    marathi_df = marathi_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    print(f"English dataset shuffled thoroughly: {english_df.shape}")
    print(f"Marathi dataset shuffled thoroughly: {marathi_df.shape}")
    
    # 6. Create a balanced dataset
    print("\nCreating balanced dataset...")
    
    # Sample equal amounts from both languages first
    english_samples = min(12000, len(english_df))
    marathi_samples = min(4000, len(marathi_df))
    
    english_sample = english_df.sample(english_samples, random_state=RANDOM_SEED)
    marathi_sample = marathi_df.sample(marathi_samples, random_state=RANDOM_SEED)
    
    # Combine samples with interleaving for better mixing
    print("Interleaving English and Marathi datasets for better mixing...")
    
    # Process to interleave the datasets
    # First, split both datasets into chunks
    n_chunks = 10  # Number of chunks to split each dataset into
    english_chunks = np.array_split(english_sample, n_chunks)
    marathi_chunks = np.array_split(marathi_sample, n_chunks)
    
    # Interleave the chunks
    interleaved_dfs = []
    for e_chunk, m_chunk in zip(english_chunks, marathi_chunks):
        interleaved_dfs.append(e_chunk)
        interleaved_dfs.append(m_chunk)
    
    # Combine the interleaved chunks
    combined_df = pd.concat(interleaved_dfs, ignore_index=True)
    
    # Shuffle again after interleaving
    combined_df = combined_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    print(f"Combined interleaved dataset shape: {combined_df.shape}")
    
    # Check class distribution
    class_distribution = combined_df['label_int'].value_counts()
    print("Initial class distribution:")
    print(class_distribution)
    
    # Balance classes
    balanced_dfs = []
    for label, count in class_distribution.items():
        # Get all samples for this class
        class_df = combined_df[combined_df['label_int'] == label]
        
        if count > MAX_SAMPLES_PER_CLASS:
            # Downsample
            class_df = class_df.sample(MAX_SAMPLES_PER_CLASS, random_state=RANDOM_SEED)
        elif count < MIN_SAMPLES_PER_CLASS:
            # Upsample by duplicating with small variations
            if count > 0:
                # Calculate multiplier for upsampling
                n_samples = min(MIN_SAMPLES_PER_CLASS, count * 3)  # Don't create too many duplicates
                
                # Create upsampled data with replacement
                upsampled = class_df.sample(n_samples, replace=True, random_state=RANDOM_SEED)
                class_df = upsampled
        
        balanced_dfs.append(class_df)
    
    # Combine balanced data
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Apply multiple rounds of shuffling to the balanced dataset
    print("\nThoroughly shuffling the combined balanced dataset...")
    # First shuffle
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    # Second shuffle with different seed
    balanced_df = balanced_df.sample(frac=1, random_state=101).reset_index(drop=True)
    # Final shuffle with RANDOM_SEED
    balanced_df = balanced_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    print(f"Balanced dataset thoroughly shuffled: {balanced_df.shape}")
    print("Balanced class distribution:")
    print(balanced_df['label_int'].value_counts())
    
    # 7. Tokenization
    print("\nTokenizing text...")
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(balanced_df['processed_text'])
    
    vocab_size = min(MAX_VOCAB_SIZE, len(tokenizer.word_index) + 1)
    print(f"Vocabulary size: {vocab_size}")
    
    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences(balanced_df['processed_text'])
    
    # Pad sequences
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    print(f"Padded sequences shape: {padded_sequences.shape}")
    
    # 8. Split data with additional shuffling
    print("\nPerforming stratified split with shuffling...")
    labels = balanced_df['label_int'].values
    
    # Create indices array and shuffle it
    indices = np.arange(padded_sequences.shape[0])
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)
    
    # Apply shuffling to both sequences and labels
    shuffled_sequences = padded_sequences[indices]
    shuffled_labels = labels[indices]
    
    # Now perform stratified split on the shuffled data
    X_train, X_test, y_train, y_test = train_test_split(
        shuffled_sequences, 
        shuffled_labels, 
        test_size=0.2, 
        random_state=RANDOM_SEED, 
        stratify=shuffled_labels  # Use stratified sampling to maintain class distribution
    )
    
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    
    # 9. Compute class weights
    print("\nComputing class weights...")
    classes = np.unique(y_train)
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # 10. Load GloVe embeddings
    print("\nLoading GloVe embeddings...")
    glove_path = download_and_extract_embeddings()
    
    # Create embedding matrix
    embedding_matrix = np.random.normal(0, 0.1, (vocab_size, EMBEDDING_DIM))  # Small random initialization
    
    if glove_path:
        # Load embeddings from file
        embeddings_index = {}
        with open(glove_path, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        
        print(f"Loaded {len(embeddings_index)} word vectors")
        
        # Fill embedding matrix with GloVe vectors
        found_words = 0
        for word, i in tokenizer.word_index.items():
            if i >= vocab_size:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                found_words += 1
        
        print(f"Found GloVe embeddings for {found_words}/{min(vocab_size, len(tokenizer.word_index))} words")
    else:
        print("Using random embeddings")
    
    # 11. Build the model
    print("\nBuilding the model...")
    num_classes = len(all_labels)
    
    model = Sequential([
        # Embedding layer - TRAINABLE for adapting to both languages
        Embedding(
            input_dim=vocab_size,
            output_dim=EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_SEQUENCE_LENGTH,
            trainable=True,  # Make it trainable
            mask_zero=True  # Support variable length inputs
        ),
        
        # Spatial dropout for better regularization of embeddings
        SpatialDropout1D(0.2),
        
        # Single LSTM layer (simplified from multiple)
        LSTM(LSTM_UNITS),
        
        # Dropout for regularization
        Dropout(0.3),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model with small learning rate and gradient clipping
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        metrics=['accuracy']
    )
    
    # Model summary
    model.summary()
    
    # 12. Train the model
    print("\nTraining the model...")
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        'multilingual_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=7,  # Increased patience
        restore_best_weights=True,
        verbose=1
    )
    
    # Reduce learning rate when plateau occurs
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.0001,
        verbose=1
    )
    
    # Train model with more epochs
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # 13. Evaluate model
    print("\nEvaluating the model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    
    # 14. Save artifacts
    print("\nSaving model artifacts...")
    
    # Save tokenizer
    with open('multilingual_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save label mappings
    with open('multilingual_label_mappings.pickle', 'wb') as handle:
        pickle.dump({
            'label_to_int': label_to_int,
            'int_to_label': int_to_label
        }, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # 15. Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('multilingual_training_history.png')
    
    # 16. Make some sample predictions
    print("\nMaking sample predictions...")
    sample_texts = [
        "I am so happy today!",
        "I'm feeling sad and depressed",
        "मला आज खूप आनंद झाला आहे",  # Marathi: I'm very happy today
        "मी आज खूप दुःखी आहे"  # Marathi: I'm very sad today
    ]
    
    # Preprocess and tokenize
    processed_texts = [preprocess_text(text) for text in sample_texts]
    sequences = tokenizer.texts_to_sequences(processed_texts)
    padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    
    # Make predictions
    predictions = model.predict(padded)
    pred_classes = np.argmax(predictions, axis=1)
    
    # Print results
    for i, text in enumerate(sample_texts):
        pred_class = pred_classes[i]
        confidence = predictions[i][pred_class]
        emotion = int_to_label[pred_class]
        print(f"Text: {text}")
        print(f"Predicted emotion: {emotion}")
        print(f"Confidence: {confidence:.4f}")
        print("---")
    
    print("\nMultilingual Emotion Detection Model completed successfully!")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 