import os
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau

# Import the model creation function from the neighboring model.py file
from models.self_attention.model import create_model

# --- HYPERPARAMETERS ---
# These parameters should match the ones used in model.py for consistency
VOCAB_SIZE = 30000
MAX_LEN = 256
EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 128
NUM_TRANSFORMER_BLOCKS = 4
DROPOUT_RATE = 0.1

# Training specific parameters
EPOCHS = 50
BATCH_SIZE = 32

# --- DATA LOADING ---

def load_data_from_jsonl(file_path, max_len):
    """
    Loads data from a .jsonl file, extracts 'input_ids' and 'label',
    and pads the sequences to a uniform length.
    """
    sequences = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            sequences.append(data['input_ids'])
            labels.append(data['target'])
            
    # Pad sequences to ensure uniform length
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return np.array(padded_sequences), np.array(labels)

# --- PLOTTING ---

def plot_training_history(history, output_path):
    """
    Plots the training and validation history for metrics and saves it to a file.
    """
    # Determine which metrics are available in the history
    metrics_to_plot = [key for key in history.history.keys() if not key.startswith('val_')]
    
    num_plots = len(metrics_to_plot)
    plt.figure(figsize=(12, 5 * num_plots))

    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(num_plots, 1, i + 1)
        plt.plot(history.history[metric], label=f'Training {metric.capitalize()}')
        if f'val_{metric}' in history.history:
            plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
        plt.title(f'Model {metric.capitalize()}')
        plt.ylabel(metric.capitalize())
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nTraining history plot saved to {output_path}")

# --- MAIN EXECUTION ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Transformer model for vulnerability detection.')
    parser.add_argument('--data_dir', type=str, default='data/preprocessed/encoded_balanced', help='Directory containing the train, val, and test .jsonl files.')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save trained models and logs.')
    args = parser.parse_args()

    # --- 1. Load Data ---
    print("Loading and preparing data...")
    train_file = os.path.join(args.data_dir, 'train_balanced_01.jsonl')
    val_file = os.path.join(args.data_dir, 'val.jsonl')

    X_train, y_train = load_data_from_jsonl(train_file, MAX_LEN)
    X_val, y_val = load_data_from_jsonl(val_file, MAX_LEN)
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")

    # --- 2. Build and Compile Model ---
    print("\nBuilding and compiling model...")
    model = create_model(
        maxlen=MAX_LEN,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
        dropout_rate=DROPOUT_RATE
    )
    # The model is compiled inside the create_model function in model.py
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy", 
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    # --- 3. Setup Callbacks ---
    # Define paths for saving artifacts
    best_model_path = os.path.join(args.output_dir, 'best_model.h5')
    final_model_path = os.path.join(args.output_dir, 'final_model.h5')
    csv_log_path = os.path.join(args.output_dir, 'training_log.csv')
    history_plot_path = os.path.join(args.output_dir, 'training_history.png')

    callbacks = [
        ModelCheckpoint(filepath=best_model_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1),
        EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=1),
        CSVLogger(csv_log_path),
        # ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ]

    # --- 4. Run Training ---
    print("\n--- Starting Training ---")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    print("--- Training Finished ---")

    # --- 5. Save Artifacts ---
    print(f"\nSaving final model to {final_model_path}...")
    model.save(final_model_path)
    
    plot_training_history(history, history_plot_path)
