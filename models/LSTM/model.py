import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.mixed_precision import Policy, set_global_policy
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    print("Using GPU:", physical_devices[0])
    set_global_policy(Policy('mixed_float16'))
else:
    print("No GPU detected, falling back to CPU.")


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
ENCODED_DIR = os.path.join(ROOT_DIR, "data", "preprocessed", "encoded")
VOCAB_PATH = os.path.join(ROOT_DIR, "data", "preprocessed", "vocab", "vocab.json")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Constants
VOCAB_SIZE = 149576  
EMBEDDING_DIM = 100  
MAX_LEN = 256  
LSTM_UNITS = 128  
DROPOUT_RATE = 0.3  
BATCH_SIZE = 32 
EPOCHS = 1

def load_data(file_path):
    """Reads data from a JSONL file"""
    X, y = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading {os.path.basename(file_path)}"):
            obj = json.loads(line)
            X.append(obj['input_ids'])
            y.append(obj['target'])
    return np.array(X), np.array(y)

def build_lstm_model(vocab_size, embedding_dim, max_len, lstm_units):
    """Build LSTM model."""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        LSTM(units=lstm_units, return_sequences=False),
        Dropout(DROPOUT_RATE),
        Dense(64, activation='relu', dtype='float32'),
        Dropout(DROPOUT_RATE),
        Dense(1, activation='sigmoid', dtype='float32')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def compute_class_weights(labels):
    """Calculate class weights for imbalanced datasets."""
    classes = np.array([0, 1])
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return {0: weights[0], 1: weights[1]}

def main():
    print("TensorFlow version:", tf.__version__)
    print("Devices:", tf.config.list_physical_devices())

    # Read data
    train_file = os.path.join(ENCODED_DIR, "train.jsonl")
    val_file = os.path.join(ENCODED_DIR, "val.jsonl")
    test_file = os.path.join(ENCODED_DIR, "test.jsonl")

    X_train, y_train = load_data(train_file)
    X_val, y_val = load_data(val_file)
    X_test, y_test = load_data(test_file)

    # Calculate class weights
    class_weights = compute_class_weights(y_train)
    print(f"Class weights: {class_weights}")

    # Build model
    model = build_lstm_model(VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN, LSTM_UNITS)
    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(MODEL_DIR, "lstm_model_best.h5"),
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # training model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stopping]
    )

    # evaluate model
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["non-vulnerable", "vulnerable"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save final model
    model.save(os.path.join(MODEL_DIR, "lstm_model_final.h5"))

if __name__ == "__main__":
    main()