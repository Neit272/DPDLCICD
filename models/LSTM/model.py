import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization, GlobalMaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')


# INPUT_DIR = "/data/preprocessed/"
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
VOCAB_PATH = os.path.join(ROOT_DIR, "data", "preprocessed", "vocab", "vocab.json")
TRAIN_FILE = os.path.join(ROOT_DIR, "data", "preprocessed", "encoded_balanced", "train_balanced_01.jsonl")
VAL_FILE = os.path.join(ROOT_DIR, "data", "preprocessed", "encoded_balanced", "val.jsonl")
TEST_FILE = os.path.join(ROOT_DIR, "data", "preprocessed", "encoded_balanced", "test.jsonl")
MODEL_DIR = os.path.join(ROOT_DIR, "models", "LSTM")
os.makedirs(MODEL_DIR, exist_ok=True)

# Optimized Hyperparameters
EMBEDDING_DIM = 100 
MAX_LEN = 256
LSTM_UNITS = 128    
DROPOUT_RATE = 0.1
BATCH_SIZE = 256
EPOCHS = 30       
LEARNING_RATE = 0.0005

def setup_gpu():
    """Setup GPU configuration for optimal performance"""
    print("🔧 Setting up GPU configuration...")
    
    physical_devices = tf.config.list_physical_devices()
    print(f"Available devices: {physical_devices}")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU configured: {len(gpus)} GPU(s) available")
        except RuntimeError as e:
            print(f"⚠️ GPU configuration error: {e}")
    else:
        print("⚠️ No GPU found, using CPU")
    
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("✅ Mixed precision enabled")
    except:
        print("⚠️ Mixed precision not available")

def validate_data_files():
    """Validate that all required data files exist"""
    print("🔍 Validating data files...")
    
    required_files = {
        "train_balanced_14.jsonl": TRAIN_FILE,
        "val.jsonl": VAL_FILE,
        "test.jsonl": TEST_FILE,
        "vocab.json": VOCAB_PATH
    }
    
    for name, file_path in required_files.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {name} at {file_path}")
        
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  ✅ {name}: {size_mb:.1f} MB")
    
    print("✅ All required files validated")

def load_data(file_path, max_samples=None):
    """Load and validate encoded data with memory optimization"""
    print(f"📥 Loading data from {os.path.basename(file_path)}...")
    
    X, y = [], []
    invalid_count = 0
    total_lines = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    
    print(f"Total lines to process: {total_lines:,}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, total=total_lines, desc=f"Loading {os.path.basename(file_path)}")):
            if max_samples and len(X) >= max_samples:
                break
                
            try:
                obj = json.loads(line.strip())
                input_ids = obj.get('input_ids', [])
                target = obj.get('target', 0)  # Change to 'labels' if your dataset uses that key
                
                if not input_ids or len(input_ids) != MAX_LEN:
                    invalid_count += 1
                    continue
                
                if target not in [0, 1]:
                    invalid_count += 1
                    continue
                
                try:
                    input_ids = [int(x) for x in input_ids]
                    if any(x < 0 for x in input_ids):
                        invalid_count += 1
                        continue
                except (ValueError, TypeError):
                    invalid_count += 1
                    continue
                
                X.append(input_ids)
                y.append(target)
                
            except (json.JSONDecodeError, KeyError) as e:
                invalid_count += 1
                continue
    
    if invalid_count > 0:
        print(f"⚠️ Skipped {invalid_count:,} invalid samples")
    
    X = np.array(X, dtype=np.int32)
    y = np.array(y, dtype=np.int32)
    
    print(f"✅ Loaded {len(X):,} samples - Shape: {X.shape}")
    return X, y

def analyze_data_distribution(y_train, y_val, y_test):
    """Analyze and print data distribution"""
    print("\n📊 DATA DISTRIBUTION ANALYSIS:")
    print("=" * 50)
    
    def print_dist(y, name):
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        print(f"\n{name} Set:")
        for label, count in zip(unique, counts):
            label_name = "Vulnerable" if label == 1 else "Non-vulnerable"
            percentage = count / total * 100
            print(f"  {label_name}: {count:,} ({percentage:.1f}%)")
        print(f"  Total: {total:,} samples")
        if len(unique) == 2 and abs(counts[0] - counts[1]) / total < 0.05:
            print(f"  ✅ Dataset appears balanced")
        else:
            print(f"  ⚠️ Dataset may be imbalanced")
    
    print_dist(y_train, "Training")
    print_dist(y_val, "Validation")
    print_dist(y_test, "Test")

def build_gpu_optimized_lstm_model(vocab_size, embedding_dim, max_len, lstm_units, learning_rate):
    """Build GPU-optimized LSTM model without masking for cuDNN compatibility"""
    print("🏗️ Building GPU-optimized LSTM model...")
    
    model = Sequential([
        Embedding(
            input_dim=vocab_size, 
            output_dim=embedding_dim, 
            input_length=max_len,
            mask_zero=False,
            name='embedding'
        ),
        tf.keras.layers.Bidirectional(
            LSTM(
                units=lstm_units, 
                return_sequences=True, 
                dropout=DROPOUT_RATE,
                recurrent_dropout=0.0,
            ),
            name='bidirectional_lstm'
        ),
        GlobalMaxPooling1D(name='global_max_pool'),
        BatchNormalization(name='batch_norm_1'),
        Dense(256, activation='relu', name='dense_1'),
        Dropout(DROPOUT_RATE, name='dropout_1'),
        BatchNormalization(name='batch_norm_2'),
        Dense(128, activation='relu', name='dense_2'),
        Dropout(DROPOUT_RATE, name='dropout_2'),
        Dense(64, activation='relu', name='dense_3'),
        Dropout(DROPOUT_RATE / 2, name='dropout_3'),
        Dense(1, activation='sigmoid', dtype='float32', name='output')
    ])
    
    model.build((None, max_len))
    
    optimizer = Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model

def setup_callbacks(model_dir):
    """Setup advanced training callbacks"""
    print("⚙️ Setting up training callbacks...")
    
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(model_dir, "best_model.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1,
            save_freq='epoch'
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            mode='max',
            verbose=1,
            min_delta=0.001
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.3,
            patience=5,
            min_lr=1e-7,
            mode='max',
            verbose=1,
            cooldown=2
        )
    ]
    
    return callbacks

def plot_training_history(history, model_dir):
    """Plot and save comprehensive training history"""
    print("📊 Plotting training history...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('LSTM Training History', fontsize=16, fontweight='bold')
    
    axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Model Accuracy', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    if 'auc' in history.history:
        axes[1, 0].plot(history.history['auc'], label='Training AUC', linewidth=2)
        axes[1, 0].plot(history.history['val_auc'], label='Validation AUC', linewidth=2)
        axes[1, 0].set_title('Model AUC', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    if 'precision' in history.history and 'recall' in history.history:
        axes[1, 1].plot(history.history['precision'], label='Training Precision', linewidth=2)
        axes[1, 1].plot(history.history['val_precision'], label='Validation Precision', linewidth=2)
        axes[1, 1].plot(history.history['recall'], label='Training Recall', linewidth=2)
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall', linewidth=2)
        axes[1, 1].set_title('Precision & Recall', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    with open(os.path.join(model_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)
    
    print(f"✅ Training history saved to {model_dir}")

def evaluate_model(model, X_test, y_test, model_dir):
    """Comprehensive model evaluation with multiple metrics"""
    print("\n🔍 COMPREHENSIVE MODEL EVALUATION:")
    print("=" * 60)
    
    print("Making predictions...")
    y_pred_proba = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    y_pred_proba_flat = y_pred_proba.flatten()
    
    best_threshold = 0.8
    y_pred = (y_pred_proba_flat > best_threshold).astype(int)
    best_f1 = f1_score(y_test, y_pred)
    
    print(f"🎯 Using fixed threshold: {best_threshold:.3f} (F1-Score: {best_f1:.4f})")
    
    print(f"\n📋 CLASSIFICATION REPORT (Threshold: {best_threshold:.3f}):")
    print("-" * 60)
    report = classification_report(
        y_test, y_pred, 
        target_names=["Non-vulnerable", "Vulnerable"],
        digits=4,
        output_dict=True
    )
    print(classification_report(y_test, y_pred, target_names=["Non-vulnerable", "Vulnerable"], digits=4))
    
    print(f"\n🔢 CONFUSION MATRIX:")
    print("-" * 30)
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Negatives:  {cm[0, 0]:,}")
    print(f"False Positives: {cm[0, 1]:,}")
    print(f"False Negatives: {cm[1, 0]:,}")
    print(f"True Positives:  {cm[1, 1]:,}")
    
    try:
        auc_score = roc_auc_score(y_test, y_pred_proba_flat)
        print(f"\n📈 ROC AUC Score: {auc_score:.4f}")
        
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_flat)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(model_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"⚠️ Could not compute ROC AUC: {e}")
        auc_score = None
    
    print(f"\n🎯 PERFORMANCE SUMMARY:")
    print("-" * 40)
    print(f"Accuracy:     {report['accuracy']:.4f}")
    print(f"Precision:    {report['Vulnerable']['precision']:.4f}")
    print(f"Recall:       {report['Vulnerable']['recall']:.4f}")
    print(f"F1-Score:     {report['Vulnerable']['f1-score']:.4f}")
    if auc_score:
        print(f"ROC AUC:      {auc_score:.4f}")
    
    results = {
        'optimal_threshold': float(best_threshold),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'roc_auc_score': float(auc_score) if auc_score else None,
        'y_true': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'y_pred_proba': y_pred_proba_flat.tolist()
    }
    
    with open(os.path.join(model_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Evaluation results saved to {model_dir}")
    
    return results

def main():
    """Main training pipeline"""
    print("🚀 LSTM VULNERABILITY DETECTION TRAINING ON KAGGLE")
    print("=" * 70)
    print(f"TensorFlow version: {tf.__version__}")
    
    setup_gpu()
    validate_data_files()
    
    print("\n📖 Loading vocabulary...")
    try:
        with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        vocab_size = len(vocab)
        print(f"✅ Vocabulary loaded: {vocab_size:,} tokens")
        
        if vocab_size > 50000:
            print(f"⚠️ Very large vocabulary ({vocab_size:,}) - may cause memory issues")
            
    except Exception as e:
        raise Exception(f"Error loading vocabulary: {e}")
    
    print("\n📥 Loading datasets...")
    X_train, y_train = load_data(TRAIN_FILE)
    X_val, y_val = load_data(VAL_FILE)
    X_test, y_test = load_data(TEST_FILE)
    
    analyze_data_distribution(y_train, y_val, y_test)
    
    print(f"\n🏗️ Building model architecture...")
    model = build_gpu_optimized_lstm_model(vocab_size, EMBEDDING_DIM, MAX_LEN, LSTM_UNITS, LEARNING_RATE)
    
    print("\n📊 Model Architecture:")
    model.summary()
    total_params = model.count_params()
    print(f"\n📈 Total parameters: {total_params:,}")
    
    callbacks = setup_callbacks(MODEL_DIR)
    
    print(f"\n🎯 STARTING TRAINING:")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print("-" * 50)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    
    plot_training_history(history, MODEL_DIR)
    
    best_model_path = os.path.join(MODEL_DIR, "best_model.h5")
    if os.path.exists(best_model_path):
        print(f"\n📥 Loading best model for evaluation...")
        model = tf.keras.models.load_model(best_model_path)
    
    evaluation_results = evaluate_model(model, X_test, y_test, MODEL_DIR)
    
    final_model_path = os.path.join(MODEL_DIR, "final_model.h5")
    model.save(final_model_path)
    print(f"\n💾 Final model saved to: {final_model_path}")
    
    print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"📁 All outputs saved to: {MODEL_DIR}")
    
    try:
        f1_score_result = evaluation_results.get('classification_report', {}).get('Vulnerable', {}).get('f1-score', 'N/A')
        auc_score_result = evaluation_results.get('roc_auc_score', 'N/A')
        
        if isinstance(f1_score_result, (int, float)):
            print(f"🎯 Best F1-Score: {f1_score_result:.4f}")
        else:
            print(f"🎯 Best F1-Score: {f1_score_result}")
            
        if isinstance(auc_score_result, (int, float)):
            print(f"📈 ROC AUC Score: {auc_score_result:.4f}")
        else:
            print(f"📈 ROC AUC Score: {auc_score_result}")
            
    except Exception as e:
        print(f"⚠️ Could not display final scores: {e}")
    
    return model, history, evaluation_results

if __name__ == "__main__":
    try:
        model, history, results = main()
    except Exception as e:
        print(f"\n❌ TRAINING FAILED:")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise
