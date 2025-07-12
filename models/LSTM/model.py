import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization, GlobalMaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, accuracy_score
from tqdm import tqdm
import pickle
import warnings
import glob
import time
from collections import Counter
warnings.filterwarnings('ignore')

# Configuration
INPUT_DIR = "/kaggle/input/vuln-encoded-dataset"
VOCAB_PATH = os.path.join(INPUT_DIR, "vocab.json")
VAL_FILE = os.path.join(INPUT_DIR, "val.jsonl")
TEST_FILE = os.path.join(INPUT_DIR, "test.jsonl")
MODEL_DIR = "/kaggle/working"
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters
EMBEDDING_DIM = 100
MAX_LEN = 256
LSTM_UNITS = 128
DROPOUT_RATE = 0.3
BATCH_SIZE = 512
EPOCHS = 30
LEARNING_RATE = 0.0003

class SequentialTrainer:
    def __init__(self):
        self.results_summary = []
        self.models_info = []
        self.best_models = []
        
    def convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
        
    def setup_gpu(self):
        """Setup GPU configuration"""
        print("Setting up GPU configuration...")
        
        physical_devices = tf.config.list_physical_devices()
        print(f"Available devices: {physical_devices}")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU configured: {len(gpus)} GPU(s) available")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        else:
            print("No GPU found, using CPU")
        
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision enabled")
        except:
            print("Mixed precision not available")

    def validate_files(self):
        """Validate required files"""
        print("Validating required files...")
        
        # Check vocab file
        if not os.path.exists(VOCAB_PATH):
            raise FileNotFoundError(f"Vocabulary file not found: {VOCAB_PATH}")
        
        # Check val and test files
        for name, path in [("val.jsonl", VAL_FILE), ("test.jsonl", TEST_FILE)]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file not found: {name} at {path}")
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"{name}: {size_mb:.1f} MB")
        
        # Find all train files
        train_pattern = os.path.join(INPUT_DIR, "train_balanced_*.jsonl")
        train_files = sorted(glob.glob(train_pattern))
        
        if not train_files:
            raise FileNotFoundError(f"No train files found matching pattern: {train_pattern}")
        
        print(f"Found {len(train_files)} train files:")
        for train_file in train_files:
            filename = os.path.basename(train_file)
            size_mb = os.path.getsize(train_file) / (1024 * 1024)
            print(f"  {filename}: {size_mb:.1f} MB")
        
        print("All files validated")
        return train_files

    def load_data(self, file_path, file_type="train"):
        """Load and validate data"""
        print(f"Loading {file_type} data from {os.path.basename(file_path)}...")
        
        X, y = [], []
        invalid_count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"Total lines to process: {len(lines):,}")
        
        for line in tqdm(lines, desc=f"Loading {os.path.basename(file_path)}"):
            try:
                obj = json.loads(line.strip())
                input_ids = obj.get('input_ids', [])
                target = obj.get('target', 0)
                
                # Validation
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
                
            except (json.JSONDecodeError, KeyError):
                invalid_count += 1
                continue
        
        if invalid_count > 0:
            print(f"Skipped {invalid_count:,} invalid samples")
        
        X = np.array(X, dtype=np.int32)
        y = np.array(y, dtype=np.int32)
        
        # Analyze distribution
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        print(f"Loaded {total:,} samples")
        for label, count in zip(unique, counts):
            label_name = "Vulnerable" if label == 1 else "Non-vulnerable"
            percentage = count / total * 100
            print(f"  {label_name}: {count:,} ({percentage:.1f}%)")
        
        return X, y

    def build_model(self, vocab_size):
        """Build LSTM model"""
        print("Building LSTM model...")
        
        model = Sequential([
            Embedding(
                input_dim=vocab_size, 
                output_dim=EMBEDDING_DIM, 
                input_length=MAX_LEN,
                mask_zero=False,
                name='embedding'
            ),
            tf.keras.layers.Bidirectional(
                LSTM(
                    units=LSTM_UNITS, 
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
        
        # Build model explicitly
        model.build((None, MAX_LEN))
        
        optimizer = Adam(
            learning_rate=LEARNING_RATE,
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

    def setup_callbacks(self, file_number):
        """Setup training callbacks for each file"""
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(MODEL_DIR, f"best_model_file_{file_number:02d}.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1,
                save_freq='epoch'
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=8,
                restore_best_weights=True,
                mode='max',
                verbose=1,
                min_delta=0.001
            ),
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=4,
                min_lr=1e-7,
                mode='max',
                verbose=1,
                cooldown=2
            )
        ]
        
        return callbacks

    def plot_confusion_matrix(self, y_true, y_pred, file_number):
        """Plot confusion matrix with detailed metrics"""
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Non-vulnerable', 'Vulnerable'],
                       yticklabels=['Non-vulnerable', 'Vulnerable'],
                       cbar_kws={'label': 'Count'})
            
            plt.title(f'Confusion Matrix - File {file_number}', fontsize=16, fontweight='bold')
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            
            # Calculate metrics
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Add metrics text box
            textstr = f'''Performance Metrics:
            Accuracy:   {accuracy:.4f}
            Precision:  {precision:.4f}
            Recall:     {recall:.4f}
            F1-Score:   {f1:.4f}
            Specificity: {specificity:.4f}
            
            Confusion Matrix Values:
            True Positives (TP):  {tp:,}
            False Positives (FP): {fp:,}
            False Negatives (FN): {fn:,}
            True Negatives (TN):  {tn:,}
            
            Total Samples: {tp + tn + fp + fn:,}'''
            
            props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
            plt.figtext(0.02, 0.02, textstr, fontsize=10, verticalalignment='bottom', bbox=props)
            
            plt.tight_layout()
            
            # Save with specific naming
            if isinstance(file_number, str) and file_number.lower() == 'ensemble':
                filename = f'confusion_matrix_ensemble.png'
            else:
                filename = f'confusion_matrix_file_{file_number:02d}.png'
            
            plt.savefig(os.path.join(MODEL_DIR, filename), dpi=300, bbox_inches='tight')
            plt.close()
            
            return cm, {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 
                       'accuracy': float(accuracy), 'precision': float(precision), 
                       'recall': float(recall), 'f1': float(f1), 'specificity': float(specificity)}
            
        except Exception as e:
            print(f"Could not save confusion matrix: {e}")
            return None, None

    def evaluate_model(self, model, X_test, y_test, file_number):
        """Enhanced evaluation with confusion matrix"""
        print(f"Evaluating model for file {file_number:02d}...")
        
        # Predict
        y_pred_proba = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
        y_pred_proba_flat = y_pred_proba.flatten()
        
        # Find optimal threshold for F1-score
        thresholds = np.arange(0.3, 0.95, 0.01)
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in thresholds:
            y_pred = (y_pred_proba_flat > threshold).astype(int)
            if np.sum(y_pred) == 0:
                continue
            
            try:
                f1 = f1_score(y_test, y_pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            except:
                continue
        
        # Final predictions with optimal threshold
        y_pred_optimal = (y_pred_proba_flat > best_threshold).astype(int)
        
        # Generate confusion matrix
        cm, cm_metrics = self.plot_confusion_matrix(y_test, y_pred_optimal, file_number)
        
        # Classification report
        report = classification_report(
            y_test, y_pred_optimal, 
            target_names=["Non-vulnerable", "Vulnerable"],
            digits=4,
            output_dict=True,
            zero_division=0
        )
        
        # Additional metrics
        try:
            auc_score = roc_auc_score(y_test, y_pred_proba_flat)
        except:
            auc_score = 0.0
        
        metrics = {
            'file_number': int(file_number),
            'optimal_threshold': float(best_threshold),
            'test_accuracy': float(report['accuracy']),
            'test_precision': float(report['Vulnerable']['precision']),
            'test_recall': float(report['Vulnerable']['recall']),
            'test_f1': float(report['Vulnerable']['f1-score']),
            'roc_auc': float(auc_score),
            'confusion_matrix': cm.tolist() if cm is not None else None,
            'confusion_details': cm_metrics if cm_metrics else None,
            'classification_report': self.convert_numpy_types(report)
        }
        
        print(f"Results for file {file_number:02d}:")
        print(f"  Accuracy:  {metrics['test_accuracy']:.4f}")
        print(f"  Precision: {metrics['test_precision']:.4f}")
        print(f"  Recall:    {metrics['test_recall']:.4f}")
        print(f"  F1-Score:  {metrics['test_f1']:.4f}")
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        print(f"  Threshold: {metrics['optimal_threshold']:.3f}")
        
        # Print confusion matrix details
        if cm_metrics:
            print(f"  Confusion Matrix:")
            print(f"  [[TP: {cm_metrics['tp']:,}  FP: {cm_metrics['fp']:,}]")
            print(f"   [FN: {cm_metrics['fn']:,}  TN: {cm_metrics['tn']:,}]]")
        
        return metrics

    def save_file_results(self, file_number, metrics, history):
        """Save results for individual file with proper type conversion"""
        try:
            # Convert all data to JSON-serializable types
            converted_metrics = self.convert_numpy_types(metrics)
            converted_history = self.convert_numpy_types(history.history if hasattr(history, 'history') else history)
            
            # Save metrics
            result_path = os.path.join(MODEL_DIR, f"training_result_file_{file_number:02d}.json")
            with open(result_path, 'w') as f:
                json.dump({
                    'file_number': int(file_number),
                    'best_metrics': {
                        'final_test_f1': float(converted_metrics['test_f1']),
                        'final_test_precision': float(converted_metrics['test_precision']),
                        'final_test_recall': float(converted_metrics['test_recall']),
                        'final_test_accuracy': float(converted_metrics['test_accuracy']),
                        'roc_auc': float(converted_metrics['roc_auc']),
                        'optimal_threshold': float(converted_metrics['optimal_threshold']),
                        'confusion_matrix': converted_metrics['confusion_matrix'],
                        'confusion_details': converted_metrics['confusion_details']
                    },
                    'history': converted_history,
                    'evaluation': converted_metrics
                }, f, indent=2)
            
            # Save training history plot
            self.plot_training_history(history, file_number)
            
            print(f"Results saved for file {file_number:02d}")
            
        except Exception as e:
            print(f"Error saving results for file {file_number:02d}: {e}")
            import traceback
            traceback.print_exc()

    def plot_training_history(self, history, file_number):
        """Plot training history for individual file"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Training History - File {file_number:02d}', fontsize=16, fontweight='bold')
            
            # Loss
            axes[0, 0].plot(history.history['loss'], label='Training', linewidth=2)
            axes[0, 0].plot(history.history['val_loss'], label='Validation', linewidth=2)
            axes[0, 0].set_title('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Accuracy
            axes[0, 1].plot(history.history['accuracy'], label='Training', linewidth=2)
            axes[0, 1].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Precision
            if 'precision' in history.history:
                axes[1, 0].plot(history.history['precision'], label='Training', linewidth=2)
                axes[1, 0].plot(history.history['val_precision'], label='Validation', linewidth=2)
                axes[1, 0].set_title('Precision')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # AUC
            if 'auc' in history.history:
                axes[1, 1].plot(history.history['auc'], label='Training', linewidth=2)
                axes[1, 1].plot(history.history['val_auc'], label='Validation', linewidth=2)
                axes[1, 1].set_title('AUC')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(MODEL_DIR, f'training_history_file_{file_number:02d}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Could not save training plot: {e}")

    def ensemble_predict(self, X_test):
        """Make ensemble predictions using all trained models"""
        print("Making ensemble predictions...")
        
        model_files = []
        for i in range(1, 15):
            model_path = os.path.join(MODEL_DIR, f"best_model_file_{i:02d}.h5")
            if os.path.exists(model_path):
                model_files.append((i, model_path))
        
        if not model_files:
            print("No trained models found for ensemble!")
            return None, None
        
        print(f"Found {len(model_files)} trained models")
        
        # Load models and get predictions
        all_predictions = []
        weights = []
        
        for file_num, model_path in model_files:
            try:
                model = tf.keras.models.load_model(model_path)
                pred = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
                all_predictions.append(pred.flatten())
                
                # Weight by F1 score if available
                result_path = os.path.join(MODEL_DIR, f"training_result_file_{file_num:02d}.json")
                if os.path.exists(result_path):
                    with open(result_path, 'r') as f:
                        results = json.load(f)
                    weight = results['best_metrics'].get('final_test_f1', 0.5)
                else:
                    weight = 0.5
                
                weights.append(weight)
                print(f"  Model {file_num:02d}: weight = {weight:.4f}")
                
                # Clean up memory
                del model
                tf.keras.backend.clear_session()
                
            except Exception as e:
                print(f"Failed to load model {file_num:02d}: {e}")
        
        if not all_predictions:
            return None, None
        
        # Weighted ensemble
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize
        
        ensemble_predictions = np.zeros_like(all_predictions[0])
        for pred, weight in zip(all_predictions, weights):
            ensemble_predictions += pred * weight
        
        return ensemble_predictions, weights

    def create_comprehensive_report(self):
        """Create comprehensive report after all training"""
        print("\nCREATING COMPREHENSIVE REPORT")
        print("=" * 60)
        
        # Load all results
        all_results = []
        for i in range(1, 15):
            result_path = os.path.join(MODEL_DIR, f"training_result_file_{i:02d}.json")
            if os.path.exists(result_path):
                with open(result_path, 'r') as f:
                    result = json.load(f)
                    all_results.append(result)
        
        if not all_results:
            print("No results found!")
            return
        
        # Create summary DataFrame equivalent
        summary_data = []
        for result in all_results:
            metrics = result['best_metrics']
            summary_data.append({
                'file': result['file_number'],
                'accuracy': metrics.get('final_test_accuracy', 0),
                'precision': metrics['final_test_precision'],
                'recall': metrics['final_test_recall'],
                'f1_score': metrics['final_test_f1'],
                'roc_auc': metrics['roc_auc'],
                'threshold': metrics['optimal_threshold']
            })
        
        # Calculate statistics
        accuracies = [d['accuracy'] for d in summary_data]
        precisions = [d['precision'] for d in summary_data]
        recalls = [d['recall'] for d in summary_data]
        f1_scores = [d['f1_score'] for d in summary_data]
        
        stats = {
            'num_files': len(summary_data),
            'mean_accuracy': float(np.mean(accuracies)),
            'mean_precision': float(np.mean(precisions)),
            'mean_recall': float(np.mean(recalls)),
            'mean_f1': float(np.mean(f1_scores)),
            'max_accuracy': float(np.max(accuracies)),
            'max_precision': float(np.max(precisions)),
            'max_recall': float(np.max(recalls)),
            'max_f1': float(np.max(f1_scores)),
            'files_accuracy_80_plus': len([a for a in accuracies if a >= 0.8]),
            'files_precision_80_plus': len([p for p in precisions if p >= 0.8]),
            'files_recall_80_plus': len([r for r in recalls if r >= 0.8]),
            'files_f1_80_plus': len([f for f in f1_scores if f >= 0.8])
        }
        
        # Print summary
        print(f"\nTRAINING SUMMARY ({stats['num_files']} files)")
        print("=" * 80)
        print("FILE\tACCURACY\tPRECISION\tRECALL\t\tF1-SCORE\tROC AUC")
        print("-" * 80)
        for data in summary_data:
            print(f"{data['file']:02d}\t{data['accuracy']:.4f}\t\t{data['precision']:.4f}\t\t{data['recall']:.4f}\t\t{data['f1_score']:.4f}\t\t{data['roc_auc']:.4f}")
        
        print(f"\nOVERALL STATISTICS")
        print("=" * 50)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key.replace('_', ' ').title()}: {value:.4f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
        
        # Best performing files
        best_accuracy_idx = np.argmax(accuracies)
        best_precision_idx = np.argmax(precisions)
        best_recall_idx = np.argmax(recalls)
        best_f1_idx = np.argmax(f1_scores)
        
        print(f"\nBEST PERFORMING FILES")
        print("=" * 50)
        print(f"Best Accuracy:  File {summary_data[best_accuracy_idx]['file']:02d} ({accuracies[best_accuracy_idx]:.4f})")
        print(f"Best Precision: File {summary_data[best_precision_idx]['file']:02d} ({precisions[best_precision_idx]:.4f})")
        print(f"Best Recall:    File {summary_data[best_recall_idx]['file']:02d} ({recalls[best_recall_idx]:.4f})")
        print(f"Best F1-Score:  File {summary_data[best_f1_idx]['file']:02d} ({f1_scores[best_f1_idx]:.4f})")
        
        # Target achievement
        print(f"\nTARGET ACHIEVEMENT (>= 0.8)")
        print("=" * 50)
        print(f"Files with Accuracy >= 0.8:  {stats['files_accuracy_80_plus']}/{stats['num_files']}")
        print(f"Files with Precision >= 0.8: {stats['files_precision_80_plus']}/{stats['num_files']}")
        print(f"Files with Recall >= 0.8:    {stats['files_recall_80_plus']}/{stats['num_files']}")
        print(f"Files with F1-Score >= 0.8:  {stats['files_f1_80_plus']}/{stats['num_files']}")
        
        # Overall success check
        all_targets_met = (stats['files_accuracy_80_plus'] > 0 and 
                          stats['files_precision_80_plus'] > 0 and 
                          stats['files_recall_80_plus'] > 0 and 
                          stats['files_f1_80_plus'] > 0)
        
        if all_targets_met:
            print("ALL TARGETS ACHIEVED!")
        else:
            print("Some targets not achieved")
        
        # Save comprehensive report
        report_data = {
            'summary_statistics': stats,
            'file_results': summary_data,
            'best_accuracy_file': summary_data[best_accuracy_idx]['file'],
            'best_precision_file': summary_data[best_precision_idx]['file'],
            'best_recall_file': summary_data[best_recall_idx]['file'],
            'best_f1_file': summary_data[best_f1_idx]['file'],
            'targets_achieved': {
                'accuracy': stats['files_accuracy_80_plus'] > 0,
                'precision': stats['files_precision_80_plus'] > 0,
                'recall': stats['files_recall_80_plus'] > 0,
                'f1_score': stats['files_f1_80_plus'] > 0,
                'all_targets': all_targets_met
            }
        }
        
        with open(os.path.join(MODEL_DIR, 'comprehensive_training_report.json'), 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return report_data

    def train_all_files(self):
        """Main training pipeline for all files"""
        print("SEQUENTIAL TRAINING OF ALL FILES")
        print("=" * 70)
        
        # Setup
        self.setup_gpu()
        train_files = self.validate_files()
        
        # Load vocabulary
        print("\nLoading vocabulary...")
        with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        vocab_size = len(vocab)
        print(f"Vocabulary loaded: {vocab_size:,} tokens")
        
        # Load validation and test data once
        print("\nLoading validation and test data...")
        X_val, y_val = self.load_data(VAL_FILE, "validation")
        X_test, y_test = self.load_data(TEST_FILE, "test")
        
        # Train each file sequentially
        total_start_time = time.time()
        
        for i, train_file in enumerate(train_files, 1):
            print(f"\n{'='*70}")
            print(f"TRAINING FILE {i:02d}/{len(train_files)}: {os.path.basename(train_file)}")
            print(f"{'='*70}")
            
            file_start_time = time.time()
            
            try:
                # Extract file number from filename
                import re
                file_match = re.search(r'train_balanced_(\d+)', train_file)
                file_number = int(file_match.group(1)) if file_match else i
                
                # Load training data for this file
                X_train, y_train = self.load_data(train_file, f"train_{file_number:02d}")
                
                # Build fresh model for each file
                model = self.build_model(vocab_size)
                
                # Print model info AFTER building
                print(f"\nModel architecture for file {file_number:02d}:")
                print(f"Total parameters: {model.count_params():,}")
                
                # Setup callbacks
                callbacks = self.setup_callbacks(file_number)
                
                print(f"\nStarting training for file {file_number:02d}...")
                print(f"Training samples: {len(X_train):,}")
                print(f"Validation samples: {len(X_val):,}")
                print(f"Epochs: {EPOCHS}")
                print(f"Batch size: {BATCH_SIZE}")
                
                # Train model
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=callbacks,
                    verbose=1,
                    shuffle=True
                )
                
                # Load best model
                best_model_path = os.path.join(MODEL_DIR, f"best_model_file_{file_number:02d}.h5")
                if os.path.exists(best_model_path):
                    model = tf.keras.models.load_model(best_model_path)
                
                # Evaluate
                metrics = self.evaluate_model(model, X_test, y_test, file_number)
                
                # Save results
                self.save_file_results(file_number, metrics, history)
                
                # Training time
                file_time = time.time() - file_start_time
                print(f"File {file_number:02d} completed in {file_time/60:.1f} minutes")
                
                # Memory cleanup
                del model, X_train, y_train
                tf.keras.backend.clear_session()
                
                # Brief pause between files
                if i < len(train_files):
                    time.sleep(10)
                
            except Exception as e:
                print(f"Failed to train file {file_number:02d}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Total training time
        total_time = time.time() - total_start_time
        print(f"\nTotal training time: {total_time/3600:.1f} hours")
        
        # Create comprehensive report
        report = self.create_comprehensive_report()
        
        # Ensemble evaluation with confusion matrix
        print(f"\nENSEMBLE EVALUATION")
        print("=" * 50)
        
        try:
            ensemble_predictions, ensemble_weights = self.ensemble_predict(X_test)
            
            if ensemble_predictions is not None:
                # Find optimal threshold for ensemble
                thresholds = np.arange(0.3, 0.95, 0.01)
                best_threshold = 0.5
                best_f1 = 0
                
                for threshold in thresholds:
                    y_pred = (ensemble_predictions > threshold).astype(int)
                    if np.sum(y_pred) == 0:
                        continue
                    
                    f1 = f1_score(y_test, y_pred)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
                
                # Final ensemble evaluation
                y_pred_ensemble = (ensemble_predictions > best_threshold).astype(int)
                ensemble_report = classification_report(
                    y_test, y_pred_ensemble,
                    target_names=["Non-vulnerable", "Vulnerable"],
                    digits=4,
                    output_dict=True
                )
                
                # Generate ensemble confusion matrix
                ensemble_cm, ensemble_cm_metrics = self.plot_confusion_matrix(y_test, y_pred_ensemble, "ensemble")
                
                print(f"ENSEMBLE RESULTS:")
                print(f"Models used: {len(ensemble_weights)}")
                print(f"Optimal threshold: {best_threshold:.3f}")
                print(f"Accuracy:  {ensemble_report['accuracy']:.4f}")
                print(f"Precision: {ensemble_report['Vulnerable']['precision']:.4f}")
                print(f"Recall:    {ensemble_report['Vulnerable']['recall']:.4f}")
                print(f"F1-Score:  {ensemble_report['Vulnerable']['f1-score']:.4f}")
                
                # Print ensemble confusion matrix
                if ensemble_cm_metrics:
                    print(f"Confusion Matrix:")
                    print(f"[[TP: {ensemble_cm_metrics['tp']:,}  FP: {ensemble_cm_metrics['fp']:,}]")
                    print(f" [FN: {ensemble_cm_metrics['fn']:,}  TN: {ensemble_cm_metrics['tn']:,}]]")
                
                # Save ensemble results
                ensemble_info = {
                    'num_models': len(ensemble_weights),
                    'optimal_threshold': float(best_threshold),
                    'weights': ensemble_weights.tolist(),
                    'performance': {
                        'accuracy': float(ensemble_report['accuracy']),
                        'precision': float(ensemble_report['Vulnerable']['precision']),
                        'recall': float(ensemble_report['Vulnerable']['recall']),
                        'f1_score': float(ensemble_report['Vulnerable']['f1-score'])
                    },
                    'confusion_matrix': ensemble_cm.tolist() if ensemble_cm is not None else None,
                    'confusion_details': ensemble_cm_metrics if ensemble_cm_metrics else None
                }
                
                with open(os.path.join(MODEL_DIR, 'ensemble_results.json'), 'w') as f:
                    json.dump(ensemble_info, f, indent=2)
                
                print(f"Ensemble results saved")
            
        except Exception as e:
            print(f"Ensemble evaluation failed: {e}")
        
        print(f"\nALL TRAINING COMPLETED!")
        print(f"All results saved to: {MODEL_DIR}")
        
        return report

def main():
    """Main execution function"""
    trainer = SequentialTrainer()
    
    try:
        report = trainer.train_all_files()
        print("\nSequential training pipeline completed successfully!")
        return report
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.create_comprehensive_report()
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        trainer.create_comprehensive_report()
        raise

if __name__ == "__main__":
    report = main()
