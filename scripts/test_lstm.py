import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import json
import numpy as np
import tensorflow as tf
from scripts.preprocessing_pipeline import VulnerabilityPreprocessor

def load_model_and_vocab():
    """Load trained model and vocab"""
    base_dir = Path(__file__).parent
    model_path = base_dir / "models" / "LSTM" / "best_model.h5"
    vocab_path = base_dir / "data" / "preprocessed" / "vocab" / "vocab.json"
    
    model = tf.keras.models.load_model(str(model_path))
    
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    return model, vocab

def predict(code_text):
    """Quick prediction"""
    model, vocab = load_model_and_vocab()
    preprocessor = VulnerabilityPreprocessor()
    
    # Process code
    tokens = preprocessor.process_and_tokenize(code_text)
    input_ids = [vocab.get(token, vocab.get('<UNK>', 1)) for token in tokens[:512]]
    
    # Pad
    input_ids += [0] * (512 - len(input_ids))
    
    # Predict
    prob = model.predict(np.array([input_ids]), verbose=0)[0][0]
    pred = int(prob > 0.5)
    
    return {
        'prediction': 'Vulnerable' if pred else 'Safe',
        'probability': float(prob),
        'confidence': float(abs(prob - 0.5) * 2)
    }

def main():
    if len(sys.argv) < 2:
        # Test with examples
        test_codes = [
            'void test() { char buf[10]; strcpy(buf, input); }',  # Vulnerable
            'void test() { char buf[10]; strncpy(buf, input, 9); buf[9] = 0; }'  # Safe
        ]
        
        for i, code in enumerate(test_codes, 1):
            result = predict(code)
            print(f"Test {i}: {result['prediction']} (conf: {result['confidence']:.3f})")
    else:
        # Predict from argument
        code = sys.argv[1]
        result = predict(code)
        print(f"Prediction: {result['prediction']}")
        print(f"Probability: {result['probability']:.4f}")
        print(f"Confidence: {result['confidence']:.4f}")

if __name__ == "__main__":
    main()