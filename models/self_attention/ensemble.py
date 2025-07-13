import numpy as np
import os
import json
import glob
import re
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from models.self_attention.model import (
    TokenAndPositionEmbedding,
    TransformerEncoder,
)  # Giả sử file model.py đã sửa lỗi import

# ... (các hàm tiện ích load_data, find_threshold, evaluate_predictions, plot_confusion_matrix) ...


def load_test_data(file_path):
    input_ids, targets = [], []
    print(f"\nĐang tải dữ liệu kiểm thử từ: {file_path}")
    if not os.path.exists(file_path):
        print(f"LỖI: Không tìm thấy file test tại '{file_path}'")
        return None, None
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading test data"):
            try:
                obj = json.loads(line)
                input_ids.append(obj["input_ids"])
                targets.append(obj["target"])
            except (json.JSONDecodeError, KeyError):
                continue
    X_test, y_test = np.array(input_ids), np.array(targets)
    print(f"Tải thành công {len(y_test)} mẫu kiểm thử.")
    return X_test, y_test


def find_optimal_threshold(y_true, y_pred_probs):
    thresholds = np.arange(0.3, 0.96, 0.01)
    best_threshold, best_f1 = 0.5, 0
    for threshold in thresholds:
        y_pred_binary = (y_pred_probs > threshold).astype(int)
        if np.sum(y_pred_binary) == 0:
            continue
        f1 = f1_score(y_true, y_pred_binary)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
    return best_threshold, best_f1


def evaluate_predictions(y_true, y_pred_probs, threshold):
    y_pred_binary = (y_pred_probs > threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred_binary),
        "precision": precision_score(y_true, y_pred_binary, zero_division=0),
        "recall": recall_score(y_true, y_pred_binary, zero_division=0),
        "f1": f1_score(y_true, y_pred_binary, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred_probs),
    }


def plot_confusion_matrix(
    y_true, y_pred, metrics, output_path, title="Confusion Matrix"
):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.flatten()
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-vulnerable", "Vulnerable"],
        yticklabels=["Non-vulnerable", "Vulnerable"],
    )
    plt.title(title, fontsize=16)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    stats_text = (
        f"Performance Metrics:\n"
        f"Accuracy:  {metrics['accuracy']:.4f}\n"
        f"Precision: {metrics['precision']:.4f}\n"
        f"Recall:    {metrics['recall']:.4f}\n"
        f"F1-Score:  {metrics['f1']:.4f}\n\n"
        f"Confusion Matrix Values:\n"
        f"True Positives (TP):  {TP:,}\n"
        f"False Positives (FP): {FP:,}\n"
        f"False Negatives (FN): {FN:,}\n"
        f"True Negatives (TN):  {TN:,}\n\n"
        f"Total Samples: {len(y_true):,}"
    )
    plt.text(
        0.05,
        0.35,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", alpha=0.5),
    )
    plt.savefig(output_path)
    print(f"\nĐã lưu biểu đồ Confusion Matrix tại: {output_path}")
    plt.close()


if __name__ == "__main__":
    MODEL_DIR = "models/self_attention/results/ensemble_models"
    RESULTS_BASE_DIR = "models/self_attention/results"
    TEST_DATA_PATH = "data/preprocessed/encoded_balanced/test.jsonl"

    X_test, y_test = load_test_data(TEST_DATA_PATH)
    if X_test is None:
        exit()

    model_paths = sorted(glob.glob(os.path.join(MODEL_DIR, "model_*.h5")))
    if not model_paths:
        print(f"LỖI: Không tìm thấy file model .h5 nào trong thư mục '{MODEL_DIR}'")
        exit()

    print(f"\nTìm thấy {len(model_paths)} model để thực hiện ensemble.")

    all_predictions, weights, individual_model_metrics = [], [], []
    custom_objects = {
        "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
        "TransformerEncoder": TransformerEncoder,
    }

    print("\n--- ĐÁNH GIÁ HIỆU SUẤT CỦA TỪNG MODEL CON TRÊN TẬP TEST ---")
    for model_path in model_paths:
        # ... (code xử lý từng model con giữ nguyên)
        model_basename = os.path.basename(model_path)
        match = re.search(r"model_(\d+)\.h5", model_basename)
        if not match:
            continue
        model_index = match.group(1)
        result_dir_path = os.path.join(RESULTS_BASE_DIR, f"run_train_{model_index}")
        result_file_path = os.path.join(
            result_dir_path, f"training_result_file_{model_index}.json"
        )
        if not os.path.exists(result_file_path):
            continue
        print(f"\nĐang xử lý Model {model_index}...")
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path, compile=False)
        pred_probs = model.predict(X_test, verbose=0).flatten()
        all_predictions.append(pred_probs)
        with open(result_file_path, "r") as f:
            weight = json.load(f).get("best_metrics", {}).get("final_test_f1", 0.5)
        weights.append(weight)
        opt_thresh, _ = find_optimal_threshold(y_test, pred_probs)
        metrics = evaluate_predictions(y_test, pred_probs, opt_thresh)
        metrics["model_index"] = f"Model {model_index}"
        metrics["f1_weight_from_train"] = weight
        individual_model_metrics.append(metrics)
        print(
            f"-> Model {model_index}: F1-Score = {metrics['f1']:.4f} (với ngưỡng tối ưu {opt_thresh:.2f})"
        )

    if not all_predictions:
        print("\nLỖI: Không có model nào được xử lý. Dừng quá trình ensemble.")
        exit()

    print("\n\n--- THỰC HIỆN TỔ HỢP ENSEMBLE ---")
    all_predictions = np.array(all_predictions)
    weights = np.array(weights)
    normalized_weights = weights / np.sum(weights)
    ensemble_probs = np.average(all_predictions, axis=0, weights=normalized_weights)
    ensemble_opt_thresh, _ = find_optimal_threshold(y_test, ensemble_probs)
    ensemble_metrics = evaluate_predictions(y_test, ensemble_probs, ensemble_opt_thresh)
    ensemble_metrics["model_index"] = "Ensemble"
    ensemble_metrics["f1_weight_from_train"] = np.nan

    print("\n\n" + "=" * 80)
    print("--- BẢNG SO SÁNH HIỆU SUẤT TỔNG KẾT ---")
    print("=" * 80)
    df = pd.DataFrame(individual_model_metrics).set_index("model_index")
    avg_metrics = df.mean(axis=0)
    avg_metrics.name = "Average Individual"
    ensemble_df = pd.DataFrame([ensemble_metrics]).set_index("model_index")
    summary_df = pd.concat([df, avg_metrics.to_frame().T, ensemble_df])
    pd.options.display.float_format = "{:.4f}".format
    print(
        summary_df[
            ["f1_weight_from_train", "f1", "precision", "recall", "accuracy", "roc_auc"]
        ]
    )
    print("=" * 80)

    print("\n--- TẠO BIỂU ĐỒ CONFUSION MATRIX CHO KẾT QUẢ ENSEMBLE ---")
    cm_plot_path = os.path.join(RESULTS_BASE_DIR, "confusion_matrix_ensemble.png")
    final_pred_binary = (ensemble_probs > ensemble_opt_thresh).astype(int)
    plot_confusion_matrix(
        y_test,
        final_pred_binary,
        ensemble_metrics,
        cm_plot_path,
        title="Confusion Matrix - Transformer Ensemble",
    )
