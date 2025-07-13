import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

# --- CÁC HÀM TIỆN ÍCH (Giữ nguyên) ---


def load_vocabulary_size(vocab_path):
    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        vocab_size = len(vocab)
        print(f"Tải thành công từ điển. Kích thước: {vocab_size}")
        return vocab_size
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file từ điển tại: {vocab_path}")
        return None


def load_data_from_jsonl(file_path, max_len):
    if not os.path.exists(file_path):
        print(f"LỖI: Không tìm thấy file dữ liệu: {file_path}")
        return None, None
    sequences, labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            sequences.append(data["input_ids"])
            labels.append(data["target"])
    padded_sequences = pad_sequences(
        sequences, maxlen=max_len, padding="post", truncating="post"
    )
    return np.array(padded_sequences), np.array(labels)


def plot_training_history(history, output_path):
    metrics_to_plot = ["loss", "accuracy", "precision", "recall", "auc"]
    plt.figure(figsize=(15, 12))
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(3, 2, i + 1)
        if metric in history.history:
            plt.plot(history.history[metric], label=f"Training {metric.capitalize()}")
        if f"val_{metric}" in history.history:
            plt.plot(
                history.history[f"val_{metric}"],
                label=f"Validation {metric.capitalize()}",
            )
        plt.title(f"Lịch sử {metric.capitalize()}")
        plt.ylabel(metric.capitalize())
        plt.xlabel("Epoch")
        plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nĐã lưu biểu đồ lịch sử huấn luyện tại: {output_path}")


# --- HÀM HUẤN LUYỆN LÕI ---


def train_single_model(args, train_index):
    """
    Hàm thực hiện huấn luyện cho một model duy nhất.
    """
    # Thêm đường dẫn của file model.py vào hệ thống để có thể import
    sys.path.append(args.model_script_dir)
    from models.self_attention.model import create_model

    # --- 1. Thiết lập đường dẫn ---
    train_filename = f"train_balanced_{train_index:02d}.jsonl"
    train_file_path = os.path.join(args.input_dir, train_filename)
    val_file_path = os.path.join(args.input_dir, "val.jsonl")
    vocab_file_path = os.path.join(args.vocab_dir, "vocab.json")

    # Tạo các thư mục output cần thiết
    run_specific_output_dir = os.path.join(
        args.output_dir, f"run_train_{train_index:02d}"
    )
    ensemble_output_dir = os.path.join(args.output_dir, "ensemble_models")
    os.makedirs(run_specific_output_dir, exist_ok=True)
    os.makedirs(ensemble_output_dir, exist_ok=True)

    print("\n" + "=" * 50)
    print(f"--- BẮT ĐẦU HUẤN LUYỆN CHO MODEL {train_index:02d} ---")
    print(f"File huấn luyện: {train_file_path}")
    print("=" * 50)

    # --- 2. Tải Dữ liệu ---
    print("Bắt đầu tải và chuẩn bị dữ liệu...")
    actual_vocab_size = load_vocabulary_size(vocab_file_path)
    if actual_vocab_size is None:
        return
    X_train, y_train = load_data_from_jsonl(train_file_path, args.max_len)
    X_val, y_val = load_data_from_jsonl(val_file_path, args.max_len)
    if X_train is None or X_val is None:
        return

    # --- 3. Xây dựng và Compile Model ---
    print("\nBắt đầu xây dựng và compile model...")
    model = create_model(
        maxlen=args.max_len,
        vocab_size=actual_vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_transformer_blocks=args.num_blocks,
        dropout_rate=args.dropout,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    model.summary()

    # --- 4. Thiết lập Callbacks ---
    best_model_run_path = os.path.join(run_specific_output_dir, "best_model_of_run.h5")
    csv_log_path = os.path.join(run_specific_output_dir, "training_log.csv")
    history_plot_path = os.path.join(run_specific_output_dir, "training_history.png")

    callbacks = [
        ModelCheckpoint(
            filepath=best_model_run_path,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
            verbose=1,
            restore_best_weights=True,
        ),
        CSVLogger(csv_log_path),
    ]

    # --- 5. Bắt đầu Huấn luyện ---
    print("\n--- BẮT ĐẦU HUẤN LUYỆN ---")
    history = model.fit(
        X_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )
    print("--- HUẤN LUYỆN KẾT THÚC ---")

    # --- 6. Lưu các kết quả ---
    plot_training_history(history, history_plot_path)

    final_ensemble_model_path = os.path.join(
        ensemble_output_dir, f"model_{train_index:02d}.h5"
    )
    print(
        f"\nĐang lưu model TỐT NHẤT cho quá trình ensemble tại: {final_ensemble_model_path}"
    )
    model.save(final_ensemble_model_path)

    # Lưu file JSON chứa các thông số và kết quả tốt nhất
    best_val_loss = min(history.history["val_loss"])
    final_metrics_path = os.path.join(
        run_specific_output_dir, f"training_result_file_{train_index:02d}.json"
    )
    final_metrics = {
        "train_file": train_filename,
        "ensemble_model_path": final_ensemble_model_path,
        "best_val_loss": best_val_loss,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
        },
    }
    with open(final_metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=4)
    print(f"Đã lưu thông số và kết quả tại: {final_metrics_path}")
    print(f"--- HOÀN TẤT MODEL {train_index:02d} ---")


# --- HÀM MAIN ĐỂ ĐIỀU PHỐI ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Transformer model(s) for vulnerability detection on a local machine."
    )

    # Tham số về đường dẫn
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the encoded_balanced data.",
    )
    parser.add_argument(
        "--vocab_dir",
        type=str,
        required=True,
        help="Directory containing the vocab_balanced.json file.",
    )
    parser.add_argument(
        "--model_script_dir",
        type=str,
        default=".",
        help="Directory containing the model.py script.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base directory to save all trained models, logs, and plots.",
    )

    # Tham số về việc huấn luyện model nào
    parser.add_argument(
        "--train_index",
        type=str,
        default="all",
        help='Index of the training file to use (e.g., 1 for train_balanced_01.jsonl), or "all" to train on all 14 files.',
    )

    # Tham số về siêu tham số của mô hình và training
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=128)
    parser.add_argument(
        "--num_blocks", type=int, default=4, help="Number of Transformer blocks."
    )
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")

    args = parser.parse_args()

    if args.train_index.lower() == "all":
        print("Chế độ huấn luyện: TẤT CẢ CÁC MODEL (1 đến 14)")
        for i in range(1, 15):  # Lặp từ 1 đến 14
            train_single_model(args, i)
        print("\n\nĐÃ HUẤN LUYỆN XONG TẤT CẢ 14 MODEL!")
    else:
        try:
            index = int(args.train_index)
            if 1 <= index <= 14:
                print(f"Chế độ huấn luyện: MODEL ĐƠN LẺ (index = {index})")
                train_single_model(args, index)
            else:
                print("LỖI: --train_index phải là một số từ 1 đến 14, hoặc 'all'.")
        except ValueError:
            print("LỖI: --train_index không hợp lệ. Vui lòng nhập một số hoặc 'all'.")
