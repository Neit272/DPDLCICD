import tensorflow as tf

# Liệt kê các thiết bị vật lý có thể sử dụng
devices = tf.config.list_physical_devices('GPU')
if devices:
    print("Có GPU sẵn sàng cho TensorFlow:")
    for device in devices:
        print(f"- {device}")
else:
    print("Không có GPU phát hiện. Đang sử dụng CPU.")
