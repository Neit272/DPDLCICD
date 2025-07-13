import tensorflow as tf
from tensorflow.python.client import device_lib
tf.debugging.set_log_device_placement(True)

devices = device_lib.list_local_devices()
for d in devices:
    if d.device_type == "GPU":
        print(f"Name: {d.name}")
        print(f"Device: {d.physical_device_desc}")
