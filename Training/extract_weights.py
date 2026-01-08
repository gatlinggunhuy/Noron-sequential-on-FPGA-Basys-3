import tensorflow as tf
import numpy as np
import os

# Kiểm tra file mô hình
model_path = 'C:/Users/tntha/Desktop/pyproject/pythoncode/food_classifier.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"File {model_path} không tồn tại. Vui lòng chạy train_model.py để tạo mô hình trước.")

# Tải mô hình
model = tf.keras.models.load_model(model_path)

# Lấy weights và biases
dense_layer = model.get_layer(index=1)
weights, biases = dense_layer.get_weights()

# Chuyển đổi sang 8-bit
def convert_to_8bit(array):
    array = np.clip(array, -1, 1)
    array = (array + 1) * 127.5
    return array.astype(np.uint8)

weights_8bit = convert_to_8bit(weights)
biases_8bit = convert_to_8bit(biases)

# Lưu vào file hex với đường dẫn tuyệt đối
weights_path = 'C:/Users/tntha/Desktop/pyproject/pythoncode/weights_hex.txt'
biases_path = 'C:/Users/tntha/Desktop/pyproject/pythoncode/biases_hex.txt'

with open(weights_path, 'w') as f:
    for i in range(weights_8bit.shape[0]):  # 3072
        for j in range(weights_8bit.shape[1]):  # 3
            f.write(f"{weights_8bit[i,j]:02x}\n")

with open(biases_path, 'w') as f:
    for i in range(biases_8bit.shape[0]):  # 3
        f.write(f"{biases_8bit[i]:02x}\n")

print("Weights shape:", weights_8bit.shape)
print("Biases shape:", biases_8bit.shape)
print(f"Weights đã lưu vào {weights_path}")
print(f"Biases đã lưu vào {biases_path}")