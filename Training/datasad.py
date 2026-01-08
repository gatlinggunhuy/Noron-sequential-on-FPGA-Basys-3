import tensorflow_datasets as tfds
import tensorflow as tf
import os
import shutil

# Tải Food-101
dataset, info = tfds.load('food101', with_info=True, as_supervised=True, split=['train', 'validation'])
train_data, val_data = dataset[0], dataset[1]

# Lấy danh sách các lớp trong Food-101
all_classes = info.features['label'].names
print("Danh sách các lớp trong Food-101:", all_classes)

# Định nghĩa các lớp muốn lọc (chỉ 3 lớp)
selected_classes = ['pizza', 'hamburger', 'caesar_salad']  # Điều chỉnh nếu cần
class_indices = {name: idx for idx, name in enumerate(all_classes)}
selected_class_indices = [class_indices[class_name] for class_name in selected_classes if class_name in class_indices]

if not selected_class_indices:
    raise ValueError("Không có lớp nào trong selected_classes được tìm thấy trong Food-101.")

def filter_classes(image, label):
    class_tensor = tf.constant(selected_class_indices, dtype=tf.int64)
    return tf.reduce_any(tf.equal(label, class_tensor))

# Lọc dữ liệu với giới hạn
train_filtered = train_data.filter(filter_classes).take(2000)  # Giới hạn 1000 ảnh train
val_filtered = val_data.filter(filter_classes).take(400)      # Giới hạn 200 ảnh validation

# Định nghĩa đường dẫn thư mục
base_dir = r'C:/Users/tntha/Desktop/pyproject/pythoncode/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')

# Tạo hoặc làm sạch thư mục
for dirpath in [train_dir, val_dir]:
    if os.path.exists(dirpath):
        for class_name in os.listdir(dirpath):
            if class_name not in selected_classes:
                shutil.rmtree(os.path.join(dirpath, class_name))
    os.makedirs(dirpath, exist_ok=True)

# Hàm lưu ảnh
def save_images(data, split_dir):
    for i, (image, label) in enumerate(data):
        class_name = info.features['label'].int2str(label.numpy())
        if class_name in selected_classes:
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            image_resized = tf.image.resize(image, [224, 224])
            tf.keras.preprocessing.image.save_img(
                os.path.join(class_dir, f'{i}.jpg'),
                image_resized
            )
            print(f"Lưu ảnh: {class_dir}/{i}.jpg")

# Lưu dữ liệu
save_images(train_filtered, train_dir)
save_images(val_filtered, val_dir)

print("Dữ liệu đã được lưu vào dataset/train và dataset/validation")