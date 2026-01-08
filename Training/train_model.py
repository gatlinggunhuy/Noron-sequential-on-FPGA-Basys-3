import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Kiểm tra thiết bị và version
print("TensorFlow version:", tf.__version__)
print("Keras available:", hasattr(tf, 'keras'))
physical_devices = tf.config.list_physical_devices()
print("Danh sách thiết bị:", physical_devices)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU được phát hiện:", gpus)
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
else:
    print("Không tìm thấy GPU, sẽ sử dụng CPU.")

# Đường dẫn đến thư mục dữ liệu
train_dir = 'C:/Users/tntha/Desktop/pyproject/pythoncode/dataset/train'
validation_dir = 'C:/Users/tntha/Desktop/pyproject/pythoncode/dataset/validation'

# Tạo ImageDataGenerator để tiền xử lý ảnh
train_datagen = ImageDataGenerator(
    channel_shift_range=0.1,
    rescale=1./255,
    rotation_range=40,  # Tăng từ 30
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,  # Thêm shear để biến dạng
    horizontal_flip=True,
    vertical_flip=True,  # Thêm flip dọc nếu ảnh thức ăn cho phép
    zoom_range=0.3,  # Tăng từ 0.2
    brightness_range=[0.7, 1.3],  # Mở rộng
    fill_mode='nearest'  # Để fill pixel sau transform
)
val_datagen = ImageDataGenerator(rescale=1./255)

# Tải dữ liệu huấn luyện
batch_size = 64
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb'
)

# Tải dữ liệu validation
validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb'
)

# In ra nhãn lớp
print("Class indices:", train_generator.class_indices)

# Xây dựng mô hình
model = Sequential([
    Conv2D(128, (3, 3), activation='relu', input_shape=(224, 224, 3), kernel_regularizer=l2(0.0005)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.40),
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.0005)),
    BatchNormalization(),

    GlobalAveragePooling2D(),  # Thay MaxPooling2D thứ hai để giảm tham số
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(3, activation='softmax')  # 3 lớp: pizza, burger, salad
])

# Biên dịch mô hình
model.compile(
    optimizer=SGD(learning_rate=0.001,momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Tính steps_per_epoch và validation_steps
steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

# Thêm callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# Huấn luyện mô hình
print("Bắt đầu huấn luyện trên:", tf.config.get_visible_devices())
history = model.fit(
    train_generator,
    steps_per_epoch=None,
    validation_data=validation_generator,
    validation_steps=None,
    epochs=100,
    callbacks=[early_stopping, lr_scheduler]
)
validation_generator.reset()
preds = np.argmax(model.predict(validation_generator, steps=validation_steps+1), axis=1)
true = validation_generator.classes[:len(preds)]
print(confusion_matrix(true, preds))

# Lưu mô hình
model.save('C:/Users/tntha/Desktop/pyproject/pythoncode/food_classifier.h5')
print("Mô hình đã được lưu vào food_classifier.h5")

# Vẽ biểu đồ loss
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Vẽ biểu đồ accuracy
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()