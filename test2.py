import sqlite3
from datetime import datetime
import hashlib
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from flask import Flask, request, jsonify
from flask_cors import CORS
import serial
import time
import sys
import io

# Force UTF-8 encoding for console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False
BASE_IMAGE_FOLDER = "user_images_api"
os.makedirs(BASE_IMAGE_FOLDER, exist_ok=True)

# Cấu hình cổng nối tiếp
SERIAL_PORT = os.environ.get('SERIAL_PORT', 'COM4')
BAUD_RATE = 115200
SERIAL_TIMEOUT = 10  # Thời gian chờ 10 giây ,r
OUTPUT_SIZE_FPGA = 1  # Chỉ nhận 1 byte (chỉ số lớp)
MAX_RETRIES = 3  # Số lần thử lại tối đa

# Tải mô hình MobileNetV2 một lần
MOBILENET_MODEL = MobileNetV2(weights='imagenet')

def init_database():
    with sqlite3.connect('food_recognition_api.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                image_folder TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                image_path TEXT NOT NULL,
                category TEXT,
                confidence FLOAT,
                upload_date DATETIME,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS food_categories (
                food_name TEXT PRIMARY KEY,
                category TEXT
            )
        ''')
        food_category_data = [
            ('pizza', 'Đồ ăn nhanh'),
            ('burger', 'Đồ ăn nhanh'),
            ('salad', 'Đồ ăn chay'),
            ('cake', 'Bánh ngọt'),
            ('ice_cream', 'Kem'),
            ('cookie', 'Bánh quy'),
            ('pancake', 'Ăn sáng'),
            ('pasta', 'Ăn trưa'),
            ('chips', 'Ăn vặt'),
            ('steak', 'Ăn tối'),
            ('dessert', 'Tráng Miệng'),
            ('sushi', 'Đồ ăn Nhật'),
            ('kimchi', 'Đồ ăn Hàn'),
            ('dumpling', 'Đồ ăn Trung'),
            ('banh_mi', 'Bánh mì'),
            ('sticky_rice', 'Xôi'),
            ('fried_rice', 'Cơm chiên')
        ]
        cursor.executemany('INSERT OR IGNORE INTO food_categories (food_name, category) VALUES (?, ?)', food_category_data)
        conn.commit()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    with sqlite3.connect('food_recognition_api.db') as conn:
        cursor = conn.cursor()
        try:
            hashed_password = hash_password(password)
            user_folder = os.path.join(BASE_IMAGE_FOLDER, username)
            os.makedirs(user_folder, exist_ok=True)
            cursor.execute('INSERT INTO users (username, password, image_folder) VALUES (?, ?, ?)',
                           (username, hashed_password, user_folder))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

def login_user(username, password):
    with sqlite3.connect('food_recognition_api.db') as conn:
        cursor = conn.cursor()
        hashed_password = hash_password(password)
        cursor.execute('SELECT id, image_folder FROM users WHERE username = ? AND password = ?',
                       (username, hashed_password))
        user_data = cursor.fetchone()
        return user_data if user_data else (None, None)

def save_image_record(user_id, image_path, category, confidence):
    with sqlite3.connect('food_recognition_api.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO images (user_id, image_path, category, confidence, upload_date)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, image_path, category, confidence, datetime.now()))
        conn.commit()

def load_and_prepare_image_for_fpga(image_path):
    try:
        img = image.load_img(image_path, target_size=(32, 32), color_mode='grayscale')
        x = image.img_to_array(img)
        x = x.flatten().astype('uint8')
        if len(x) != 1024:  # 32x32 = 1024 byte (grayscale)
            print(f"Kích thước dữ liệu ảnh không khớp (mong đợi 1024, nhận được {len(x)})")
            return None
        return x
    except Exception as e:
        print(f"Lỗi tiền xử lý ảnh cho FPGA: {e}")
        return None

def establish_serial_connection():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
        print(f"Đã kết nối với cổng {SERIAL_PORT} ở tốc độ {BAUD_RATE}")
        return ser
    except serial.SerialException as e:
        print(f"Không thể mở cổng {SERIAL_PORT}: {e}")
        return None

def predict_with_mobilenet(image_path):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = MOBILENET_MODEL.predict(x)
        decoded_preds = decode_predictions(preds, top=5)[0]
        
        food_map = {
            'pizza': 'pizza',
            'cheeseburger': 'burger',
            'salad': 'salad',
            'cake': 'cake',
            'ice_cream': 'ice_cream',
            'cookie': 'cookie',
            'pancake': 'pancake',
            'spaghetti_bolognese': 'pasta',
            'french_fries': 'chips',
            'steak': 'steak',
            'chocolate_mousse': 'dessert',
            'sushi': 'sushi',
            'kimchi': 'kimchi',
            'dumpling': 'dumpling',
            'sandwich': 'banh_mi',
            'sticky_rice': 'sticky_rice',
            'fried_rice': 'fried_rice'
        }
        
        for pred in decoded_preds:
            pred_label = pred[1]
            if pred_label in food_map:
                return food_map[pred_label], pred[2]
        return 'unknown', 0.0
    except Exception as e:
        print(f"Lỗi dự đoán MobileNetV2: {e}")
        return 'unknown', 0.0

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        username = data.get('username')  # Đây là email từ form đăng ký
        password = data.get('password')
        if username and password and register_user(username, password):
            with sqlite3.connect('food_recognition_api.db') as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id, image_folder FROM users WHERE username = ?', (username,))
                user_data = cursor.fetchone()
                user_id, user_folder = user_data[0], user_data[1]
            return jsonify({
                'message': 'User registered successfully',
                'user_id': user_id,
                'user_folder': user_folder,
                'email': username
            }), 201
        return jsonify({'message': 'Registration failed'}), 400
    except Exception as e:
        print(f"Lỗi trong endpoint /register: {e}")
        return jsonify({'error': f'Lỗi trong endpoint /register: {e}'}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        print(f"Received data: {data}")
        username = data.get('username')
        password = data.get('password')
        print(f"Username: {username}, Password: {password}")
        user_id, user_folder = login_user(username, password)
        if user_id:
            return jsonify({'message': 'Login successful', 'user_id': user_id, 'user_folder': user_folder}), 200
        return jsonify({'message': 'Invalid credentials'}), 401
    except Exception as e:
        print(f"Lỗi trong endpoint /login: {e}")
        return jsonify({'error': f'Lỗi trong endpoint /login: {e}'}), 500

@app.route('/upload_to_fpga', methods=['POST'])
def upload_image_to_fpga():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image part in the request'}), 400
        file = request.files['image']
        user_id = request.form.get('user_id')
        if not user_id:
            return jsonify({'error': 'User ID is required'}), 400

        with sqlite3.connect('food_recognition_api.db') as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT image_folder FROM users WHERE id = ?', (user_id,))
            result = cursor.fetchone()
            if not result:
                return jsonify({'error': 'User ID does not exist'}), 404
            user_folder = result[0]
        os.makedirs(user_folder, exist_ok=True)

        if file.filename == '':
            return jsonify({'error': 'No selected image'}), 400
        
        filename = os.path.join(user_folder, f"{datetime.now().timestamp()}.jpg")
        file.save(filename)

        # Thử dự đoán với MobileNetV2 trước
        predicted_category, confidence = predict_with_mobilenet(filename)
        
        # Nếu dự đoán thuộc 3 category FPGA, sử dụng FPGA
        if predicted_category in ['pizza', 'burger', 'salad']:
            img_bytes = load_and_prepare_image_for_fpga(filename)
            if img_bytes is None:
                return jsonify({'error': 'Failed to prepare image for FPGA'}), 500

            ser = establish_serial_connection()
            if ser is None:
                return jsonify({'error': 'Không thể kết nối với FPGA'}), 503

            try:
                print("Đang gửi dữ liệu ảnh đến FPGA...")
                ser.write(img_bytes)
                print(f"Đã gửi {len(img_bytes)} byte dữ liệu ảnh đến FPGA.")

                time.sleep(1)

                print("Đang nhận kết quả từ FPGA...")
                prediction_result = []
                retries = 0
                while retries < MAX_RETRIES:
                    prediction_result = []
                    for i in range(OUTPUT_SIZE_FPGA):
                        byte = ser.read(1)
                        if byte:
                            prediction_result.append(int.from_bytes(byte, byteorder='big'))
                            print(f"Byte {i+1} nhận được: {prediction_result[-1]}")
                        else:
                            print(f"Không nhận được byte {i+1} trong lần thử {retries+1}")
                            break
                    if len(prediction_result) == OUTPUT_SIZE_FPGA:
                        break
                    retries += 1
                    print(f"Thử lại lần {retries}: Chỉ nhận được {len(prediction_result)} byte, mong đợi {OUTPUT_SIZE_FPGA} byte.")
                    time.sleep(1)

                if len(prediction_result) != OUTPUT_SIZE_FPGA:
                    print(f"Không nhận đủ byte sau {MAX_RETRIES} lần thử.")
                    ser.close()
                    return jsonify({'error': 'Không nhận được đủ byte từ FPGA'}), 503

                print(f"Kết quả dự đoán từ FPGA (byte): {prediction_result}")

                prediction_map = {
                    0: "pizza",
                    1: "burger",
                    2: "salad"
                }
                predicted_category = prediction_map.get(prediction_result[0], "unknown")
                confidence = 1.0  # FPGA không trả về confidence, giả sử 1.0

                ser.close()
            except serial.SerialException as e:
                print(f"Lỗi giao tiếp nối tiếp: {e}")
                if ser.is_open:
                    ser.close()
                return jsonify({'error': f'Lỗi giao tiếp nối tiếp: {e}'}), 500
            finally:
                if ser.is_open:
                    ser.close()
        else:
            # Sử dụng kết quả từ MobileNetV2
            confidence = float(confidence)

        # Lưu kết quả vào cơ sở dữ liệu
        save_image_record(user_id, filename, predicted_category, confidence)

        return jsonify({
            'predicted_category': predicted_category,
            'confidence': confidence
        }), 200

    except Exception as e:
        print(f"Lỗi trong endpoint /upload_to_fpga: {e}")
        return jsonify({'error': f'Lỗi trong endpoint /upload_to_fpga: {e}'}), 500

@app.route('/recommend', methods=['GET'])
def recommend_food():
    try:
        search_term = request.args.get('query')
        if not search_term:
            return jsonify({'error': 'Search query is required'}), 400

        with sqlite3.connect('food_recognition_api.db') as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT category FROM food_categories WHERE food_name LIKE ?', ('%' + search_term + '%',))
            categories = [row[0] for row in cursor.fetchall()]

            recommended_foods = []
            if categories:
                placeholders = ','.join(['?'] * len(categories))
                cursor.execute(f'SELECT food_name FROM food_categories WHERE category IN ({placeholders}) LIMIT 10', categories)
                recommended_foods = [row[0] for row in cursor.fetchall()]

        return jsonify({'recommendations': recommended_foods}), 200
    except Exception as e:
        print(f"Lỗi trong endpoint /recommend: {e}")
        return jsonify({'error': f'Lỗi trong endpoint /recommend: {e}'}), 500

if __name__ == '__main__':
    init_database()
    app.run(debug=True)