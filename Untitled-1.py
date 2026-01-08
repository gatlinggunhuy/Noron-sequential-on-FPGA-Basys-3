import sqlite3
from datetime import datetime
import hashlib
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
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
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200
SERIAL_TIMEOUT = 10  # Thời gian chờ 10 giây
OUTPUT_SIZE_FPGA = 1  # Chỉ nhận 1 byte (chỉ số lớp)
MAX_RETRIES = 3  # Số lần thử lại tối đa

def init_database():
    conn = sqlite3.connect('food_recognition_api.db')
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
        ('caesar_salad', 'healthy'),  # Thay 'salad' bằng 'caesar_salad'
        ('hamburger', 'fast food'),   # Thay 'burger' bằng 'hamburger'
        ('pizza', 'fast food')
    ]
    cursor.executemany('INSERT OR IGNORE INTO food_categories (food_name, category) VALUES (?, ?)', food_category_data)

    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    conn = sqlite3.connect('food_recognition_api.db')
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
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect('food_recognition_api.db')
    cursor = conn.cursor()
    hashed_password = hash_password(password)
    cursor.execute('SELECT id, image_folder FROM users WHERE username = ? AND password = ?',
                   (username, hashed_password))
    user_data = cursor.fetchone()
    conn.close()
    if user_data:
        return user_data[0], user_data[1]
    return None, None

def save_image_record(user_id, image_path, category, confidence):
    conn = sqlite3.connect('food_recognition_api.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO images (user_id, image_path, category, confidence, upload_date)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, image_path, category, confidence, datetime.now()))
    conn.commit()
    conn.close()

def load_and_prepare_image_for_fpga(image_path):
    try:
        img = image.load_img(image_path, target_size=(32, 32), color_mode='rgb')  # Thay grayscale bằng rgb
        x = image.img_to_array(img)
        x = x.flatten().astype('uint8')
        if len(x) != 3072:  # 32x32x3 = 3072 byte (RGB)
            print(f"Kích thước dữ liệu ảnh không khớp (mong đợi 3072, nhận được {len(x)})")
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

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        if username and password and register_user(username, password):
            return jsonify({'message': 'User registered successfully'}), 201
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





@app.route('/recommend', methods=['GET'])
def recommend_food():
    try:
        search_term = request.args.get('query')
        if not search_term:
            return jsonify({'error': 'Search query is required'}), 400

        conn = sqlite3.connect('food_recognition_api.db')
        cursor = conn.cursor()

        cursor.execute('SELECT DISTINCT category FROM food_categories WHERE food_name LIKE ?', ('%' + search_term + '%',))
        category_result = cursor.fetchall()
        categories = [row[0] for row in category_result]

        recommended_foods = []
        if categories:
            cursor.execute('SELECT food_name FROM food_categories WHERE category IN ({})'.format(','.join('?'*len(categories))), categories)
            food_results = cursor.fetchall()
            recommended_foods = [row[0] for row in food_results]

        conn.close()
        return jsonify({'recommendations': recommended_foods}), 200
    except Exception as e:
        print(f"Lỗi trong endpoint /recommend: {e}")
        return jsonify({'error': f'Lỗi trong endpoint /recommend: {e}'}), 500

if __name__ == '__main__':
    init_database()
    app.run(debug=True)














@app.route('/upload_to_fpga', methods=['POST'])
def upload_image_to_fpga():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image part in the request'}), 400
        file = request.files['image']
        user_id = request.form.get('user_id')
        if not user_id:
            return jsonify({'error': 'User ID is required'}), 400

        user_folder_conn = sqlite3.connect('food_recognition_api.db')
        cursor = user_folder_conn.cursor()
        cursor.execute('SELECT image_folder FROM users WHERE id = ?', (user_id,))
        result = cursor.fetchone()
        user_folder = result[0] if result else os.path.join(BASE_IMAGE_FOLDER, 'default')
        user_folder_conn.close()
        os.makedirs(user_folder, exist_ok=True)

        if file.filename == '':
            return jsonify({'error': 'No selected image'}), 400
        
        filename = os.path.join(user_folder, f"{datetime.now().timestamp()}.jpg")
        file.save(filename)

        img_bytes = load_and_prepare_image_for_fpga(filename)
        if img_bytes is None:
            return jsonify({'error': 'Failed to prepare image for FPGA'}), 500

        ser = establish_serial_connection()
        if ser is None:
            return jsonify({'error': 'Không thể kết nối với FPGA'}), 503

        try:
            print("Đang gửi dữ liệu ảnh đến FPGA...")
            print(f"Dữ liệu gửi đi (đầu 10 byte): {img_bytes[:10]}")
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
                0: "caesar_salad",
                1: "hamburger",
                2: "pizza"
            }
            predicted_category = prediction_map.get(prediction_result[0], "unknown")

            ser.close()
            return jsonify({
                'prediction_from_fpga_bytes': prediction_result,
                'predicted_category': predicted_category
            }), 200

        except Exception as e:
            print(f"Lỗi giao tiếp nối tiếp: {e}")
            if ser.is_open:
                ser.close()
            return jsonify({'error': f'Lỗi giao tiếp nối tiếp: {e}'}), 500
        finally:
            if ser.is_open:
                ser.close()

    except Exception as e:
        print(f"Lỗi trong endpoint /upload_to_fpga: {e}")
        return jsonify({'error': f'Lỗi trong endpoint /upload_to_fpga: {e}'}), 500

