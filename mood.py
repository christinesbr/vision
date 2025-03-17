import os
import cv2
import numpy as np
import datetime
import csv
import time
import traceback
import json
import requests
import base64
from io import BytesIO
from PIL import Image

# Tambahkan import psycopg2
try:
    import psycopg2
    from psycopg2 import sql

    PSYCOPG2_AVAILABLE = True
except ImportError:
    print("Warning: psycopg2 not installed. Database functionality disabled.")
    print("Install with: pip install psycopg2-binary")
    PSYCOPG2_AVAILABLE = False

# Konfigurasi
MODEL_PATH = "Model"  # Path folder foto model
# OUTPUT_FILE = "6.csv"  # File CSV untuk output
CONFIDENCE_THRESHOLD = 30  # Threshold untuk testing

# Konfigurasi Ollama Llava untuk deteksi mood
OLLAMA_ENABLED = True  # Set ke False jika tidak ingin menggunakan Llava
OLLAMA_URL = "http://localhost:11434/api/generate"  # URL API Ollama
OLLAMA_MODEL = "llava"  # Nama default, akan otomatis mencari model llava yang tersedia

# Konfigurasi Database PostgreSQL
DB_CONFIG = {
    "host": "localhost",
    "database": "localhost-chr",  # Ganti dengan nama database Anda
    "user": "postgres",  # Ganti dengan username Anda
    "password": "postgres",  # Ganti dengan password Anda
    "port": "5432"  # Ganti jika port berbeda
}
DB_TABLE = "data_absensi"  # Nama tabel untuk menyimpan data absensi

# Konfigurasi Kelas - gunakan kelas yang tetap tanpa nama orang
KELAS_MAP = {
    "CHRISTINE": "HRK",
    "CHRISTINE S": "CHR",
    "DENIS": "DEN",
    "FADLY": "FDL",
    "FAISAL": "FAI"
    # Tambahkan mapping lain sesuai kebutuhan
    # Jika tidak ada di mapping, akan menggunakan "Kelas Default"
}
DEFAULT_KELAS = "Kelas Default"

# Inisialisasi face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inisialisasi recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Variable untuk koneksi database
db_conn = None
use_database = False

# Cache untuk mood untuk menghindari terlalu banyak panggilan ke Llava
mood_cache = {}


# Fungsi untuk menghubungkan ke database
def connect_to_database():
    global use_database

    if not PSYCOPG2_AVAILABLE:
        print("Database functionality disabled (psycopg2 not installed)")
        return None

    try:
        print(f"Connecting to database at {DB_CONFIG['host']}:{DB_CONFIG['port']}...")
        conn = psycopg2.connect(
            host=DB_CONFIG["host"],
            database=DB_CONFIG["database"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            port=DB_CONFIG["port"],
            # Tambahkan timeout untuk menghindari hanging
            connect_timeout=5
        )
        print("Database connection successful")
        use_database = True

        # Coba buat tabel jika belum ada
        create_table(conn)

        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        print("Continuing without database connection. Data will be saved to CSV only.")
        use_database = False
        return None


# Fungsi untuk membuat tabel jika belum ada
def create_table(conn):
    if not conn:
        return False

    try:
        cursor = conn.cursor()

        # Cek apakah tabel sudah ada
        cursor.execute(f"SELECT to_regclass('public.{DB_TABLE}')")
        table_exists = cursor.fetchone()[0]

        if not table_exists:
            # Buat tabel baru dengan kolom mood
            create_table_query = f"""
            CREATE TABLE {DB_TABLE} (
                id SERIAL PRIMARY KEY,
                nama TEXT NOT NULL,
                kelas TEXT,
                tanggal DATE NOT NULL,
                waktu TIME NOT NULL,
                status TEXT,
                confidence NUMERIC(5, 2),
                mood TEXT
            )
            """
            cursor.execute(create_table_query)
            conn.commit()
            print(f"Table {DB_TABLE} created successfully")
        else:
            # Cek apakah kolom mood sudah ada
            try:
                check_column_query = f"""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = '{DB_TABLE}' AND column_name = 'mood'
                """
                cursor.execute(check_column_query)
                mood_column_exists = cursor.fetchone()

                # Jika kolom mood belum ada, tambahkan
                if not mood_column_exists:
                    add_column_query = f"""
                    ALTER TABLE {DB_TABLE} 
                    ADD COLUMN mood TEXT
                    """
                    cursor.execute(add_column_query)
                    conn.commit()
                    print(f"Added 'mood' column to {DB_TABLE}")
            except Exception as e:
                print(f"Error checking/adding mood column: {e}")

        cursor.close()
        return True
    except Exception as e:
        print(f"Error creating table: {e}")
        conn.rollback()
        return False


# Tambahkan fungsi ini ke kode Anda untuk menampilkan model yang tersedia
def list_available_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            print("\nDaftar model Ollama yang tersedia:")
            for model in models:
                print(f"- {model['name']}")
            return models
        else:
            print(f"Error mendapatkan daftar model: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error saat mengakses API Ollama: {e}")
        return []


# Fungsi yang ditingkatkan untuk memeriksa koneksi Ollama
def check_ollama_connection():
    global OLLAMA_ENABLED, OLLAMA_MODEL

    try:
        # Cek koneksi ke Ollama dengan simple health check
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("Terhubung ke server Ollama.")

            # Tampilkan semua model yang tersedia
            models = list_available_ollama_models()

            # Jika model llava tidak ditemukan, coba cari model llava dengan nama lain
            if not any(model['name'] == OLLAMA_MODEL for model in models):
                print(f"Model '{OLLAMA_MODEL}' tidak ditemukan.")

                # Cari model llava dengan nama yang mirip
                llava_models = [model['name'] for model in models if 'llava' in model['name'].lower()]
                if llava_models:
                    print(f"Model llava ditemukan dengan nama alternatif: {llava_models}")
                    OLLAMA_MODEL = llava_models[0]  # Gunakan model llava pertama yang ditemukan
                    print(f"Menggunakan model: {OLLAMA_MODEL}")
                    OLLAMA_ENABLED = True
                else:
                    print("Tidak ada model llava yang ditemukan.")
                    print("Silakan install dengan perintah: ollama pull llava")
                    OLLAMA_ENABLED = False
            else:
                print(f"Model {OLLAMA_MODEL} tersedia.")
                OLLAMA_ENABLED = True
        else:
            print("Tidak dapat terhubung ke API Ollama.")
            OLLAMA_ENABLED = False
    except Exception as e:
        print(f"Error saat menghubungi Ollama: {e}")
        OLLAMA_ENABLED = False

    return OLLAMA_ENABLED

# Fungsi untuk menganalisis mood menggunakan Ollama Llava
def analyze_mood(face_image):
    global mood_cache, OLLAMA_MODEL

    # Jika Ollama dimatikan, return mood default
    if not OLLAMA_ENABLED:
        return "Normal"

    try:
        # Konversi image ke base64
        _, buffer = cv2.imencode('.jpg', face_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Buat payload untuk API Ollama
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": "Analyze this person's facial expression and determine their mood. Only respond with a single word: either 'Stress' or 'Good'. Don't explain your reasoning or provide any other text.",
            "stream": False,
            "images": [img_base64]
        }

        # Kirim request ke Ollama
        print(f"Sending request to Ollama for mood analysis using model {OLLAMA_MODEL}...")
        response = requests.post(OLLAMA_URL, json=payload, timeout=15)  # Increased timeout

        if response.status_code == 200:
            result = response.json()

            # Extract mood dari response
            mood_text = result.get('response', '').strip()
            print(f"Raw response from Ollama: '{mood_text}'")

            # Normalize mood (Stress or Good)
            if 'stress' in mood_text.lower():
                mood = "Stress"
            elif 'good' in mood_text.lower() or 'happy' in mood_text.lower() or 'positive' in mood_text.lower():
                mood = "Good"
            else:
                mood = "Normal"  # Default jika tidak terdeteksi dengan jelas

            print(f"Detected mood: {mood}")
            return mood
        else:
            print(f"Error from Ollama API: {response.status_code}")
            if response.text:
                print(f"Error details: {response.text}")
            return "Normal"  # Default jika error
    except Exception as e:
        print(f"Error analyzing mood: {e}")
        traceback.print_exc()
        return "Normal"  # Default jika error


# Fungsi untuk preprocessing gambar
def preprocess_face(image):
    # Konversi ke grayscale jika belum
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Normalisasi histogram untuk menangani perbedaan pencahayaan
    gray = cv2.equalizeHist(gray)

    return gray


# Fungsi untuk mendapatkan kelas berdasarkan nama/ID
def get_class_for_student(student_id):
    # Student ID disini adalah nama file tanpa ekstensi
    # Gunakan mapping kelas jika ada, atau default jika tidak ada
    return KELAS_MAP.get(student_id, DEFAULT_KELAS)


# Fungsi untuk training model dari gambar di folder Model
def train_face_recognizer():
    print("Loading face data...")
    faces = []
    labels = []
    label_map = {}
    label_count = 0

    # Periksa semua file dalam folder Model
    for image_file in os.listdir(MODEL_PATH):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing {image_file}")
            image_path = os.path.join(MODEL_PATH, image_file)

            # Load gambar
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error loading {image_path}. File mungkin rusak.")
                continue

            # Preprocessing
            gray = preprocess_face(img)

            # Deteksi wajah
            detected_faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(detected_faces) > 0:
                # Ambil wajah terbesar
                (x, y, w, h) = sorted(detected_faces, key=lambda x: x[2] * x[3], reverse=True)[0]

                # Ekstrak wajah
                face_roi = gray[y:y + h, x:x + w]

                # Resize ke ukuran yang konsisten
                face_roi = cv2.resize(face_roi, (100, 100))

                # Gunakan nama file (tanpa ekstensi) sebagai nama orang
                name = os.path.splitext(image_file)[0]

                # Tentukan label numerik untuk nama
                if name not in label_map:
                    label_map[name] = label_count
                    label_count += 1

                faces.append(face_roi)
                labels.append(label_map[name])
                print(f"Loaded {name}")
            else:
                print(f"Warning: No face found in {image_file}")

    # Training recognizer jika ada data
    if len(faces) > 0 and len(faces) == len(labels):
        print(f"Training with {len(faces)} faces...")
        try:
            recognizer.train(faces, np.array(labels))
            print("Training complete!")

            # Buat lookup table label->name untuk digunakan saat prediksi
            label_to_name = {v: k for k, v in label_map.items()}
            return label_to_name
        except Exception as e:
            print(f"Error during training: {e}")
            return {}
    else:
        print("Error: No faces found for training.")
        return {}


# # Siapkan file CSV
# def setup_csv():
#     try:
#         # Cek apakah file sudah ada
#         file_exists = os.path.isfile(OUTPUT_FILE)
#
#         # Buat file CSV jika belum ada
#         if not file_exists:
#             with open(OUTPUT_FILE, 'w', newline='') as csvfile:
#                 fieldnames = ['Nama', 'Kelas', 'Tanggal', 'Waktu', 'Status', 'Confidence', 'Mood']
#                 writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#                 writer.writeheader()
#         # Jika file sudah ada, cek header untuk kompatibilitas
#         else:
#             with open(OUTPUT_FILE, 'r', newline='') as csvfile:
#                 reader = csv.reader(csvfile)
#                 header = next(reader, None)
#
#                 # Jika header lama tidak memiliki kolom Mood, buat file baru
#                 if header and 'Mood' not in header:
#                     print("Updating CSV format to include Mood column...")
#                     # Backup file lama
#                     backup_file = f"{OUTPUT_FILE}.bak"
#                     os.rename(OUTPUT_FILE, backup_file)
#
#                     # Buat file baru dengan header yang benar
#                     with open(OUTPUT_FILE, 'w', newline='') as new_csvfile:
#                         fieldnames = ['Nama', 'Kelas', 'Tanggal', 'Waktu', 'Status', 'Confidence', 'Mood']
#                         writer = csv.DictWriter(new_csvfile, fieldnames=fieldnames)
#                         writer.writeheader()
#
#                         # Copy data lama ke file baru
#                         with open(backup_file, 'r', newline='') as old_csvfile:
#                             reader = csv.DictReader(old_csvfile)
#                             for row in reader:
#                                 # Tambahkan kolom Mood jika tidak ada
#                                 if 'Mood' not in row:
#                                     row['Mood'] = 'Normal'
#                                 writer.writerow(row)
#
#                     print(f"CSV file updated. Backup saved as {backup_file}")
#
#         return True
#     except Exception as e:
#         print(f"Error setting up CSV: {e}")
#         return False


# Fungsi untuk mencatat absensi ke database
def record_attendance_to_db(conn, name, confidence, mood):
    if not conn or not use_database:
        return False

    try:
        # Buat timestamp untuk absensi
        now = datetime.datetime.now()
        date_string = now.strftime('%Y-%m-%d')
        time_string = now.strftime('%H:%M:%S')

        # Dapatkan kelas dari mapping
        kelas = get_class_for_student(name)

        # Tambahkan data ke database
        cursor = conn.cursor()
        insert_query = sql.SQL("""
            INSERT INTO {} (nama, kelas, tanggal, waktu, status, confidence, mood)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """).format(sql.Identifier(DB_TABLE))

        cursor.execute(insert_query, (name, kelas, date_string, time_string, 'Hadir', confidence, mood))
        conn.commit()
        cursor.close()

        print(f"Recorded attendance to database for {name} ({kelas}) at {time_string} - Mood: {mood}")
        return True
    except Exception as e:
        print(f"Error recording to database: {e}")
        if conn:
            try:
                conn.rollback()
            except:
                pass
        return False


# Fungsi untuk mencatat absensi ke CSV
def record_attendance_to_csv(name, confidence, mood):
    try:
        # Buat timestamp untuk absensi
        now = datetime.datetime.now()
        date_string = now.strftime('%Y-%m-%d')
        time_string = now.strftime('%H:%M:%S')

        # Dapatkan kelas dari mapping berdasarkan ID/nama
        kelas = get_class_for_student(name)

        # Tambahkan data ke CSV
        # with open(OUTPUT_FILE, 'a', newline='') as csvfile:
        #     writer = csv.DictWriter(csvfile,
        #                             fieldnames=['Nama', 'Kelas', 'Tanggal', 'Waktu', 'Status', 'Confidence', 'Mood'])
        #     writer.writerow({
        #         'Nama': name,
        #         'Kelas': kelas,
        #         'Tanggal': date_string,
        #         'Waktu': time_string,
        #         'Status': 'Hadir',
        #         'Confidence': f"{confidence:.2f}",
        #         'Mood': mood
        #     })

        print(f"Recorded attendance to CSV for {name} ({kelas}) at {time_string} - Mood: {mood}")
        return True
    except Exception as e:
        print(f"Error recording to CSV: {e}")
        return False


# Fungsi untuk mencatat absensi (ke database dan CSV)
def record_attendance(name, confidence, face_img=None):
    # Analisis mood jika gambar wajah tersedia
    mood = "Normal"  # Default mood
    if face_img is not None and OLLAMA_ENABLED:
        # Cek cache dulu
        cache_key = f"{name}_{int(time.time() / 60)}"  # Cache per menit per orang
        if cache_key in mood_cache:
            mood = mood_cache[cache_key]
        else:
            # Analisis mood dengan Ollama
            mood = analyze_mood(face_img)
            # Simpan ke cache
            mood_cache[cache_key] = mood

    # Catat ke database jika tersedia
    db_success = False
    if db_conn and use_database:
        db_success = record_attendance_to_db(db_conn, name, confidence, mood)

    # Catat ke CSV (sebagai backup atau jika database tidak tersedia)
    csv_success = record_attendance_to_csv(name, confidence, mood)

    # Return True jika salah satu berhasil
    return (db_success or csv_success), mood


# Fungsi untuk memeriksa koneksi Ollama
def check_ollama_connection():
    global OLLAMA_ENABLED

    try:
        # Cek koneksi ke Ollama dengan simple health check
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            # Periksa apakah model llava tersedia
            if any(model['name'] == OLLAMA_MODEL for model in models):
                print(f"Ollama connected. {OLLAMA_MODEL} model is available.")
                OLLAMA_ENABLED = True
            else:
                print(f"Ollama connected, but {OLLAMA_MODEL} model is not available.")
                print(f"Please install the model with: ollama pull {OLLAMA_MODEL}")
                OLLAMA_ENABLED = False
        else:
            print("Could not connect to Ollama API.")
            OLLAMA_ENABLED = False
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        OLLAMA_ENABLED = False

    return OLLAMA_ENABLED


# Main function
def main():
    global db_conn, use_database, OLLAMA_ENABLED

    print("=" * 50)
    print("SISTEM ABSENSI WAJAH DENGAN DETEKSI MOOD")
    print("=" * 50)

    # Cek koneksi Ollama
    if OLLAMA_ENABLED:
        OLLAMA_ENABLED = check_ollama_connection()

    # Mencoba koneksi ke database
    db_conn = connect_to_database()

    # Training model
    label_to_name = train_face_recognizer()
    if not label_to_name:
        print("Error: No face data available. Exiting.")
        if db_conn:
            db_conn.close()
        return

    # Setup CSV
    # if not setup_csv():
    #     print("Error: CSV setup failed. Exiting.")
    #     if db_conn:
    #         db_conn.close()
    #     return

    # Mulai video capture
    print("Opening webcam...")
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open webcam!")
        if db_conn:
            db_conn.close()
        return

    print("\nPetunjuk Penggunaan:")
    print("- Posisikan wajah pada kamera dengan pencahayaan yang baik")
    print("- Tekan 'r' untuk mencatat kehadiran secara manual")
    print("- Tekan 's' untuk menyimpan gambar saat ini")
    print("- Tekan 'd' untuk mencoba koneksi database lagi")
    print("- Tekan 'o' untuk mencoba koneksi Ollama lagi")
    print("- Tekan 'q' untuk keluar\n")

    # Inisialisasi variabel
    attendance_recorded = set()
    mood_record = {}  # Untuk menyimpan mood per orang
    frame_count = 0
    start_time = time.time()
    force_record = False

    while True:
        # Ambil frame dari video
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_count += 1

        # Buat salinan untuk tampilan
        display_frame = frame.copy()

        # Process setiap 3 frame untuk efisiensi atau ketika dipaksa
        if frame_count % 3 == 0 or force_record:
            # Deteksi wajah menggunakan Haar Cascade
            gray = preprocess_face(frame)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Loop semua wajah yang terdeteksi
            for (x, y, w, h) in faces:
                try:
                    # Ekstrak dan preprocessing wajah
                    face_roi = gray[y:y + h, x:x + w]
                    face_color = frame[y:y + h, x:x + w]  # For mood analysis

                    # Resize ke ukuran yang sama dengan training
                    face_roi = cv2.resize(face_roi, (100, 100))

                    # Prediksi wajah
                    label, confidence = recognizer.predict(face_roi)

                    # Konversi confidence 0-100
                    confidence_score = 100 - min(100, confidence)

                    # Dapatkan nama dan status
                    name = label_to_name.get(label, "Unknown")
                    verified = name in attendance_recorded

                    # Dapatkan kelas
                    kelas = get_class_for_student(name)

                    # Tentukan warna box berdasarkan confidence
                    if confidence_score >= CONFIDENCE_THRESHOLD or force_record:
                        box_color = (0, 255, 0)  # Hijau jika confidence tinggi

                        # Jika belum terekam atau force record, catat absensi
                        if (not verified or force_record) and name != "Unknown":
                            success, detected_mood = record_attendance(name, confidence_score, face_color)
                            if success:
                                attendance_recorded.add(name)
                                mood_record[name] = detected_mood
                        elif name in mood_record:
                            detected_mood = mood_record[name]
                        else:
                            detected_mood = "Normal"
                    else:
                        box_color = (0, 165, 255)  # Orange jika confidence rendah
                        detected_mood = "Unknown"

                    # Gambar box di sekitar wajah
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), box_color, 2)

                    # Label dengan nama, kelas, dan status
                    status = "✓ Terverifikasi" if verified else f"Mendeteksi... ({confidence_score:.1f}%)"
                    label_text = f"{name} | {kelas} | {status}"
                    cv2.putText(display_frame, label_text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                    # Tampilkan mood di bawah nama
                    if name != "Unknown" and detected_mood != "Unknown":
                        mood_color = (0, 255, 0) if detected_mood == "Good" else (
                        0, 0, 255) if detected_mood == "Stress" else (255, 165, 0)
                        cv2.putText(display_frame, f"Mood: {detected_mood}", (x, y + h + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mood_color, 2)

                except Exception as e:
                    print(f"Error processing face: {e}")
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Reset force record flag
        force_record = False

        # Hitung dan tampilkan FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # Tampilkan informasi
        cv2.putText(display_frame, f"Terdeteksi: {len(attendance_recorded)}/{len(label_to_name)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Tampilkan nama-nama yang sudah terabsen
        y_pos = 60
        for name in attendance_recorded:
            kelas = get_class_for_student(name)
            mood = mood_record.get(name, "Normal")
            mood_text = f" - Mood: {mood}" if mood != "Unknown" else ""
            cv2.putText(display_frame, f"✓ {name} ({kelas}){mood_text}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 25

        # Tampilkan status database
        db_status = "Terhubung ke Database" if (db_conn and use_database) else "Tidak Terhubung ke DB!"
        db_color = (0, 255, 255) if (db_conn and use_database) else (0, 0, 255)
        cv2.putText(display_frame, db_status, (10, display_frame.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, db_color, 2)

        # Tampilkan status Ollama
        ollama_status = "Ollama Aktif" if OLLAMA_ENABLED else "Ollama Tidak Aktif"
        ollama_color = (0, 255, 255) if OLLAMA_ENABLED else (0, 0, 255)
        cv2.putText(display_frame, ollama_status, (10, display_frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ollama_color, 2)

        # Tampilkan instruksi di bagian bawah
        cv2.putText(display_frame, "r: Record | s: Save | d: DB | o: Ollama | q: Quit",
                    (10, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Tampilkan hasil
        cv2.imshow('Sistem Absensi Wajah dengan Deteksi Mood', display_frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # Quit
            break
        elif key == ord('s'):
            # Save current frame
            filename = f"capture_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Gambar disimpan: {filename}")
        elif key == ord('r'):
            # Force record attendance for all detected faces
            force_record = True
            print("Memaksa pencatatan kehadiran...")
        elif key == ord('d'):
            # Try to connect to database again
            print("Mencoba menghubungkan ke database...")
            if db_conn:
                try:
                    db_conn.close()
                except:
                    pass
            db_conn = connect_to_database()
        elif key == ord('o'):
            # Try to connect to Ollama again
            print("Mencoba menghubungkan ke Ollama...")
            OLLAMA_ENABLED = check_ollama_connection()

    # Bersihkan
    video_capture.release()
    cv2.destroyAllWindows()
    if db_conn:
        db_conn.close()

    print("\nPencatatan Kehadiran Selesai")
    # print(f"Data absensi tersimpan di: {os.path.abspath(OUTPUT_FILE)}")
    if use_database:
        print(f"Data absensi juga tersimpan di database: {DB_CONFIG['database']}.{DB_TABLE}")
    print(f"Total absensi: {len(attendance_recorded)} orang")
    for name in attendance_recorded:
        kelas = get_class_for_student(name)
        mood = mood_record.get(name, "Normal")
        print(f"- {name} ({kelas}) - Mood: {mood}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unhandled error: {e}")
        traceback.print_exc()