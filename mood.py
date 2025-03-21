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
CONFIDENCE_THRESHOLD = 10  # Threshold untuk testing

# Konfigurasi Ollama Llava untuk deteksi mood
OLLAMA_ENABLED = True  # Set ke False jika tidak ingin menggunakan Llava
# OLLAMA_URL = "http://localhost:11434/api/generate"  # URL API Ollama
OLLAMA_URL = "http://10.53.25.239:11434/api"  # URL API Ollama
OLLAMA_MODEL = "llava:latest"  # Nama default, akan otomatis mencari model llava yang tersedia

# Konfigurasi Database PostgreSQL
DB_CONFIG = {
    "host": "103.149.230.107",
    "database": "demo_staging_bi",  # Ganti dengan nama database Anda
    "user": "postgres",  # Ganti dengan username Anda
    "password": "P@ssw0rd",  # Ganti dengan password Anda
    "port": "6623"  # Ganti jika port berbeda
}
DB_TABLE = "data_absensi"  # Nama tabel untuk menyimpan data absensi

# Konfigurasi Kelas - gunakan kelas yang tetap tanpa nama orang
KELAS_MAP = {
    "CHRISTINE": "CHR",
    "DENIS": "DEN",
    "FADLY": "FDL",
    "MUMTAZ": "MTZ",
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
            # Buat tabel baru dengan kolom mood dan mood_analysis
            create_table_query = f"""
            CREATE TABLE {DB_TABLE} (
                id SERIAL PRIMARY KEY,
                nama TEXT NOT NULL,
                kelas TEXT,
                tanggal DATE NOT NULL,
                waktu TIME NOT NULL,
                status TEXT,
                confidence NUMERIC(5, 2),
                mood TEXT,
                mood_analysis TEXT
            )
            """
            cursor.execute(create_table_query)
            conn.commit()
            print(f"Table {DB_TABLE} created successfully")
        else:
            # Cek apakah kolom mood_analysis sudah ada
            try:
                check_column_query = f"""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = '{DB_TABLE}' AND column_name = 'mood_analysis'
                """
                cursor.execute(check_column_query)
                analysis_column_exists = cursor.fetchone()

                # Jika kolom mood_analysis belum ada, tambahkan
                if not analysis_column_exists:
                    add_column_query = f"""
                    ALTER TABLE {DB_TABLE} 
                    ADD COLUMN mood_analysis TEXT
                    """
                    cursor.execute(add_column_query)
                    conn.commit()
                    print(f"Added 'mood_analysis' column to {DB_TABLE}")
            except Exception as e:
                print(f"Error checking/adding mood_analysis column: {e}")

        cursor.close()
        return True
    except Exception as e:
        print(f"Error creating table: {e}")
        conn.rollback()
        return False


# Tambahkan fungsi ini ke kode Anda untuk menampilkan model yang tersedia
def list_available_ollama_models():
    try:
        response = requests.get(OLLAMA_URL+"/tags", timeout=5)
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
        response = requests.get(OLLAMA_URL+"/tags", timeout=5)
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
        return {"category": "Normal", "analysis": "Fitur analisis mood tidak aktif"}

    try:
        # Konversi image ke base64
        _, buffer = cv2.imencode('.jpg', face_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Buat prompt yang meminta kategori dan penjelasan
        detailed_prompt = """
        Analisis ekspresi wajah orang ini dan berikan:
        1. Kategori mood (pilih satu: Stress, Good, atau Normal)
        2. Penjelasan singkat (maks 50 kata) mengapa Anda memilih kategori tersebut berdasarkan fitur wajah yang terlihat.

        Format jawaban:
        KATEGORI: [kategori mood]
        ANALISIS: [penjelasan singkat]
        """

        # Buat payload untuk API Ollama
        payload = {
            "model": "llava:latest",  # Pastikan menggunakan nama yang tepat
            "prompt": detailed_prompt,
            "stream": False,
            "images": [img_base64]
        }

        # Debug: Log payload size
        print(f"Image payload size: {len(img_base64)} bytes")

        # Kirim request ke Ollama dengan timeout lebih lama
        print(f"Sending request to Ollama using model {payload['model']}...")
        response = requests.post(OLLAMA_URL+"/chat", json=payload, timeout=30)

        # Debug: Log full response
        print(f"Response status: {response.status_code}")
        print(f"Response text: {response.text[:200]}...")  # Print first 200 chars

        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', '').strip()

            # Parsing respons untuk ekstrak kategori dan analisis
            category = "Normal"  # Default
            analysis = ""

            # Coba ekstrak kategori dan analisis dari respons
            if "KATEGORI:" in response_text:
                parts = response_text.split("KATEGORI:", 1)[1].split("ANALISIS:", 1)
                if len(parts) > 0:
                    category = parts[0].strip()
                if len(parts) > 1:
                    analysis = parts[1].strip()
            else:
                # Fallback jika format tidak sesuai
                analysis = response_text[:200]  # Batasi panjang
                # Tentukan kategori dari teks
                if "stress" in response_text.lower() or "tegang" in response_text.lower():
                    category = "Stress"
                elif "bahagia" in response_text.lower() or "senang" in response_text.lower() or "positif" in response_text.lower():
                    category = "Good"
                else:
                    category = "Normal"

            # Return objek yang berisi kategori dan analisis
            return {"category": category, "analysis": analysis}
        else:
            return {"category": "Error", "analysis": f"HTTP Error: {response.status_code}"}
    except Exception as e:
        print(f"Error analyzing mood: {e}")
        traceback.print_exc()
        return {"category": "Error", "analysis": str(e)[:100]}


def record_attendance(name, confidence, face_img=None):
    # Analisis mood jika gambar wajah tersedia
    mood_data = {"category": "Normal", "analysis": ""}  # Default mood

    if face_img is not None and OLLAMA_ENABLED:
        # Cek cache dulu
        cache_key = f"{name}_{int(time.time() / 60)}"  # Cache per menit per orang
        if cache_key in mood_cache:
            mood_data = mood_cache[cache_key]
        else:
            # Analisis mood dengan Ollama
            mood_data = analyze_mood(face_img)
            # Simpan ke cache
            mood_cache[cache_key] = mood_data

    # Catat ke database jika tersedia
    db_success = False
    if db_conn and use_database:
        db_success = record_attendance_to_db(db_conn, name, confidence, mood_data)

    # Catat ke CSV (sebagai backup atau jika database tidak tersedia)
    csv_success = record_attendance_to_csv(name, confidence, mood_data)

    # Return True jika salah satu berhasil
    return (db_success or csv_success), mood_data


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

    # Mengambil gambar dari setiap subfolder (kelas) di dalam folder Model
    for class_folder in os.listdir(MODEL_PATH):
        class_path = os.path.join(MODEL_PATH, class_folder)
        if os.path.isdir(class_path):  # Pastikan itu folder (kelas)
            for image_file in os.listdir(class_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    print(f"Processing {image_file} from {class_folder}")
                    image_path = os.path.join(class_path, image_file)

                    # Load gambar
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"Error loading {image_path}. File mungkin rusak.")
                        continue

                    # Preprocessing gambar
                    gray = preprocess_face(img)

                    # Deteksi wajah
                    detected_faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                    if len(detected_faces) > 0:
                        (x, y, w, h) = detected_faces[0]

                        # Ekstrak wajah
                        face_roi = gray[y:y + h, x:x + w]

                        # Resize wajah
                        face_roi = cv2.resize(face_roi, (100, 100))

                        # Tentukan label untuk kelas berdasarkan folder
                        if class_folder not in label_map:
                            label_map[class_folder] = label_count
                            label_count += 1

                        faces.append(face_roi)
                        labels.append(label_map[class_folder])
                        print(f"Loaded {class_folder}")
                    else:
                        print(f"Warning: No face found in {image_file}")

    # Melatih model dengan data wajah
    if len(faces) > 0 and len(faces) == len(labels):
        print(f"Training with {len(faces)} faces...")
        try:
            recognizer.train(faces, np.array(labels))
            print("Training complete!")

            # Lookup table untuk label ke nama kelas
            label_to_name = {v: k for k, v in label_map.items()}
            return label_to_name
        except Exception as e:
            print(f"Error during training: {e}")
            return {}
    else:
        print("Error: No faces found for training.")
        return {}

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

        # Ekstrak kategori mood dan analisis dari dict mood
        mood_category = mood.get('category', 'Normal')  # Default 'Normal' jika tidak ada kategori
        mood_analysis = mood.get('analysis', '')  # Default kosong jika tidak ada analisis

        # Tambahkan data ke database
        cursor = conn.cursor()
        insert_query = sql.SQL("""
            INSERT INTO {} (nama, kelas, tanggal, waktu, status, confidence, mood, mood_analysis)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """).format(sql.Identifier(DB_TABLE))

        cursor.execute(insert_query, (name, kelas, date_string, time_string, 'Hadir', confidence, mood_category, mood_analysis))
        conn.commit()
        cursor.close()

        print(f"Recorded attendance to database for {name} ({kelas}) at {time_string} - Mood: {mood_category}")
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
        response = requests.get(OLLAMA_URL+"/tags", timeout=2)
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

    # Mulai video capture
    print("Opening webcam...")
    video_capture = cv2.VideoCapture(2)

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
                    if name != "Unknown" and detected_mood["category"] != "Unknown":
                        mood_color = (0, 255, 0) if detected_mood["category"] == "Good" else (
                            0, 0, 255) if detected_mood["category"] == "Stress" else (255, 165, 0)

                        # Tampilkan kategori mood
                        cv2.putText(display_frame, f"Mood: {detected_mood['category']}", (x, y + h + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mood_color, 2)

                        # Tampilkan analisis mood (maksimal 60 karakter)
                        analysis_text = detected_mood.get("analysis", "")
                        if analysis_text:
                            # Batasi panjang teks dan bagi menjadi beberapa baris jika perlu
                            max_length = 60
                            if len(analysis_text) > max_length:
                                analysis_text = analysis_text[:max_length] + "..."

                            cv2.putText(display_frame, f"Analisis: {analysis_text}", (x, y + h + 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

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
