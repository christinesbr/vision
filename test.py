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
import face_recognition  # Perlu diinstall: pip install face_recognition
import pickle

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
FACE_ENCODINGS_FILE = "face_encodings.pkl"  # File untuk menyimpan encoding wajah
CONFIDENCE_THRESHOLD = 0.6  # Threshold untuk matching (lebih rendah = lebih ketat)

# Konfigurasi Ollama Llava untuk deteksi mood
OLLAMA_ENABLED = True  # Set ke False jika tidak ingin menggunakan Llava
OLLAMA_URL = "http://localhost:11434/api/generate"  # URL API Ollama
OLLAMA_MODEL = "llava:latest"  # Ubah dari "llava" menjadi "llava:latest"

# Konfigurasi Database PostgreSQL
DB_CONFIG = {
    "host": "localhost",
    "database": "localhost-chr",  # Ganti dengan nama database Anda
    "user": "postgres",  # Ganti dengan username Anda
    "password": "postgres",  # Ganti dengan password Anda
    "port": "5432"  # Ganti jika port berbeda
}
DB_TABLE = "data_absensi"  # Nama tabel untuk menyimpan data absensi

# Variable untuk koneksi database
db_conn = None
use_database = False

# Cache untuk mood untuk menghindari terlalu banyak panggilan ke Llava
mood_cache = {}


# Fungsi untuk membaca struktur folder dan menghasilkan KELAS_MAP
def generate_class_mapping():
    kelas_map = {}

    # Periksa apakah folder model ada
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model path {MODEL_PATH} does not exist")
        return kelas_map

    # Iterasi melalui semua folder di dalam MODEL_PATH
    for item in os.listdir(MODEL_PATH):
        item_path = os.path.join(MODEL_PATH, item)

        # Jika item adalah folder, ini adalah kelas
        if os.path.isdir(item_path):
            kelas_code = item  # Kode kelas adalah nama folder

            # Iterasi melalui file di dalam folder kelas
            for file in os.listdir(item_path):
                file_path = os.path.join(item_path, file)

                # Jika itu adalah file gambar
                if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Nama orang adalah nama file tanpa ekstensi
                    person_name = os.path.splitext(file)[0]

                    # Tambahkan ke mapping
                    kelas_map[person_name] = kelas_code
                    print(f"Mapped {person_name} to class {kelas_code}")

        # Jika item adalah file gambar di root MODEL_PATH
        elif os.path.isfile(item_path) and item.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Nama orang adalah nama file tanpa ekstensi
            person_name = os.path.splitext(item)[0]

            # Gunakan nama file sebagai kelas juga (untuk gambar yang tidak di dalam subfolder)
            kelas_map[person_name] = person_name
            print(f"Mapped {person_name} (root file) with default class {person_name}")

    return kelas_map


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


# Fungsi untuk menampilkan model Ollama yang tersedia
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


# Fungsi untuk memeriksa koneksi Ollama
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


def analyze_mood(face_image):
    global mood_cache, OLLAMA_MODEL

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
            "model": OLLAMA_MODEL,
            "prompt": detailed_prompt,
            "stream": False,
            "images": [img_base64]
        }

        # Kirim request ke Ollama
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)

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


# Fungsi untuk mendapatkan kelas berdasarkan nama/ID dari mapping dinamis
def get_class_for_student(student_id, kelas_map, default_kelas="Kelas Default"):
    # Student ID disini adalah nama file tanpa ekstensi
    # Gunakan mapping kelas jika ada, atau default jika tidak ada
    return kelas_map.get(student_id, default_kelas)


# Fungsi untuk memproses dan enkode semua gambar wajah
def train_face_recognizer():
    print("Loading face data and creating face encodings...")

    # Kamus untuk menyimpan encoding wajah dan informasi terkait
    known_face_encodings = []
    known_face_names = []
    person_to_folder = {}

    # Periksa jika folder Model ada
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model folder '{MODEL_PATH}' not found")
        return [], [], {}

    # Coba load face encodings dari file jika ada
    if os.path.exists(FACE_ENCODINGS_FILE):
        try:
            with open(FACE_ENCODINGS_FILE, 'rb') as f:
                data = pickle.load(f)
                known_face_encodings = data['encodings']
                known_face_names = data['names']
                person_to_folder = data.get('folders', {})
                print(f"Loaded {len(known_face_names)} face encodings from file.")
                return known_face_encodings, known_face_names, person_to_folder
        except Exception as e:
            print(f"Error loading face encodings from file: {e}")
            # Continue with training

    # Proses semua subfolder (kelas)
    for item in os.listdir(MODEL_PATH):
        item_path = os.path.join(MODEL_PATH, item)

        # Jika item adalah folder
        if os.path.isdir(item_path):
            kelas_code = item  # Kode kelas adalah nama folder
            print(f"Processing class folder: {kelas_code}")

            # Proses semua file gambar dalam folder kelas
            for image_file in os.listdir(item_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(item_path, image_file)
                    person_name = os.path.splitext(image_file)[0]

                    # Proses file gambar
                    process_image_for_encoding(
                        image_path,
                        person_name,
                        kelas_code,
                        known_face_encodings,
                        known_face_names,
                        person_to_folder
                    )

        # Jika item adalah file gambar di root folder Model
        elif os.path.isfile(item_path) and item.lower().endswith(('.png', '.jpg', '.jpeg')):
            person_name = os.path.splitext(item)[0]

            # Proses file gambar
            process_image_for_encoding(
                item_path,
                person_name,
                None,  # Tidak ada kelas karena file di root
                known_face_encodings,
                known_face_names,
                person_to_folder
            )

    # Simpan face encodings ke file untuk digunakan kembali
    if known_face_encodings and known_face_names:
        try:
            with open(FACE_ENCODINGS_FILE, 'wb') as f:
                pickle.dump({
                    'encodings': known_face_encodings,
                    'names': known_face_names,
                    'folders': person_to_folder
                }, f)
            print(f"Saved {len(known_face_names)} face encodings to file.")
        except Exception as e:
            print(f"Error saving face encodings to file: {e}")

    return known_face_encodings, known_face_names, person_to_folder


# Fungsi helper untuk memproses gambar dan mendapatkan encoding
# def process_image_for_encoding(image_path, person_name, class_name, known_face_encodings, known_face_names,
#                                person_to_folder):
#     try:
#         print(f"Processing {image_path}")
#
#         # Load gambar dengan face_recognition
#         image = face_recognition.load_image_file(image_path)
#
#         # Deteksi wajah dalam gambar (menggunakan CNN model untuk akurasi lebih tinggi)
#         face_locations = face_recognition.face_locations(image, model="cnn")
#
#         if face_locations:
#             # Ambil encoding wajah (menggunakan wajah pertama yang ditemukan)
#             face_encoding = face_recognition.face_encodings(image, face_locations, num_jitters=5)[0]
#
#             # Simpan encoding dan nama
#             known_face_encodings.append(face_encoding)
#             known_face_names.append(person_name)
#
#             # Simpan informasi folder/kelas
#             person_to_folder[person_name] = class_name if class_name else person_name
#
#             print(f"Successfully encoded face for {person_name}")
#         else:
#             print(f"Warning: No face found in {image_path}")
#     except Exception as e:
#         print(f"Error processing {image_path}: {e}")

def process_image_for_encoding(image_path, person_name, class_name, known_face_encodings, known_face_names,
                               person_to_folder):
    try:
        print(f"Processing {image_path}")

        # Load gambar dengan face_recognition
        try:
            image = face_recognition.load_image_file(image_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return

        try:
            # Resize gambar jika terlalu besar untuk menghindari masalah memori
            h, w = image.shape[:2]
            if max(h, w) > 1500:  # Jika dimensi > 1500 pixel
                scale = 1500 / max(h, w)
                image = cv2.resize(image, (int(w * scale), int(h * scale)))

            # Deteksi wajah dalam gambar (gunakan model HOG yang lebih ringan jika CNN bermasalah)
            face_locations = face_recognition.face_locations(image, model="hog")  # Coba 'hog' jika 'cnn' crash

            if face_locations:
                # Ambil encoding wajah (kurangi num_jitters jika memori bermasalah)
                face_encoding = face_recognition.face_encodings(image, face_locations, num_jitters=1)[0]

                # Simpan encoding dan nama
                known_face_encodings.append(face_encoding)
                known_face_names.append(person_name)

                # Simpan informasi folder/kelas
                person_to_folder[person_name] = class_name if class_name else person_name

                print(f"Successfully encoded face for {person_name}")
            else:
                print(f"Warning: No face found in {image_path}")
        except Exception as e:
            print(f"Error during face detection/encoding in {image_path}: {e}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")


# Fungsi untuk mencatat absensi ke database
def record_attendance_to_db(conn, name, confidence, mood_data, kelas):
    if not conn or not use_database:
        return False

    try:
        # Buat timestamp untuk absensi
        now = datetime.datetime.now()
        date_string = now.strftime('%Y-%m-%d')
        time_string = now.strftime('%H:%M:%S')

        # Tambahkan data ke database
        cursor = conn.cursor()
        insert_query = sql.SQL("""
            INSERT INTO {} (nama, kelas, tanggal, waktu, status, confidence, mood, mood_analysis)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """).format(sql.Identifier(DB_TABLE))

        cursor.execute(insert_query, (
            name,
            kelas,
            date_string,
            time_string,
            'Hadir',
            confidence * 100,  # Konversi ke persentase
            mood_data["category"],
            mood_data["analysis"]
        ))
        conn.commit()
        cursor.close()

        print(f"Recorded attendance to database for {name} ({kelas}) at {time_string} - Mood: {mood_data['category']}")
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
def record_attendance_to_csv(name, confidence, mood_data, kelas):
    try:
        # Buat timestamp untuk absensi
        now = datetime.datetime.now()
        date_string = now.strftime('%Y-%m-%d')
        time_string = now.strftime('%H:%M:%S')

        # Output CSV filename disesuaikan dengan tanggal
        output_file = f"attendance_{now.strftime('%Y%m%d')}.csv"

        # Cek apakah file sudah ada
        file_exists = os.path.isfile(output_file)

        # Buat atau append ke file CSV
        with open(output_file, 'a', newline='') as csvfile:
            fieldnames = ['Nama', 'Kelas', 'Tanggal', 'Waktu', 'Status', 'Confidence', 'Mood', 'Mood_Analysis']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Tulis header jika file baru
            if not file_exists:
                writer.writeheader()

            # Tulis data
            writer.writerow({
                'Nama': name,
                'Kelas': kelas,
                'Tanggal': date_string,
                'Waktu': time_string,
                'Status': 'Hadir',
                'Confidence': f"{confidence * 100:.2f}",  # Konversi ke persentase
                'Mood': mood_data["category"],
                'Mood_Analysis': mood_data["analysis"]
            })

        print(f"Recorded attendance to CSV for {name} ({kelas}) at {time_string} - Mood: {mood_data['category']}")
        return True
    except Exception as e:
        print(f"Error recording to CSV: {e}")
        return False


# Fungsi untuk mencatat absensi (ke database dan CSV)
def record_attendance(name, confidence, face_img=None, kelas_map=None):
    if kelas_map is None:
        kelas_map = {}

    # Dapatkan kelas dari mapping
    kelas = get_class_for_student(name, kelas_map)

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
        db_success = record_attendance_to_db(db_conn, name, confidence, mood_data, kelas)

    # Catat ke CSV (sebagai backup atau jika database tidak tersedia)
    csv_success = record_attendance_to_csv(name, confidence, mood_data, kelas)

    # Return True jika salah satu berhasil
    return (db_success or csv_success), mood_data


# Main function
def main():
    global db_conn, use_database, OLLAMA_ENABLED

    print("=" * 50)
    print("SISTEM ABSENSI WAJAH DENGAN DETEKSI MOOD (CNN-BASED)")
    print("=" * 50)

    # Cek koneksi Ollama
    if OLLAMA_ENABLED:
        OLLAMA_ENABLED = check_ollama_connection()

    # Mencoba koneksi ke database
    db_conn = connect_to_database()

    # Generate mapping kelas dari struktur folder
    kelas_map = generate_class_mapping()
    print(f"Generated class mapping from folder structure: {kelas_map}")

    # Training model (load/generate face encodings)
    known_face_encodings, known_face_names, person_to_folder = train_face_recognizer()

    if not known_face_encodings or not known_face_names:
        print("Error: No face data available. Exiting.")
        if db_conn:
            db_conn.close()
        return

    print(f"Loaded {len(known_face_names)} faces for recognition")

    # Jika kelas_map kosong, gunakan person_to_folder
    if not kelas_map and person_to_folder:
        kelas_map = person_to_folder
        print(f"Using folder structure for class mapping: {kelas_map}")

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
    process_this_frame = True

    try:
        while True:
            # Ambil frame dari video
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame_count += 1

            # Hanya proses setiap frame ke-2 untuk performa lebih baik (kecuali jika force_record)
            if process_this_frame or force_record:
                # Resize frame untuk pemrosesan lebih cepat
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Konversi dari BGR (OpenCV) ke RGB (face_recognition)
                rgb_small_frame = small_frame[:, :, ::-1]

                # Deteksi wajah dan encoding
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                # Reset daftar nama wajah yang terdeteksi pada frame ini
                face_names = []
                face_confidences = []

                for face_encoding in face_encodings:
                    # Bandingkan dengan semua wajah yang diketahui
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding,
                                                             tolerance=CONFIDENCE_THRESHOLD)
                    name = "Unknown"
                    confidence = 0.0

                    # Gunakan face_distance untuk mendapatkan yang paling mendekati wajah saat ini
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]
                            # Konversi face_distance ke confidence score (1.0 - distance)
                            # face_distance mendekati 0 berarti cocok, mendekati 1 berarti tidak cocok
                            confidence = 1.0 - face_distances[best_match_index]

                    face_names.append(name)
                    face_confidences.append(confidence)

            # Alihkan flag process_this_frame
            process_this_frame = not process_this_frame

            # Buat salinan untuk tampilan
            display_frame = frame.copy()

            # Tampilkan hasil
            for (top, right, bottom, left), name, confidence in zip(face_locations, face_names, face_confidences):
                # Scale koordinat dari small_frame ke ukuran asli
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Tentukan apakah wajah dikenali dengan baik
                verified = name in attendance_recorded

                # Ekstrak gambar wajah untuk analisis mood
                face_img = frame[top:bottom, left:right]

                # Dapatkan kelas
                kelas = get_class_for_student(name, kelas_map)

                # Tentukan warna box berdasarkan confidence
                if confidence >= CONFIDENCE_THRESHOLD or force_record:
                    box_color = (0, 255, 0)  # Hijau jika confidence tinggi

                    # Jika belum terekam atau force record, catat absensi
                    if (not verified or force_record) and name != "Unknown":
                        success, detected_mood = record_attendance(name, confidence, face_img, kelas_map)
                        if success:
                            attendance_recorded.add(name)
                            mood_record[name] = detected_mood
                    elif name in mood_record:
                        detected_mood = mood_record[name]
                    else:
                        detected_mood = {"category": "Normal", "analysis": ""}
                else:
                    box_color = (0, 165, 255)  # Orange jika confidence rendah
                    detected_mood = {"category": "Unknown", "analysis": ""}

                # Gambar box di sekitar wajah
                cv2.rectangle(display_frame, (left, top), (right, bottom), box_color, 2)

                # Label dengan nama, kelas, dan status
                status = "✓ Terverifikasi" if verified else f"Mendeteksi... ({confidence * 100:.1f}%)"
                label_text = f"{name} | {kelas} | {status}"
                cv2.putText(display_frame, label_text, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                # Tampilkan mood di bawah nama
                if name != "Unknown" and detected_mood["category"] != "Unknown":
                    mood_color = (0, 255, 0) if detected_mood["category"] == "Good" else (
                        0, 0, 255) if detected_mood["category"] == "Stress" else (255, 165, 0)

                    # Tampilkan kategori mood
                    cv2.putText(display_frame, f"Mood: {detected_mood['category']}", (left, bottom + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, mood_color, 2)

                    # Tampilkan analisis mood (maksimal 60 karakter)
                    analysis_text = detected_mood.get("analysis", "")
                    if analysis_text:
                        # Batasi panjang teks dan bagi menjadi beberapa baris jika perlu
                        max_length = 60
                        if len(analysis_text) > max_length:
                            analysis_text = analysis_text[:max_length] + "..."

                        cv2.putText(display_frame, f"Analisis: {analysis_text}", (left, bottom + 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Reset force record flag
            force_record = False

            # Hitung dan tampilkan FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            # Tampilkan informasi
            cv2.putText(display_frame, f"Terdeteksi: {len(attendance_recorded)}/{len(known_face_names)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Tampilkan nama-nama yang sudah terabsen
            y_pos = 60
            for name in attendance_recorded:
                kelas = get_class_for_student(name, kelas_map)
                mood = mood_record.get(name, {"category": "Normal"})
                mood_text = f" - Mood: {mood['category']}" if mood['category'] != "Unknown" else ""
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
            cv2.imshow('Sistem Absensi Wajah dengan Deteksi Mood (CNN)', display_frame)

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

    except KeyboardInterrupt:
        print("Program terminated by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
    finally:
        # Bersihkan
        video_capture.release()
        cv2.destroyAllWindows()
        if db_conn:
            db_conn.close()

        print("\nPencatatan Kehadiran Selesai")
        print(f"Data absensi tersimpan di CSV dan/atau database")
        if use_database:
            print(f"Data absensi tersimpan di database: {DB_CONFIG['database']}.{DB_TABLE}")
        print(f"Total absensi: {len(attendance_recorded)} orang")
        for name in attendance_recorded:
            kelas = get_class_for_student(name, kelas_map)
            mood = mood_record.get(name, {"category": "Normal"})
            print(f"- {name} ({kelas}) - Mood: {mood['category']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unhandled error: {e}")
        traceback.print_exc()