import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import datetime

# Inisialisasi MTCNN (deteksi wajah) dan FaceNet (untuk pengenalan wajah)
mtcnn = MTCNN(keep_all=True)  # MTCNN digunakan untuk mendeteksi wajah
model = InceptionResnetV1(pretrained='vggface2').eval()  # Model FaceNet untuk pengenalan wajah

# Folder tempat gambar model disimpan
MODEL_PATH = "Model"
CONFIDENCE_THRESHOLD = 11  # Threshold untuk testing


# Fungsi untuk preprocessing wajah dan mendapatkan embeddings dengan FaceNet
def preprocess_face_with_facenet(image):
    # Deteksi wajah menggunakan MTCNN (hanya mengembalikan wajah yang terdeteksi)
    faces = mtcnn(image)

    if faces is not None and len(faces) > 0:
        # Jika wajah terdeteksi, ambil embeddings untuk pengenalan wajah
        face_embeddings = model(faces)
        return face_embeddings
    else:
        print("No faces detected.")
        return None

# Fungsi untuk melatih pengenalan wajah menggunakan FaceNet (menggunakan embeddings)
def train_face_recognizer():
    print("Loading face data...")
    faces = []
    labels = []
    label_map = {}
    label_count = 0

    for class_folder in os.listdir(MODEL_PATH):
        class_path = os.path.join(MODEL_PATH, class_folder)
        if os.path.isdir(class_path):  # Jika itu adalah folder (kelas)
            for image_file in os.listdir(class_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    print(f"Processing {image_file} from {class_folder}")
                    image_path = os.path.join(class_path, image_file)

                    # Load gambar
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"Error loading {image_path}. File mungkin rusak.")
                        continue

                    # Preprocessing gambar dengan FaceNet
                    face_embeddings = preprocess_face_with_facenet(img)

                    if face_embeddings is not None:
                        # Menambahkan embeddings wajah dan label kelas
                        embeddings_array = face_embeddings.detach().cpu().numpy()
                        faces.append(embeddings_array)
                        label_map[class_folder] = label_count
                        labels.append(label_count)
                        label_count += 1
                    else:
                        print(f"Warning: No face found in {image_file}")

    # Melatih model dengan embeddings
    if len(faces) > 0 and len(faces) == len(labels):
        print(f"Training with {len(faces)} faces...")
        try:
            faces = np.array(faces)
            labels = np.array(labels)
            return faces, labels, label_map
        except Exception as e:
            print(f"Error during training: {e}")
            return {}
    else:
        print("Error: No faces found for training.")
        return {}


# Fungsi untuk mengenali wajah dengan FaceNet menggunakan cosine similarity
def recognize_face_with_facenet(face_img, known_face_embeddings, known_labels):
    face_embeddings = preprocess_face_with_facenet(face_img)

    if face_embeddings is not None:
        # Menghitung similarity antara embeddings wajah yang terdeteksi dan yang diketahui
        similarities = cosine_similarity(face_embeddings.detach().cpu().numpy(), known_face_embeddings)

        best_match_idx = np.argmax(similarities)
        best_match_similarity = similarities[0, best_match_idx]

        if best_match_similarity > 0.8:  # Threshold similarity
            recognized_label = known_labels[best_match_idx]
            return recognized_label
        else:
            return "Unknown"
    else:
        return "No face detected"


# Fungsi utama untuk menghubungkan semua bagian
def main():
    # Melatih FaceNet dengan embeddings wajah
    faces, labels, label_map = train_face_recognizer()

    # Mulai video capture untuk mendeteksi wajah secara real-time
    video_capture = cv2.VideoCapture(1)  # Gunakan webcam default

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Deteksi wajah menggunakan MTCNN
        faces = mtcnn(frame)  # Dapatkan wajah yang terdeteksi

        # Jika tidak ada wajah yang terdeteksi, lanjutkan ke frame berikutnya
        if faces is None or len(faces) == 0:
            print("No faces detected in the frame.")
            continue  # Skip frame ini dan lanjutkan ke frame berikutnya

        # Proses wajah yang terdeteksi
        for face in faces:
            # Mengenali wajah dengan FaceNet (cocokkan embeddings)
            recognized_label = recognize_face_with_facenet(face, faces, labels)

            # Menggambar kotak di sekitar wajah yang dikenali
            if recognized_label != "Unknown":
                cv2.putText(frame, recognized_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Misalnya jika menggunakan bounding box
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        # Keluar jika tekan 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
