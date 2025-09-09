import cv2
import pickle
import json
import os
from datetime import datetime
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity


def load_database(path="faces_db.pkl"):
    """Load face embeddings database"""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError("❌ faces_db.pkl not found! Please register faces first.")


def compute_mean_embeddings(database):
    """Compute mean embedding per person"""
    return {name: np.mean(embeds, axis=0) for name, embeds in database.items()}


def log_recognition(name, similarity, log_path="recognition_log.json"):
    """Log recognition result into JSON file with timestamp"""
    entry = {
        "name": name,
        "similarity": float(similarity),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Load existing log
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    # Append and save
    logs.append(entry)
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=4)

    print(f"📝 Logged recognition: {entry}")


def recognize_faces(threshold=0.6):
    """Run face recognition loop using webcam"""

    print("✅ Loading database...")
    database = load_database()
    mean_embeddings = compute_mean_embeddings(database)

    print("📷 Starting face recognition...")
    detector = MTCNN()
    embedder = FaceNet()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_img)

        for face in faces:
            x, y, w, h = face['box']
            face_crop = rgb_img[y:y+h, x:x+w]

            if face_crop.size == 0:
                continue

            # Get embedding
            embedding = embedder.embeddings([face_crop])[0]

            # Compare with stored embeddings
            name = "Unknown"
            max_sim = -1

            for person, db_embed in mean_embeddings.items():
                sim = cosine_similarity([embedding], [db_embed])[0][0]
                if sim > max_sim and sim > threshold:
                    name = person
                    max_sim = sim

            # Draw results
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({max_sim:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Log recognition (only if recognized, not "Unknown")
            if name != "Unknown":
                log_recognition(name, max_sim)

        cv2.imshow("Face Recognition", frame)
        cv2.waitKey(1)  # refresh loop

    cap.release()
    cv2.destroyAllWindows()
