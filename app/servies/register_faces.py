import cv2
import pickle
import os
from mtcnn import MTCNN
from keras_facenet import FaceNet


def load_database(path="faces_db.pkl"):
    """Load face embeddings database if exists, else return empty dict"""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def save_database(database, path="faces_db.pkl"):
    """Save face embeddings database to file"""
    with open(path, "wb") as f:
        pickle.dump(database, f)


def capture_face_samples(name, detector, embedder, samples_to_collect=30):
    """Capture face embeddings from webcam for a given person"""
    cap = cv2.VideoCapture(0)
    embeddings = []
    collected = 0

    print(f"📸 Capturing {samples_to_collect} samples for {name}...")

    while collected < samples_to_collect:
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
            embeddings.append(embedding)
            collected += 1

            print(f"✅ Captured sample {collected}/{samples_to_collect}")

            # Draw box and counter
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {collected}/{samples_to_collect}",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        cv2.imshow("Register Face", frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

    return embeddings


def register_face(name, samples_to_collect=30):
    """Main function to register a new face"""
    # Load models
    detector = MTCNN()
    embedder = FaceNet()

    # Load DB
    database = load_database()

    # Capture samples
    embeddings = capture_face_samples(name, detector, embedder, samples_to_collect)

    # Save embeddings
    if name not in database:
        database[name] = []
    database[name].extend(embeddings)

    save_database(database)
    print(f"🎉 Done! Collected {len(embeddings)} samples for {name} and saved to faces_db.pkl")


if __name__ == "__main__":
    person_name = input("Enter the name of the person: ")
    register_face(person_name, samples_to_collect=30)
