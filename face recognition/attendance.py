import os
import csv
import datetime
import pickle
import cv2
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine

# ==============================
# SETTINGS
# ==============================

KNOWN_DIR = "known_faces"
ATT_FILE = "attendance.csv"
ENCODINGS_FILE = "encodings.pkl"

TOLERANCE = 0.45          # Matching threshold
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"   # Best accuracy

# ==============================
# INIT ATTENDANCE FILE
# ==============================

def init_attendance_file():

    if not os.path.exists(ATT_FILE):

        with open(
            ATT_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                ["Name", "Date", "Time"]
            )


def normalize_embedding(embedding):

    vector = np.array(
        embedding,
        dtype="float32"
    )

    norm = np.linalg.norm(vector)

    if norm == 0:
        raise ValueError("Embedding norm is zero.")

    return vector / norm

# ==============================
# MARK ATTENDANCE
# ==============================

def mark_attendance(name):

    now = datetime.datetime.now()

    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    if os.path.exists(ATT_FILE):

        with open(
            ATT_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            reader = csv.reader(f)

            for row in reader:

                if (
                    len(row) >= 2
                    and row[0] == name
                    and row[1] == date
                ):
                    return

    with open(
        ATT_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [name, date, time]
        )

    print("✓ Attendance marked:", name)

# ==============================
# LOAD KNOWN FACES
# ==============================

def load_known_faces():

    faces = []

    if not os.path.exists(KNOWN_DIR):

        os.makedirs(KNOWN_DIR)

        print(
            "Created 'known_faces' folder."
        )

        print(
            "Add images and restart."
        )

        return faces

    print("Loading known faces...")

    cached_faces = load_cached_faces()

    if cached_faces:
        print(
            "Loaded encodings from cache:",
            len(cached_faces)
        )
        return cached_faces

    for filename in os.listdir(KNOWN_DIR):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        name = os.path.splitext(filename)[0]

        path = os.path.join(
            KNOWN_DIR,
            filename
        )

        try:

            embedding = DeepFace.represent(
                img_path=path,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False
            )

            if isinstance(embedding, list):
                embedding = embedding[0]

            if isinstance(embedding, dict):
                embedding = embedding["embedding"]

            embedding = normalize_embedding(
                embedding
            )

            faces.append(
                (name, embedding)
            )

            print("Loaded:", name)

        except Exception as e:

            print(
                "Failed:",
                filename,
                e
            )

    print(
        "Total faces loaded:",
        len(faces)
    )

    return faces


def load_cached_faces():

    if not os.path.exists(ENCODINGS_FILE):
        return []

    try:

        with open(
            ENCODINGS_FILE,
            "rb"
        ) as f:

            payload = pickle.load(f)

        if not isinstance(payload, dict):
            print("Ignoring invalid cache format.")
            return []

        if payload.get("model_name") != MODEL_NAME:
            print("Cached encodings model does not match current attendance model.")
            return []

        if payload.get("detector_backend") != DETECTOR_BACKEND:
            print("Cached encodings detector does not match current attendance detector.")
            return []

        names = payload.get("names", [])
        encodings = payload.get("encodings", [])

        if len(names) != len(encodings):
            print("Ignoring cache because names and encodings do not align.")
            return []

        faces = []

        for name, embedding in zip(
            names,
            encodings
        ):
            faces.append(
                (
                    name,
                    normalize_embedding(embedding)
                )
            )

        return faces

    except Exception as e:

        print(
            "Failed to load cached encodings:",
            e
        )

        return []


def crop_face(frame, facial_area):

    frame_height, frame_width = frame.shape[:2]

    x = max(facial_area["x"], 0)
    y = max(facial_area["y"], 0)
    w = max(facial_area["w"], 0)
    h = max(facial_area["h"], 0)

    x2 = min(x + w, frame_width)
    y2 = min(y + h, frame_height)

    return frame[y:y2, x:x2]

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    init_attendance_file()

    known_faces = load_known_faces()

    if len(known_faces) == 0:

        print(
            "No known faces found."
        )

        exit()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print(
            "Cannot open camera"
        )

        exit()

    print(
        "Camera started..."
    )

    print(
        "Press Q to exit"
    )

    marked = False

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        try:

            # Detect faces

            detections = DeepFace.extract_faces(

                img_path=frame,

                detector_backend=DETECTOR_BACKEND,

                enforce_detection=False

            )

            for face in detections:

                if marked:
                    break

                x = face["facial_area"]["x"]
                y = face["facial_area"]["y"]
                w = face["facial_area"]["w"]
                h = face["facial_area"]["h"]

                # Draw rectangle

                cv2.rectangle(

                    frame,

                    (x, y),

                    (x + w, y + h),

                    (0, 255, 0),

                    2

                )

                # Extract the embedding from the detected face crop

                face_crop = crop_face(
                    frame,
                    face["facial_area"]
                )

                if face_crop.size == 0:
                    continue

                embedding = DeepFace.represent(

                    img_path=face_crop,

                    model_name=MODEL_NAME,

                    detector_backend="skip",

                    enforce_detection=False

                )

                if isinstance(embedding, list):
                    embedding = embedding[0]

                if isinstance(embedding, dict):
                    embedding = embedding["embedding"]

                embedding = normalize_embedding(
                    embedding
                )

                best_distance = float("inf")
                best_name = None

                for name, known_emb in known_faces:

                    distance = cosine(
                        embedding,
                        known_emb
                    )

                    if distance < best_distance:

                        best_distance = distance
                        best_name = name

                print(
                    "Distance:",
                    round(best_distance, 3)
                )

                # MATCH FOUND

                if best_distance < TOLERANCE:

                    mark_attendance(
                        best_name
                    )

                    cv2.putText(

                        frame,

                        "Marked: "
                        + best_name,

                        (30, 40),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        1,

                        (0, 255, 0),

                        2

                    )

                    cv2.imshow(
                        "Face Attendance System",
                        frame
                    )

                    cv2.waitKey(2000)

                    marked = True

                else:

                    cv2.putText(

                        frame,

                        "Unknown",

                        (30, 40),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        1,

                        (0, 0, 255),

                        2

                    )

            if marked:
                break

        except Exception as e:

            print(
                "Error:",
                e
            )

        cv2.imshow(
            "Face Attendance System",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

        if marked:
            break

    cap.release()

    cv2.destroyAllWindows()
