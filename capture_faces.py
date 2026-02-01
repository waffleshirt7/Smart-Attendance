import cv2
import os
import numpy as np

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def preprocess_face_for_dataset(gray_img):
    """
    Preprocess a face patch before saving it to the dataset.
    This is kept consistent with training/runtime:
    - histogram equalization
    - light blur
    - normalization
    - resize to 200x200
    """
    gray = cv2.equalizeHist(gray_img)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.resize(gray, (200, 200))
    return gray


def face_quality_ok(face_roi, min_sharpness: float = 20.0, min_brightness: float = 40.0, max_brightness: float = 220.0) -> bool:
    """Simple quality check so we don't save very blurry or too-dark/bright faces."""
    sharpness = cv2.Laplacian(face_roi, cv2.CV_64F).var()
    brightness = face_roi.mean()
    if sharpness < min_sharpness:
        return False
    if not (min_brightness <= brightness <= max_brightness):
        return False
    return True


def pick_largest_face(faces):
    """Return the (x,y,w,h) of the largest face, or None."""
    if faces is None or len(faces) == 0:
        return None
    faces = [tuple(map(int, f)) for f in faces]
    return max(faces, key=lambda r: r[2] * r[3])

num_students = int(input("Enter number of students: "))

for student_num in range(num_students):
    print(f"\n--- Student {student_num + 1} ---")
    
    student_id = input("Enter Student ID: ")
    student_name = input("Enter Student Name: ")
    
    cam = cv2.VideoCapture(0)
    
    # Try camera index 1 if 0 doesn't work
    if not cam.isOpened():
        cam = cv2.VideoCapture(1)
    
    if not cam.isOpened():
        print(f"Error: Cannot open camera for {student_name}. Check camera connection.")
        continue
    
    count = 0
    path = f"dataset/{student_name}_{student_id}"
    os.makedirs(path, exist_ok=True)
    
    # Set camera properties for better quality
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FPS, 30)
    
    print(f"Capturing {student_name}'s face images. Press ESC to finish or collect 50 samples.")
    print("Try to move your head slightly at different angles for better accuracy.")
    
    while True:
        ret, img = cam.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(80, 80),  # avoid tiny/false detections and background faces
        )

        # To keep the dataset clean, capture only ONE face per frame (largest one).
        picked = pick_largest_face(faces)
        if picked is not None:
            (x, y, w, h) = picked
            face_roi = gray[y:y+h, x:x+w]
            # Basic quality check to avoid saving bad samples
            if not face_quality_ok(face_roi):
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(
                    img,
                    "Move closer / better light",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            else:
                face_roi = preprocess_face_for_dataset(face_roi)
                count += 1
                cv2.imwrite(f"{path}/{count}.jpg", face_roi)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"Samples: {count}/100",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                # Add a short delay to slow down capture
                cv2.waitKey(100)  # 100 ms pause after each capture
    
        cv2.imshow('Capturing Faces - Move head slightly for different angles', img)
    
        if cv2.waitKey(1) == 27 or count >= 100:  # Increased from 20 to 100 samples
            break
    
    cam.release()
    cv2.destroyAllWindows()
    print(f"✓ Face samples collected for {student_name}: {count} images")

print("\nAll students' face data collected successfully!")