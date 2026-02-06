import cv2
import os
import numpy as np
from face_alignment_utils import preprocess_face_advanced, get_face_quality_score

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def preprocess_face_for_dataset(gray_img):
    """
    Preprocess a face patch before saving it to the dataset.
    Uses advanced preprocessing with CLAHE and face alignment:
    - CLAHE for contrast (better lighting invariance)
    - Face alignment by eyes (10-15% accuracy improvement)
    - Histogram equalization
    - Gaussian blur for noise reduction
    - Normalization
    - Resize to 200x200
    
    This preprocessing is consistent with training and runtime in improved_attendance.py
    """
    return preprocess_face_advanced(gray_img, target_size=(200, 200), 
                                   use_alignment=True, use_clahe=True)


def face_quality_ok(face_roi, min_sharpness: float = 20.0, min_brightness: float = 40.0, max_brightness: float = 220.0) -> bool:
    """
    Comprehensive quality check using multiple metrics to ensure high-quality training data.
    Checks: sharpness, brightness, and contrast.
    Better quality training data = better recognition accuracy.
    """
    quality_scores = get_face_quality_score(face_roi)
    return quality_scores['is_good']


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
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam.set(cv2.CAP_PROP_FPS, 30)
    
    print(f"\n📸 Manual Capturing {student_name}'s face images.")
    print(f"   Target: 30 high-quality samples")
    print(f"   Instructions:")
    print(f"   - Move head left, right, up, down slightly")
    print(f"   - Vary distance from camera")
    print(f"   - Ensure good lighting")
    print(f"   - Blink naturally")
    print(f"   - Press 'C' to capture when ready")
    print(f"   - Press 'ESC' to finish")
    
    frame_count = 0
    while True:
        ret, img = cam.read()
        if not ret:
            break
        
        frame_count += 1
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,  # Increased from 3 for better reliability
            minSize=(80, 80),
        )

        # To keep the dataset clean, only detect ONE face per frame (largest one).
        picked = pick_largest_face(faces)
        if picked is not None:
            (x, y, w, h) = picked
            face_roi = gray[y:y+h, x:x+w]
            
            # Get quality scores for feedback
            quality_scores = get_face_quality_score(face_roi)
            
            # Show quality feedback
            if not face_quality_ok(face_roi):
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                feedback = "❌ Poor quality - "
                if not quality_scores['is_sharp']:
                    feedback += "Too blurry"
                elif not quality_scores['is_well_lit']:
                    feedback += "Bad lighting"
                elif not quality_scores['is_good_contrast']:
                    feedback += "Low contrast"
                
                cv2.putText(
                    img,
                    feedback,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            else:
                # Show good quality feedback (but don't save yet - wait for 'C' press)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"✓ Ready: {count}/30 | Press 'C' to capture",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                # Feedback on quality
                cv2.putText(
                    img,
                    f"Sharpness: {quality_scores['sharpness']:.1f} | Brightness: {quality_scores['brightness']:.0f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 0),
                    1,
                )
        else:
            # No face detected
            cv2.putText(
                img,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
            )
    
        cv2.imshow('Manual Face Capture - Press C to capture, ESC to finish', img)
    
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'C' to manually capture
        if key == ord('c') or key == ord('C'):
            if picked is not None and face_quality_ok(face_roi):
                # Preprocess and save
                face_roi_processed = preprocess_face_for_dataset(face_roi)
                count += 1
                cv2.imwrite(f"{path}/{count}.jpg", face_roi_processed)
                print(f"   ✓ Image {count} saved!")
            else:
                print(f"   ❌ Cannot capture - face quality is not good enough")
        
        # Press ESC to finish
        if key == 27 or count >= 30:
            break
    
    cam.release()
    cv2.destroyAllWindows()
    print(f"✓ Face samples collected for {student_name}: {count}/30 images")

print("\n✨ All students' face data collected successfully!")
