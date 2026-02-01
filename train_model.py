import cv2
import numpy as np
import os


def preprocess_face_for_training(gray_img):
    """
    Apply the same style of preprocessing used at runtime in `attendance.py`:
    - Histogram equalization
    - Light Gaussian blur
    - Normalization
    - Resize to 200x200
    This helps the recognizer see training data in the same form as live data.
    """
    gray = cv2.equalizeHist(gray_img)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.resize(gray, (200, 200))
    return gray


def main():
    # Create LBPH recognizer with optimized parameters
    # radius: 1 for local patterns
    # neighbors: 8 for better feature detection
    # grid_x/grid_y: spatial histograms
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)

    faces = []
    ids = []

    dataset_path = "dataset"
    min_images_per_person = 10  # ignore identities with too few samples (very unreliable)

    label = 0
    label_dict = {}

    for folder in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder)

        if not os.path.isdir(folder_path):
            continue

        label_dict[label] = folder

        image_paths = [
            os.path.join(folder_path, img)
            for img in os.listdir(folder_path)
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(image_paths) < min_images_per_person:
            print(
                f"Skipping '{folder}' (only {len(image_paths)} images; "
                f"need at least {min_images_per_person} for reliable training)."
            )
            continue

        person_loaded = 0
        for img_path in image_paths:
            try:
                gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if gray is None:
                    print(f"Warning: Could not read {img_path}")
                    continue

                gray = preprocess_face_for_training(gray)

                faces.append(gray)
                ids.append(label)
                person_loaded += 1

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        print(f"Loaded {person_loaded} images for {folder}")

        # Only advance label if we actually loaded images for this person
        if person_loaded > 0:
            label += 1

    print(f"\nTotal faces loaded: {len(faces)}")
    if len(faces) == 0:
        print("Error: No training data found. Please run capture_faces.py first.")
        return

    print("Training model...")
    recognizer.train(faces, np.array(ids))
    os.makedirs("trainer", exist_ok=True)
    recognizer.save("trainer/trainer.yml")
    print("✓ Model trained successfully!")
    print(f"Model parameters: radius=1, neighbors=8, grid_x=8, grid_y=8")


if __name__ == "__main__":
    main()