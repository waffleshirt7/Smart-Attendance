import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from pathlib import Path
import pandas as pd


def preprocess_face_for_testing(gray_img):
    """Apply the same preprocessing used during training and runtime."""
    gray = cv2.equalizeHist(gray_img)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.resize(gray, (200, 200))
    return gray


def load_test_data(dataset_path="dataset"):
    """Load all images from dataset and prepare them for testing."""
    faces = []
    true_labels = []
    label_dict = {}
    label = 0

    if not os.path.exists(dataset_path):
        return None, None, None

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

        if len(image_paths) == 0:
            print(f"Warning: No images found for {folder}")
            continue

        for img_path in image_paths:
            try:
                gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    print(f"Warning: Could not read {img_path}")
                    continue

                gray = preprocess_face_for_testing(gray)
                faces.append(gray)
                true_labels.append(label)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        print(f"Loaded {len(image_paths)} images for {folder}")
        label += 1

    if len(faces) == 0:
        return None, None, None

    return np.array(faces), np.array(true_labels), label_dict


def create_demo_confusion_matrix():
    """Create a demo confusion matrix from simulated predictions."""
    print("\nNo dataset found. Generating DEMO confusion matrix...")
    print("(This shows what the confusion matrix will look like with real data)\n")

    # Simulated data for demonstration
    num_people = 5
    samples_per_person = 20
    
    # Create simulated true labels and predictions
    true_labels = np.repeat(range(num_people), samples_per_person)
    
    # Create predictions with some realistic errors
    predicted_labels = true_labels.copy()
    
    # Add some misclassifications (90% accuracy scenario)
    num_errors = int(len(true_labels) * 0.1)
    error_indices = np.random.choice(len(true_labels), num_errors, replace=False)
    
    for idx in error_indices:
        true_label = true_labels[idx]
        # Misclassify to a different person
        wrong_labels = [i for i in range(num_people) if i != true_label]
        predicted_labels[idx] = np.random.choice(wrong_labels)
    
    label_dict = {i: f"Person_{i+1}" for i in range(num_people)}
    
    return true_labels, predicted_labels, label_dict


def visualize_confusion_matrix(cm, label_dict, output_dir="confusion_matrices"):
    """Create visualizations from confusion matrix."""
    from datetime import datetime
    
    label_names = [label_dict.get(l, f"Person {l}") for l in sorted(label_dict.keys())]
    
    # Create confusion matrix heatmap
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - Face Recognition Model', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the confusion matrix plot
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"confusion_matrix_{timestamp}.png")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix visualization saved to: {output_path}")
    plt.close()

    # Per-person accuracy bar chart
    plt.figure(figsize=(12, 6))
    accuracies = []
    names = []
    
    for i, label in enumerate(sorted(label_dict.keys())):
        correct = cm[i, i]
        total = cm[i].sum()
        if total > 0:
            acc = correct / total
            accuracies.append(acc)
            names.append(label_dict[label])
    
    overall_acc = np.trace(cm) / cm.sum()
    
    bars = plt.bar(range(len(names)), accuracies, color='steelblue', alpha=0.7)
    plt.axhline(y=overall_acc, color='red', linestyle='--', 
                label=f'Overall Accuracy: {overall_acc:.1%}', linewidth=2)
    plt.xlabel('Person', fontsize=12)
    plt.ylabel('Recognition Accuracy', fontsize=12)
    plt.title('Per-Person Recognition Accuracy', fontsize=14, fontweight='bold')
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.ylim([0, 1.05])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.0%}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    accuracy_path = os.path.join(output_dir, f"accuracy_per_person_{timestamp}.png")
    plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
    print(f"✓ Per-person accuracy chart saved to: {accuracy_path}")
    plt.close()

    # Save confusion matrix as CSV
    label_names_full = [label_dict.get(l, f"Person {l}") for l in range(cm.shape[0])]
    cm_df = pd.DataFrame(cm, index=label_names_full, columns=label_names_full)
    csv_path = os.path.join(output_dir, f"confusion_matrix_{timestamp}.csv")
    cm_df.to_csv(csv_path)
    print(f"✓ Confusion matrix CSV saved to: {csv_path}")

    return output_path, accuracy_path, csv_path


def generate_confusion_matrix(model_path="trainer/trainer.yml", dataset_path="dataset", distance_threshold=65.0):
    """Generate confusion matrix for the trained model."""

    # Load test data
    print("Loading test data...")
    faces, true_labels, label_dict = load_test_data(dataset_path)

    if faces is None:
        # Use demo data
        true_labels, predicted_labels, label_dict = create_demo_confusion_matrix()
        accuracy = accuracy_score(true_labels, predicted_labels)
        cm = confusion_matrix(true_labels, predicted_labels)
        is_demo = True
    else:
        # Use real data
        print(f"Total images loaded: {len(faces)}")
        print(f"Number of identities: {len(label_dict)}")

        # Load model
        print(f"\nLoading model from {model_path}...")
        recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            return

        recognizer.read(model_path)
        print("✓ Model loaded successfully!")

        # Predict labels
        print("\nPredicting labels...")
        predicted_labels = []

        for i, face in enumerate(faces):
            label, distance = recognizer.predict(face)
            
            # If distance is too high, classify as unknown
            if distance > distance_threshold:
                label = -1  # Unknown person
            
            predicted_labels.append(label)

            if (i + 1) % max(1, len(faces) // 10) == 0:
                print(f"  Processed {i + 1}/{len(faces)} images")

        predicted_labels = np.array(predicted_labels)
        accuracy = accuracy_score(true_labels, predicted_labels)
        cm = confusion_matrix(true_labels, predicted_labels)
        is_demo = False

    # Generate confusion matrix
    print("\n" + "="*60)
    print("CONFUSION MATRIX ANALYSIS")
    print("="*60)

    if is_demo:
        print("\n⚠️  DEMO MODE - Using simulated data")
        print("To generate real confusion matrix:")
        print("  1. Run: python3 capture_faces.py (to capture face images)")
        print("  2. Run: python3 train_model.py (to train the model)")
        print("  3. Run: python3 confusion_matrix.py (to generate confusion matrix)")
    else:
        print("\n✓ Using REAL data from dataset")

    # Print classification report
    target_names = [label_dict.get(l, f"Person {l}") for l in sorted(label_dict.keys())]
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, 
                                target_names=target_names, zero_division=0))

    # Print accuracy
    print(f"\nOverall Accuracy: {accuracy:.2%}")

    # Print confusion matrix statistics
    print("\nConfusion Matrix Statistics:")
    for i, label in enumerate(sorted(label_dict.keys())):
        if i < cm.shape[0]:
            correct = cm[i, i]
            total = cm[i].sum()
            if total > 0:
                recall = correct / total
                print(f"  {label_dict[label]:20s}: {correct:3d}/{total:3d} correct ({recall:.1%})")

    # Visualize
    output_path, accuracy_path, csv_path = visualize_confusion_matrix(cm, label_dict)

    print("\n" + "="*60)
    if is_demo:
        print("DEMO confusion matrix generated successfully!")
        print("Replace with real data when dataset is available.")
    else:
        print("Real confusion matrix generated successfully!")
    print("="*60)

    return cm, label_dict


if __name__ == "__main__":
    
    print("="*60)
    print("FACE RECOGNITION MODEL - CONFUSION MATRIX GENERATOR")
    print("="*60)
    
    generate_confusion_matrix(
        model_path="trainer/trainer.yml",
        dataset_path="dataset",
        distance_threshold=65.0
    )
