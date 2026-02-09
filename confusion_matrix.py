import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import argparse

try:
    from face_alignment_utils import preprocess_face_advanced
    USE_ADVANCED_PREPROCESS = True
except ImportError:
    USE_ADVANCED_PREPROCESS = False


def preprocess(img):
    """Use same preprocessing as train_model.py for accurate metrics."""
    if USE_ADVANCED_PREPROCESS:
        return preprocess_face_advanced(img, target_size=(200, 200), use_alignment=True, use_clahe=True)
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (200, 200))
    return img


def load_dataset(dataset_path):
    names = []
    images = []
    labels = []
    if not os.path.exists(dataset_path):
        return None, None, None
    for idx, folder in enumerate(sorted(os.listdir(dataset_path))):
        fp = os.path.join(dataset_path, folder)
        if not os.path.isdir(fp):
            continue
        for f in os.listdir(fp):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            p = os.path.join(fp, f)
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            images.append(preprocess(img))
            labels.append(folder)
            names.append(folder)
    if not images:
        return None, None, None
    return np.array(images), np.array(labels), sorted(list(set(names)))


def infer_model_map(recognizer, dataset_path):
    # majority-vote mapping: model_label -> folder name
    from collections import Counter
    mapping = {}
    for folder in sorted(os.listdir(dataset_path)):
        fp = os.path.join(dataset_path, folder)
        if not os.path.isdir(fp):
            continue
        preds = []
        for f in os.listdir(fp):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            p = os.path.join(fp, f)
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = preprocess(img)
            try:
                lab, _ = recognizer.predict(img)
                preds.append(int(lab))
            except Exception:
                pass
        if preds:
            mapping[Counter(preds).most_common(1)[0][0]] = folder
    return mapping


def simple_confusion_image(cm, labels, out_path):
    # cm: square numpy array, labels: list of names
    cm_sum = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        pct = np.nan_to_num(cm / cm_sum)

    annot = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{int(cm[i, j])}\n{pct[i,j]*100:0.1f}%"

    plt.figure(figsize=(8, 6))
    sns.heatmap(pct, annot=annot, fmt='', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Row proportion'})
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title('Confusion Matrix (count + % row)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main(model_path, dataset_path, out_path, no_threshold=False):
    imgs, true_labels, label_names = load_dataset(dataset_path)
    if imgs is None:
        print("Error: No dataset found.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=3, neighbors=10, grid_x=10, grid_y=10)
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        return
    recognizer.read(model_path)

    # predict
    preds = []
    for img in imgs:
        lab, dist = recognizer.predict(img)
        if (not no_threshold) and dist > 65.0:
            preds.append('Unknown')
        else:
            preds.append(lab)

    # if preds contain numeric labels, map numeric entries to names, keep 'Unknown' as-is
    if preds and any(isinstance(p, (int, np.integer)) for p in preds):
        model_map = infer_model_map(recognizer, dataset_path)
        mapped = []
        for p in preds:
            if isinstance(p, (int, np.integer)):
                mapped.append(model_map.get(int(p), 'Unknown'))
            else:
                mapped.append(p)
        preds = mapped

    # now true_labels are folder names already
    labels_union = sorted(list(set(true_labels) | set(preds)))
    cm = confusion_matrix(list(true_labels), list(preds), labels=labels_union)

    simple_confusion_image(cm, labels_union, out_path)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='trainer/trainer.yml')
    p.add_argument('--dataset', default='dataset')
    p.add_argument('--out', default='confusion_matrices/confusion_matrix_simple.png')
    p.add_argument('--no-threshold', action='store_true')
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    main(args.model, args.dataset, args.out, no_threshold=args.no_threshold)
    # End of script
