# Smart Attendance System

Face recognition-based attendance system with DeepFace (99%+ accuracy) or LBPH fallback. Captures faces, trains a model, and marks attendance via camera.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. For higher accuracy (optional): install DeepFace + TensorFlow
   ```bash
   pip install deepface tensorflow
   ```

## Workflow

1. **Capture Faces** – Collect face samples for each student
2. **Train Model** – Train the recognition model on the dataset
3. **Take Attendance** – Run attendance; press Enter in the camera window to finish

## Usage

### GUI Launcher (recommended)

```bash
python simple_gui.py
```

- **Capture Faces** – Add students to the dataset
- **Train Model** – Train after capturing
- **Start Attendance** – Take attendance; press Enter to exit and generate the sheet
- **Today's Report** – View today's attendance report
- Folder buttons – Open dataset, trainer, records, sheets

### Command line

```bash
# 1. Capture faces (interactive)
python capture_faces.py

# 2. Train the model
python train_model.py

# 3. Run attendance (press Enter to exit)
python attendance.py
```

## Project Structure

- `attendance.py` – Main attendance system (DeepFace + LBPH)
- `capture_faces.py` – Face capture (press C to capture, ESC to finish)
- `train_model.py` – LBPH model training
- `semester_register.py` – Generates Excel/HTML/CSV register
- `simple_gui.py` – GUI launcher
- `face_alignment_utils.py` – Preprocessing (CLAHE, alignment)
- `dataset/` – Student face images (`Name_RollNo/` folders)
- `trainer/` – Trained model (`trainer.yml`)
- `attendance_records/` – Daily attendance CSV/JSON
- `attendance_sheets/` – Semester register outputs

## Dataset Format

Create folders as `Name_RollNo` (e.g. `John Doe_42`). Each folder should have at least 10–30 face images.

## Requirements

- Python 3.8+
- Webcam
- OpenCV, pandas, numpy, openpyxl (core)
- DeepFace, TensorFlow (optional, for higher accuracy)
