import cv2
import pandas as pd
import os
import json
from datetime import datetime
from pathlib import Path



import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

@dataclass(frozen=True)
class AttendanceSettings:
    """Runtime settings for the attendance system."""

    model_path: str = "trainer/trainer.yml"
    cascade_path: str = "haarcascade_frontalface_default.xml"
    dataset_path: str = "dataset"
    output_dir: str = "attendance_records"

    # Recognition tuning (LBPH returns a distance; lower is better)
    # Slightly relaxed so real students are accepted more easily, but still safe.
    max_lbph_distance: float = 65.0
    # Keep several agreeing frames to avoid random mistakes.
    required_consecutive_frames: int = 4
    min_face_size: int = 50  # pixels; ignore tiny detections
    min_sharpness: float = 20.0  # variance of Laplacian; filter blurry frames
    min_brightness: float = 40.0  # mean pixel lower bound
    max_brightness: float = 220.0  # mean pixel upper bound

    # Debug overlay
    show_debug: bool = True

    # Detection + performance tuning
    detection_scale: float = 1.1
    neighbors: int = 4
    # Suppress duplicate/overlapping face boxes (common with Haar cascades)
    enable_nms: bool = True
    nms_iou_threshold: float = 0.35
    max_faces_per_frame: int = 8
    # If True, only the single best-matching face in each frame
    # is allowed to update attendance (good when only one person is at the camera).
    single_person_mode: bool = True
    frame_width: int = 480
    frame_height: int = 360
    skip_frames: int = 1  # process every Nth frame (1 = process all)

class AttendanceSystem:
    """Smart Attendance System using face recognition (optimized for speed and accuracy)."""

    def __init__(self, settings: Optional[AttendanceSettings] = None):
        self.settings = settings or AttendanceSettings()
        # IMPORTANT: match the LBPH parameters used in `train_model.py`
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        if not os.path.exists(self.settings.model_path):
            raise FileNotFoundError(f"Model not found at {self.settings.model_path}. Please train the model first.")
        self.recognizer.read(self.settings.model_path)
        self.face_cascade = cv2.CascadeClassifier(self.settings.cascade_path)
        if self.face_cascade.empty():
            raise FileNotFoundError(f"Cascade classifier not found at {self.settings.cascade_path}")
        # Track who is already recorded in this run: key is (name, roll_no)
        self.attendance = set()
        self.df = pd.DataFrame(columns=["Name", "Roll No.", "Date", "Time", "Status", "Confidence (%)"])
        self.label_dict = self._load_label_dict()  # id -> {"name": str, "roll": str}
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.running = True
        # Require multiple agreeing frames before accepting a match
        self.match_counts: Dict[str, int] = {}

    # -------------------------- UI / drawing helpers -------------------------- #

    def _draw_label(
        self,
        frame,
        text: str,
        x: int,
        y: int,
        font_scale: float = 0.7,
        text_color=(255, 255, 255),
        bg_color=(0, 140, 255),
        thickness: int = 2,
    ):
        """
        Draw text with a solid background bar for better readability.
        (x, y) is the bottom-left corner of the text baseline.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        # Background rectangle
        cv2.rectangle(
            frame,
            (x - 2, y - text_h - baseline - 4),
            (x + text_w + 2, y + baseline + 2),
            bg_color,
            cv2.FILLED,
        )
        # Text on top
        cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    @staticmethod
    def _nms_rectangles(rects: List[Tuple[int, int, int, int]], iou_thresh: float) -> List[Tuple[int, int, int, int]]:
        """
        Non-maximum suppression for (x, y, w, h) rectangles.
        Keeps the largest boxes and removes highly-overlapping duplicates.
        """
        if not rects:
            return []

        # Sort by area (desc) so we keep the biggest/most stable boxes.
        rects_sorted = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
        kept: List[Tuple[int, int, int, int]] = []

        def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            ax2, ay2 = ax + aw, ay + ah
            bx2, by2 = bx + bw, by + bh

            inter_x1 = max(ax, bx)
            inter_y1 = max(ay, by)
            inter_x2 = min(ax2, bx2)
            inter_y2 = min(ay2, by2)
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter = inter_w * inter_h
            if inter <= 0:
                return 0.0
            union = (aw * ah) + (bw * bh) - inter
            return float(inter) / float(union) if union > 0 else 0.0

        for r in rects_sorted:
            if all(iou(r, k) < iou_thresh for k in kept):
                kept.append(r)

        return kept

    def _parse_folder_identity(self, folder_name: str) -> Tuple[str, str]:
        """
        Expected dataset folder format from `capture_faces.py`: {student_name}_{student_id}
        Example: "jikmet_1" -> ("jikmet", "1")
        """
        if "_" not in folder_name:
            return folder_name, ""
        name, roll = folder_name.rsplit("_", 1)
        return name, roll

    def _load_label_dict(self) -> Dict[int, Dict[str, str]]:
        label_dict: Dict[int, Dict[str, str]] = {}
        dataset_path = self.settings.dataset_path
        if not os.path.exists(dataset_path):
            return label_dict
        label = 0
        for folder in sorted(os.listdir(dataset_path)):
            folder_path = os.path.join(dataset_path, folder)
            if os.path.isdir(folder_path):
                name, roll = self._parse_folder_identity(folder)
                label_dict[label] = {"name": name, "roll": roll}
                label += 1
        return label_dict

    def get_student_identity(self, student_id: int) -> Dict[str, str]:
        return self.label_dict.get(student_id, {"name": f"Unknown_{student_id}", "roll": str(student_id)})

    def _status_from_confidence(self, confidence_pct: float) -> str:
        # Keep wording similar to your sample file.
        if confidence_pct >= 45:
            return "Present (High confidence)"
        if confidence_pct >= 30:
            return "Present (Medium confidence)"
        return "Present (Low confidence)"

    def record_attendance(self, name: str, roll_no: str, confidence_pct: float):
        key = (name, roll_no)
        if key in self.attendance:
            return
        now = datetime.now()
        self.attendance.add(key)
        date_str = now.strftime("%d-%m-%Y")
        time_str = now.strftime("%H:%M:%S")
        status = self._status_from_confidence(confidence_pct)
        self.df.loc[len(self.df)] = [name, roll_no, date_str, time_str, status, f"{confidence_pct:.2f}"]
        print(f"✓ Attendance recorded for {name} ({roll_no}) (Confidence: {confidence_pct:.2f}%)")

    def align_face(self, face_img):
        # Optionally add alignment logic here (e.g., eye detection)
        # For now, just return the resized face
        return cv2.resize(face_img, (200, 200))

    def preprocess_face(self, face_roi):
        # Histogram equalization, normalization, and alignment
        face_roi = cv2.equalizeHist(face_roi)
        face_roi = cv2.GaussianBlur(face_roi, (3, 3), 0)
        face_roi = cv2.normalize(face_roi, None, 0, 255, cv2.NORM_MINMAX)
        face_roi = self.align_face(face_roi)
        return face_roi

    def _face_quality_ok(self, face_roi) -> bool:
        """Filter out very blurry or poorly lit faces to improve accuracy."""
        sharpness = cv2.Laplacian(face_roi, cv2.CV_64F).var()
        brightness = face_roi.mean()
        if sharpness < self.settings.min_sharpness:
            return False
        if not (self.settings.min_brightness <= brightness <= self.settings.max_brightness):
            return False
        return True

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.settings.detection_scale,
            minNeighbors=self.settings.neighbors,
            minSize=(self.settings.min_face_size, self.settings.min_face_size),
        )
        faces_list: List[Tuple[int, int, int, int]] = [tuple(map(int, f)) for f in faces] if len(faces) else []

        # Suppress overlapping duplicate detections (one real face -> multiple boxes).
        if self.settings.enable_nms and faces_list:
            faces_list = self._nms_rectangles(faces_list, float(self.settings.nms_iou_threshold))

        # Hard cap to avoid pathological over-detections in noisy frames.
        if self.settings.max_faces_per_frame and len(faces_list) > int(self.settings.max_faces_per_frame):
            faces_list = sorted(faces_list, key=lambda r: r[2] * r[3], reverse=True)[: int(self.settings.max_faces_per_frame)]

        # Collect candidate matches so we can optionally pick only one per frame.
        candidates: List[Dict[str, Any]] = []

        for (x, y, w, h) in faces_list:
            if w < self.settings.min_face_size or h < self.settings.min_face_size:
                continue  # Skip very small faces
            face_roi = gray[y:y+h, x:x+w]
            face_roi = self.preprocess_face(face_roi)
            if not self._face_quality_ok(face_roi):
                continue  # Skip blurry/poorly lit faces to reduce false matches
            try:
                student_id, confidence = self.recognizer.predict(face_roi)
            except Exception:
                continue
            # OpenCV LBPH "confidence" is a distance (lower is better). Convert to a 0-100-ish score for display.
            distance = float(confidence)
            confidence_percentage = max(0.0, min(100.0, 100.0 - distance))
            ident = self.get_student_identity(student_id)
            predicted_name = ident["name"]
            is_match = distance <= float(self.settings.max_lbph_distance)
            if is_match:
                # In single-person mode we postpone the actual update and
                # pick the best (lowest-distance) candidate after the loop.
                if self.settings.single_person_mode:
                    candidates.append(
                        {
                            "distance": distance,
                            "confidence_pct": confidence_percentage,
                            "ident": ident,
                            "rect": (x, y, w, h),
                        }
                    )
                else:
                    key = f'{ident["name"]}|{ident["roll"]}'
                    self.match_counts[key] = self.match_counts.get(key, 0) + 1
                    count = self.match_counts[key]
                    if count >= self.settings.required_consecutive_frames:
                        self.record_attendance(ident["name"], ident["roll"], confidence_percentage)
                        # Prevent counter from growing unbounded
                        self.match_counts[key] = self.settings.required_consecutive_frames
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 0), 2)
                    display_label = (
                        f'{ident["name"]} ({ident["roll"]}) '
                        f'{confidence_percentage:.0f}% '
                        f'[{count}/{self.settings.required_consecutive_frames}]'
                    )
                    self._draw_label(frame, display_label, x, max(20, y - 10), bg_color=(0, 120, 0))
            else:
                # Reset counts for this identity to avoid stale votes
                for key in list(self.match_counts.keys()):
                    if key.startswith(f'{predicted_name}|'):
                        self.match_counts.pop(key, None)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 2)
                unknown_label = f"Unknown ({confidence_percentage:.0f}%)"
                self._draw_label(frame, unknown_label, x, max(20, y - 10), bg_color=(0, 0, 120))

            if self.settings.show_debug:
                # Show raw LBPH distance so you can tune `max_lbph_distance`
                debug = f"id={student_id} {predicted_name} dist={distance:.1f} thr={self.settings.max_lbph_distance:.0f}"
                self._draw_label(
                    frame,
                    debug,
                    x,
                    y + h + 22,
                    font_scale=0.5,
                    bg_color=(60, 60, 60),
                    text_color=(255, 255, 0),
                    thickness=1,
                )

        # In single-person mode, at most ONE identity per frame can
        # contribute to attendance; pick the best (smallest distance).
        if self.settings.single_person_mode and candidates:
            best = min(candidates, key=lambda c: c["distance"])
            ident = best["ident"]
            distance = best["distance"]
            confidence_percentage = best["confidence_pct"]
            (x, y, w, h) = best["rect"]

            key = f'{ident["name"]}|{ident["roll"]}'
            self.match_counts[key] = self.match_counts.get(key, 0) + 1
            count = self.match_counts[key]
            if count >= self.settings.required_consecutive_frames:
                self.record_attendance(ident["name"], ident["roll"], confidence_percentage)
                self.match_counts[key] = self.settings.required_consecutive_frames

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 0), 2)
            display_label = (
                f'{ident["name"]} ({ident["roll"]}) '
                f'{confidence_percentage:.0f}% '
                f'[{count}/{self.settings.required_consecutive_frames}]'
            )
            self._draw_label(frame, display_label, x, max(20, y - 10), bg_color=(0, 120, 0))

            if self.settings.show_debug:
                debug = (
                    f"id=? {ident['name']} dist={distance:.1f} "
                    f"thr={self.settings.max_lbph_distance:.0f}"
                )
                self._draw_label(
                    frame,
                    debug,
                    x,
                    y + h + 22,
                    font_scale=0.5,
                    bg_color=(60, 60, 60),
                    text_color=(255, 255, 0),
                    thickness=1,
                )
        return frame

    def camera_thread(self, cam):
        while self.running:
            ret, frame = cam.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.settings.frame_width, self.settings.frame_height))
            with self.frame_lock:
                self.latest_frame = frame.copy()

    def run(self):
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            raise RuntimeError("Cannot open camera. Please check camera connection.")
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.frame_width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.frame_height)
        print("Attendance System Started. Press ENTER/RETURN to exit.")
        print("-" * 50)
        self.running = True
        t = threading.Thread(target=self.camera_thread, args=(cam,), daemon=True)
        t.start()
        frame_count = 0
        try:
            while True:
                with self.frame_lock:
                    frame = self.latest_frame.copy() if self.latest_frame is not None else None
                if frame is None:
                    continue
                frame_count += 1
                if self.settings.skip_frames > 1 and (frame_count % self.settings.skip_frames != 0):
                    # Show frame but skip processing for speed
                    display_frame = frame.copy()
                else:
                    display_frame = self.process_frame(frame.copy())
                status_text = f"Attendance Records: {len(self.attendance)}"
                cv2.putText(display_frame, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, "Press ENTER/RETURN to exit", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.imshow('Smart Attendance System', display_frame)
                key = cv2.waitKey(1) & 0xFF
                # Enter/Return: 13 (CR). Some systems may also send 10 (LF).
                if key in (13, 10):
                    break
        finally:
            self.running = False
            t.join(timeout=1)
            self.save_attendance()
            cam.release()
            cv2.destroyAllWindows()

    def _output_paths(self):
        """
        Match `simple_interface.py` which searches for today's date in filenames (%d-%m-%Y).
        """
        Path(self.settings.output_dir).mkdir(exist_ok=True)
        date_str = datetime.now().strftime("%d-%m-%Y")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"attendance_{date_str}_{timestamp}"
        return (
            str(Path(self.settings.output_dir) / f"{base}.csv"),
            str(Path(self.settings.output_dir) / f"{base}.json"),
        )

    def save_attendance(self):
        if len(self.df) == 0:
            print("No attendance records to save.")
            return
        csv_file, json_file = self._output_paths()
        self.df.to_csv(csv_file, index=False)
        print(f"\n✓ Attendance saved to {csv_file}")
        # JSON format should match the CSV schema/keys
        self.df.to_json(json_file, orient="records", indent=2)
        print(f"✓ Attendance saved to {json_file}")
        print(f"Total attendees: {len(self.attendance)}")
        print(f"Records: {len(self.df)}")

def main():
    try:
        system = AttendanceSystem()
        system.run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except RuntimeError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    main()