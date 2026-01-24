import cv2
import pandas as pd
import os
import json
from datetime import datetime
from pathlib import Path


class AttendanceSystem:
    """Smart Attendance System using face recognition."""
    
    CONFIDENCE_THRESHOLD = 70
    DETECTION_SCALE = 1.2
    NEIGHBORS = 5
    
    def __init__(self, model_path="trainer/trainer.yml", cascade_path="haarcascade_frontalface_default.xml"):
        """Initialize the attendance system with face recognizer and cascade classifier."""
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
        
        self.recognizer.read(model_path)
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise FileNotFoundError(f"Cascade classifier not found at {cascade_path}")
        
        self.attendance = {}
        self.df = pd.DataFrame(columns=["Name", "Date", "Time", "Confidence"])
        self.label_dict = self._load_label_dict()
    
    def _load_label_dict(self):
        """Load label dictionary from training data."""
        label_dict = {}
        dataset_path = "dataset"
        
        if not os.path.exists(dataset_path):
            return label_dict
        
        label = 0
        for folder in sorted(os.listdir(dataset_path)):
            folder_path = os.path.join(dataset_path, folder)
            if os.path.isdir(folder_path):
                label_dict[label] = folder
                label += 1
        
        return label_dict
    
    def get_student_name(self, student_id):
        """Get student name from ID using label dictionary."""
        return self.label_dict.get(student_id, f"Unknown_{student_id}")
    
    def record_attendance(self, name, confidence):
        """Record attendance for a student if not already recorded."""
        if name not in self.attendance:
            now = datetime.now()
            self.attendance[name] = True
            self.df.loc[len(self.df)] = [name, now.date(), now.strftime("%H:%M:%S"), f"{confidence:.2f}"]
            print(f"✓ Attendance recorded for {name} (Confidence: {confidence:.2f}%)")
    
    def process_frame(self, frame):
        """Process a single frame for face detection and recognition."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, self.DETECTION_SCALE, self.NEIGHBORS)
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            student_id, confidence = self.recognizer.predict(face_roi)
            
            # Confidence should be lower for better match (0 = perfect match)
            confidence_percentage = 100 - confidence
            
            if confidence < self.CONFIDENCE_THRESHOLD:
                name = self.get_student_name(student_id)
                self.record_attendance(name, confidence_percentage)
                
                # Draw green rectangle and label for recognized face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({confidence_percentage:.0f}%)", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                # Draw red rectangle and label for unknown face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, f"Unknown ({confidence_percentage:.0f}%)", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return frame
    
    def run(self):
        """Main loop for the attendance system."""
        cam = cv2.VideoCapture(0)
        
        if not cam.isOpened():
            raise RuntimeError("Cannot open camera. Please check camera connection.")
        
        print("Attendance System Started. Press ESC to exit.")
        print("-" * 50)
        
        try:
            while True:
                ret, frame = cam.read()
                
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Add status bar
                status_text = f"Attendance Records: {len(self.attendance)}"
                cv2.putText(processed_frame, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(processed_frame, "Press ESC to exit", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                cv2.imshow('Smart Attendance System', processed_frame)
                
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    break
        
        finally:
            self.save_attendance()
            cam.release()
            cv2.destroyAllWindows()
    
    def save_attendance(self):
        """Save attendance records to CSV and JSON."""
        if len(self.df) == 0:
            print("No attendance records to save.")
            return
        
        # Create output directory if it doesn't exist
        Path("attendance_records").mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"attendance_records/attendance_{timestamp}.csv"
        json_file = f"attendance_records/attendance_{timestamp}.json"
        
        # Save to CSV
        self.df.to_csv(csv_file, index=False)
        print(f"\n✓ Attendance saved to {csv_file}")
        
        # Save to JSON
        self.df.to_json(json_file, orient='records', indent=2)
        print(f"✓ Attendance saved to {json_file}")
        
        print(f"Total attendees: {len(self.attendance)}")
        print(f"Records: {len(self.df)}")


def main():
    """Main entry point."""
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