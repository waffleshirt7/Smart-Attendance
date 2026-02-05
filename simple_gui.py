#!/usr/bin/env python3
"""
Simple GUI launcher for the SmartAttendance project.

This GUI does not modify existing scripts — it simply runs them and
shows output. Buttons available:
 - Capture Faces (capture_faces.py)
 - Quality Capture (capture_quality_check.py)
 - Train Model (train_model.py)
 - Run Attendance (attendance.py) [start/stop]
 - Download DeepFace Models (deepface_download.py)
 - Run DeepFace Verify Test (test_deepface_verify.py)
 - Open dataset / trainer folders

Designed to be lightweight and safe; uses subprocess to execute scripts.
"""
import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import subprocess
import threading
import os
import shlex


ROOT = os.path.dirname(os.path.abspath(__file__))


class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SmartAttendance Launcher")
        self.geometry("800x520")

        btn_frame = tk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=8, pady=6)

        # Buttons
        actions = [
            ("Capture Faces", f"python3 {os.path.join(ROOT, 'capture_faces.py')}") ,
            ("Quality Capture", f"python3 {os.path.join(ROOT, 'capture_quality_check.py')}") ,
            ("Train Model", f"python3 {os.path.join(ROOT, 'train_model.py')}") ,
            ("Download DeepFace Models", f"python3 {os.path.join(ROOT, 'deepface_download.py')}") ,
            ("DeepFace Verify Test", f"python3 {os.path.join(ROOT, 'test_deepface_verify.py')}") ,
        ]

        for (label, cmd) in actions:
            b = tk.Button(btn_frame, text=label, width=18, command=lambda c=cmd: self.run_command(c))
            b.pack(side=tk.LEFT, padx=4, pady=4)

        # Attendance start/stop
        self.att_proc = None
        self.start_btn = tk.Button(btn_frame, text="Start Attendance", width=16, command=self.start_attendance)
        self.start_btn.pack(side=tk.LEFT, padx=4)
        self.stop_btn = tk.Button(btn_frame, text="Stop Attendance", width=16, command=self.stop_attendance, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=4)

        # Open folders
        folder_frame = tk.Frame(self)
        folder_frame.pack(fill=tk.X, padx=8)
        tk.Button(folder_frame, text="Open dataset", command=lambda: self.open_path(os.path.join(ROOT, 'dataset'))).pack(side=tk.LEFT, padx=4)
        tk.Button(folder_frame, text="Open trainer", command=lambda: self.open_path(os.path.join(ROOT, 'trainer'))).pack(side=tk.LEFT, padx=4)
        tk.Button(folder_frame, text="Open attendance_records", command=lambda: self.open_path(os.path.join(ROOT, 'attendance_records'))).pack(side=tk.LEFT, padx=4)

        # Output console
        self.console = scrolledtext.ScrolledText(self, height=22)
        self.console.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.console.configure(state='disabled')

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status = tk.Label(self, textvariable=self.status_var, anchor='w')
        status.pack(fill=tk.X, side=tk.BOTTOM)

    def append(self, text: str):
        self.console.configure(state='normal')
        self.console.insert(tk.END, text)
        self.console.see(tk.END)
        self.console.configure(state='disabled')

    def run_command(self, cmd: str):
        def target():
            self.status_var.set(f"Running: {cmd}")
            self.append(f"$ {cmd}\n")
            try:
                p = subprocess.Popen(shlex.split(cmd), cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in p.stdout:
                    self.append(line)
                p.wait()
                self.append(f"[EXIT {p.returncode}]\n")
            except Exception as e:
                self.append(f"Error starting command: {e}\n")
            finally:
                self.status_var.set("Ready")

        threading.Thread(target=target, daemon=True).start()

    def start_attendance(self):
        if self.att_proc is not None:
            messagebox.showinfo("Attendance", "Attendance already running")
            return

        cmd = f"python3 {os.path.join(ROOT, 'attendance.py')}"
        self.status_var.set("Starting attendance...")
        self.append(f"$ {cmd}\n")

        try:
            self.att_proc = subprocess.Popen(shlex.split(cmd), cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except Exception as e:
            self.append(f"Failed to start attendance: {e}\n")
            self.att_proc = None
            return

        # start a thread to read output
        def reader():
            for line in self.att_proc.stdout:
                self.append(line)
            self.append(f"[ATTENDANCE EXIT {self.att_proc.returncode}]\n")
            self.att_proc = None
            self.start_btn.configure(state=tk.NORMAL)
            self.stop_btn.configure(state=tk.DISABLED)
            self.status_var.set("Ready")

        threading.Thread(target=reader, daemon=True).start()
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.status_var.set("Attendance running")

    def stop_attendance(self):
        if not self.att_proc:
            return
        self.append("Stopping attendance (terminate)...\n")
        try:
            self.att_proc.terminate()
        except Exception as e:
            self.append(f"Error terminating process: {e}\n")

    def open_path(self, path: str):
        if not os.path.exists(path):
            messagebox.showwarning("Open path", f"Path does not exist: {path}")
            return
        if os.name == 'posix':
            subprocess.Popen(['open', path])
        else:
            subprocess.Popen(['explorer', path])


if __name__ == '__main__':
    app = Launcher()
    app.mainloop()
