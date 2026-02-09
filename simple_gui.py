#!/usr/bin/env python3
"""
Simple GUI launcher for the SmartAttendance project.

This GUI does not modify existing scripts — it simply runs them and
shows output. Buttons available:
 - Capture Faces (capture_faces.py)
 - Quality Capture (capture_quality_check.py)
 - Train Model (train_model.py)
 - Run Attendance (attendance.py) - press Enter in camera window to exit and generate sheet
 - Download DeepFace Models (deepface_download.py)
 - Run DeepFace Verify Test (test_deepface_verify.py)
 - Open dataset / trainer folders

Designed to be lightweight and safe; uses subprocess to execute scripts.
"""
import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import subprocess
import sys
import threading
import os
import shlex
import json
from pathlib import Path
from datetime import datetime
import pandas as pd


ROOT = os.path.dirname(os.path.abspath(__file__))


class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SmartAttendance")
        self.geometry("800x520")

        btn_frame = tk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=8, pady=6)

        # Buttons
        actions = [
            # ("Capture Faces", f"python3 {os.path.join(ROOT, 'capture_faces.py')}") ,
            ("Train Model", f"python3 {os.path.join(ROOT, 'train_model.py')}") ,
        ]

        for (label, cmd) in actions:
            b = tk.Button(btn_frame, text=label, width=18, command=lambda c=cmd: self.run_command(c))
            b.pack(side=tk.LEFT, padx=4, pady=4)

        # Attendance
        self.att_proc = None
        self.start_btn = tk.Button(btn_frame, text="Start Attendance", width=16, command=self.start_attendance)
        self.start_btn.pack(side=tk.LEFT, padx=4)
        self.report_btn = tk.Button(btn_frame, text="Today's Report", width=16, command=self.view_today_report)
        self.report_btn.pack(side=tk.LEFT, padx=4)

        # Open folders
        folder_frame = tk.Frame(self)
        folder_frame.pack(fill=tk.X, padx=8)
        tk.Button(folder_frame, text="Open dataset", command=lambda: self.open_path(os.path.join(ROOT, 'dataset'))).pack(side=tk.LEFT, padx=4)
        tk.Button(folder_frame, text="Open trainer", command=lambda: self.open_path(os.path.join(ROOT, 'trainer'))).pack(side=tk.LEFT, padx=4)
        tk.Button(folder_frame, text="Open attendance_records", command=lambda: self.open_path(os.path.join(ROOT, 'attendance_records'))).pack(side=tk.LEFT, padx=4)
        tk.Button(folder_frame, text="Open attendance_sheets", command=lambda: self.open_path(os.path.join(ROOT, 'attendance_sheets'))).pack(side=tk.LEFT, padx=4)

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
            # Generate attendance sheet (same as simple_interface.py)
            self.append("Generating attendance sheet...\n")
            try:
                subprocess.run(
                    [sys.executable, os.path.join(ROOT, 'semester_register.py')],
                    cwd=ROOT, capture_output=True, text=True
                )
                self.append("[Sheet generated from attendance_records]\n")
            except Exception as e:
                self.append(f"Error generating sheet: {e}\n")
            self.start_btn.configure(state=tk.NORMAL)
            self.status_var.set("Ready")

        threading.Thread(target=reader, daemon=True).start()
        self.start_btn.configure(state=tk.DISABLED)
        self.status_var.set("Attendance running - Press ENTER in camera window to exit")

    def view_today_report(self):
        """View today's attendance report (fast, non-blocking)."""
        self.status_var.set("Loading today's report...")
        self.report_btn.configure(state=tk.DISABLED)

        def run_report():
            try:
                records_dir = Path(ROOT) / "attendance_records"
                if not records_dir.exists():
                    self.after(0, lambda: messagebox.showwarning("No Data", "No attendance records found yet."))
                    return
                today = datetime.now().strftime("%d-%m-%Y")
                json_files = list(records_dir.glob(f"*{today}*.json"))
                if not json_files:
                    self.after(0, lambda: messagebox.showwarning("No Data", f"No attendance records for {today}"))
                    return
                latest_file = max(json_files, key=os.path.getctime)
                with open(latest_file, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)

                def show_report():
                    display_window = tk.Toplevel(self)
                    display_window.title(f"Today's Attendance Report - {today}")
                    display_window.geometry("700x500")
                    text_widget = scrolledtext.ScrolledText(display_window, wrap=tk.WORD, font=("Courier", 10))
                    text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                    report = "=" * 80 + "\n"
                    report += f"ATTENDANCE REPORT - {today}\n"
                    report += "=" * 80 + "\n\n"
                    report += df.to_string(index=False)
                    report += "\n\n" + "=" * 80 + "\n"
                    report += f"Total Present: {len(df)} students\n"
                    report += "=" * 80
                    text_widget.insert(tk.END, report)
                    text_widget.config(state=tk.DISABLED)
                    self.status_var.set(f"Showing report for {today}")

                self.after(0, show_report)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", f"Failed to load report: {e}"))
            finally:
                self.after(0, lambda: self.report_btn.configure(state=tk.NORMAL))

        threading.Thread(target=run_report, daemon=True).start()

    def open_path(self, path: str):
        if not os.path.exists(path):
            messagebox.showwarning("Open path", f"Path does not exist: {path}")
            return
        if sys.platform == 'darwin':
            subprocess.Popen(['open', path])
        elif sys.platform == 'win32':
            subprocess.Popen(['explorer', path])
        else:
            subprocess.Popen(['xdg-open', path])


if __name__ == '__main__':
    app = Launcher()
    app.mainloop()
