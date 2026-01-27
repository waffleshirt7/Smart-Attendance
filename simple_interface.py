#!/usr/bin/env python3
"""
Simple Unified Attendance Interface
Combines: Attendance marking + Register generation + Report viewing
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import subprocess
import json
import os
from datetime import datetime
from pathlib import Path
import threading
import pandas as pd


class SimpleAttendanceApp:
    """Simple unified attendance application."""

    def __init__(self, root):
        self.root = root
        self.root.title("Smart Attendance System - Simple Interface")
        self.root.geometry("600x500")
        self.root.configure(bg="#f0f2f5")
        self.setup_ui()

    def update_status(self, message):
        """Update status message (non-blocking)."""
        self.status_label.config(text=message)
        self.root.after(10, lambda: None)

    def start_attendance(self):
        """Start attendance, then generate register (single click)."""
        self.update_status("Starting attendance system...")
        self.btn_start.config(state=tk.DISABLED)
        self.btn_report.config(state=tk.DISABLED)
        def run_attendance():
            try:
                # 1) Attendance
                subprocess.run(['python3', 'attendance.py'], check=True)
                self.root.after(0, lambda: self.update_status("Attendance completed. Generating register..."))

                # 2) Register generation
                subprocess.run(['python3', 'semester_register.py'], capture_output=True, text=True, check=True)

                def done_message():
                    self.update_status("Attendance + Register completed!")
                    messagebox.showinfo(
                        "Success",
                        "✓ Attendance completed!\n"
                        "✓ Register generated!\n\n"
                        "Register location: attendance_sheets/\n"
                        "Formats: Excel, HTML, CSV",
                    )
                self.root.after(0, done_message)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed: {e}"))
                self.root.after(0, lambda: self.update_status("Error running attendance/register"))
            finally:
                self.root.after(0, lambda: self.btn_start.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.btn_report.config(state=tk.NORMAL))
        threading.Thread(target=run_attendance, daemon=True).start()

    def setup_ui(self):
        """Setup the user interface with a clean, modern look."""
        # Gradient background using a canvas
        self.bg_canvas = tk.Canvas(self.root, width=600, height=500, highlightthickness=0)
        self.bg_canvas.pack(fill=tk.BOTH, expand=True)
        for i in range(0, 500, 2):
            color = f"#{int(31 + (102-31)*i/500):02x}{int(78 + (238-78)*i/500):02x}{int(120 + (170-120)*i/500):02x}"
            self.bg_canvas.create_rectangle(0, i, 600, i+2, outline="", fill=color)

        # Main frame
        main_frame = tk.Frame(self.root, bg="#f8fafc", bd=0, highlightthickness=0)
        main_frame.place(relx=0.05, rely=0.07, relwidth=0.9, relheight=0.86)

        # Title
        title = tk.Label(main_frame, text="Smart Attendance System", font=("Segoe UI", 22, "bold"), bg="#f8fafc", fg="#1F4E78")
        title.pack(pady=(30, 10))

        # Button frame
        button_frame = tk.Frame(main_frame, bg="#f8fafc")
        button_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Modern.TButton', font=("Segoe UI", 15, "bold"), padding=18, borderwidth=0, relief="flat",
                        background="#4CAF50", foreground="white")
        style.map('Modern.TButton',
                  foreground=[('pressed', 'white'), ('active', 'white')],
                  background=[('pressed', '#388E3C'), ('active', '#2196F3'), ("!active", "#4CAF50")],
                  relief=[('pressed', 'groove'), ('active', 'raised')])

        self.btn_start = ttk.Button(button_frame, text="▶ Start Attendance", command=self.start_attendance, style='Modern.TButton')
        self.btn_start.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=18, pady=14, ipadx=10, ipady=10)

        style.configure('Orange.TButton', font=("Segoe UI", 15, "bold"), padding=18, borderwidth=0, relief="flat",
                        background="#FF9800", foreground="white")
        style.map('Orange.TButton',
                  foreground=[('pressed', 'white'), ('active', 'white')],
                  background=[('pressed', '#F57C00'), ('active', '#FFA726'), ("!active", "#FF9800")],
                  relief=[('pressed', 'groove'), ('active', 'raised')])

        self.btn_report = ttk.Button(button_frame, text="📊 Today's Report", command=self.view_today_report, style='Orange.TButton')
        self.btn_report.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=18, pady=14, ipadx=10, ipady=10)

        # Make grid responsive
        button_frame.grid_rowconfigure(0, weight=1)
        button_frame.grid_rowconfigure(1, weight=1)
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)

        # Status bar
        self.status_label = tk.Label(main_frame, text="Ready", bg="#e0e0e0", 
                                     font=("Segoe UI", 11), pady=10, bd=0, relief="flat")
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))

    
    def view_today_report(self):
        """View today's attendance report (fast, non-blocking)."""
        self.update_status("Loading today's report...")
        self.btn_report.config(state=tk.DISABLED)
        def run_report():
            try:
                records_dir = Path("attendance_records")
                if not records_dir.exists():
                    self.root.after(0, lambda: messagebox.showwarning("No Data", "No attendance records found yet."))
                    return
                today = datetime.now().strftime("%d-%m-%Y")
                json_files = list(records_dir.glob(f"*{today}*.json"))
                if not json_files:
                    self.root.after(0, lambda: messagebox.showwarning("No Data", f"No attendance records for {today}"))
                    return
                latest_file = max(json_files, key=os.path.getctime)
                with open(latest_file, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                def show_report():
                    display_window = tk.Toplevel(self.root)
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
                    self.update_status(f"Showing report for {today}")
                self.root.after(0, show_report)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load report: {e}"))
            finally:
                self.root.after(0, lambda: self.btn_report.config(state=tk.NORMAL))
        threading.Thread(target=run_report, daemon=True).start()
    
        # ...existing code...


def main():
    """Main entry point."""
    root = tk.Tk()
    app = SimpleAttendanceApp(root)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    main()
