import os
import csv
import tkinter as tk
from tkinter import ttk, messagebox

import sys
import subprocess

ATT_FILE = "attendance.csv"


def open_attendance():
    if not os.path.exists(ATT_FILE):
        messagebox.showwarning("No file", "No attendance file found. Run attendance.py first.")
        return

    # Clear existing rows
    for row in tree.get_children():
        tree.delete(row)

    try:
        with open(ATT_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                messagebox.showerror("Error", "Attendance file is empty or corrupted.")
                return
            
            count = 0
            for r in reader:
                if not r or "Name" not in r:
                    continue
                name = r.get("Name", "Unknown").strip()
                date = r.get("Date", "").strip()
                time = r.get("Time", "").strip()
                
                if name and name != "Unknown":
                    tree.insert("", "end", values=(name, date, time))
                    count += 1
            
            if count == 0:
                messagebox.showinfo("Info", "No attendance records found.")
            else:
                messagebox.showinfo("Success", f"Loaded {count} attendance records.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load attendance: {e}")


def start_attendance():
    target_script = "attendance.py"
    if not os.path.exists(target_script):
        messagebox.showerror("Error", "No attendance script found. Check your folder contents.")
        return

    try:
        subprocess.Popen([sys.executable, target_script], shell=False)
        messagebox.showinfo("Started", "Attendance camera started.\nPress Q in the camera window to stop it.")
    except Exception as e:
        messagebox.showerror("Error", f"Could not start attendance: {e}")


def clear_log():
    if os.path.exists(ATT_FILE):
        os.remove(ATT_FILE)
    for row in tree.get_children():
        tree.delete(row)
    messagebox.showinfo("Cleared", "Attendance log cleared.")


root = tk.Tk()
root.title("Smart Attendance - Face Recognition")
root.geometry("650x400")
root.resizable(False, False)

button_frame = tk.Frame(root)
button_frame.pack(pady=10, fill="x")

start_btn = tk.Button(button_frame, text="Start Attendance", command=start_attendance, width=16)
start_btn.pack(side="left", padx=10)

open_btn = tk.Button(button_frame, text="Load Attendance", command=open_attendance, width=16)
open_btn.pack(side="left", padx=10)

refresh_btn = tk.Button(button_frame, text="Refresh", command=open_attendance, width=16)
refresh_btn.pack(side="left", padx=10)

clear_btn = tk.Button(button_frame, text="Clear Log", command=clear_log, width=16)
clear_btn.pack(side="left", padx=10)

cols = ("Name", "Date", "Time")
tree = ttk.Treeview(root, columns=cols, show="headings")
for c in cols:
    tree.heading(c, text=c)
    tree.column(c, width=200, anchor="center")

tree.pack(fill="both", expand=True, padx=10, pady=5)

# Auto-load attendance on startup
open_attendance()

root.mainloop()
