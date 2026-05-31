import tkinter as tk
import subprocess

process = None

def start_detection():
    global process

    status_label.config(text="Loading model and camera... Please wait")
    start_button.config(state="disabled")
    root.update()

    if process is None or process.poll() is not None:
        process = subprocess.Popen(["python", "live.py"])

    status_label.config(text="Loading model and camera... Please wait")
    start_button.config(state="normal")

def close_all():
    global process
    if process is not None and process.poll() is None:
        process.terminate()
    root.destroy()

root = tk.Tk()
root.title("Emergency Sign Language Support System")
root.geometry("600x400")

title = tk.Label(
    root,
    text="Emergency Sign Language Support System",
    font=("Arial", 18, "bold")
)
title.pack(pady=30)

description = tk.Label(
    root,
    text="Real-time recognition of emergency sign language gestures",
    font=("Arial", 12)
)
description.pack(pady=10)

start_button = tk.Button(
    root,
    text="Start Detection",
    font=("Arial", 14),
    command=start_detection
)
start_button.pack(pady=20)

status_label = tk.Label(
    root,
    text="",
    font=("Arial", 11)
)
status_label.pack(pady=10)

exit_button = tk.Button(
    root,
    text="Exit",
    font=("Arial", 12),
    command=close_all
)
exit_button.pack()

root.protocol("WM_DELETE_WINDOW", close_all)
root.mainloop()