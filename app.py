import soundcard as sc
from soundcard import SoundcardRuntimeWarning
from vosk import Model, KaldiRecognizer
import numpy as np
import json
import threading
import tkinter as tk
from tkinter import font
import ctypes
import time  # Added for time tracking

# Configuration
MODEL_PATH = "model"
SAMPLE_RATE = 16000
CHUNK_SIZE = 5  # Seconds
DISPLAY_FONT = ('Arial', 24)
WINDOW_OPACITY = 0.7

# Load Vosk model
model = Model(MODEL_PATH)

# Get default speaker with loopback
speaker = sc.default_speaker()
mic = sc.get_microphone(speaker.id, include_loopback=True)

# Shared transcription and timing
transcriptions = []
last_text_time = 0.0  # Track last update time
lock = threading.Lock()

def update_display():
    with lock:
        current_time = time.time()
        # Check if 15 seconds have passed since last text
        if current_time - last_text_time <= 5 and transcriptions:
            current_text = transcriptions[-1]
        else:
            current_text = ''
    label.config(text=current_text)
    root.after(100, update_display)  # Continue updating

def capture_and_transcribe():
    global transcriptions, last_text_time
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    
    with mic.recorder(samplerate=SAMPLE_RATE, 
                    blocksize=SAMPLE_RATE,  # 1-second chunks
                    channels=1) as recorder:
        while True:
            try:
                audio_data = recorder.record(numframes=SAMPLE_RATE)
                if audio_data.size == 0:
                    continue
                
                audio_int16 = (audio_data * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()

                recognizer.AcceptWaveform(audio_bytes)
                result = json.loads(recognizer.PartialResult())
                text = result.get('partial', '')
                
                # Update only if there's new text
                with lock:
                    if text:
                        transcriptions = [text]
                        last_text_time = time.time()

            except Exception as e:
                if "data discontinuity" not in str(e):
                    print(f"Audio error: {str(e)}")
                continue

# Start capture thread
thread = threading.Thread(target=capture_and_transcribe, daemon=True)
thread.start()

# Overlay setup
root = tk.Tk()
root.attributes('-alpha', WINDOW_OPACITY)
root.attributes('-topmost', True)
root.overrideredirect(True)
# Adjusted geometry for top-left and smaller size
root.geometry('800x200+0+0')  # Width:800, Height:200, Position:Top-Left
root.configure(bg='black')
root.wm_attributes("-transparent", "black")

# Windows API for click-through
try:
    hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
    WS_EX_LAYERED = 0x80000
    WS_EX_TRANSPARENT = 0x20
    GWL_EXSTYLE = -20

    current_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, 
                                      current_style | WS_EX_LAYERED | WS_EX_TRANSPARENT)
    ctypes.windll.user32.SetLayeredWindowAttributes(hwnd, 0x000000, 0, 0x1)
    ctypes.windll.user32.UpdateWindow(hwnd)
except Exception as e:
    print(f"Couldn't set click-through: {str(e)}")

label = tk.Label(
    root,
    text='',
    font=DISPLAY_FONT,
    fg='white',
    bg='black',
    wraplength=780,  # Adjusted for smaller window
    justify='left'
)
label.pack(expand=True, fill='both')

update_display()

try:
    root.mainloop()
except KeyboardInterrupt:
    root.destroy()
    print("Exiting...")