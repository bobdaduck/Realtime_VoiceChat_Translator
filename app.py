import soundcard as sc
from vosk import Model, KaldiRecognizer
import numpy as np
import json
import threading
import tkinter as tk
from tkinter import font

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

# Shared transcription
transcriptions = []
lock = threading.Lock()

def capture_and_transcribe():
    global transcriptions 
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    
    with mic.recorder(samplerate=SAMPLE_RATE) as recorder:
        while True:
            # Capture audio directly from system output
            audio_data = recorder.record(numframes=SAMPLE_RATE * CHUNK_SIZE)
            # Convert to mono and 16-bit PCM
            audio_mono = np.mean(audio_data, axis=1)
            audio_int16 = (audio_mono * 32767).astype(np.int16).tobytes()
            
            if recognizer.AcceptWaveform(audio_int16):
                result = json.loads(recognizer.Result())
                with lock:
                    transcriptions.insert(0, result.get('text', ''))
                    transcriptions = transcriptions[:5]  # Keep last 5

# Start capture thread
thread = threading.Thread(target=capture_and_transcribe, daemon=True)
thread.start()

# Create overlay window
root = tk.Tk()
root.attributes('-alpha', WINDOW_OPACITY)
root.attributes('-topmost', True)
root.overrideredirect(True)
root.geometry('1000x300+50+50')
root.configure(bg='black')

label = tk.Label(
    root,
    text='',
    font=DISPLAY_FONT,
    fg='white',
    bg='black',
    wraplength=980,
    justify='left'
)
label.pack(expand=True, fill='both')

def update_display():
    with lock:
        current_text = '\n\n'.join(transcriptions)
    label.config(text=current_text)
    root.after(500, update_display)

update_display()

try:
    root.mainloop()
except KeyboardInterrupt:
    root.destroy()
    print("Exiting...")