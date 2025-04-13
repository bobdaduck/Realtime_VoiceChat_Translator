import soundcard as sc
from soundcard import SoundcardRuntimeWarning
from vosk import Model, KaldiRecognizer
import numpy as np
import json
import threading
import tkinter as tk
from tkinter import font
import ctypes
import time
import pypinyin
import warnings

# Filter out the specific warning
warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning, message="data discontinuity in recording")

# Configuration
ENGLISH_MODEL_PATH = "model"
CHINESE_MODEL_PATH = "chinese-model"  # You'll need to download a Chinese Vosk model
SAMPLE_RATE = 16000
BUFFER_SIZE = 1024  # Smaller buffer size may help with discontinuities
DISPLAY_FONT = ('Arial', 24)
WINDOW_OPACITY = 0.7
TEXT_COLOR = 'teal'  # Changed text color from white to teal

# Load Vosk models
english_model = Model(ENGLISH_MODEL_PATH)
chinese_model = Model(CHINESE_MODEL_PATH)

# Get default speaker with loopback and microphone
speaker = sc.default_speaker()
loopback_mic = sc.get_microphone(speaker.id, include_loopback=True)
regular_mic = sc.default_microphone()

# Shared transcription and timing
english_transcriptions = []
chinese_transcriptions = []
chinese_pinyin = []
last_english_text_time = 0.0
last_chinese_text_time = 0.0
lock = threading.Lock()

def update_displays():
    with lock:
        current_time = time.time()
        
        # Update English display
        if current_time - last_english_text_time <= 5 and english_transcriptions:
            english_text = english_transcriptions[-1]
        else:
            english_text = ''
        english_label.config(text=english_text)
        
        # Update Chinese display
        if current_time - last_chinese_text_time <= 5 and chinese_pinyin:
            chinese_text = chinese_pinyin[-1]
        else:
            chinese_text = ''
        chinese_label.config(text=chinese_text)
        
    root.after(100, update_displays)  # Continue updating

def capture_english_audio():
    global english_transcriptions, last_english_text_time
    recognizer = KaldiRecognizer(english_model, SAMPLE_RATE)
    
    # Error recovery retry logic
    while True:
        try:
            with loopback_mic.recorder(samplerate=SAMPLE_RATE, 
                                      blocksize=BUFFER_SIZE, 
                                      channels=1) as recorder:
                while True:
                    try:
                        audio_data = recorder.record(numframes=SAMPLE_RATE)
                        if audio_data.size == 0:
                            time.sleep(0.01)  # Small delay to reduce CPU usage
                            continue
                        
                        audio_int16 = (audio_data * 32767).astype(np.int16)
                        audio_bytes = audio_int16.tobytes()
                        
                        if recognizer.AcceptWaveform(audio_bytes):
                            result = json.loads(recognizer.Result())
                            text = result.get('text', '')
                            if text:
                                with lock:
                                    english_transcriptions = [text]
                                    last_english_text_time = time.time()
                        else:
                            partial = json.loads(recognizer.PartialResult())
                            text = partial.get('partial', '')
                            if text:
                                with lock:
                                    english_transcriptions = [text]
                                    last_english_text_time = time.time()
                                    
                    except Exception as e:
                        print(f"English processing error: {str(e)}")
                        time.sleep(0.1)  # Brief pause before continuing
                        continue
        except Exception as e:
            print(f"English recorder error: {str(e)}")
            time.sleep(1)  # Wait a bit before trying to reconnect
            # Reset recognizer in case it's corrupted
            recognizer = KaldiRecognizer(english_model, SAMPLE_RATE)

def capture_chinese_audio():
    global chinese_transcriptions, chinese_pinyin, last_chinese_text_time
    recognizer = KaldiRecognizer(chinese_model, SAMPLE_RATE)
    
    # Error recovery retry logic
    while True:
        try:
            with regular_mic.recorder(samplerate=SAMPLE_RATE, 
                                     blocksize=BUFFER_SIZE, 
                                     channels=1) as recorder:
                while True:
                    try:
                        audio_data = recorder.record(numframes=SAMPLE_RATE)
                        if audio_data.size == 0:
                            time.sleep(0.01)  # Small delay to reduce CPU usage
                            continue
                        
                        audio_int16 = (audio_data * 32767).astype(np.int16)
                        audio_bytes = audio_int16.tobytes()
                        
                        if recognizer.AcceptWaveform(audio_bytes):
                            result = json.loads(recognizer.Result())
                            text = result.get('text', '')
                            if text:
                                pinyin_result = ' '.join(pypinyin.lazy_pinyin(text))
                                with lock:
                                    chinese_transcriptions = [text]
                                    chinese_pinyin = [pinyin_result]
                                    last_chinese_text_time = time.time()
                        else:
                            partial = json.loads(recognizer.PartialResult())
                            text = partial.get('partial', '')
                            if text:
                                pinyin_result = ' '.join(pypinyin.lazy_pinyin(text))
                                with lock:
                                    chinese_transcriptions = [text]
                                    chinese_pinyin = [pinyin_result]
                                    last_chinese_text_time = time.time()
                                    
                    except Exception as e:
                        print(f"Chinese processing error: {str(e)}")
                        time.sleep(0.1)  # Brief pause before continuing
                        continue
        except Exception as e:
            print(f"Chinese recorder error: {str(e)}")
            time.sleep(1)  # Wait a bit before trying to reconnect
            # Reset recognizer in case it's corrupted
            recognizer = KaldiRecognizer(chinese_model, SAMPLE_RATE)

# Start capture threads
english_thread = threading.Thread(target=capture_english_audio, daemon=True)
english_thread.start()

chinese_thread = threading.Thread(target=capture_chinese_audio, daemon=True)
chinese_thread.start()

# Create a single root window
root = tk.Tk()
root.attributes('-alpha', WINDOW_OPACITY)
root.attributes('-topmost', True)
root.overrideredirect(True)
screen_width = root.winfo_screenwidth()
root.geometry(f'{screen_width}x200+0+0')  # Full width at top of screen
root.configure(bg='black')
root.wm_attributes("-transparent", "black")

# Windows API for click-through
try:
    hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
    WS_EX_LAYERED = 0x80000
    WS_EX_TRANSPARENT = 0x20
    GWL_EXSTYLE = -20
    
    current_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, current_style | WS_EX_LAYERED | WS_EX_TRANSPARENT)
    ctypes.windll.user32.SetLayeredWindowAttributes(hwnd, 0x000000, 0, 0x1)
    ctypes.windll.user32.UpdateWindow(hwnd)
except Exception as e:
    print(f"Couldn't set click-through: {str(e)}")

# Create two frames side by side
left_frame = tk.Frame(root, bg='black')
left_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

right_frame = tk.Frame(root, bg='black')
right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

# Labels in each frame with teal text color
english_label = tk.Label(
    left_frame,
    text='',
    font=DISPLAY_FONT,
    fg=TEXT_COLOR,  # Changed to teal
    bg='black',
    wraplength=screen_width//2-20,
    justify='left'
)
english_label.pack(expand=True, fill='both')

chinese_label = tk.Label(
    right_frame,
    text='',
    font=DISPLAY_FONT,
    fg=TEXT_COLOR,  # Changed to teal
    bg='black',
    wraplength=screen_width//2-20,
    justify='right'
)
chinese_label.pack(expand=True, fill='both')

# Start display updates
root.after(100, update_displays)

# Add a callback for the window close button
def on_closing():
    print("Closing application...")
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Add keyboard shortcut to exit (Esc key)
def on_escape(event):
    on_closing()

root.bind('<Escape>', on_escape)

# Run the main window
try:
    root.mainloop()
except KeyboardInterrupt:
    root.destroy()
    print("Exiting...")
except Exception as e:
    print(f"Application error: {str(e)}")
    root.destroy()