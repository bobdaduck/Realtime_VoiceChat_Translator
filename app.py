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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Filter out the specific warning
warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning, message="data discontinuity in recording")

# Configuration
ENGLISH_MODEL_PATH = "model"
CHINESE_MODEL_PATH = "chinese-model"
SAMPLE_RATE = 16000
BUFFER_SIZE = 1024  # Smaller buffer size may help with discontinuities
DISPLAY_FONT = ('Arial', 24)
WINDOW_OPACITY = 0.7
TEXT_COLOR = 'teal'  # Changed text color from white to teal

# Load Vosk models
english_model = Model(ENGLISH_MODEL_PATH)
chinese_model = Model(CHINESE_MODEL_PATH)

# Load Hugging Face translation models
print("Downloading and loading translation models... (this may take a few minutes on first run)")
zh_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
zh_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

en_zh_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
en_zh_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
print("Translation models loaded successfully!")

# Get default speaker with loopback and microphone
speaker = sc.default_speaker()
loopback_mic = sc.get_microphone(speaker.id, include_loopback=True)
regular_mic = sc.default_microphone()

# Shared transcription and timing
english_transcriptions = []
english_translations = []  # Store translations of English to Chinese
english_pinyin = []        # Store pinyin of English translations
chinese_transcriptions = []
chinese_pinyin = []
chinese_translations = []  # Store translations of Chinese to English
last_english_text_time = 0.0
last_chinese_text_time = 0.0
lock = threading.Lock()

def translate_chinese_to_english(text):
    """Translate Chinese text to English using Hugging Face model"""
    try:
        # Skip translation if text is empty
        if not text.strip():
            return ""
            
        inputs = zh_en_tokenizer(text, return_tensors="pt", padding=True)
        output_tokens = zh_en_model.generate(**inputs, max_length=128)
        translation = zh_en_tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return translation
    except Exception as e:
        print(f"Chinese to English translation error: {str(e)}")
        return f"[Translation Error: {str(e)}]"

def translate_english_to_chinese(text):
    """Translate English text to Chinese using Hugging Face model"""
    try:
        # Skip translation if text is empty
        if not text.strip():
            return ""
            
        inputs = en_zh_tokenizer(text, return_tensors="pt", padding=True)
        output_tokens = en_zh_model.generate(**inputs, max_length=128)
        translation = en_zh_tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return translation
    except Exception as e:
        print(f"English to Chinese translation error: {str(e)}")
        return f"[Translation Error: {str(e)}]"

def update_displays():
    with lock:
        current_time = time.time()
        
        # Update English display (microphone)
        if current_time - last_english_text_time <= 5:
            if english_transcriptions and english_pinyin:
                english_text = f"{english_transcriptions[-1]}\n{english_pinyin[-1]}"
            elif english_transcriptions:
                english_text = english_transcriptions[-1]
            else:
                english_text = ''
        else:
            english_text = ''
        english_label.config(text=english_text)
        
        # Update Chinese display (system audio)
        if current_time - last_chinese_text_time <= 5:
            if chinese_pinyin and chinese_translations:
                chinese_text = f"{chinese_pinyin[-1]}\n{chinese_translations[-1]}"
            elif chinese_pinyin:
                chinese_text = chinese_pinyin[-1]
            else:
                chinese_text = ''
        else:
            chinese_text = ''
        chinese_label.config(text=chinese_text)
        
    root.after(100, update_displays)  # Continue updating

def capture_english_audio():
    global english_transcriptions, english_translations, english_pinyin, last_english_text_time
    recognizer = KaldiRecognizer(english_model, SAMPLE_RATE)
    
    # Error recovery retry logic
    while True:
        try:
            # Using regular_mic for English
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
                                # Translate English to Chinese
                                chinese_translation = translate_english_to_chinese(text)
                                
                                # Generate pinyin for the Chinese translation
                                pinyin_result = ' '.join(pypinyin.lazy_pinyin(
                                    chinese_translation, 
                                    style=pypinyin.Style.TONE  # Add tone markers
                                ))
                                
                                with lock:
                                    english_transcriptions = [text]
                                    english_translations = [chinese_translation]
                                    english_pinyin = [pinyin_result]
                                    last_english_text_time = time.time()
                        else:
                            partial = json.loads(recognizer.PartialResult())
                            text = partial.get('partial', '')
                            if text:
                                # Optional: Only translate complete sentences to reduce processing
                                if len(text) > 5:  # Only translate substantial text
                                    # Translate English to Chinese
                                    chinese_translation = translate_english_to_chinese(text)
                                    
                                    # Generate pinyin for the Chinese translation
                                    pinyin_result = ' '.join(pypinyin.lazy_pinyin(
                                        chinese_translation, 
                                        style=pypinyin.Style.TONE  # Add tone markers
                                    ))
                                    
                                    with lock:
                                        english_transcriptions = [text]
                                        english_translations = [chinese_translation]
                                        english_pinyin = [pinyin_result]
                                        last_english_text_time = time.time()
                                else:
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
    global chinese_transcriptions, chinese_pinyin, chinese_translations, last_chinese_text_time
    recognizer = KaldiRecognizer(chinese_model, SAMPLE_RATE)
    
    # Error recovery retry logic
    while True:
        try:
            # Using loopback_mic for Chinese
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
                                # Use tone marks in pinyin conversion
                                pinyin_result = ' '.join(pypinyin.lazy_pinyin(
                                    text, 
                                    style=pypinyin.Style.TONE  # Add tone markers
                                ))
                                
                                # Translate Chinese to English
                                translation = translate_chinese_to_english(text)
                                
                                with lock:
                                    chinese_transcriptions = [text]
                                    chinese_pinyin = [pinyin_result]
                                    chinese_translations = [translation]
                                    last_chinese_text_time = time.time()
                        else:
                            partial = json.loads(recognizer.PartialResult())
                            text = partial.get('partial', '')
                            if text:
                                # Use tone marks in pinyin conversion
                                pinyin_result = ' '.join(pypinyin.lazy_pinyin(
                                    text, 
                                    style=pypinyin.Style.TONE  # Add tone markers
                                ))
                                
                                # Optional: Only translate substantial text
                                if len(text) > 5:
                                    # Translate Chinese to English
                                    translation = translate_chinese_to_english(text)
                                    
                                    with lock:
                                        chinese_transcriptions = [text]
                                        chinese_pinyin = [pinyin_result]
                                        chinese_translations = [translation]
                                        last_chinese_text_time = time.time()
                                else:
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