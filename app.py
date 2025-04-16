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
WINDOW_OPACITY = 0.8
TEXT_COLOR = '#20EADA'  

# Windows-specific constants for transparent/click-through windows
WS_EX_TRANSPARENT = 0x00000020
WS_EX_LAYERED = 0x00080000
GWL_EXSTYLE = -20
LWA_ALPHA = 0x00000002
LWA_COLORKEY = 0x00000001

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

# Shared transcription and timing - using dictionaries for better organization
english_display = {
    "transcription": "",
    "translation": "",
    "pinyin": "",
    "last_update_time": 0.0
}

chinese_display = {
    "transcription": "",
    "translation": "",
    "pinyin": "",
    "last_update_time": 0.0
}

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
        return f"[Translation Error]"

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
        return f"[Translation Error]"

def generate_pinyin(chinese_text):
    """Generate pinyin with tone marks for Chinese text"""
    if not chinese_text.strip():
        return ""
    try:
        return ' '.join(pypinyin.lazy_pinyin(
            chinese_text, 
            style=pypinyin.Style.TONE  # Add tone markers
        ))
    except Exception as e:
        print(f"Pinyin generation error: {str(e)}")
        return "[Pinyin Error]"

def update_displays():
    with lock:
        current_time = time.time()
        
        # Update English display (microphone)
        if current_time - english_display["last_update_time"] <= 15:
            english_text = f"{english_display['transcription']}\n{english_display['translation']}"
        else:
            english_text = ''
        english_label.config(text=english_text)
        
        # Update Chinese display (system audio)
        if current_time - chinese_display["last_update_time"] <= 5:
            chinese_text = f"{chinese_display['pinyin']}\n{chinese_display['translation']}"
        else:
            chinese_text = ''
        chinese_label.config(text=chinese_text)
        
    root.after(100, update_displays)  # Continue updating

def process_text(text, is_english):
    """Process text once to ensure consistency between transcription and translation"""
    if not text.strip():
        return {"transcription": "", "translation": "", "pinyin": ""}
    
    current_time = time.time()
    result = {}
    
    # Clean text for stuttering
    words = text.split()
    cleaned_words = []
    prev_word = None
    repeat_count = 0
    
    for word in words:
        if word == prev_word:
            repeat_count += 1
            if repeat_count < 2:  # Allow max 2 repeats
                cleaned_words.append(word)
        else:
            repeat_count = 0
            cleaned_words.append(word)
        prev_word = word
    
    cleaned_text = ' '.join(cleaned_words)
    result["transcription"] = cleaned_text
    
    # Process based on language
    if is_english:
        # For English input
        if len(cleaned_text) > 5:  # Only translate substantial text
            # Translate English to Chinese
            chinese_translation = translate_english_to_chinese(cleaned_text)
            # Generate pinyin for the Chinese translation
            pinyin_result = generate_pinyin(chinese_translation)
            
            result["translation"] = chinese_translation
            result["pinyin"] = pinyin_result
        else:
            result["translation"] = ""
            result["pinyin"] = ""
    else:
        # For Chinese input
        if len(cleaned_text) > 2:  # Only translate substantial text
            # Generate pinyin for Chinese text
            pinyin_result = generate_pinyin(cleaned_text)
            # Translate Chinese to English
            english_translation = translate_chinese_to_english(cleaned_text)
            
            result["pinyin"] = pinyin_result
            result["translation"] = english_translation
        else:
            result["pinyin"] = generate_pinyin(cleaned_text)
            result["translation"] = ""
    
    return result

def capture_english_audio():
    global english_display
    recognizer = KaldiRecognizer(english_model, SAMPLE_RATE)
    
    # For debouncing and preventing stuttering
    last_partial_text = ""
    last_update_time = 0
    debounce_delay = 0.4  # seconds
    
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
                        current_time = time.time()
                        
                        if recognizer.AcceptWaveform(audio_bytes):
                            # This is a complete phrase/sentence
                            result = json.loads(recognizer.Result())
                            text = result.get('text', '')
                            if text:
                                # Process text once to ensure consistency
                                processed = process_text(text, is_english=True)
                                
                                with lock:
                                    english_display = {
                                        "transcription": processed["transcription"],
                                        "translation": processed["pinyin"],  # Chinese text goes on bottom
                                        "pinyin": "",  # Not used for English input
                                        "last_update_time": current_time
                                    }
                                    
                                # Reset the partial text tracking
                                last_partial_text = ""
                        else:
                            # This is a partial result
                            partial = json.loads(recognizer.PartialResult())
                            text = partial.get('partial', '')
                            
                            # Only process if text is different from last partial and not empty
                            if text and (text != last_partial_text) and (current_time - last_update_time > debounce_delay):
                                # Process text once to ensure consistency
                                processed = process_text(text, is_english=True)
                                
                                with lock:
                                    english_display = {
                                        "transcription": processed["transcription"],
                                        "translation": processed["pinyin"],  # Chinese text goes on bottom
                                        "pinyin": "",  # Not used for English input
                                        "last_update_time": current_time
                                    }
                                
                                # Update tracking variables
                                last_partial_text = text
                                last_update_time = current_time
                                    
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
    global chinese_display
    recognizer = KaldiRecognizer(chinese_model, SAMPLE_RATE)
    
    # For debouncing and preventing stuttering
    last_partial_text = ""
    last_update_time = 0
    debounce_delay = 0.4  # seconds
    
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
                        current_time = time.time()
                        
                        if recognizer.AcceptWaveform(audio_bytes):
                            # This is a complete phrase/sentence
                            result = json.loads(recognizer.Result())
                            text = result.get('text', '')
                            if text:
                                # Process text once to ensure consistency
                                processed = process_text(text, is_english=False)
                                
                                with lock:
                                    chinese_display = {
                                        "transcription": processed["transcription"],
                                        "translation": processed["translation"],
                                        "pinyin": processed["pinyin"],
                                        "last_update_time": current_time
                                    }
                                    
                                # Reset the partial text tracking
                                last_partial_text = ""
                        else:
                            # This is a partial result
                            partial = json.loads(recognizer.PartialResult())
                            text = partial.get('partial', '')
                            
                            # Only process if text is different from last partial and not empty
                            if text and (text != last_partial_text) and (current_time - last_update_time > debounce_delay):
                                # Process text once to ensure consistency
                                processed = process_text(text, is_english=False)
                                
                                with lock:
                                    chinese_display = {
                                        "transcription": processed["transcription"],
                                        "translation": processed["translation"],
                                        "pinyin": processed["pinyin"],
                                        "last_update_time": current_time
                                    }
                                
                                # Update tracking variables
                                last_partial_text = text
                                last_update_time = current_time
                                    
                    except Exception as e:
                        print(f"Chinese processing error: {str(e)}")
                        time.sleep(0.1)  # Brief pause before continuing
                        continue
        except Exception as e:
            print(f"Chinese recorder error: {str(e)}")
            time.sleep(1)  # Wait a bit before trying to reconnect
            # Reset recognizer in case it's corrupted
            recognizer = KaldiRecognizer(chinese_model, SAMPLE_RATE)

# Create a transparent overlay window (special implementation)
class ClickThroughWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Configure window properties
        self.attributes('-alpha', WINDOW_OPACITY)
        self.attributes('-topmost', True)
        self.overrideredirect(True)
        screen_width = self.winfo_screenwidth()
        self.geometry(f'{screen_width}x200+0+0')  # Full width at top of screen
        self.configure(bg='black')
        self.wm_attributes("-transparent", "black")
        
        # This flag is needed for Windows to properly handle the window
        self.wm_attributes("-toolwindow", True)
        
        # Create layout
        self.setup_ui(screen_width)
        
        # Set up Windows-specific transparency behaviors
        self.after(10, self.setup_window_transparency)

    def setup_ui(self, screen_width):
        # Create two frames side by side
        self.left_frame = tk.Frame(self, bg='black')
        self.left_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        self.right_frame = tk.Frame(self, bg='black')
        self.right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        # Create labels
        self.english_label = tk.Label(
            self.left_frame,
            text='',
            font=DISPLAY_FONT,
            fg=TEXT_COLOR,
            bg='black',
            wraplength=screen_width//2-30,
            justify='left'
        )
        self.english_label.pack(expand=True, fill='both')
        
        self.chinese_label = tk.Label(
            self.right_frame,
            text='',
            font=DISPLAY_FONT,
            fg=TEXT_COLOR,
            bg='black',
            wraplength=screen_width//2-15,
            justify='right'
        )
        self.chinese_label.pack(expand=True, fill='both')
        
        # Add keyboard shortcut to exit (Esc key)
        self.bind('<Escape>', lambda e: self.destroy())
        self.protocol("WM_DELETE_WINDOW", lambda: self.destroy())

    def setup_window_transparency(self):
        """Use Windows-specific API to properly set click-through transparency"""
        try:
            # Get the window handle
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            
            # Set the window to layered and transparent (this is the key for true click-through)
            ex_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                hwnd, 
                GWL_EXSTYLE, 
                ex_style | WS_EX_LAYERED | WS_EX_TRANSPARENT
            )
            
            # Apply transparency color key
            ctypes.windll.user32.SetLayeredWindowAttributes(
                hwnd,
                0,  # RGB black 
                int(WINDOW_OPACITY * 255),  # Alpha value
                LWA_ALPHA | LWA_COLORKEY  # Using both alpha blending and color key
            )
            
            # Create a subclass for all child windows
            def subclass_func(hwnd, msg, wparam, lparam, uid, data):
                return ctypes.windll.user32.DefWindowProcW(hwnd, msg, wparam, lparam)
            
            # Find all child windows recursively and make them non-interactive
            def make_all_children_noninteractive(parent_widget):
                for child in parent_widget.winfo_children():
                    # Try to get hwnd for this child
                    if hasattr(child, 'winfo_id'):
                        child_hwnd = child.winfo_id() 
                        if child_hwnd:
                            # Get current style
                            child_style = ctypes.windll.user32.GetWindowLongW(child_hwnd, GWL_EXSTYLE)
                            # Set transparent and layered
                            ctypes.windll.user32.SetWindowLongW(
                                child_hwnd, 
                                GWL_EXSTYLE, 
                                child_style | WS_EX_TRANSPARENT | WS_EX_LAYERED
                            )
                            
                            # Also apply transparency attributes
                            ctypes.windll.user32.SetLayeredWindowAttributes(
                                child_hwnd, 
                                0,  # RGB black
                                int(WINDOW_OPACITY * 255),  # Alpha
                                LWA_ALPHA | LWA_COLORKEY  # Both modes
                            )
                    
                    # Recursively process its children
                    make_all_children_noninteractive(child)
            
            # Apply to all children
            make_all_children_noninteractive(self)
            
        except Exception as e:
            print(f"Error setting up transparency: {e}")

# Start capture threads
english_thread = threading.Thread(target=capture_english_audio, daemon=True)
english_thread.start()

chinese_thread = threading.Thread(target=capture_chinese_audio, daemon=True)
chinese_thread.start()

# Create the main window using our custom class
root = ClickThroughWindow()
english_label = root.english_label
chinese_label = root.chinese_label

# Start display updates
root.after(100, update_displays)

# Run the main window
try:
    root.mainloop()
except KeyboardInterrupt:
    root.destroy()
    print("Exiting...")
except Exception as e:
    print(f"Application error: {str(e)}")
    root.destroy()