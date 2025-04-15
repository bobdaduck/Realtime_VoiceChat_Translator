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
        if current_time - last_english_text_time <= 15:
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
    
    # For debouncing and preventing stuttering
    last_partial_text = ""
    last_update_time = 0
    debounce_delay = 0.2  # seconds
    
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
                                    last_english_text_time = current_time
                                    
                                # Reset the partial text tracking
                                last_partial_text = ""
                        else:
                            # This is a partial result
                            partial = json.loads(recognizer.PartialResult())
                            text = partial.get('partial', '')
                            
                            # Only process if text is different from last partial and not empty
                            if text and (text != last_partial_text) and (current_time - last_update_time > debounce_delay):
                                # Check for stuttering pattern (simplified approach)
                                words = text.split()
                                
                                # Simple stutter detection - if more than 3 identical words in a row
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
                                
                                # Use the cleaned text
                                cleaned_text = ' '.join(cleaned_words)
                                
                                # Optional: Only translate substantial text
                                if len(cleaned_text) > 5:  # Only translate substantial text
                                    # Translate English to Chinese
                                    chinese_translation = translate_english_to_chinese(cleaned_text)
                                    
                                    # Generate pinyin for the Chinese translation
                                    pinyin_result = ' '.join(pypinyin.lazy_pinyin(
                                        chinese_translation, 
                                        style=pypinyin.Style.TONE  # Add tone markers
                                    ))
                                    
                                    with lock:
                                        english_transcriptions = [cleaned_text]
                                        english_translations = [chinese_translation]
                                        english_pinyin = [pinyin_result]
                                        last_english_text_time = current_time
                                else:
                                    with lock:
                                        english_transcriptions = [cleaned_text]
                                        last_english_text_time = current_time
                                
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
    global chinese_transcriptions, chinese_pinyin, chinese_translations, last_chinese_text_time
    recognizer = KaldiRecognizer(chinese_model, SAMPLE_RATE)
    
    # For debouncing and preventing stuttering
    last_partial_text = ""
    last_update_time = 0
    debounce_delay = 0.2  # seconds
    
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
                                    last_chinese_text_time = current_time
                                    
                                # Reset the partial text tracking
                                last_partial_text = ""
                        else:
                            # This is a partial result
                            partial = json.loads(recognizer.PartialResult())
                            text = partial.get('partial', '')
                            
                            # Only process if text is different from last partial and not empty
                            if text and (text != last_partial_text) and (current_time - last_update_time > debounce_delay):
                                # Check for stuttering pattern (simplified approach)
                                words = text.split()
                                
                                # Simple stutter detection - if more than 3 identical words in a row
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
                                
                                # Use the cleaned text
                                cleaned_text = ' '.join(cleaned_words)
                                
                                # Use tone marks in pinyin conversion
                                pinyin_result = ' '.join(pypinyin.lazy_pinyin(
                                    cleaned_text, 
                                    style=pypinyin.Style.TONE  # Add tone markers
                                ))
                                
                                # Only translate substantial text
                                if len(cleaned_text) > 5:
                                    # Translate Chinese to English
                                    translation = translate_chinese_to_english(cleaned_text)
                                    
                                    with lock:
                                        chinese_transcriptions = [cleaned_text]
                                        chinese_pinyin = [pinyin_result]
                                        chinese_translations = [translation]
                                        last_chinese_text_time = current_time
                                else:
                                    with lock:
                                        chinese_transcriptions = [cleaned_text]
                                        chinese_pinyin = [pinyin_result]
                                        last_chinese_text_time = current_time
                                
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