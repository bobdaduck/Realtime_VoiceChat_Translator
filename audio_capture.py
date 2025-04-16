import soundcard as sc
from soundcard import SoundcardRuntimeWarning
from vosk import Model, KaldiRecognizer
import numpy as np
import json
import threading
import time
import pypinyin
import warnings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Filter out the specific warning
warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning, 
                       message="data discontinuity in recording")

# Configuration
ENGLISH_MODEL_PATH = "model"
CHINESE_MODEL_PATH = "chinese-model"
SAMPLE_RATE = 16000
BUFFER_SIZE = 1024  # Smaller buffer size may help with discontinuities

# Shared data and locks
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

# Initialize audio devices
def initialize_audio_devices():
    speaker = sc.default_speaker()
    loopback_mic = sc.get_microphone(speaker.id, include_loopback=True)
    regular_mic = sc.default_microphone()
    return loopback_mic, regular_mic

# Initialize translation models
def initialize_translation_models():
    print("Downloading and loading translation models... (this may take a few minutes on first run)")
    
    # Load Vosk models
    english_model = Model(ENGLISH_MODEL_PATH)
    chinese_model = Model(CHINESE_MODEL_PATH)
    
    # Load Hugging Face translation models
    zh_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    zh_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    
    en_zh_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    en_zh_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    
    print("Translation models loaded successfully!")
    
    return (english_model, chinese_model, 
            zh_en_tokenizer, zh_en_model, 
            en_zh_tokenizer, en_zh_model)

def translate_chinese_to_english(text, zh_en_tokenizer, zh_en_model):
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

def translate_english_to_chinese(text, en_zh_tokenizer, en_zh_model):
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

def process_text(text, is_english, zh_en_tokenizer, zh_en_model, en_zh_tokenizer, en_zh_model):
    """Process text once to ensure consistency between transcription and translation"""
    if not text.strip():
        return {"transcription": "", "translation": "", "pinyin": ""}
    
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
            chinese_translation = translate_english_to_chinese(cleaned_text, en_zh_tokenizer, en_zh_model)
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
            english_translation = translate_chinese_to_english(cleaned_text, zh_en_tokenizer, zh_en_model)
            
            result["pinyin"] = pinyin_result
            result["translation"] = english_translation
        else:
            result["pinyin"] = generate_pinyin(cleaned_text)
            result["translation"] = ""
    
    return result

def capture_english_audio(english_model, regular_mic, zh_en_tokenizer, zh_en_model, en_zh_tokenizer, en_zh_model):
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
                                processed = process_text(text, is_english=True, 
                                                        zh_en_tokenizer=zh_en_tokenizer, 
                                                        zh_en_model=zh_en_model, 
                                                        en_zh_tokenizer=en_zh_tokenizer, 
                                                        en_zh_model=en_zh_model)
                                
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
                                processed = process_text(text, is_english=True, 
                                                        zh_en_tokenizer=zh_en_tokenizer, 
                                                        zh_en_model=zh_en_model, 
                                                        en_zh_tokenizer=en_zh_tokenizer, 
                                                        en_zh_model=en_zh_model)
                                
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

def capture_chinese_audio(chinese_model, loopback_mic, zh_en_tokenizer, zh_en_model, en_zh_tokenizer, en_zh_model):
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
                                processed = process_text(text, is_english=False, 
                                                        zh_en_tokenizer=zh_en_tokenizer, 
                                                        zh_en_model=zh_en_model, 
                                                        en_zh_tokenizer=en_zh_tokenizer, 
                                                        en_zh_model=en_zh_model)
                                
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
                                processed = process_text(text, is_english=False, 
                                                        zh_en_tokenizer=zh_en_tokenizer, 
                                                        zh_en_model=zh_en_model, 
                                                        en_zh_tokenizer=en_zh_tokenizer, 
                                                        en_zh_model=en_zh_model)
                                
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

def start_audio_threads(english_model, chinese_model, regular_mic, loopback_mic, 
                        zh_en_tokenizer, zh_en_model, en_zh_tokenizer, en_zh_model):
    # Start capture threads
    english_thread = threading.Thread(
        target=capture_english_audio, 
        args=(english_model, regular_mic, zh_en_tokenizer, zh_en_model, en_zh_tokenizer, en_zh_model),
        daemon=True
    )
    english_thread.start()

    chinese_thread = threading.Thread(
        target=capture_chinese_audio, 
        args=(chinese_model, loopback_mic, zh_en_tokenizer, zh_en_model, en_zh_tokenizer, en_zh_model),
        daemon=True
    )
    chinese_thread.start()
    
    return english_thread, chinese_thread

def get_display_data():
    with lock:
        return english_display.copy(), chinese_display.copy()