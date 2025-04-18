import soundcard as sc
from soundcard import SoundcardRuntimeWarning
from vosk import KaldiRecognizer
import numpy as np
import json
import threading
import time
import warnings
import model_work

# Filter out the specific warning
warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning, 
                       message="data discontinuity in recording")

# Configuration
SAMPLE_RATE = 16000
BUFFER_SIZE = 128     # Smaller buffer size may help with discontinuities
VOLUME_THRESHOLD = 0.01  # Adjust based on testing (0.0-1.0 scale)
SILENCE_DURATION = 0.5   # Time in seconds of silence before stopping processing

# Shared data and locks
english_display = {
"transcription": "",
    "translation": "",
    "pinyin": "",
    "last_update_time": 0.0,
    "start_time": 0.0,
    "accumulated_text": ""
}

chinese_display = {
    "transcription": "",
    "translation": "",
    "pinyin": "",
    "last_update_time": 0.0,
    "start_time": 0.0,
    "accumulated_text": ""
}

lock = threading.Lock()

# Initialize audio devices
def initialize_audio_devices():
    speaker = sc.default_speaker()
    loopback_mic = sc.get_microphone(speaker.id, include_loopback=True)
    regular_mic = sc.default_microphone()
    return loopback_mic, regular_mic

def is_above_volume_threshold(audio_data, threshold=VOLUME_THRESHOLD):
    """Check if audio volume exceeds the threshold"""
    # Calculate RMS (root mean square) as volume indicator
    rms = np.sqrt(np.mean(np.square(audio_data)))
    return rms > threshold

def capture_english_audio(english_model, regular_mic, zh_en_tokenizer, zh_en_model, en_zh_tokenizer, en_zh_model):
    global english_display
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
                                # Process text once to ensure consistency
                                processed = model_work.process_text(
                                    text, is_english=True, 
                                    zh_en_tokenizer=zh_en_tokenizer, 
                                    zh_en_model=zh_en_model, 
                                    en_zh_tokenizer=en_zh_tokenizer, 
                                    en_zh_model=en_zh_model
                                )
                                
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
                                processed = model_work.process_text(
                                    text, is_english=True, 
                                    zh_en_tokenizer=zh_en_tokenizer, 
                                    zh_en_model=zh_en_model, 
                                    en_zh_tokenizer=en_zh_tokenizer, 
                                    en_zh_model=en_zh_model
                                )
                                
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
                                # Process text once to ensure consistency
                                processed = model_work.process_text(
                                    text, is_english=False, 
                                    zh_en_tokenizer=zh_en_tokenizer, 
                                    zh_en_model=zh_en_model, 
                                    en_zh_tokenizer=en_zh_tokenizer, 
                                    en_zh_model=en_zh_model
                                )
                                
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
                                processed = model_work.process_text(
                                    text, is_english=False, 
                                    zh_en_tokenizer=zh_en_tokenizer, 
                                    zh_en_model=zh_en_model, 
                                    en_zh_tokenizer=en_zh_tokenizer, 
                                    en_zh_model=en_zh_model
                                )
                                
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