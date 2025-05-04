import soundcard as sc
from soundcard import SoundcardRuntimeWarning
from vosk import KaldiRecognizer
import numpy as np
import json
import threading
import time
import warnings
import model_work
import logging
from collections import deque
import audio_preprocessing as ap  # Import audio preprocessing module

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Filter out the specific warning
warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning, 
                       message="data discontinuity in recording")

# Configuration
SAMPLE_RATE = 16000
BUFFER_SIZE = 1       # Smaller buffer size may help with discontinuities
VOLUME_THRESHOLD = 0.01  # Adjust based on testing (0.0-1.0 scale)
SILENCE_DURATION = 0.5   # Time in seconds of silence before stopping processing
CHINESE_TEXT_WINDOW = 4  # Number of seconds to keep Chinese text in rolling window

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

# Rolling window for Chinese text segments
chinese_text_segments = deque(maxlen=CHINESE_TEXT_WINDOW)
chinese_last_processed_time = 0

lock = threading.Lock()

# Initialize audio devices
def initialize_audio_devices():
    speaker = sc.default_speaker()
    loopback_mic = sc.get_microphone(speaker.id, include_loopback=True)
    regular_mic = sc.default_microphone()
    return loopback_mic, regular_mic

def capture_english_audio(english_model, regular_mic, *args):
    """
    Capture English audio and process it
    Note: *args is used to maintain compatibility with the original function signature
    """
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
                        
                        # Apply simple filtering to the audio
                        processed_audio, _ = ap.preprocess_buffer(audio_data)
                        
                        audio_int16 = (processed_audio * 32767).astype(np.int16)
                        audio_bytes = audio_int16.tobytes()
                        current_time = time.time()
                        
                        if recognizer.AcceptWaveform(audio_bytes):
                            # This is a complete phrase/sentence
                            result = json.loads(recognizer.Result())
                            text = result.get('text', '')
                            if text:
                                # Process text 
                                processed = model_work.process_text(text, is_english=True)
                                
                                with lock:
                                    english_display = {
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
                                # Process text
                                processed = model_work.process_text(text, is_english=True)
                                
                                with lock:
                                    english_display = {
                                        "transcription": processed["transcription"],
                                        "translation": processed["translation"], 
                                        "pinyin": processed["pinyin"],
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

def capture_chinese_audio(chinese_model, loopback_mic, *args):
    """
    Capture Chinese audio and process it with rolling window approach
    Note: *args is used to maintain compatibility with the original function signature
    """
    global chinese_display, chinese_text_segments, chinese_last_processed_time
    
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
                            time.sleep(0.001)  # Small delay to reduce CPU usage
                            continue

                        # Apply simple filtering to the audio
                        processed_audio, _ = ap.preprocess_buffer(audio_data)
                        
                        # Convert to int16 for Vosk
                        audio_int16 = (processed_audio * 32767).astype(np.int16)
                        audio_bytes = audio_int16.tobytes()
                        current_time = time.time()
                        
                        # Let Vosk process audio in its natural way
                        if recognizer.AcceptWaveform(audio_bytes):
                            # This is a complete phrase/sentence determined by Vosk
                            result = json.loads(recognizer.Result())
                            text = result.get('text', '')
                            
                            if text and text.strip():
                                logger.info("-------------- COMPLETE CHINESE PHRASE --------------")
                                logger.info(f"Chinese text recognized: {text}")
                                
                                # Add the completed phrase to our rolling window
                                with lock:
                                    chinese_text_segments.append(text)
                                    
                                    # Check if we need to process the combined text
                                    full_text = ' '.join(chinese_text_segments)
                                    processed = model_work.process_text(full_text, is_english=False)
                                    
                                    chinese_display = {
                                        "transcription": full_text,
                                        "translation": processed["translation"],
                                        "pinyin": processed["pinyin"],
                                        "last_update_time": current_time
                                    }
                                    
                                    chinese_last_processed_time = current_time
                                
                                logger.info(f"Window size: {len(chinese_text_segments)} segments")
                                logger.info(f"Combined text: {full_text}")
                                logger.info(f"Translation: {processed['translation']}")
                                logger.info("------------------------------------------------------")
                        else:
                            # This is a partial result - we don't need to do anything with it
                            # Vosk will eventually produce a complete result when appropriate
                            pass
                                    
                    except Exception as e:
                        print(f"Chinese processing error: {str(e)}")
                        time.sleep(0.01)  # Brief pause before continuing
                        continue
        except Exception as e:
            print(f"Chinese recorder error: {str(e)}")
            time.sleep(1)  # Wait a bit before trying to reconnect
            # Reset recognizer in case it's corrupted
            recognizer = KaldiRecognizer(chinese_model, SAMPLE_RATE)

def start_audio_threads(english_model, chinese_model, regular_mic, loopback_mic, *args):
    """
    Start audio capture threads
    Note: *args is used to maintain compatibility with the original function signature
    """
    # Start capture threads
    english_thread = threading.Thread(
        target=capture_english_audio, 
        args=(english_model, regular_mic),
        daemon=True
    )
    english_thread.start()

    chinese_thread = threading.Thread(
        target=capture_chinese_audio, 
        args=(chinese_model, loopback_mic),
        daemon=True
    )
    chinese_thread.start()
    
    return english_thread, chinese_thread

def get_display_data():
    with lock:
        return english_display.copy(), chinese_display.copy()