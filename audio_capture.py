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

# Filter out specific warnings and progres bars
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
                        # ap.play_after_delay(audio_int16, SAMPLE_RATE, delay=2.0)
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
    Capture Chinese audio and process it with funASR's Paraformer model using a timer-based approach
    Note: *args is used to maintain compatibility with the original function signature
    """
    global chinese_display, chinese_text_segments, chinese_last_processed_time
    
    # Configuration for timer-based processing
    PROCESS_INTERVAL = 3.0  # Process every 3 seconds of audio
    
    # Buffer for collecting audio
    audio_buffer = []
    last_process_time = time.time()
    
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

                        # Apply existing audio preprocessing
                        processed_audio, _ = ap.preprocess_buffer(audio_data)
                        ###################only for debugging
                        # audio_int16 = (processed_audio * 32767).astype(np.int16)
                        # ap.play_after_delay(audio_int16, SAMPLE_RATE, delay=2.0)
                        #########################debugging done

                        current_time = time.time()
                        
                        # Add the processed audio to our buffer
                        audio_buffer.append(processed_audio)
                        
                        # Check if it's time to process the buffer
                        if current_time - last_process_time >= PROCESS_INTERVAL:
                            # Process the collected audio buffer
                            if audio_buffer:
                                logger.info("-------------- PROCESSING CHINESE AUDIO --------------")
                                
                                try:
                                    # Combine all audio chunks
                                    combined_audio = np.concatenate(audio_buffer)
                                    
                                    # IMPORTANT: Ensure the audio is 1D for funASR
                                    # This is specific to fixing the dimension issue
                                    if len(combined_audio.shape) > 1:
                                        # Extract first channel if multi-dimensional
                                        combined_audio = combined_audio[:, 0]
                                    
                                    # Make sure audio is float32 in [-1.0, 1.0] range
                                    if combined_audio.dtype != np.float32:
                                        combined_audio = combined_audio.astype(np.float32)
                                    
                                    if np.max(np.abs(combined_audio)) > 1.0:
                                        combined_audio = combined_audio / np.max(np.abs(combined_audio))
                                    
                                    # Instead of converting to int16, keep the audio as float32 in [-1.0, 1.0]
                                    audio_for_model = combined_audio  # This remains in float32

                                    logger.info(f"Audio buffer size: {len(audio_buffer)}, Combined shape: {audio_for_model.shape}")

                                    # Process with funASR model
                                    try:
                                        # Make sure the audio is not empty and has valid data
                                        if audio_for_model.size > 0 and np.any(audio_for_model != 0):
                                                            
                                            # The generate method expects waveform data in float32 format
                                            result = chinese_model.generate(audio_for_model)
                                            # logger.warn(f"funASR generate result: {result}")
                                            # print(f"funASR generate result: {result}")

                                            # Extract the text from the result (check exact structure)
                                            if result and len(result) > 0 and len(result[0]) > 0:
                                                
                                                text = result[0].get("text", "")
                                                
                                                if text and text.strip():
                                                    logger.info(f"Chinese text recognized: {text}")
                                                    
                                                    # Add the completed phrase to our rolling window
                                                    with lock:
                                                        chinese_text_segments.append(text)
                                                        
                                                        # Process the combined text
                                                        full_text = ' '.join(chinese_text_segments)
                                                        processed = model_work.process_text(full_text, is_english=False)
                                                        
                                                        chinese_display = {
                                                            "transcription": full_text,
                                                            "translation": processed["translation"],
                                                            "pinyin": processed["pinyin"],
                                                            "last_update_time": current_time
                                                        }
                                                        
                                                        chinese_last_processed_time = current_time
                                            else:
                                                logger.info("No speech detected in this audio segment")
                                        else:
                                            logger.info("Audio buffer contains no valid data")
                                    except Exception as e:
                                        logger.error(f"Error in funASR model processing: {str(e)}")
                                        # Print more detailed error info
                                        import traceback
                                        logger.error(traceback.format_exc())
                                except Exception as e:
                                    logger.error(f"Error preparing audio data: {str(e)}")
                                    import traceback
                                    logger.error(traceback.format_exc())
                                
                                # Reset buffer and timer for next interval
                                audio_buffer = []
                                last_process_time = current_time
                                
                                logger.info("------------------------------------------------------")
                                    
                    except Exception as e:
                        print(f"Chinese processing error: {str(e)}")
                        time.sleep(0.01)  # Brief pause before continuing
                        continue
        except Exception as e:
            print(f"Chinese recorder error: {str(e)}")
            time.sleep(1)  # Wait a bit before trying to reconnect

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