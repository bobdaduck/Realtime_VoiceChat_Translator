import soundcard as sc
from soundcard import SoundcardRuntimeWarning
from vosk import KaldiRecognizer
import numpy as np
import json
import threading
import time
import warnings
import translation_framework.model_work as model_work
from translation_framework.model_init import get_cedict
from translation_framework.dictionary_translators import filter_chinese_text, remove_pattern_repetitions
import logging
from collections import deque
import audio_preprocessing as ap
import jieba
import pypinyin

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Filter out specific warnings
warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning, 
                       message="data discontinuity in recording")

# Configuration
SAMPLE_RATE = 16000
BUFFER_SIZE = 1
MAX_CHINESE_WORDS = 20  # Maximum number of Chinese words to keep in rolling window

total_audio = np.array([])

# Shared data and locks
english_display = {
    "transcription": "",
    "translation": "",
    "pinyin": "",
    "last_update_time": 0.0,
}

# Chinese text segments stored as dictionaries in a deque (no maxlen, we manage it manually)
chinese_text_segments = deque()

# Chinese display with cascade_words for word-by-word display
chinese_display = {
    "transcription": "",
    "translation": "",
    "pinyin": "",
    "last_update_time": 0.0,
    "text_segments": chinese_text_segments,
    "cascade_words": [],  # Individual words for cascade display
}

chinese_last_processed_time = 0

lock = threading.Lock()


def count_chinese_words(text):
    """Count the number of Chinese words in text using jieba"""
    if not text or not text.strip():
        return 0
    segments = list(jieba.cut(text))
    return len([seg for seg in segments if seg.strip()])


def get_word_by_word_cedict(text):
    """
    Break down Chinese text into individual words and get CEDICT translation for each.
    Applies filtering BEFORE word breakdown to remove artifacts.
    Returns list of {"pinyin": "...", "cedict": "..."} dictionaries
    """
    if not text or not text.strip():
        return []
    
    # CRITICAL: Apply filters BEFORE word breakdown to remove bad audio artifacts
    filtered_text = filter_chinese_text(text)
    filtered_text = remove_pattern_repetitions(filtered_text)
    
    if not filtered_text or not filtered_text.strip():
        logger.info(f"Text filtered out completely: '{text}'")
        return []
    
    logger.info(f"Filtered text for cascade: '{text}' -> '{filtered_text}'")
    
    cedict = get_cedict()
    if not cedict:
        return []
    
    # Segment the filtered text using jieba
    segments = list(jieba.cut(filtered_text))
    
    word_list = []
    for segment in segments:
        if not segment.strip():
            continue
        
        # Get pinyin for this segment
        pinyin = ' '.join(pypinyin.lazy_pinyin(segment, style=pypinyin.Style.TONE))
        
        # Look up in CEDICT
        cedict_translation = cedict.get(segment, segment)  # Use original if not found
        
        word_list.append({
            "pinyin": pinyin,
            "cedict": cedict_translation
        })
    
    logger.info(f"Created {len(word_list)} cascade words from '{filtered_text}'")
    return word_list


def trim_segments_to_word_limit(segments, max_words):
    """
    Remove oldest segments until total word count <= max_words
    Returns the trimmed list of segments
    """
    if not segments:
        return []
    
    segments_list = list(segments)
    
    # Count total words across all segments
    total_words = sum(count_chinese_words(seg.get("transcription", "")) for seg in segments_list)
    
    logger.info(f"Total words before trim: {total_words}, max allowed: {max_words}")
    
    # Remove oldest segments until we're under the limit
    while total_words > max_words and len(segments_list) > 0:
        removed_segment = segments_list.pop(0)  # Remove oldest (first) segment
        removed_words = count_chinese_words(removed_segment.get("transcription", ""))
        total_words -= removed_words
        logger.info(f"Removed segment with {removed_words} words: '{removed_segment.get('transcription', '')}'. Remaining: {total_words} words")
    
    return segments_list


# Initialize audio devices
def initialize_audio_devices():
    speaker = sc.default_speaker()
    loopback_mic = sc.get_microphone(speaker.id, include_loopback=True)
    regular_mic = sc.default_microphone()
    return loopback_mic, regular_mic


def capture_english_audio(english_model, regular_mic, *args):
    """
    Capture English audio and process it
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
                logger.info("[ENGLISH] Audio recorder started")
                while True:
                    try:
                        audio_data = recorder.record(numframes=SAMPLE_RATE)
                        if audio_data.size == 0:
                            time.sleep(0.01)
                            continue
                        
                        # Apply audio preprocessing
                        processed_audio = ap.process_audio(audio_data, False)
                        
                        audio_int16 = (processed_audio * 32767).astype(np.int16)
                        audio_bytes = audio_int16.tobytes()
                        current_time = time.time()
                        
                        if recognizer.AcceptWaveform(audio_bytes):
                            # Complete phrase/sentence
                            result = json.loads(recognizer.Result())
                            text = result.get('text', '')
                            if text and text.strip():
                                logger.info(f"[ENGLISH FINAL] '{text}'")
                                
                                # Process text 
                                processed = model_work.process_text(text, is_english=True)
                                
                                logger.info(f"[ENGLISH PROCESSED] Trans: '{processed.get('translation', '')}' | Pinyin: '{processed.get('pinyin', '')}'")
                                
                                with lock:
                                    english_display = {
                                        "transcription": processed.get("transcription", ""),
                                        "translation": processed.get("translation", ""), 
                                        "pinyin": processed.get("pinyin", ""), 
                                        "last_update_time": current_time
                                    }
                                
                                last_partial_text = ""
                        else:
                            # Partial result
                            partial = json.loads(recognizer.PartialResult())
                            text = partial.get('partial', '')
                            
                            # Only process if text is different and enough time has passed
                            if text and text.strip() and (text != last_partial_text) and (current_time - last_update_time > debounce_delay):
                                logger.info(f"[ENGLISH PARTIAL] '{text}'")
                                
                                # Process text
                                processed = model_work.process_text(text, is_english=True)
                                
                                with lock:
                                    english_display = {
                                        "transcription": processed.get("transcription", ""),
                                        "translation": processed.get("translation", ""), 
                                        "pinyin": processed.get("pinyin", ""),
                                        "last_update_time": current_time
                                    }
                                
                                last_partial_text = text
                                last_update_time = current_time
                                    
                    except Exception as e:
                        logger.error(f"[ENGLISH] Processing error: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
                        time.sleep(0.1)
                        continue
        except Exception as e:
            logger.error(f"[ENGLISH] Recorder error: {str(e)}")
            time.sleep(1)
            recognizer = KaldiRecognizer(english_model, SAMPLE_RATE)


def capture_chinese_audio(chinese_model, loopback_mic, *args):
    """
    Capture Chinese audio and process it with funASR's Paraformer model
    """
    global chinese_display, chinese_text_segments, chinese_last_processed_time, total_audio 
    
    # Configuration
    PROCESS_INTERVAL = 3.0  # Process every 3 seconds
    
    # Buffer for collecting audio
    audio_buffer = []
    last_process_time = time.time()
    
    while True:
        try:
            with loopback_mic.recorder(samplerate=SAMPLE_RATE, 
                                     blocksize=BUFFER_SIZE, 
                                     channels=1) as recorder:
                logger.info("[CHINESE] Audio recorder started")
                while True:
                    try:
                        audio_data = recorder.record(numframes=SAMPLE_RATE)
                        if audio_data.size == 0:
                            time.sleep(0.001)
                            continue

                        processed_audio = ap.process_audio(audio_data)
                        current_time = time.time()
                        
                        audio_buffer.append(processed_audio)
                        
                        if current_time - last_process_time >= PROCESS_INTERVAL:
                            if audio_buffer:
                                logger.info("-------------- PROCESSING CHINESE AUDIO --------------")
                                
                                try:
                                    combined_audio = np.concatenate(audio_buffer)
                                    
                                    if len(combined_audio.shape) > 1:
                                        combined_audio = combined_audio[:, 0]
                                    
                                    if combined_audio.dtype != np.float32:
                                        combined_audio = combined_audio.astype(np.float32)
                                    
                                    if np.max(np.abs(combined_audio)) > 1.0:
                                        combined_audio = combined_audio / np.max(np.abs(combined_audio))
                                    
                                    audio_for_model = combined_audio

                                    logger.info(f"Audio buffer size: {len(audio_buffer)}, Combined shape: {audio_for_model.shape}")

                                    try:
                                        if audio_for_model.size > 0 and np.any(audio_for_model != 0):
                                            result = chinese_model.generate(audio_for_model)

                                            if result and len(result) > 0 and len(result[0]) > 0:
                                                text = result[0].get("text", "").replace(" ", "")
                                                
                                                if text and text.strip():
                                                    logger.info(f"[CHINESE RAW] '{text}'")
                                                    
                                                    # Process this segment individually
                                                    processed = model_work.process_individual_segment(text, is_english=False)
                                                    
                                                    logger.info(f"[CHINESE PROCESSED] Trans: '{processed.get('translation', '')}' | Pinyin: '{processed.get('pinyin', '')}'")
                                                    
                                                    # Create segment with all needed fields
                                                    segment = {
                                                        "transcription": processed.get("transcription", ""),
                                                        "translation": processed.get("translation", ""),
                                                        "pinyin": processed.get("pinyin", ""),
                                                        "dict_translation": processed.get("dict_translation", ""),
                                                        "audio_data": audio_for_model
                                                    }
                                                    
                                                    with lock:
                                                        # Add new segment
                                                        chinese_text_segments.append(segment)
                                                        
                                                        # Trim segments to word limit
                                                        trimmed_segments = trim_segments_to_word_limit(
                                                            chinese_text_segments, 
                                                            MAX_CHINESE_WORDS
                                                        )
                                                        
                                                        # Update the deque with trimmed segments
                                                        chinese_text_segments.clear()
                                                        chinese_text_segments.extend(trimmed_segments)
                                                        
                                                        # Get word-by-word CEDICT breakdown for ALL current segments
                                                        all_cascade_words = []
                                                        for seg in chinese_text_segments:
                                                            seg_words = get_word_by_word_cedict(seg.get("transcription", ""))
                                                            all_cascade_words.extend(seg_words)
                                                        
                                                        # Process all segments together for combined display
                                                        processed_full = model_work.process_chinese_segments(list(chinese_text_segments))
                                                        
                                                        # Update display with current segments
                                                        chinese_display = {
                                                            "transcription": processed_full.get("transcription", ""),
                                                            "translation": processed_full.get("translation", ""),
                                                            "pinyin": processed_full.get("pinyin", ""),
                                                            "last_update_time": current_time,
                                                            "text_segments": chinese_text_segments,
                                                            "cascade_words": all_cascade_words,
                                                        }
                                                        
                                                        chinese_last_processed_time = current_time
                                                        
                                                        logger.info(f"Current segments ({len(chinese_text_segments)}):")
                                                        for i, seg in enumerate(chinese_text_segments):
                                                            word_count = count_chinese_words(seg['transcription'])
                                                            logger.info(f"  {i}: {seg['transcription']} ({word_count} words) → {seg['pinyin']}")
                                                        
                                                        total_words = sum(count_chinese_words(s.get('transcription', '')) for s in chinese_text_segments)
                                                        logger.info(f"Total words: {total_words}/{MAX_CHINESE_WORDS}")
                                                        logger.info(f"Full text: {processed_full.get('transcription', '')}")
                                                        logger.info(f"Full pinyin: {processed_full.get('pinyin', '')}")
                                                        logger.info(f"Full AI translation: {processed_full.get('translation', '')}")
                                                        logger.info(f"Cascade words: {len(all_cascade_words)} total")
                                            else:
                                                logger.info("No speech detected in this audio segment")
                                        else:
                                            logger.info("Audio buffer contains no valid data")
                                    except Exception as e:
                                        logger.error(f"Error in funASR model processing: {str(e)}")
                                        import traceback
                                        logger.error(traceback.format_exc())
                                except Exception as e:
                                    logger.error(f"Error preparing audio data: {str(e)}")
                                    import traceback
                                    logger.error(traceback.format_exc())
                                
                                audio_buffer = []
                                last_process_time = current_time
                                
                                logger.info("------------------------------------------------------")
                                    
                    except Exception as e:
                        logger.error(f"[CHINESE] Processing error: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
                        time.sleep(0.01)
                        continue
        except Exception as e:
            logger.error(f"[CHINESE] Recorder error: {str(e)}")
            time.sleep(1)


def start_audio_threads(english_model, chinese_model, regular_mic, loopback_mic, *args):
    """Start audio capture threads"""
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
    """Return display data for both English and Chinese"""
    with lock:
        eng_copy = english_display.copy()
        chi_copy = chinese_display.copy()
        # Make sure to copy the cascade_words list separately
        chi_copy['cascade_words'] = list(chinese_display.get('cascade_words', []))
        # Also copy text_segments as a list
        chi_copy['text_segments'] = list(chinese_display.get('text_segments', []))
        return eng_copy, chi_copy


def get_chinese_segments():
    """Get a copy of the current Chinese text segments"""
    with lock:
        return [segment.copy() for segment in chinese_text_segments]


def clear_chinese_segments():
    """Clear all Chinese text segments"""
    with lock:
        chinese_text_segments.clear()
        
        global chinese_display
        chinese_display = {
            "transcription": "",
            "translation": "",
            "pinyin": "",  
            "last_update_time": time.time(),
            "text_segments": chinese_text_segments,
            "cascade_words": [],
        }