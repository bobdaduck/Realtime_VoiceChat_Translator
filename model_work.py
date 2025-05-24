import pypinyin
from vosk import Model as VoskModel
import logging
import translators as ts
import funasr
import numpy as np
import os
import re
from itertools import groupby
import traceback
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Configuration
ENGLISH_MODEL_PATH = "vosk-en"
CHINESE_MODEL_PATH = "paraformer-zh"  # Path to funASR paraformer-zh model
SAMPLE_RATE = 16000

# Audio codex sometimes writes down english and gibberish, which we don't want to send to the translation engine.
CHINESE_FILTER_WORDS = [
    "the", "de", "AUNUAY", "dé", "lo", "UY", "u", "YUYUY", "dock", "ASUAOY"
    "HUANSEEEEEREEE", "py", "you", "be", "and", "一", "UUANNUENE", "AASUAAAAAAAOY",
    "UUUUUU", "没有", "没没", "是是", "有没", "嗯嗯", "们的", "HNENUTE", "HUNONU",
    "helhUAU","HUANERENE", "HUANYEREOY", "igrait", "IUNY"
]

def safe_execute(func, *args, default_return=None, operation_name="Unknown operation", **kwargs):
    """
    Safely execute a function with comprehensive error handling
    Returns default_return if any exception occurs
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error in {operation_name}: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Log the arguments that caused the issue (safely)
        try:
            logger.error(f"Arguments: args={args[:3] if len(args) > 3 else args}, kwargs keys={list(kwargs.keys())}")
        except:
            logger.error("Could not log arguments safely")
        
        return default_return

def initialize_translation_models():
    """Initialize and load all models needed for translation"""
    print("Loading translation models...")
    
    try:
        # Load Vosk model for English speech recognition
        english_model = VoskModel(ENGLISH_MODEL_PATH)
        print("✓ Vosk English model loaded")
    except Exception as e:
        logger.error(f"Failed to load Vosk model: {e}")
        logger.error(traceback.format_exc())
        raise
    
    # Load funASR model for Chinese speech recognition
    # Ensure the directory exists
    os.makedirs(CHINESE_MODEL_PATH, exist_ok=True)
    
    try:
        # Configure funASR with explicit model path and parameters
        chinese_model = funasr.AutoModel(
            model=CHINESE_MODEL_PATH,
            model_revision="v2.0.4",  # Specify the model version
            batch_size=1,
            device="cpu",
            vad_model=None,  # Disable VAD as we're not using it
            punc_model=None,  # Disable punctuation model if not needed
            disable_pbar=True, #disables progress bar spam
            disable_update=True, #Don't redownload every time
            # Model download options
            cache_dir=CHINESE_MODEL_PATH,
            automatic_download=True  # Allow automatic download
        )
        print("✓ funASR model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading funASR model: {e}")
        logger.error(traceback.format_exc())
        raise
    
    print("Translation models loaded successfully!")
    
    return (english_model, chinese_model, None, None, None, None)

def remove_consecutive_duplicates(text):
    """Remove consecutive duplicated characters or words"""
    def _remove_duplicates():
        if not text or len(text) <= 1:
            return text
        
        # For Chinese text (no spaces between characters)
        if ' ' not in text and any('\u4e00' <= char <= '\u9fff' for char in text):
            # Group consecutive identical characters
            chars = [char for char, _ in groupby(text)]
            return ''.join(chars)
        else:
            # For text with spaces (treat as words)
            words = text.split()
            if len(words) <= 1:
                return text
            
            # Group consecutive identical words
            result = []
            for word, _ in groupby(words):
                result.append(word)
            
            return ' '.join(result)
    
    return safe_execute(
        _remove_duplicates,
        default_return=text,
        operation_name="remove_consecutive_duplicates"
    )

def remove_pattern_repetitions(text):
    """
    Enhanced detection and removal of repetitive patterns in text.
    Handles both character-level and word-level patterns.
    """
    def _remove_patterns():
        if not text or len(text) <= 3:  # Need at least some characters to detect patterns
            return text
        
        # First remove consecutive duplicates (e.g., "有 有 有" -> "有")
        processed_text = remove_consecutive_duplicates(text)
        
        # For Chinese text, we might need to handle character-level patterns
        is_chinese = any('\u4e00' <= char <= '\u9fff' for char in processed_text)
        
        # Split into tokens (either words or characters)
        if is_chinese and ' ' not in processed_text:
            # Process as characters if it's Chinese without spaces
            tokens = list(processed_text)
        else:
            # Process as words
            tokens = processed_text.split()
        
        if len(tokens) <= 3:  # Not enough tokens to find meaningful patterns
            return processed_text
        
        # Try different pattern lengths, starting from smaller patterns
        max_pattern_length = min(10, len(tokens) // 2)  # Limit max pattern size
        
        for pattern_length in range(1, max_pattern_length + 1):
            # Skip if too short
            if pattern_length < 1:
                continue
                
            # Look for repeating patterns throughout the text
            patterns_found = False
            
            # Check for patterns that repeat at least twice
            for start_idx in range(len(tokens) - pattern_length * 2 + 1):
                pattern = tokens[start_idx:start_idx + pattern_length]
                
                # Count repetitions of this pattern
                repetition_count = 1
                next_start = start_idx + pattern_length
                
                while next_start + pattern_length <= len(tokens) and tokens[next_start:next_start + pattern_length] == pattern:
                    repetition_count += 1
                    next_start += pattern_length
                
                # If we found significant repetition (3+ times or covering >50% of text)
                if repetition_count >= 3 or (repetition_count * pattern_length > len(tokens) // 2):
                    patterns_found = True
                    pattern_str = ' '.join(pattern) if isinstance(pattern[0], str) else ''.join(pattern)
                    full_str = ' '.join(tokens) if isinstance(tokens[0], str) else ''.join(tokens)
                    
                    # Return just the pattern (if words) or single occurrence of pattern (if chars)
                    if is_chinese and ' ' not in processed_text:
                        return ''.join(pattern)
                    else:
                        return ' '.join(pattern)
            
            if patterns_found:
                break
        
        # Additional check for alternating patterns (like "没 有 没 有 没 有")
        if len(tokens) >= 4:
            # Look for A B A B pattern
            if tokens[0:2] == tokens[2:4] and len(tokens) >= 6 and tokens[0:2] == tokens[4:6]:
                pattern = tokens[0:2]
                pattern_str = ' '.join(pattern) if isinstance(pattern[0], str) else ''.join(pattern)
                full_str = ' '.join(tokens) if isinstance(tokens[0], str) else ''.join(tokens)
                logger.info(f"Found alternating pattern '{pattern_str}' in '{full_str}'")
                
                # Return just one occurrence
                if is_chinese and ' ' not in processed_text:
                    return ''.join(pattern)
                else:
                    return ' '.join(pattern)
        
        return processed_text
    
    return safe_execute(
        _remove_patterns,
        default_return=text,
        operation_name="remove_pattern_repetitions"
    )

def filter_chinese_text(text):
    """
    Filter out common hallucinated words from Chinese text
    """
    def _filter_text():
        if not text:
            return text
            
        # Create a pattern to match any of the filter words
        pattern = '|'.join(re.escape(word) for word in CHINESE_FILTER_WORDS)
        
        # Replace all occurrences with an empty string
        filtered_text = re.sub(pattern, '', text)
        
        # Remove extra spaces that might be created
        filtered_text = re.sub(r'\s+', ' ', filtered_text).strip()
        
        return filtered_text
    
    return safe_execute(
        _filter_text,
        default_return=text,
        operation_name="filter_chinese_text"
    )

def translate_chinese_to_english(text):
    """Translate Chinese text to English """
    def _translate():
        # Apply filter before translation
        filtered_text = filter_chinese_text(text)
        
        # Only translate if there's still text after filtering
        if not filtered_text:
            return ""
            
        logger.info(f"Sending for translation: {filtered_text}")
        
        # Try multiple translation attempts with different strategies
        translation_attempts = [
            lambda: ts.translate_text(filtered_text, translator='caiyun', from_language='zh', to_language='en'),
            lambda: ts.translate_text(filtered_text, translator='google', from_language='zh', to_language='en'),
            lambda: ts.translate_text(filtered_text, translator='bing', from_language='zh', to_language='en'),
        ]
        
        for i, attempt in enumerate(translation_attempts):
            try:
                translation = attempt()
                if translation and translation.strip():
                    logger.info(f"Translation returned (attempt {i+1}): {translation}")
                    return translation
            except Exception as e:
                logger.warning(f"Translation attempt {i+1} failed: {str(e)}")
                continue
        
        logger.error(f"All translation attempts failed for text: {filtered_text}")
        return f"[Translation failed: {filtered_text}]"
    
    return safe_execute(
        _translate,
        default_return="",
        operation_name="translate_chinese_to_english"
    )

def translate_english_to_chinese(text):
    """Translate English text to Chinese """
    def _translate():
        # Try multiple translation attempts
        translation_attempts = [
            lambda: ts.translate_text(text, translator='caiyun', from_language='en', to_language='zh'),
            lambda: ts.translate_text(text, translator='google', from_language='en', to_language='zh'),
            lambda: ts.translate_text(text, translator='bing', from_language='en', to_language='zh'),
        ]
        
        for i, attempt in enumerate(translation_attempts):
            try:
                translation = attempt()
                if translation and translation.strip():
                    return translation
            except Exception as e:
                logger.warning(f"Translation attempt {i+1} failed: {str(e)}")
                continue
        
        logger.error(f"All translation attempts failed for text: {text}")
        return f"[Translation failed: {text}]"
    
    return safe_execute(
        _translate,
        default_return="",
        operation_name="translate_english_to_chinese"
    )

def generate_pinyin(chinese_text):
    """Generate pinyin with tone marks for Chinese text"""
    def _generate_pinyin():
        if not chinese_text or not chinese_text.strip():
            return ""
        
        # Apply filter before generating pinyin
        filtered_text = filter_chinese_text(chinese_text)
        
        # Only generate pinyin if there's still text after filtering
        if not filtered_text:
            return ""
            
        return ' '.join(pypinyin.lazy_pinyin( #adds tone marks
            filtered_text, 
            style=pypinyin.Style.TONE
        ))
    
    return safe_execute(
        _generate_pinyin,
        default_return="[Pinyin Error]",
        operation_name="generate_pinyin"
    )

def process_text(text, is_english, *args):
    """Process text once to ensure consistency between transcription and translation"""
    def _process():
        if not text or not text.strip():
            return {"transcription": "", "translation": "", "pinyin": ""}
        
        result = {}
        processed_text = text

        # Apply filtering for Chinese text
        if not is_english:
            processed_text = filter_chinese_text(processed_text)
            processed_text = remove_pattern_repetitions(processed_text)

        # If after filtering there's no text left, return empty results
        if not processed_text or not processed_text.strip():
            return {"transcription": "", "translation": "", "pinyin": ""}
            
        result["transcription"] = processed_text
        
        if is_english:
            if len(processed_text) > 1:
                cn_trans = translate_english_to_chinese(processed_text)
                result["translation"] = cn_trans
                result["pinyin"] = generate_pinyin(cn_trans)
            else:
                result.update({"translation": "", "pinyin": ""})
        else:
            if len(processed_text) > 1:
                logger.info(f"Chinese characters received: {processed_text}")
                pin = generate_pinyin(processed_text)
                logger.info(f"Generated pinyin: {pin}")
                en_trans = translate_chinese_to_english(processed_text)
                result.update({"pinyin": pin, "translation": en_trans})
            else:
                result["pinyin"] = generate_pinyin(processed_text)
                result["translation"] = ""
        return result
    
    return safe_execute(
        _process,
        default_return={"transcription": "", "translation": "", "pinyin": ""},
        operation_name="process_text"
    )

def process_chinese_segments(segments):
    """
    Process a list of Chinese text segments as a whole to ensure consistent translation
    but preserve individual segment pinyin
    """
    def _process_segments():
        if not segments:
            return {"transcription": "", "translation": "", "pinyin": ""}
        
        # Combine all segment transcriptions for processing translation as a batch
        combined_text = ''.join([seg.get("transcription", "") for seg in segments if seg.get("transcription")])
        
        if not combined_text:
            return {"transcription": "", "translation": "", "pinyin": ""}
        
        # Process the combined text for translation only
        full_processing = process_text(combined_text, is_english=False)
        
        # Instead of using the pinyin from full processing, combine individual segment pinyin
        combined_pinyin = ' '.join([seg.get("pinyin", "") for seg in segments if seg.get("pinyin")])
        
        return {
            "transcription": full_processing.get("transcription", ""),
            "translation": full_processing.get("translation", ""),
            "pinyin": combined_pinyin  # Use combined individual pinyin values
        }
    
    return safe_execute(
        _process_segments,
        default_return={"transcription": "", "translation": "", "pinyin": ""},
        operation_name="process_chinese_segments"
    )

def process_individual_segment(text, is_english=False):
    """
    Process a single segment of text
    """
    return process_text(text, is_english)