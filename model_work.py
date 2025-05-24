import pypinyin
from vosk import Model as VoskModel
import logging
import translators as ts
import funasr
import numpy as np
import os
import re
from itertools import groupby

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
    "helhUAU","HUANERENE", "HUANYEREOY", "igrait"
]

def initialize_translation_models():
    """Initialize and load all models needed for translation"""
    print("Loading translation models...")
    
    # Load Vosk model for English speech recognition
    english_model = VoskModel(ENGLISH_MODEL_PATH)
    
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
        print("funASR model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading funASR model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    print("Translation models loaded successfully!")
    
    return (english_model, chinese_model, None, None, None, None)

def remove_consecutive_duplicates(text):
    """Remove consecutive duplicated characters or words"""
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

def remove_pattern_repetitions(text):
    """
    Enhanced detection and removal of repetitive patterns in text.
    Handles both character-level and word-level patterns.
    """
    if not text or len(text) <= 3:  # Need at least some characters to detect patterns
        return text
    
    # First remove consecutive duplicates (e.g., "有 有 有" -> "有")
    text = remove_consecutive_duplicates(text)
    
    # For Chinese text, we might need to handle character-level patterns
    is_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
    
    # Split into tokens (either words or characters)
    if is_chinese and ' ' not in text:
        # Process as characters if it's Chinese without spaces
        tokens = list(text)
    else:
        # Process as words
        tokens = text.split()
    
    if len(tokens) <= 3:  # Not enough tokens to find meaningful patterns
        return text
    
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
                if is_chinese and ' ' not in text:
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
            if is_chinese and ' ' not in text:
                return ''.join(pattern)
            else:
                return ' '.join(pattern)
    
    return text

def filter_chinese_text(text):
    """
    Filter out common hallucinated words from Chinese text
    """
    if not text:
        return text
        
    # Create a pattern to match any of the filter words
    pattern = '|'.join(re.escape(word) for word in CHINESE_FILTER_WORDS)
    
    # Replace all occurrences with an empty string
    filtered_text = re.sub(pattern, '', text)
    
    # Remove extra spaces that might be created
    filtered_text = re.sub(r'\s+', ' ', filtered_text).strip()
    
    return filtered_text

def translate_chinese_to_english(text):
    """Translate Chinese text to English """
    try:
        # Apply filter before translation
        filtered_text = filter_chinese_text(text)
        
        # Only translate if there's still text after filtering
        if not filtered_text:
            return ""
            
        logger.info(f"Sending for translation: {filtered_text}")
        translation = ts.translate_text(filtered_text, translator='caiyun', from_language='zh', to_language='en')
        logger.info(f"Translation returned: {translation}")
        return translation
    except Exception as e:
        logger.error(f"Failure in translator service attempting to send {text}: {str(e)}")
        return ""

def translate_english_to_chinese(text):
    """Translate English text to Chinese """
    try:
        return ts.translate_text(text, translator='caiyun', from_language='en', to_language='zh')
    except Exception as e:
        logger.error(f"Failure in translator service attempting to send {text}: {str(e)}")
        return ""

def generate_pinyin(chinese_text):
    """Generate pinyin with tone marks for Chinese text"""
    if chinese_text:
        if not chinese_text.strip():
            return ""
        try:
            # Apply filter before generating pinyin
            filtered_text = filter_chinese_text(chinese_text)
            
            # Only generate pinyin if there's still text after filtering
            if not filtered_text:
                return ""
                
            return ' '.join(pypinyin.lazy_pinyin( #adds tone marks
                filtered_text, 
                style=pypinyin.Style.TONE
            ))
        except Exception as e:
            print(f"Pinyin generation error: {str(e)}")
            return "[Pinyin Error]"
    return ""

def process_text(text, is_english, *args):
    """Process text once to ensure consistency between transcription and translation"""
    if not text or not text.strip():
        return {"transcription": "", "translation": "", "pinyin": ""}
    
    result = {}

    # Apply filtering for Chinese text
    if not is_english:
        text = filter_chinese_text(text)
        text = remove_pattern_repetitions(text)

    # If after filtering there's no text left, return empty results
    if not text or not text.strip():
        return {"transcription": "", "translation": "", "pinyin": ""}
        
    result["transcription"] = text
    
    if is_english:
        if len(text) > 1:
            cn_trans = translate_english_to_chinese(text)
            result["translation"] = cn_trans
            result["pinyin"] = generate_pinyin(cn_trans)
        else:
            result.update({"translation": "", "pinyin": ""})
    else:
        if len(text) > 1:
            logger.info(f"Chinese characters received: {text}")
            pin = generate_pinyin(text)
            logger.info(f"Generated pinyin: {pin}")
            en_trans = translate_chinese_to_english(text)
            result.update({"pinyin": pin, "translation": en_trans})
        else:
            result["pinyin"] = generate_pinyin(text)
            result["translation"] = ""
    return result

def process_chinese_segments(segments):
    """
    Process a list of Chinese text segments as a whole to ensure consistent translation
    but preserve individual segment pinyin
    """
    if not segments:
        return {"transcription": "", "translation": "", "pinyin": ""}
    
    # Combine all segment transcriptions for processing translation as a batch
    combined_text = ''.join([seg["transcription"] for seg in segments])
    
    # Process the combined text for translation only
    full_processing = process_text(combined_text, is_english=False)
    
    # Instead of using the pinyin from full processing, combine individual segment pinyin
    combined_pinyin = ' '.join([seg["pinyin"] for seg in segments if seg["pinyin"]])
    
    return {
        "transcription": full_processing["transcription"],
        "translation": full_processing["translation"],
        "pinyin": combined_pinyin  # Use combined individual pinyin values
    }

def process_individual_segment(text, is_english=False):
    """
    Process a single segment of text
    """
    return process_text(text, is_english)