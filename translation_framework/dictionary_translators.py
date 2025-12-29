"""
Dictionary-based translation using CC-CEDICT
Includes text filtering, pinyin generation, and CEDICT lookup
"""
import re
import logging
import traceback
from itertools import groupby
import jieba
import pypinyin

from .model_init import get_cedict

# Set up logging
logger = logging.getLogger(__name__)

# Filter words for Chinese text
CHINESE_FILTER_WORDS = [
    "the", "de", "AUNUAY", "dé", "lo", "UY", "u", "YUYUY", "dock", "ASUAOY",
    "HUANSEEEEEREEE", "py", "you", "be", "and", "一", "UUANNUENE", "AASUAAAAAAAOY",
    "UUUUUU", "没有", "没没", "是是", "有没", "嗯嗯", "们的", "HNENUTE", "HUNONU",
    "helhUAU", "HUANERENE", "HUANYEREOY", "igrait", "IUNY", "IANEATU"
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
        
        try:
            logger.error(f"Arguments: args={args[:3] if len(args) > 3 else args}, kwargs keys={list(kwargs.keys())}")
        except:
            logger.error("Could not log arguments safely")
        
        return default_return


def filter_chinese_text(text):
    """Filter out common hallucinated words from Chinese text"""
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
        if not text or len(text) <= 3:
            return text
        
        # First remove consecutive duplicates
        processed_text = remove_consecutive_duplicates(text)
        
        # For Chinese text, we might need to handle character-level patterns
        is_chinese = any('\u4e00' <= char <= '\u9fff' for char in processed_text)
        
        # Split into tokens (either words or characters)
        if is_chinese and ' ' not in processed_text:
            tokens = list(processed_text)
        else:
            tokens = processed_text.split()
        
        if len(tokens) <= 3:
            return processed_text
        
        # Try different pattern lengths, starting from smaller patterns
        max_pattern_length = min(10, len(tokens) // 2)
        
        for pattern_length in range(1, max_pattern_length + 1):
            if pattern_length < 1:
                continue
                
            patterns_found = False
            
            for start_idx in range(len(tokens) - pattern_length * 2 + 1):
                pattern = tokens[start_idx:start_idx + pattern_length]
                
                # Count repetitions of this pattern
                repetition_count = 1
                next_start = start_idx + pattern_length
                
                while next_start + pattern_length <= len(tokens) and tokens[next_start:next_start + pattern_length] == pattern:
                    repetition_count += 1
                    next_start += pattern_length
                
                # If we found significant repetition
                if repetition_count >= 3 or (repetition_count * pattern_length > len(tokens) // 2):
                    patterns_found = True
                    
                    if is_chinese and ' ' not in processed_text:
                        return ''.join(pattern)
                    else:
                        return ' '.join(pattern)
            
            if patterns_found:
                break
        
        # Check for alternating patterns (like "没 有 没 有 没 有")
        if len(tokens) >= 4:
            if tokens[0:2] == tokens[2:4] and len(tokens) >= 6 and tokens[0:2] == tokens[4:6]:
                pattern = tokens[0:2]
                pattern_str = ' '.join(pattern) if isinstance(pattern[0], str) else ''.join(pattern)
                full_str = ' '.join(tokens) if isinstance(tokens[0], str) else ''.join(tokens)
                logger.info(f"Found alternating pattern '{pattern_str}' in '{full_str}'")
                
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


def generate_pinyin(chinese_text):
    """Generate pinyin with tone marks for Chinese text"""
    def _generate_pinyin():
        if not chinese_text or not chinese_text.strip():
            return ""
        
        # Apply filter before generating pinyin
        filtered_text = filter_chinese_text(chinese_text)
        
        if not filtered_text:
            return ""
            
        return ' '.join(pypinyin.lazy_pinyin(
            filtered_text, 
            style=pypinyin.Style.TONE
        ))
    
    return safe_execute(
        _generate_pinyin,
        default_return="[Pinyin Error]",
        operation_name="generate_pinyin"
    )


def dict_translate_zh_to_en(text):
    """Translate Chinese text to English using jieba and CC-CEDICT"""
    def _translate():
        # Apply filter before translation
        filtered_text = filter_chinese_text(text)
        
        if not filtered_text:
            return ""
        
        logger.info(f"Dictionary translating: {filtered_text}")
        
        cedict = get_cedict()
        
        if not cedict:
            logger.warning("CEDICT not loaded, returning original text")
            return filtered_text
        
        # Segment the text using jieba
        segments = jieba.cut(filtered_text)
        
        # Look up each segment in dictionary
        translations = []
        for segment in segments:
            if segment in cedict:
                translations.append(cedict[segment])
            else:
                # Keep original if no translation found
                translations.append(segment)
        
        result = ' '.join(translations)
        logger.info(f"Dictionary translation: {result}")
        return result
    
    return safe_execute(
        _translate,
        default_return="",
        operation_name="dict_translate_zh_to_en"
    )