"""
Main translation orchestrator
Coordinates between AI, dictionary, and web translators with fallback logic
"""
import logging
from .model_init import initialize_transcription_models
from .dictionary_translators import (
    filter_chinese_text,
    remove_pattern_repetitions,
    generate_pinyin,
    dict_translate_zh_to_en,
    safe_execute
)
from .ai_translators import (
    ai_translate_zh_to_en,
    ai_translate_en_to_zh
)
from .web_translators import (
    web_translate_zh_to_en,
    web_translate_en_to_zh,
    enable_web_translation,
    disable_web_translation,
    is_web_translation_enabled
)

# Set up logging
logger = logging.getLogger(__name__)

# Re-export for backward compatibility
SAMPLE_RATE = 16000


def process_text(text, is_english, *args):
    """
    Process text with translation fallback logic:
    
    For Chinese (ZH->EN):
    1. Try AI translation (primary) - not yet implemented
    2. Use CEDICT translation (always shown as fallback)
    3. Use web translation only if enabled
    
    For English (EN->ZH):
    1. Generate pinyin using CEDICT only (no web lookup)
    2. Try AI translation if available
    3. Use web translation only if enabled
    """
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
            # English text processing
            if len(processed_text) > 1:
                # Try AI translation first
                cn_trans = ai_translate_en_to_zh(processed_text)
                
                # Fall back to web translation if enabled and AI failed
                if cn_trans is None and is_web_translation_enabled():
                    cn_trans = web_translate_en_to_zh(processed_text)
                
                # Use translation result or empty string
                cn_trans = cn_trans if cn_trans else ""
                
                result["translation"] = cn_trans
                # Generate pinyin from Chinese translation using CEDICT only
                result["pinyin"] = generate_pinyin(cn_trans) if cn_trans else ""
            else:
                result.update({"translation": "", "pinyin": ""})
                
        else:
            # Chinese text processing
            if len(processed_text) > 1:
                logger.info(f"Chinese characters received: {processed_text}")
                
                # Always generate pinyin using CEDICT
                pin = generate_pinyin(processed_text)
                logger.info(f"Generated pinyin: {pin}")
                
                # Try AI translation first (will return None for now)
                en_trans = ai_translate_zh_to_en(processed_text)
                
                # Fall back to CEDICT translation
                if en_trans is None:
                    en_trans = dict_translate_zh_to_en(processed_text)
                    logger.info(f"Using CEDICT translation: {en_trans}")
                
                # Only try web translation if enabled and previous methods failed/insufficient
                if (en_trans is None or en_trans == "") and is_web_translation_enabled():
                    en_trans = web_translate_zh_to_en(processed_text)
                    logger.info(f"Using web translation: {en_trans}")
                
                result.update({
                    "pinyin": pin,
                    "translation": en_trans if en_trans else "",
                    "dict_translation": dict_translate_zh_to_en(processed_text)  # Always provide CEDICT fallback
                })
            else:
                result["pinyin"] = generate_pinyin(processed_text)
                result["translation"] = ""
                result["dict_translation"] = ""
                
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
        
        result = {
            "transcription": full_processing.get("transcription", ""),
            "translation": full_processing.get("translation", ""),
            "pinyin": combined_pinyin  # Use combined individual pinyin values
        }
        
        # Include dict_translation if available
        if "dict_translation" in full_processing:
            result["dict_translation"] = full_processing["dict_translation"]
        
        return result
    
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


# Re-export functions for backward compatibility and easy access
__all__ = [
    'initialize_transcription_models',
    'process_text',
    'process_chinese_segments',
    'process_individual_segment',
    'enable_web_translation',
    'disable_web_translation',
    'is_web_translation_enabled',
    'SAMPLE_RATE'
]