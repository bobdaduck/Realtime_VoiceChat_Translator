"""
Web-based translation using third-party APIs
Only called on command to save tokens
"""
import logging
import translators as ts
from .dictionary_translators import filter_chinese_text

# Set up logging
logger = logging.getLogger(__name__)

# Configuration - toggleable at runtime
WEB_TRANSLATION_ENABLED = False


def enable_web_translation():
    """Enable web translation APIs"""
    global WEB_TRANSLATION_ENABLED
    WEB_TRANSLATION_ENABLED = True
    logger.info("Web translation enabled")


def disable_web_translation():
    """Disable web translation APIs"""
    global WEB_TRANSLATION_ENABLED
    WEB_TRANSLATION_ENABLED = False
    logger.info("Web translation disabled")


def is_web_translation_enabled():
    """Check if web translation is enabled"""
    return WEB_TRANSLATION_ENABLED


def web_translate_zh_to_en(text):
    """
    Translate Chinese text to English using third-party APIs
    Only runs if web translation is enabled
    """
    if not WEB_TRANSLATION_ENABLED:
        logger.debug("Web translation disabled, skipping")
        return None
    
    # Apply filter before translation
    filtered_text = filter_chinese_text(text)
    
    # Only translate if there's still text after filtering
    if not filtered_text:
        return ""
        
    logger.info(f"Sending for web translation: {filtered_text}")
    
    # Try translation services in priority order: Caiyun -> Baidu -> Google
    translation_services = [
        ('caiyun', lambda: ts.translate_text(filtered_text, translator='caiyun', from_language='zh', to_language='en')),
        ('baidu', lambda: ts.translate_text(filtered_text, translator='baidu', from_language='zh', to_language='en')),
        ('google', lambda: ts.translate_text(filtered_text, translator='google', from_language='zh', to_language='en')),
    ]
    
    for service_name, attempt in translation_services:
        try:
            translation = attempt()
            if translation and translation.strip():
                logger.info(f"Web translation returned from {service_name}: {translation}")
                return translation
        except Exception as e:
            logger.warning(f"Web translation failed with {service_name}: {str(e)}")
            continue
    
    logger.error(f"All web translation attempts failed for text: {filtered_text}")
    return f"[Translation failed: {filtered_text}]"


def web_translate_en_to_zh(text):
    """
    Translate English text to Chinese using third-party APIs
    Only runs if web translation is enabled
    """
    if not WEB_TRANSLATION_ENABLED:
        logger.debug("Web translation disabled, skipping")
        return None
    
    logger.info(f"Sending for web translation EN->ZH: {text}")
    
    # Try multiple translation attempts
    translation_attempts = [
        ('caiyun', lambda: ts.translate_text(text, translator='caiyun', from_language='en', to_language='zh')),
        ('google', lambda: ts.translate_text(text, translator='google', from_language='en', to_language='zh')),
        ('bing', lambda: ts.translate_text(text, translator='bing', from_language='en', to_language='zh')),
    ]
    
    for service_name, attempt in translation_attempts:
        try:
            translation = attempt()
            if translation and translation.strip():
                logger.info(f"Web translation EN->ZH returned from {service_name}: {translation}")
                return translation
        except Exception as e:
            logger.warning(f"Web translation EN->ZH failed with {service_name}: {str(e)}")
            continue
    
    logger.error(f"All web translation EN->ZH attempts failed for text: {text}")
    return f"[Translation failed: {text}]"