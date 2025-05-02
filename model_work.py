import json
import pypinyin
from vosk import Model
import logging
from chinese_english_lookup import Dictionary
import jieba
import re
import translators as ts

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize CC-CEDICT dictionary
ce_dict = Dictionary()

# Configuration
ENGLISH_MODEL_PATH = "model"
CHINESE_MODEL_PATH = "chinese-model"
SAMPLE_RATE = 16000

def initialize_translation_models():
    """Initialize and load all models needed for translation"""
    print("Loading translation models...")
    
    # Load Vosk models for speech recognition
    english_model = Model(ENGLISH_MODEL_PATH)
    chinese_model = Model(CHINESE_MODEL_PATH)
    
    print("CC-CEDICT dictionary ready to use")
    print("Translation models loaded successfully!")
    
    return (english_model, chinese_model, None, None, None, None)

def translate_chinese_to_english(text):
    """Translate Chinese text to English """
    return ts.translate_text(text, translator='caiyun', from_language='zh', to_language='en')


def translate_english_to_chinese(text):
    """Translate English text to Chinese """
    return ts.translate_text(text, translator='caiyun', from_language='en', to_language='zh')

def generate_pinyin(chinese_text):
    """Generate pinyin with tone marks for Chinese text"""
    if not chinese_text.strip():
        return ""
    try:
        return ' '.join(pypinyin.lazy_pinyin( #adds tone marks
            chinese_text, 
            style=pypinyin.Style.TONE
        ))
    except Exception as e:
        print(f"Pinyin generation error: {str(e)}")
        return "[Pinyin Error]"

def process_text(text, is_english, *args):
    """Process text once to ensure consistency between transcription and translation"""
    if not text.strip():
        return {"transcription": "", "translation": "", "pinyin": ""}
    
    result = {}
    words = text.split()
    cleaned, prev, count = [], None, 0
    for w in words:
        if w == prev:
            count += 1
            if count < 2:
                cleaned.append(w)
        else:
            count = 0
            cleaned.append(w)
        prev = w
    cleaned_text = ' '.join(cleaned)
    result["transcription"] = cleaned_text
    
    if is_english:
        if len(cleaned_text) > 1:
            cn_trans = translate_english_to_chinese(cleaned_text)
            result["translation"] = cn_trans
            result["pinyin"] = generate_pinyin(cn_trans)
        else:
            result.update({"translation": "", "pinyin": ""})
    else:
        if len(cleaned_text) > 1:
            logger.info(f"Chinese characters received: {cleaned_text}")
            pin = generate_pinyin(cleaned_text)
            logger.info(f"Generated pinyin: {pin}")
            en_trans = translate_chinese_to_english(cleaned_text)
            logger.info(f"Translation result: {en_trans}")
            result.update({"pinyin": pin, "translation": en_trans})
        else:
            result["pinyin"] = generate_pinyin(cleaned_text)
            result["translation"] = ""
    return result
