import json
import pypinyin
from vosk import Model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Configuration
ENGLISH_MODEL_PATH = "model"
CHINESE_MODEL_PATH = "chinese-model"
SAMPLE_RATE = 16000

def initialize_translation_models():
    """Initialize and load all machine learning models needed for translation"""
    print("Downloading and loading translation models... (this may take a few minutes on first run)")
    
    # Load Vosk models
    english_model = Model(ENGLISH_MODEL_PATH)
    chinese_model = Model(CHINESE_MODEL_PATH)
    
    # Load Hugging Face translation models
    zh_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    zh_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    
    en_zh_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    en_zh_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    
    print("Translation models loaded successfully!")
    
    return (english_model, chinese_model, 
            zh_en_tokenizer, zh_en_model, 
            en_zh_tokenizer, en_zh_model)

def translate_chinese_to_english(text, zh_en_tokenizer, zh_en_model):
    """Translate Chinese text to English using Hugging Face model"""
    try:
        # Skip translation if text is empty
        if not text.strip():
            return ""
            
        inputs = zh_en_tokenizer(text, return_tensors="pt", padding=True)
        output_tokens = zh_en_model.generate(**inputs, max_length=128)
        translation = zh_en_tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return translation
    except Exception as e:
        print(f"Chinese to English translation error: {str(e)}")
        return f"[Translation Error]"

def translate_english_to_chinese(text, en_zh_tokenizer, en_zh_model):
    """Translate English text to Chinese using Hugging Face model"""
    try:
        # Skip translation if text is empty
        if not text.strip():
            return ""
            
        inputs = en_zh_tokenizer(text, return_tensors="pt", padding=True)
        output_tokens = en_zh_model.generate(**inputs, max_length=128)
        translation = en_zh_tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return translation
    except Exception as e:
        print(f"English to Chinese translation error: {str(e)}")
        return f"[Translation Error]"

def generate_pinyin(chinese_text):
    """Generate pinyin with tone marks for Chinese text"""
    if not chinese_text.strip():
        return ""
    try:
        return ' '.join(pypinyin.lazy_pinyin(
            chinese_text, 
            style=pypinyin.Style.TONE  # Add tone markers
        ))
    except Exception as e:
        print(f"Pinyin generation error: {str(e)}")
        return "[Pinyin Error]"

def process_text(text, is_english, zh_en_tokenizer, zh_en_model, en_zh_tokenizer, en_zh_model):
    """Process text once to ensure consistency between transcription and translation"""
    if not text.strip():
        return {"transcription": "", "translation": "", "pinyin": ""}
    
    result = {}
    
    # Clean text for stuttering
    words = text.split()
    cleaned_words = []
    prev_word = None
    repeat_count = 0
    
    for word in words:
        if word == prev_word:
            repeat_count += 1
            if repeat_count < 2:  # Allow max 2 repeats
                cleaned_words.append(word)
        else:
            repeat_count = 0
            cleaned_words.append(word)
        prev_word = word
    
    cleaned_text = ' '.join(cleaned_words)
    result["transcription"] = cleaned_text
    
    # Process based on language
    if is_english:
        # For English input
        if len(cleaned_text) > 1:  # Only translate substantial text
            # Translate English to Chinese
            chinese_translation = translate_english_to_chinese(cleaned_text, en_zh_tokenizer, en_zh_model)
            # Generate pinyin for the Chinese translation
            pinyin_result = generate_pinyin(chinese_translation)
            
            result["translation"] = chinese_translation
            result["pinyin"] = pinyin_result
        else:
            result["translation"] = ""
            result["pinyin"] = ""
    else:
        # For Chinese input
        if len(cleaned_text) > 1:  # Only translate substantial text
            # Generate pinyin for Chinese text
            pinyin_result = generate_pinyin(cleaned_text)
            # Translate Chinese to English
            english_translation = translate_chinese_to_english(cleaned_text, zh_en_tokenizer, zh_en_model)
            
            result["pinyin"] = pinyin_result
            result["translation"] = english_translation
        else:
            result["pinyin"] = generate_pinyin(cleaned_text)
            result["translation"] = ""
    
    return result