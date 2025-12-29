"""
AI-based translation using local models
This will be the primary translation method with CEDICT as fallback
"""
import logging
from abc import ABC, abstractmethod
from typing import Optional

# Set up logging
logger = logging.getLogger(__name__)

# Configuration
AI_MODEL = None
AI_TRANSLATION_ENABLED = False


class TranslationStrategy(ABC):
    """Abstract base class for translation strategies"""
    
    @abstractmethod
    def translate(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Translate text from source to target language"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this translation method is available"""
        pass


class OpusMTTranslator(TranslationStrategy):
    """
    Helsinki-NLP OPUS-MT translator - produces more literal translations
    Best for word-for-word accuracy without hallucination
    
    Installation: pip install transformers sentencepiece torch
    """
    
    def __init__(self):
        self.model_zh_en = None
        self.tokenizer_zh_en = None
        self.model_en_zh = None
        self.tokenizer_en_zh = None
        self._initialize()
    
    def _initialize(self):
        try:
            from transformers import MarianMTModel, MarianTokenizer
            
            # Chinese to English model
            model_name_zh_en = "Helsinki-NLP/opus-mt-zh-en"
            logger.info(f"Loading {model_name_zh_en}...")
            self.tokenizer_zh_en = MarianTokenizer.from_pretrained(model_name_zh_en)
            self.model_zh_en = MarianMTModel.from_pretrained(model_name_zh_en)
            
            # English to Chinese model
            model_name_en_zh = "Helsinki-NLP/opus-mt-en-zh"
            logger.info(f"Loading {model_name_en_zh}...")
            self.tokenizer_en_zh = MarianTokenizer.from_pretrained(model_name_en_zh)
            self.model_en_zh = MarianMTModel.from_pretrained(model_name_en_zh)
            
            logger.info("OPUS-MT models loaded successfully")
            
        except ImportError:
            logger.error("transformers library not found. Install with: pip install transformers sentencepiece torch")
        except Exception as e:
            logger.error(f"Failed to load OPUS-MT models: {e}")
    
    def is_available(self) -> bool:
        return self.model_zh_en is not None and self.model_en_zh is not None
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        try:
            if source_lang == "zh" and target_lang == "en":
                if not self.model_zh_en:
                    return None
                
                # Tokenize and translate
                inputs = self.tokenizer_zh_en(text, return_tensors="pt", padding=True)
                outputs = self.model_zh_en.generate(
                    **inputs,
                    max_length=512,
                    num_beams=1,  # Use greedy decoding for more literal translation
                    temperature=0.1,  # Low temperature for less creativity
                    do_sample=False,  # No sampling for deterministic output
                )
                translation = self.tokenizer_zh_en.decode(outputs[0], skip_special_tokens=True)
                return translation
                
            elif source_lang == "en" and target_lang == "zh":
                if not self.model_en_zh:
                    return None
                
                inputs = self.tokenizer_en_zh(text, return_tensors="pt", padding=True)
                outputs = self.model_en_zh.generate(
                    **inputs,
                    max_length=512,
                    num_beams=1,
                    temperature=0.1,
                    do_sample=False,
                )
                translation = self.tokenizer_en_zh.decode(outputs[0], skip_special_tokens=True)
                return translation
            
            return None
            
        except Exception as e:
            logger.error(f"OPUS-MT translation failed: {e}")
            return None


class OllamaTranslator(TranslationStrategy):
    """
    Ollama-based translator - more flexible, can upgrade models easily
    Uses strict prompting for literal translation
    
    Installation: 
    1. Install Ollama from https://ollama.ai
    2. Pull a model: ollama pull llama3.2
    3. pip install ollama
    """
    
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        self.client = None
        self._initialize()
    
    def _initialize(self):
        try:
            import ollama
            self.client = ollama.Client()
            
            # Test if model is available
            try:
                self.client.list()
                logger.info(f"Ollama client initialized with model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Ollama not running or model not available: {e}")
                self.client = None
                
        except ImportError:
            logger.error("ollama library not found. Install with: pip install ollama")
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        if not self.client:
            return None
        
        try:
            # Create a strict prompt for literal translation
            if source_lang == "zh" and target_lang == "en":
                prompt = f"""Translate the following Chinese text to English LITERALLY and WORD-FOR-WORD. 

CRITICAL RULES:
- Do NOT fix grammar or make it sound natural
- Do NOT add words that aren't in the original
- Do NOT interpret or guess meaning
- Translate each word directly, even if the result sounds broken
- Keep the exact same word order when possible
- If a word seems wrong or doesn't make sense, translate it anyway

Chinese text: {text}

Literal English translation:"""
            
            elif source_lang == "en" and target_lang == "zh":
                prompt = f"""Translate the following English text to Chinese LITERALLY and WORD-FOR-WORD.

CRITICAL RULES:
- Do NOT fix grammar or make it sound natural
- Do NOT add words that aren't in the original
- Do NOT interpret or guess meaning
- Translate each word directly, even if the result sounds broken
- If a word seems wrong or doesn't make sense, translate it anyway

English text: {text}

Literal Chinese translation:"""
            else:
                return None
            
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.1,  # Very low temperature for consistency
                    'top_p': 0.9,
                    'top_k': 10,
                }
            )
            
            translation = response['response'].strip()
            
            # Remove any explanations or notes (some models add them despite prompting)
            # Keep only the first line if model added explanation
            if '\n' in translation:
                translation = translation.split('\n')[0].strip()
            
            return translation
            
        except Exception as e:
            logger.error(f"Ollama translation failed: {e}")
            return None


# Global translator instance
_translator: Optional[TranslationStrategy] = None


def initialize_ai_translator(strategy: str = "opus"):
    """
    Initialize the local AI translation model
    
    Args:
        strategy: Translation strategy to use
                 - "opus": Helsinki-NLP OPUS-MT (default, most literal)
                 - "ollama": Ollama with local LLM (more flexible, upgradeable)
    
    Returns:
        True if initialization successful, False otherwise
    """
    global AI_MODEL, AI_TRANSLATION_ENABLED, _translator
    
    logger.info(f"Initializing AI translator with strategy: {strategy}")
    
    if strategy == "opus":
        _translator = OpusMTTranslator()
    elif strategy == "ollama":
        _translator = OllamaTranslator()
    else:
        logger.error(f"Unknown strategy: {strategy}")
        return False
    
    if _translator.is_available():
        AI_TRANSLATION_ENABLED = True
        AI_MODEL = _translator
        logger.info("AI translation enabled successfully")
        return True
    else:
        AI_TRANSLATION_ENABLED = False
        logger.warning("AI translation initialization failed - using fallback")
        return False


def ai_translate_zh_to_en(text: str) -> Optional[str]:
    """
    Translate Chinese text to English using local AI model
    Performs LITERAL translation without interpretation
    
    Args:
        text: Chinese text to translate
        
    Returns:
        English translation or None if AI translation is not available
    """
    if not AI_TRANSLATION_ENABLED or _translator is None:
        logger.debug("AI translation not available")
        return None
    
    logger.debug(f"AI translation ZH->EN: {text}")
    
    try:
        translation = _translator.translate(text, source_lang="zh", target_lang="en")
        if translation:
            logger.info(f"Translation result: {translation}")
        return translation
        
    except Exception as e:
        logger.error(f"AI translation failed: {e}")
        return None


def ai_translate_en_to_zh(text: str) -> Optional[str]:
    """
    Translate English text to Chinese using local AI model
    Performs LITERAL translation without interpretation
    
    Args:
        text: English text to translate
        
    Returns:
        Chinese translation or None if AI translation is not available
    """
    if not AI_TRANSLATION_ENABLED or _translator is None:
        logger.debug("AI translation not available")
        return None
    
    logger.debug(f"AI translation EN->ZH: {text}")
    
    try:
        translation = _translator.translate(text, source_lang="en", target_lang="zh")
        if translation:
            logger.info(f"Translation result: {translation}")
        return translation
        
    except Exception as e:
        logger.error(f"AI translation EN->ZH failed: {e}")
        return None


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Try OPUS-MT first (recommended for literal translation)
    print("Testing OPUS-MT translator...")
    if initialize_ai_translator("opus"):
        result = ai_translate_zh_to_en("狗做猫树")  # Literal nonsense
        print(f"ZH->EN: {result}")
        
        result = ai_translate_en_to_zh("dog does cat tree")
        print(f"EN->ZH: {result}")
    
    # Try Ollama as alternative
    print("\nTesting Ollama translator...")
    if initialize_ai_translator("ollama"):
        result = ai_translate_zh_to_en("狗做猫树")
        print(f"ZH->EN: {result}")
        
        result = ai_translate_en_to_zh("dog does cat tree")
        print(f"EN->ZH: {result}")