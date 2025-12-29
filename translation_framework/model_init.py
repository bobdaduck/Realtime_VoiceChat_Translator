"""
Model initialization and loading
Handles Vosk, funASR, and CEDICT dictionary loading
"""
import os
import logging
import traceback
import urllib.request
from vosk import Model as VoskModel
import funasr

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
ENGLISH_MODEL_PATH = "vosk-en"
CHINESE_MODEL_PATH = "paraformer-zh"
SAMPLE_RATE = 16000
CEDICT_PATH = "cedict_ts.u8"
CEDICT_URL = "https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz"

# Global variables for models and dictionary
CEDICT = {}
english_model = None
chinese_model = None


def download_cedict():
    """Download CC-CEDICT if not already present"""
    if os.path.exists(CEDICT_PATH):
        logger.info(f"CC-CEDICT already exists at {CEDICT_PATH}")
        return
    
    try:
        import gzip
        logger.info("Downloading CC-CEDICT...")
        
        # Download the gzipped file
        gz_path = CEDICT_PATH + ".gz"
        urllib.request.urlretrieve(CEDICT_URL, gz_path)
        
        # Decompress
        with gzip.open(gz_path, 'rb') as f_in:
            with open(CEDICT_PATH, 'wb') as f_out:
                f_out.write(f_in.read())
        
        # Clean up gz file
        os.remove(gz_path)
        logger.info("✓ CC-CEDICT downloaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to download CC-CEDICT: {e}")
        raise


def load_cedict():
    """Load CC-CEDICT into a dictionary"""
    cedict = {}
    
    try:
        with open(CEDICT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                # Skip comments and empty lines
                if line.startswith('#') or not line.strip():
                    continue
                
                try:
                    # Parse format: 簡體 繁體 [pin1 yin1] /definition1/definition2/
                    parts = line.split('[')
                    if len(parts) < 2:
                        continue
                    
                    words = parts[0].strip().split()
                    if len(words) < 2:
                        continue
                    
                    simplified = words[1]  # Use simplified Chinese
                    
                    # Extract definitions
                    def_part = line.split('/')
                    definitions = [d.strip() for d in def_part[1:-1] if d.strip()]
                    
                    if definitions:
                        # Store first definition for simplicity
                        cedict[simplified] = definitions[0]
                
                except Exception:
                    continue
        
        logger.info(f"✓ Loaded {len(cedict)} entries from CC-CEDICT")
        return cedict
        
    except Exception as e:
        logger.error(f"Failed to load CC-CEDICT: {e}")
        return {}


def initialize_transcription_models():
    """Initialize and load all models needed for translation"""
    global CEDICT, english_model, chinese_model
    
    print("Loading translation models...")
    
    # Load Vosk model for English speech recognition
    try:
        english_model = VoskModel(ENGLISH_MODEL_PATH)
        print("✓ Vosk English model loaded")
    except Exception as e:
        logger.error(f"Failed to load Vosk model: {e}")
        logger.error(traceback.format_exc())
        raise
    
    # Load funASR model for Chinese speech recognition
    os.makedirs(CHINESE_MODEL_PATH, exist_ok=True)
    
    try:
        chinese_model = funasr.AutoModel(
            model=CHINESE_MODEL_PATH,
            model_revision="v2.0.4",
            batch_size=1,
            device="cpu",
            vad_model=None,
            punc_model=None,
            disable_pbar=True,
            disable_update=True,
            cache_dir=CHINESE_MODEL_PATH,
            automatic_download=True
        )
        print("✓ funASR model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading funASR model: {e}")
        logger.error(traceback.format_exc())
        raise
    
    # Download and load CC-CEDICT
    try:
        download_cedict()
        CEDICT = load_cedict()
        print("✓ CC-CEDICT loaded successfully")
    except Exception as e:
        logger.error(f"Error loading CC-CEDICT: {e}")
        logger.error(traceback.format_exc())
        CEDICT = {}
    
    print("Translation models loaded successfully!")
    
    return (english_model, chinese_model, None, None, None, None)


def get_cedict():
    """Get the loaded CEDICT dictionary"""
    return CEDICT