import numpy as np
import scipy.signal as signal
import logging
import threading
import time
import sounddevice as sd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Configuration constants
SAMPLE_RATE = 16000

# Bandpass filter parameters (focus on human speech: 300-3000 Hz)
BANDPASS_LOW = 300
BANDPASS_HIGH = 3200

# Noise reduction parameters
NOISE_REDUCTION_FACTOR = 1.4 

def design_bandpass_filter():
    """Design a bandpass filter to focus on human speech frequencies"""
    nyquist = SAMPLE_RATE / 2.0
    low = BANDPASS_LOW / nyquist
    high = BANDPASS_HIGH / nyquist
    
    # Create a butterworth bandpass filter (order 4)
    b, a = signal.butter(4, [low, high], btype='band')
    return b, a

# Pre-compute the filter coefficients
BANDPASS_B, BANDPASS_A = design_bandpass_filter()

def apply_bandpass_filter(audio_data):
    """Apply bandpass filter to focus on human speech frequencies"""
    try:
        # Apply filter to the audio data
        filtered_audio = signal.lfilter(BANDPASS_B, BANDPASS_A, audio_data)
        return filtered_audio
    except Exception as e:
        logger.error(f"Error applying bandpass filter: {str(e)}")
        return audio_data  # Return original on error

def apply_light_smoothing(audio_data):
    """
    Apply very light smoothing to reduce high frequency noise
    without trying to do advanced noise reduction
    """
    try:
        # Apply a very mild gain to smooth the signal
        # This just attenuates the signal slightly to reduce noise
        gain = 1.0 - NOISE_REDUCTION_FACTOR
        return audio_data * gain
    except Exception as e:
        logger.error(f"Error in audio smoothing: {str(e)}")
        return audio_data  # Return original on error

def process_audio(audio_data):
    """
    Simple audio processing function that just applies filtering
    
    Parameters:
    - audio_data: Input audio data (numpy array)
    
    Returns:
    - Processed audio data
    """
    try:
        # Step 1: Apply bandpass filter
        filtered_audio = apply_bandpass_filter(audio_data)
        
        # Step 2: Apply light smoothing
        smoothed_audio = apply_light_smoothing(filtered_audio)
        
        return smoothed_audio
        
    except Exception as e:
        logger.error(f"Error in audio processing: {str(e)}")
        # Return original audio on error
        return audio_data

def preprocess_buffer(audio_buffer, *args):
    """
    Process a buffer of audio data with simple filtering
    Accepts noise_profile argument for compatibility but ignores it
    
    Parameters:
    - audio_buffer: Buffer of audio data
    - *args: For backward compatibility (previously noise_profile)
    
    Returns:
    - Tuple of (processed_buffer, None) for compatibility
    """
    # Just process the audio without noise estimation
    processed_audio = process_audio(audio_buffer)
    
    # Return None as the second value for compatibility with existing code
    return processed_audio, None





def play_after_delay(audio_array: np.ndarray,
                     samplerate: int = SAMPLE_RATE,
                     delay: float = 2.0):
    """
    Play back a NumPy audio buffer after `delay` seconds for debugging
    """
    def _player():
        time.sleep(delay)
        sd.play(audio_array, samplerate)
        # optionally block until done:
        # sd.wait()

    threading.Thread(target=_player, daemon=True).start()