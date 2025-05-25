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

# Bandpass filter parameters (focus on human speech: 300-3000 Hz), these are divided from samplerate/2 so 8000 is max
BANDPASS_LOW = 100
BANDPASS_HIGH = 3000

volume_cutoff = 0.04

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

def apply_modulation_filterbank(audio, fs=SAMPLE_RATE,
                                mod_low=.2, mod_high=20,
                                order=4, mix=0.9, floor=0.8):
    # ensure correct dtype/contiguity
    audio = np.ascontiguousarray(audio, dtype=np.float32)

    # 1. extract envelope
    env = np.abs(signal.hilbert(audio))

    # 2. band-pass the envelope
    nyq = fs/2
    b, a = signal.butter(order, [mod_low/nyq, mod_high/nyq], btype='band')
    env_f = signal.lfilter(b, a, env)

    # 3. safe normalize
    peak = env_f.max()
    peak = peak if peak > 1e-8 else 1e-8
    env_f = env_f / peak
    env_f = np.clip(env_f, floor, 1.0)

    # 4. re-apply to carrier, mix with dry
    out = mix * (audio * env_f) + (1 - mix) * audio

    # keep same length
    return out[:len(audio)]
def remove_low_volume(audio_data, threshold=volume_cutoff, smooth_ms=50):
    """
    Remove low-volume audio below a certain threshold while preserving louder parts.
    Optimized for efficiency.
    """
    try:
        # Convert to numpy array if not already
        audio_data = np.asarray(audio_data, dtype=np.float32)
        
        # Store original shape for later
        original_shape = audio_data.shape
        flattened = False
        
        # Handle multi-channel audio efficiently
        if len(original_shape) > 1:
            audio_flat = audio_data.reshape(-1)
            flattened = True
        else:
            audio_flat = audio_data
            
        # Fast envelope calculation using absolute values
        abs_audio = np.abs(audio_flat)
        
        # Calculate window size in samples
        window_size = max(3, int(SAMPLE_RATE * smooth_ms / 1000))
        if window_size % 2 == 0:
            window_size += 1
            
        # Use scipy's efficient moving average implementation
        # This is much faster than manual convolution for large arrays
        envelope = signal.medfilt(abs_audio, kernel_size=window_size)
        
        # Fast binary masking (vectorized operation)
        mask = envelope > threshold
        
        # Apply mask (vectorized)
        processed_audio = audio_flat * mask
        
        # Reshape if needed
        if flattened:
            processed_audio = processed_audio.reshape(original_shape)
            
        logger.info(f"Applied noise gate with threshold {threshold}")
        return processed_audio
        
    except Exception as e:
        logger.error(f"Error in remove_low_volume: {str(e)}")
        return audio_data
    
def process_audio(audio_data, is_chinese=True):
    """
    Simple audio processing function that just applies filtering
    """
    processed_audio_sample = audio_data
    try:
        if is_chinese: #English audio mic doesn't require threshold
            processed_audio_sample = remove_low_volume(processed_audio_sample)

        processed_audio_sample = apply_bandpass_filter(processed_audio_sample)

        processed_audio_sample = apply_modulation_filterbank(processed_audio_sample)

        # play_after_delay(processed_audio_sample) #Debugging
        return processed_audio_sample
        
    except Exception as e:
        logger.error(f"Error in audio processing: {str(e)}")
        # Return original audio on error
        return audio_data

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