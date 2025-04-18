import audio_capture
import model_work
import window_display

def main():
    try:
        # Initialize audio devices
        print("Initializing audio devices...")
        loopback_mic, regular_mic = audio_capture.initialize_audio_devices()
        
        # Initialize translation models
        print("Loading translation models...")
        (english_model, chinese_model, 
         zh_en_tokenizer, zh_en_model, 
         en_zh_tokenizer, en_zh_model) = model_work.initialize_translation_models()
        
        # Start audio capture threads
        print("Starting audio capture threads...")
        english_thread, chinese_thread = audio_capture.start_audio_threads(
            english_model, chinese_model, regular_mic, loopback_mic,
            zh_en_tokenizer, zh_en_model, en_zh_tokenizer, en_zh_model
        )
        
        # Create and run the window
        print("Creating display window...")
        root = window_display.create_window(audio_capture.get_display_data)
        
        # Run the main window
        print("Application running. Press Esc to exit.")
        root.mainloop()
        
    except KeyboardInterrupt:
        print("Exiting due to keyboard interrupt...")
    except Exception as e:
        print(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()