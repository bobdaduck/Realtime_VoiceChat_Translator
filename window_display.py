import tkinter as tk
import ctypes
import time
import threading
from tkinter import font

# Windows-specific constants for transparent/click-through windows
WS_EX_TRANSPARENT = 0x00000020
WS_EX_LAYERED = 0x00080000
GWL_EXSTYLE = -20
LWA_ALPHA = 0x00000002
LWA_COLORKEY = 0x00000001

# Display configuration
DISPLAY_FONT = ('Arial', 14)
WINDOW_OPACITY = 0.8
TEXT_COLOR = '#20EADA'

# Distinct colors for Chinese and pinyin segments (matching pairs)
SEGMENT_COLORS = [
    '#20EADA',  # Bright cyan
    '#FFD700',  # Gold
    '#FF6B6B',  # Red
    '#4ECDC4',  # Teal
    '#45B7D1',  # Blue
    '#96CEB4',  # Green
    '#FFEAA7',  # Yellow
    '#DDA0DD'   # Plum
]

# Timing configuration for display refresh
TEXT_DISPLAY_DURATION = 4  # How long to keep text on screen with no updates (seconds)
CLEAR_DISPLAY_INTERVAL = 4  # How frequently to clear the display for new text (seconds)

# Create a transparent overlay window (special implementation)
class ClickThroughWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Configure window properties
        self.attributes('-alpha', WINDOW_OPACITY)
        self.attributes('-topmost', True)
        self.overrideredirect(True)
        screen_width = self.winfo_screenwidth()
        self.geometry(f'{screen_width}x200+0+0')  # Full width at top of screen
        self.configure(bg='black')
        self.wm_attributes("-transparent", "black")
        
        # This flag is needed for Windows to properly handle the window
        self.wm_attributes("-toolwindow", True)
        
        # Create layout
        self.setup_ui(screen_width)
        
        # Set up Windows-specific transparency behaviors
        self.after(10, self.setup_window_transparency)

    def setup_ui(self, screen_width):
        # Create two frames side by side
        self.left_frame = tk.Frame(self, bg='black')
        self.left_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        self.right_frame = tk.Frame(self, bg='black')
        self.right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        # Create labels for English side
        self.english_label = tk.Label(
            self.left_frame,
            text='',
            font=DISPLAY_FONT,
            fg=TEXT_COLOR,
            bg='black',
            wraplength=screen_width//2-30,
            justify='left'
        )
        self.english_label.pack(expand=True, fill='both')
        
        # Create text widget for Chinese side to support multiple colors
        self.chinese_text = tk.Text(
            self.right_frame,
            font=DISPLAY_FONT,
            bg='black',
            borderwidth=0,
            highlightthickness=0,
            wrap='word',
            width=screen_width//20,  # Approximate width in characters
            height=10
        )
        self.chinese_text.pack(expand=True, fill='both', anchor='e')  # Anchor to the right
        
        # Configure tags for different colors
        for i, color in enumerate(SEGMENT_COLORS):
            self.chinese_text.tag_configure(f'chinese_segment_{i}', foreground=color)
            self.chinese_text.tag_configure(f'pinyin_segment_{i}', foreground=color)  # Same color for matching segments
            
        # Configure tag for English translation
        self.chinese_text.tag_configure('translation', foreground='#AAAAAA')  # Light gray for translation
        
        # Disable text editing
        self.chinese_text.configure(state='disabled')
        
        # # Add keyboard shortcut to exit (Esc key)
        # self.bind('<Escape>', lambda e: self.destroy())
        # self.protocol("WM_DELETE_WINDOW", lambda: self.destroy())

    def setup_window_transparency(self):
        """Use Windows-specific API to properly set click-through transparency"""
        try:
            # Get the window handle
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            
            # Set the window to layered and transparent (this is the key for true click-through)
            ex_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                hwnd, 
                GWL_EXSTYLE, 
                ex_style | WS_EX_LAYERED | WS_EX_TRANSPARENT
            )
            
            # Apply transparency color key
            ctypes.windll.user32.SetLayeredWindowAttributes(
                hwnd,
                0,  # RGB black 
                int(WINDOW_OPACITY * 255),  # Alpha value
                LWA_ALPHA | LWA_COLORKEY  # Using both alpha blending and color key
            )
            
            # Create a subclass for all child windows
            def subclass_func(hwnd, msg, wparam, lparam, uid, data):
                return ctypes.windll.user32.DefWindowProcW(hwnd, msg, wparam, lparam)
            
            # Find all child windows recursively and make them non-interactive
            def make_all_children_noninteractive(parent_widget):
                for child in parent_widget.winfo_children():
                    # Try to get hwnd for this child
                    if hasattr(child, 'winfo_id'):
                        child_hwnd = child.winfo_id() 
                        if child_hwnd:
                            # Get current style
                            child_style = ctypes.windll.user32.GetWindowLongW(child_hwnd, GWL_EXSTYLE)
                            # Set transparent and layered
                            ctypes.windll.user32.SetWindowLongW(
                                child_hwnd, 
                                GWL_EXSTYLE, 
                                child_style | WS_EX_TRANSPARENT | WS_EX_LAYERED
                            )
                            
                            # Also apply transparency attributes
                            ctypes.windll.user32.SetLayeredWindowAttributes(
                                child_hwnd, 
                                0,  # RGB black
                                int(WINDOW_OPACITY * 255),  # Alpha
                                LWA_ALPHA | LWA_COLORKEY  # Both modes
                            )
                    
                    # Recursively process its children
                    make_all_children_noninteractive(child)
            
            # Apply to all children
            make_all_children_noninteractive(self)
            
        except Exception as e:
            print(f"Error setting up transparency: {e}")

def update_displays(root, english_label, chinese_text, get_display_data_func):
    """Update the display labels with the latest transcription data"""
    english_display, chinese_display = get_display_data_func()
    
    current_time = time.time()
    
    # Initialize attributes if they don't exist yet
    if not hasattr(english_label, "last_text"):
        english_label.last_text = ""
        english_label.last_update_time = current_time
        english_label.display_start_time = current_time
        english_label.should_clear = False
        
    if not hasattr(chinese_text, "last_update_time"):
        chinese_text.last_update_time = current_time
        chinese_text.display_start_time = current_time
        chinese_text.should_clear = False
    
    # Handle English display (microphone)
    if english_display["transcription"]:
        # Check if this is new text
        if english_display["transcription"] != english_label.last_text:
            # Check if we should clear the display before showing new text
            elapsed_since_last_clear = current_time - getattr(english_label, "display_start_time", 0)
            if elapsed_since_last_clear > CLEAR_DISPLAY_INTERVAL or english_label.should_clear:
                # Start with fresh text instead of appending
                # Display English transcription and translation
                english_text = f"{english_display['transcription']}"
                if english_display.get('translation'):
                    english_text += f"\n{english_display['translation']}"
                english_label.display_start_time = current_time
                english_label.should_clear = False
            else:
                # Keep existing text if within the clear interval
                english_text = english_label.last_text
                if english_display.get('translation'):
                    english_text += f"\n{english_display['translation']}"
            
            # Update the label
            english_label.config(text=english_text)
            english_label.last_text = english_display['transcription']
            english_label.last_update_time = current_time
    else:
        # If no new text, check if we should clear the display
        if hasattr(english_label, "last_update_time"):
            time_since_update = current_time - english_label.last_update_time
            if time_since_update > TEXT_DISPLAY_DURATION:
                english_label.config(text="")
                english_label.last_text = ""
                english_label.should_clear = True
    
    # Handle Chinese display (system audio) with color gradients
    if chinese_display.get("text_segments") and len(chinese_display["text_segments"]) > 0:
        # Enable text widget for editing
        chinese_text.configure(state='normal')
        
        # Clear current content
        chinese_text.delete('1.0', tk.END)
        
        # Get individual segments from the deque for coloring
        segments = list(chinese_display["text_segments"])
        num_segments = len(segments)
        
        # Split the main transcription and pinyin by segments for coloring
        # Since we have the processed transcription and pinyin, we need to map them to segments
        transcription_text = chinese_display.get('transcription', '')
        pinyin_text = chinese_display.get('pinyin', '')
        
        if transcription_text:
            # Split transcription into parts based on segments
            transcription_parts = []
            pinyin_parts = []
            
            for i, segment in enumerate(segments):
                if segment.get("transcription"):
                    transcription_parts.append(segment["transcription"])
                if segment.get("pinyin"):
                    pinyin_parts.append(segment["pinyin"])
            
            # Display Chinese text segments with distinct colors
            for i, part in enumerate(transcription_parts):
                color_idx = i % len(SEGMENT_COLORS)  # Cycle through distinct colors
                chinese_text.insert(tk.END, part, f'chinese_segment_{color_idx}')
                if i < len(transcription_parts) - 1:
                    chinese_text.insert(tk.END, " ")
            
            # Add newline before pinyin
            chinese_text.insert(tk.END, "\n")
            
            # Display pinyin segments with matching colors
            for i, part in enumerate(pinyin_parts):
                color_idx = i % len(SEGMENT_COLORS)  # Same color as corresponding Chinese segment
                chinese_text.insert(tk.END, part, f'pinyin_segment_{color_idx}')
                if i < len(pinyin_parts) - 1:
                    chinese_text.insert(tk.END, " ")
        
        # Add translation (if available) at the bottom in light gray
        if chinese_display.get('translation'):
            chinese_text.insert(tk.END, f"\n{chinese_display['translation']}", 'translation')
        
        # Update last update time
        chinese_text.last_update_time = current_time
        
        # Disable text widget again
        chinese_text.configure(state='disabled')
        
    else:
        # If no segments, check if we should clear the display
        if hasattr(chinese_text, "last_update_time"):
            time_since_update = current_time - chinese_text.last_update_time
            if time_since_update > TEXT_DISPLAY_DURATION:
                chinese_text.configure(state='normal')
                chinese_text.delete('1.0', tk.END)
                chinese_text.configure(state='disabled')
                chinese_text.should_clear = True
    
    # Continue updating
    root.after(100, lambda: update_displays(root, english_label, chinese_text, get_display_data_func))
    
def create_window(get_display_data_func):
    """Create and run the main window"""
    # Create the main window using our custom class
    root = ClickThroughWindow()
    english_label = root.english_label
    chinese_text = root.chinese_text

    # Start display updates
    root.after(100, lambda: update_displays(root, english_label, chinese_text, get_display_data_func))
    
    return root