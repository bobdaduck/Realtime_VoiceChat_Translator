import tkinter as tk
import ctypes
import time
import threading
import random
import colorsys
from tkinter import font
from audio_capture import MAX_CHINESE_WORDS

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
PINYIN_COLOR = '#FFA500'  # Orange for pinyin
CEDICT_COLOR = '#90EE90'  # Light green for CEDICT

# Generate colors for pinyin words
def generate_pinyin_word_colors(num_colors=20):
    """Generate distinct colors for pinyin words"""
    colors = []
    random.seed(42)  # Fixed seed for consistency
    
    color_ranges = [
        (0.55, 0.75),  # Blues
        (0.13, 0.20),  # Yellows
        (0.03, 0.12),  # Oranges
        (0.85, 0.95),  # Pinks/Magentas
    ]
    
    for i in range(num_colors):
        range_idx = i % len(color_ranges)
        hue_min, hue_max = color_ranges[range_idx]
        
        hue_offset = (i // len(color_ranges)) * 0.618033988749895
        hue_range = hue_max - hue_min
        hue = hue_min + ((hue_offset % 1.0) * hue_range)
        
        saturation = 0.65 + (random.random() * 0.25)
        value = 0.80 + (random.random() * 0.15)
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    
    return colors

PINYIN_WORD_COLORS = generate_pinyin_word_colors()

class ClickThroughWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Configure window properties
        self.attributes('-alpha', WINDOW_OPACITY)
        self.attributes('-topmost', True)
        self.overrideredirect(True)
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f'{screen_width}x{screen_height}+0+0')
        self.configure(bg='black')
        self.wm_attributes("-transparent", "black")
        self.wm_attributes("-toolwindow", True)
        
        # Create layout
        self.setup_ui(screen_width, screen_height)
        
        # Set up Windows-specific transparency
        self.after(10, self.setup_window_transparency)

    def setup_ui(self, screen_width, screen_height):
        """
        Create three display areas:
        1. Top-left quadrant: English text + Pinyin directly below
        2. Top-right quadrant: Chinese Pinyin (colored) + AI Translation directly below (colored)
        3. Right side vertical (middle): Cascading individual Chinese words with Pinyin → CEDICT
        """
        
        # Top section container (for the two quadrants)
        top_section = tk.Frame(self, bg='black', height=200)
        top_section.pack(side=tk.TOP, fill=tk.X)
        top_section.pack_propagate(False)
        
        # === TOP LEFT QUADRANT: English + Pinyin ===
        self.top_left_frame = tk.Frame(top_section, bg='black')
        self.top_left_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10)
        
        # Container for vertical stacking
        left_container = tk.Frame(self.top_left_frame, bg='black')
        left_container.pack(expand=True)
        
        # English text label
        self.english_text_label = tk.Label(
            left_container,
            text='',
            font=DISPLAY_FONT,
            fg=TEXT_COLOR,
            bg='black',
            wraplength=screen_width//2 - 40,
            justify='center'
        )
        self.english_text_label.pack(pady=(0, 2))
        
        # Pinyin for English (directly below, no gap)
        self.english_pinyin_label = tk.Label(
            left_container,
            text='',
            font=DISPLAY_FONT,
            fg=PINYIN_COLOR,
            bg='black',
            wraplength=screen_width//2 - 40,
            justify='center'
        )
        self.english_pinyin_label.pack(pady=(0, 0))
        
        # === TOP RIGHT QUADRANT: Chinese Pinyin (colored) + AI Translation (colored) ===
        self.top_right_frame = tk.Frame(top_section, bg='black')
        self.top_right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10)
        
        # Container for vertical stacking
        right_container = tk.Frame(self.top_right_frame, bg='black')
        right_container.pack(expand=True)
        
        # Use Text widget for color-coded pinyin
        self.chinese_pinyin_text = tk.Text(
            right_container,
            font=DISPLAY_FONT,
            bg='black',
            borderwidth=0,
            highlightthickness=0,
            wrap='word',
            height=2,
            width=40
        )
        self.chinese_pinyin_text.pack(pady=(0, 2))
        self.chinese_pinyin_text.configure(state='disabled')
        
        # Configure color tags for pinyin words
        for i, color in enumerate(PINYIN_WORD_COLORS):
            self.chinese_pinyin_text.tag_configure(f'color_{i}', foreground=color)
        
        # AI Translation label (single green color like CEDICT)
        self.chinese_translation_label = tk.Label(
            right_container,
            text='',
            font=DISPLAY_FONT,
            fg=CEDICT_COLOR,  # Use same green as CEDICT
            bg='black',
            wraplength=screen_width//2 - 40,
            justify='center'
        )
        self.chinese_translation_label.pack(pady=(0, 0))
        
        # === RIGHT SIDE VERTICAL CASCADE (MIDDLE): Individual Chinese Words + CEDICT ===
        # Position at middle of screen instead of top
        self.cascade_frame = tk.Frame(self, bg='black', width=450)
        self.cascade_frame.place(x=screen_width-470, y=screen_height//3, width=450, height=screen_height//2)
        
        # Text widget for vertical cascade with scrolling
        self.cascade_text = tk.Text(
            self.cascade_frame,
            font=DISPLAY_FONT,
            bg='black',
            fg=PINYIN_COLOR,
            borderwidth=0,
            highlightthickness=0,
            wrap='none',  # Changed to 'none' to keep word pairs on single lines
            width=40
        )
        self.cascade_text.pack(expand=True, fill=tk.BOTH)
        
        # Configure tags for cascade display
        self.cascade_text.tag_configure('pinyin', foreground=PINYIN_COLOR)
        self.cascade_text.tag_configure('cedict', foreground=CEDICT_COLOR)
        
        # Disable text editing
        self.cascade_text.configure(state='disabled')

    def setup_window_transparency(self):
        """Use Windows-specific API to set click-through transparency"""
        try:
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            
            ex_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                hwnd, 
                GWL_EXSTYLE, 
                ex_style | WS_EX_LAYERED | WS_EX_TRANSPARENT
            )
            
            ctypes.windll.user32.SetLayeredWindowAttributes(
                hwnd,
                0,
                int(WINDOW_OPACITY * 255),
                LWA_ALPHA | LWA_COLORKEY
            )
            
            def make_all_children_noninteractive(parent_widget):
                for child in parent_widget.winfo_children():
                    if hasattr(child, 'winfo_id'):
                        child_hwnd = child.winfo_id() 
                        if child_hwnd:
                            child_style = ctypes.windll.user32.GetWindowLongW(child_hwnd, GWL_EXSTYLE)
                            ctypes.windll.user32.SetWindowLongW(
                                child_hwnd, 
                                GWL_EXSTYLE, 
                                child_style | WS_EX_TRANSPARENT | WS_EX_LAYERED
                            )
                            ctypes.windll.user32.SetLayeredWindowAttributes(
                                child_hwnd, 
                                0,
                                int(WINDOW_OPACITY * 255),
                                LWA_ALPHA | LWA_COLORKEY
                            )
                    make_all_children_noninteractive(child)
            
            make_all_children_noninteractive(self)
            
        except Exception as e:
            print(f"Error setting up transparency: {e}")


def update_displays(root, get_display_data_func):
    """Update all three display areas with the latest transcription data"""
    english_display, chinese_display = get_display_data_func()
    
    current_time = time.time()
    
    # Initialize tracking attributes if needed
    if not hasattr(root, "last_texts"):
        root.last_texts = {
            'english': "",
            'english_pinyin': "",
            'chinese_pinyin': "",
        }
    
    # === UPDATE TOP LEFT: English text + Pinyin (from segments) ===
    english_text = english_display.get("transcription", "")
    english_pinyin = english_display.get("pinyin", "")
    
    # Update if content changed
    if english_text != root.last_texts['english'] or english_pinyin != root.last_texts['english_pinyin']:
        root.english_text_label.config(text=english_text)
        root.english_pinyin_label.config(text=english_pinyin)
        root.last_texts['english'] = english_text
        root.last_texts['english_pinyin'] = english_pinyin
        if english_text:
            print(f"[DISPLAY] English: '{english_text}'")
    
    # === UPDATE TOP RIGHT: Chinese Pinyin (colored) + AI Translation (green) ===
    chinese_pinyin = chinese_display.get("pinyin", "")
    chinese_translation = chinese_display.get("translation", "")
    
    # Update if content changed
    if chinese_pinyin != root.last_texts['chinese_pinyin']:
        
        # Update colored pinyin
        if chinese_pinyin:
            pinyin_words = chinese_pinyin.split()
            root.chinese_pinyin_text.configure(state='normal')
            root.chinese_pinyin_text.delete('1.0', tk.END)
            for i, word in enumerate(pinyin_words):
                root.chinese_pinyin_text.insert(tk.END, word, f'color_{i % len(PINYIN_WORD_COLORS)}')
                if i < len(pinyin_words) - 1:
                    root.chinese_pinyin_text.insert(tk.END, " ")
            root.chinese_pinyin_text.configure(state='disabled')
        else:
            root.chinese_pinyin_text.configure(state='normal')
            root.chinese_pinyin_text.delete('1.0', tk.END)
            root.chinese_pinyin_text.configure(state='disabled')
        
        # Update colored translation (single green color)
        root.chinese_translation_label.config(text=chinese_translation if chinese_translation else "")
        
        root.last_texts['chinese_pinyin'] = chinese_pinyin
    
    # === UPDATE RIGHT SIDE CASCADE: Individual Chinese Words + CEDICT ===
    cascade_words = chinese_display.get("cascade_words", [])
    
    if cascade_words:
        root.cascade_text.configure(state='normal')
        root.cascade_text.delete('1.0', tk.END)
        
        for word_data in cascade_words:
            pinyin = word_data.get("pinyin", "")
            cedict_trans = word_data.get("cedict", "")
            
            if pinyin:
                root.cascade_text.insert(tk.END, pinyin, 'pinyin')
                if cedict_trans:
                    root.cascade_text.insert(tk.END, f"  →  {cedict_trans}", 'cedict')
                root.cascade_text.insert(tk.END, "\n")
        
        root.cascade_text.configure(state='disabled')
        root.cascade_text.see(tk.END)
    else:
        root.cascade_text.configure(state='normal')
        root.cascade_text.delete('1.0', tk.END)
        root.cascade_text.configure(state='disabled')
    
    # Continue updating
    root.after(100, lambda: update_displays(root, get_display_data_func))


def create_window(get_display_data_func):
    """Create and run the main window"""
    root = ClickThroughWindow()
    
    # Start display updates
    root.after(100, lambda: update_displays(root, get_display_data_func))
    
    return root