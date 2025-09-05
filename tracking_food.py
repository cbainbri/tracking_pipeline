#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IMPROVED Worm Tracker - With Trajectory-Aware Assignment AND Nose Detection
MAJOR IMPROVEMENTS:
- Trajectory prediction to prevent ID swapping at intersections
- Velocity-aware assignment scoring
- Configurable trajectory weight parameter
- Better handling of worm crossings
- FIXED UI scaling issues for different screen sizes
- NEW: Nose detection based on locomotion direction
"""

import os
import sys
import gc
import cv2
import random
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Dict, Tuple, Optional
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import threading
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


class WormTracker:
    """
    IMPROVED worm tracking system with trajectory-aware assignment and nose detection
    """

    def __init__(self):
        # Data & state
        self.image_dir: Optional[str] = None
        self.image_files: List[str] = []
        self.background: Optional[np.ndarray] = None
        self.sample_images: List[np.ndarray] = []
        self.sample_indices: List[int] = []
        self.processed_samples: List[np.ndarray] = []
        self.threshold_overlays: List[np.ndarray] = []

        # IMPROVED tracking parameters with trajectory awareness
        self.threshold_min: int = 12
        self.threshold_max: int = 255
        self.min_blob_size: int = 120
        self.max_blob_size: int = 2400
        self.max_distance: int = 75
        self.trajectory_weight: float = 0.7  # Trajectory prediction weight
        self.min_track_length: int = 50
        self.use_hungarian: bool = False  # Greedy is now default

        # NEW: Nose detection parameters
        self.nose_detection_enabled: bool = True
        self.nose_smoothing_frames: int = 2  # CHANGED: Reduced from 4 to 2 for more detections
        self.min_movement_threshold: float = 2.0  # Movement per frame in pixels

        # Tracking results (MODIFIED to include nose positions)
        self.tracks: Dict[int, List[Tuple[float, float, int]]] = {}  # centroid positions
        self.nose_tracks: Dict[int, List[Tuple[float, float, int]]] = {}  # nose positions
        self.track_data: List[Dict[str, float]] = []
        self.track_statistics: List[Dict[str, float]] = []

        # GUI components
        self.root: Optional[tk.Tk] = None
        self.notebook = None
        self.setup_tab = None
        self.threshold_tab = None
        self.results_tab = None

        # Widgets/vars created later
        self.bg_button = None
        self.bg_progress = None
        self.bg_display_frame = None
        self.dir_label = None
        self.start_tracking_button = None
        self.track_progress = None
        self.hist_fig: Optional[Figure] = None
        self.hist_canvas: Optional[FigureCanvasTkAgg] = None
        self.threshold_fig: Optional[Figure] = None
        self.threshold_canvas: Optional[FigureCanvasTkAgg] = None
        self.results_fig: Optional[Figure] = None
        self.results_canvas: Optional[FigureCanvasTkAgg] = None
        self.summary_text = None

        # Controls
        self.min_scale = None
        self.max_scale = None
        self.min_entry_var = None
        self.max_entry_var = None
        self.min_blob_var = None
        self.max_blob_var = None
        self.min_track_var = None
        self.max_distance_var = None
        self.trajectory_weight_var = None
        self.algo_var = None

        # NEW: Nose detection controls
        self.nose_detection_var = None
        self.nose_smoothing_var = None
        self.min_movement_var = None

        self.show_threshold_var = None
        self.image_var = None
        self.image_scale = None
        self.image_label = None

        self.bg_type_var = None
        self.show_detections_var = None
        self.show_noses_var = None  # NEW: Toggle nose visualization

        # Viewer/plot state
        self.current_image_idx: int = 0
        self._cached_track_colors = None

        # Threads
        self._background_thread: Optional[threading.Thread] = None
        self._tracking_thread: Optional[threading.Thread] = None

        # Build UI
        self.setup_gui()

    # ------------------------ GUI & Tabs ------------------------

    def setup_gui(self):
        """Initialize the main GUI with better scaling"""
        self.root = tk.Tk()
        self.root.title("IMPROVED Worm Tracker - Trajectory-Aware Assignment + Nose Detection")
        
        # Get screen dimensions for responsive sizing
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Scale based on 1920x1080 as baseline
        scale_factor = min(screen_width / 1920, screen_height / 1080)
        
        # Set window size with scaling
        window_width = int(1600 * scale_factor)
        window_height = int(1000 * scale_factor)
        self.root.geometry(f"{window_width}x{window_height}")
        
        # Set minimum size
        min_width = int(1400 * min(scale_factor, 1.0))
        min_height = int(900 * min(scale_factor, 1.0))
        self.root.minsize(min_width, min_height)

        # Resizable grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Ensure proper cleanup on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Tab 1: Setup and Background Subtraction
        self.setup_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.setup_tab, text="1. Setup & Background")
        self.create_setup_tab()

        # Tab 2: Thresholding and QC
        self.threshold_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.threshold_tab, text="2. Threshold & QC")
        self.create_threshold_tab()

        # Tab 3: Tracking Results
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="3. Tracking Results")
        self.create_results_tab()

    def on_closing(self):
        """Handle window closing properly with cleanup"""
        try:
            plt.close('all')
            self.root.quit()
            self.root.destroy()
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def create_setup_tab(self):
        """Create the setup and background subtraction tab"""
        # Directory selection
        dir_frame = ttk.LabelFrame(self.setup_tab, text="Image Directory Selection", padding=10)
        dir_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(dir_frame, text="Select Image Directory",
                   command=self.select_directory).pack(side='left')
        self.dir_label = ttk.Label(dir_frame, text="No directory selected")
        self.dir_label.pack(side='left', padx=(10, 0))

        # Background generation
        bg_frame = ttk.LabelFrame(self.setup_tab, text="Background Generation", padding=10)
        bg_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(bg_frame, text="Background will be generated from sampled images").pack(anchor='w')
        self.bg_button = ttk.Button(bg_frame, text="Generate Background",
                                    command=self.generate_background, state='disabled')
        self.bg_button.pack(side='left')

        self.bg_progress = ttk.Progressbar(bg_frame, mode='determinate')
        self.bg_progress.pack(side='left', padx=(10, 0), fill='x', expand=True)

        # Background display
        self.bg_display_frame = ttk.LabelFrame(self.setup_tab, text="Background Image", padding=10)
        self.bg_display_frame.pack(fill='both', expand=True, padx=10, pady=5)

    def create_threshold_tab(self):
        """Create the thresholding and QC tab with IMPROVED tracking parameters and nose detection"""
        # Use PanedWindow for resizable split between controls and viewer
        main_paned = ttk.PanedWindow(self.threshold_tab, orient='horizontal')
        main_paned.pack(fill='both', expand=True, padx=10, pady=5)

        # Left side: Controls with fixed width and internal scrolling
        left_container = ttk.Frame(main_paned)
        main_paned.add(left_container, weight=0)
        
        # Create scrollable area for controls
        control_canvas = tk.Canvas(left_container, width=420, highlightthickness=0)
        v_scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=control_canvas.yview)
        scrollable_frame = ttk.Frame(control_canvas)
        
        # Configure scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all"))
        )
        control_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        control_canvas.configure(yscrollcommand=v_scrollbar.set)
        
        # Pack scrolling components
        control_canvas.pack(side="left", fill="both", expand=True)
        v_scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            control_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        control_canvas.bind("<MouseWheel>", _on_mousewheel)

        # TOP SECTION: Sample generation and tracking
        top_section = ttk.LabelFrame(scrollable_frame, text="Sample Generation & Tracking", padding=10)
        top_section.pack(fill='x', pady=(0, 10))
        
        # Sample generation controls in one row
        sample_controls_frame = ttk.Frame(top_section)
        sample_controls_frame.pack(fill='x', pady=(0, 10))
        
        sample_button = ttk.Button(sample_controls_frame, text="Generate Samples",
                                   command=self.generate_samples, state='disabled')
        sample_button.pack(side='left', padx=(0, 10))

        resample_button = ttk.Button(sample_controls_frame, text="Re-sample Images",
                                     command=self.generate_samples, state='disabled')
        resample_button.pack(side='left')

        # Store references for state management
        self.sample_button2 = sample_button
        self.resample_button2 = resample_button
        
        # Tracking controls in same section
        tracking_controls = ttk.Frame(top_section)
        tracking_controls.pack(fill='x')
        
        # Algorithm selection
        algo_frame = ttk.Frame(tracking_controls)
        algo_frame.pack(fill='x', pady=(0, 5))
        ttk.Label(algo_frame, text="Algorithm:", font=('TkDefaultFont', 10, 'bold')).pack(anchor='w')

        algo_select_frame = ttk.Frame(algo_frame)
        algo_select_frame.pack(fill='x', pady=(2, 0))

        self.algo_var = tk.StringVar(value="Greedy")
        algo_combo = ttk.Combobox(algo_select_frame, textvariable=self.algo_var,
                                  values=["Greedy", "Hungarian"], width=15, state="readonly")
        algo_combo.pack(side='left')

        def show_algo_tooltip():
            tooltip_text = ("Greedy: Fast assignment, better for preventing ID swapping\n"
                            "Hungarian: Optimal assignment, may cause more ID swaps\n\n"
                            "Greedy is recommended for worm tracking!")
            messagebox.showinfo("Algorithm Info", tooltip_text)

        ttk.Button(algo_select_frame, text="?", width=3, command=show_algo_tooltip).pack(side='left', padx=(5, 0))
        
        # START TRACKING BUTTON
        self.start_tracking_button = ttk.Button(tracking_controls, text="BEGIN TRAJECTORY-AWARE TRACKING + NOSE DETECTION",
                                                command=self.start_tracking, state='disabled')
        self.start_tracking_button.pack(fill='x', pady=(10, 5))

        self.track_progress = ttk.Progressbar(tracking_controls, mode='determinate')
        self.track_progress.pack(fill='x')

        # Histogram section
        hist_frame = ttk.LabelFrame(scrollable_frame, text="Pixel Intensity Histogram", padding=5)
        hist_frame.pack(fill='x', pady=(0, 10))

        self.hist_fig = Figure(figsize=(4.5, 2.5))
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, hist_frame)
        self.hist_canvas.get_tk_widget().pack(fill='x')

        # Threshold sliders - COMPACT DESIGN
        slider_frame = ttk.Frame(hist_frame)
        slider_frame.pack(fill='x', pady=(5, 0))

        ttk.Label(slider_frame, text="Threshold Range:", font=('TkDefaultFont', 9, 'bold')).pack()

        # Min threshold - COMPACT
        min_frame = ttk.Frame(slider_frame)
        min_frame.pack(fill='x', pady=1)
        ttk.Label(min_frame, text="Min:", width=4).pack(side='left')

        ttk.Button(min_frame, text="◀", width=2,
                   command=lambda: self.adjust_threshold('min', -1)).pack(side='left')

        self.min_scale = tk.Scale(min_frame, from_=0, to=255, orient='horizontal',
                                  command=self.update_threshold, resolution=1, length=100)
        self.min_scale.set(self.threshold_min)
        self.min_scale.pack(side='left', fill='x', expand=True, padx=(1, 1))

        ttk.Button(min_frame, text="▶", width=2,
                   command=lambda: self.adjust_threshold('min', 1)).pack(side='left')

        self.min_entry_var = tk.StringVar(value=str(self.threshold_min))
        min_entry = ttk.Entry(min_frame, textvariable=self.min_entry_var, width=5)
        min_entry.pack(side='left', padx=(3, 0))
        min_entry.bind('<Return>', self.update_threshold_from_text)
        min_entry.bind('<FocusOut>', self.update_threshold_from_text)

        # Max threshold - COMPACT
        max_frame = ttk.Frame(slider_frame)
        max_frame.pack(fill='x', pady=1)
        ttk.Label(max_frame, text="Max:", width=4).pack(side='left')

        ttk.Button(max_frame, text="◀", width=2,
                   command=lambda: self.adjust_threshold('max', -1)).pack(side='left')

        self.max_scale = tk.Scale(max_frame, from_=0, to=255, orient='horizontal',
                                  command=self.update_threshold, resolution=1, length=100)
        self.max_scale.set(self.threshold_max)
        self.max_scale.pack(side='left', fill='x', expand=True, padx=(1, 1))

        ttk.Button(max_frame, text="▶", width=2,
                   command=lambda: self.adjust_threshold('max', 1)).pack(side='left')

        self.max_entry_var = tk.StringVar(value=str(self.threshold_max))
        max_entry = ttk.Entry(max_frame, textvariable=self.max_entry_var, width=5)
        max_entry.pack(side='left', padx=(3, 0))
        max_entry.bind('<Return>', self.update_threshold_from_text)
        max_entry.bind('<FocusOut>', self.update_threshold_from_text)

        # Blob size controls - COMPACT
        blob_frame = ttk.LabelFrame(scrollable_frame, text="Blob Size Filters", padding=5)
        blob_frame.pack(fill='x', pady=(0, 10))

        # Single row for blob controls
        blob_row1 = ttk.Frame(blob_frame)
        blob_row1.pack(fill='x', pady=(0, 3))
        ttk.Label(blob_row1, text="Min size:", width=8).pack(side='left')
        self.min_blob_var = tk.StringVar(value=str(self.min_blob_size))
        min_blob_entry = ttk.Entry(blob_row1, textvariable=self.min_blob_var, width=8)
        min_blob_entry.pack(side='left', padx=(0, 10))
        min_blob_entry.bind('<KeyRelease>', self.update_blob_params)
        ttk.Label(blob_row1, text="Max size:", width=8).pack(side='left')
        self.max_blob_var = tk.StringVar(value=str(self.max_blob_size))
        max_blob_entry = ttk.Entry(blob_row1, textvariable=self.max_blob_var, width=8)
        max_blob_entry.pack(side='left')
        max_blob_entry.bind('<KeyRelease>', self.update_blob_params)

        # Trajectory parameters - COMPACT
        track_frame = ttk.LabelFrame(scrollable_frame, text="Trajectory Parameters", padding=5)
        track_frame.pack(fill='x', pady=(0, 10))

        # Compact parameter layout
        param_row1 = ttk.Frame(track_frame)
        param_row1.pack(fill='x', pady=(0, 3))
        ttk.Label(param_row1, text="Max Distance:", width=12).pack(side='left')
        self.max_distance_var = tk.StringVar(value=str(self.max_distance))
        distance_entry = ttk.Entry(param_row1, textvariable=self.max_distance_var, width=8)
        distance_entry.pack(side='left')
        distance_entry.bind('<KeyRelease>', self.update_tracking_params)

        param_row2 = ttk.Frame(track_frame)
        param_row2.pack(fill='x', pady=(0, 3))
        ttk.Label(param_row2, text="Traj Weight:", width=12, font=('TkDefaultFont', 8, 'bold')).pack(side='left')
        self.trajectory_weight_var = tk.StringVar(value=str(self.trajectory_weight))
        trajectory_entry = ttk.Entry(param_row2, textvariable=self.trajectory_weight_var, width=8)
        trajectory_entry.pack(side='left')
        trajectory_entry.bind('<KeyRelease>', self.update_tracking_params)

        param_row3 = ttk.Frame(track_frame)
        param_row3.pack(fill='x', pady=(0, 3))
        ttk.Label(param_row3, text="Min Track Len:", width=12).pack(side='left')
        self.min_track_var = tk.StringVar(value=str(self.min_track_length))
        length_entry = ttk.Entry(param_row3, textvariable=self.min_track_var, width=8)
        length_entry.pack(side='left')
        length_entry.bind('<KeyRelease>', self.update_tracking_params)

        # NEW: Nose Detection Parameters
        nose_frame = ttk.LabelFrame(scrollable_frame, text="Nose Detection Parameters", padding=5)
        nose_frame.pack(fill='x', pady=(0, 10))

        # Nose detection enable/disable
        nose_enable_frame = ttk.Frame(nose_frame)
        nose_enable_frame.pack(fill='x', pady=(0, 3))
        self.nose_detection_var = tk.BooleanVar(value=self.nose_detection_enabled)
        nose_check = ttk.Checkbutton(nose_enable_frame, text="Enable Nose Detection",
                                     variable=self.nose_detection_var,
                                     command=self.update_nose_params)
        nose_check.pack(side='left')

        # Nose smoothing frames
        nose_row1 = ttk.Frame(nose_frame)
        nose_row1.pack(fill='x', pady=(0, 3))
        ttk.Label(nose_row1, text="Smooth Frames:", width=12).pack(side='left')
        self.nose_smoothing_var = tk.StringVar(value=str(self.nose_smoothing_frames))
        nose_smooth_entry = ttk.Entry(nose_row1, textvariable=self.nose_smoothing_var, width=8)
        nose_smooth_entry.pack(side='left')
        nose_smooth_entry.bind('<KeyRelease>', self.update_nose_params)

        # Min movement threshold
        nose_row2 = ttk.Frame(nose_frame)
        nose_row2.pack(fill='x', pady=(0, 3))
        ttk.Label(nose_row2, text="Min Movement:", width=12).pack(side='left')
        self.min_movement_var = tk.StringVar(value=str(self.min_movement_threshold))
        nose_movement_entry = ttk.Entry(nose_row2, textvariable=self.min_movement_var, width=8)
        nose_movement_entry.pack(side='left')
        nose_movement_entry.bind('<KeyRelease>', self.update_nose_params)

        # Explanation
        ttk.Label(nose_frame, text="Nose detection finds front of worm based on locomotion direction", 
                  font=('TkDefaultFont', 7), wraplength=350).pack(pady=(5, 0))

        # Right side: Image viewer
        viewer_frame = ttk.LabelFrame(main_paned, text="Sample Images Viewer", padding=10)
        main_paned.add(viewer_frame, weight=1)

        # Image navigation controls
        nav_frame = ttk.Frame(viewer_frame)
        nav_frame.pack(fill='x', pady=(0, 10))

        self.show_threshold_var = tk.BooleanVar(value=True)
        threshold_check = ttk.Checkbutton(nav_frame, text="Show Thresholding",
                                          variable=self.show_threshold_var,
                                          command=self.refresh_display)
        threshold_check.pack(side='left')

        ttk.Label(nav_frame, text="Navigate Images:").pack(side='left', padx=(20, 5))
        self.image_var = tk.IntVar(value=0)
        self.image_scale = tk.Scale(nav_frame, from_=0, to=4, orient='horizontal',
                                    variable=self.image_var, command=self.update_displayed_image)
        self.image_scale.pack(side='left', fill='x', expand=True, padx=(10, 0))
        self.image_scale.config(state='disabled')

        self.image_label = ttk.Label(nav_frame, text="Image 1 of 5")
        self.image_label.pack(side='right')

        # Large image display
        self.threshold_fig = Figure(figsize=(10, 8))
        self.threshold_canvas = FigureCanvasTkAgg(self.threshold_fig, viewer_frame)
        self.threshold_canvas.get_tk_widget().pack(fill='both', expand=True)

        self.current_image_idx = 0

    def create_results_tab(self):
        """Create the results display tab with nose visualization options"""
        summary_frame = ttk.LabelFrame(self.results_tab, text="IMPROVED Tracking Summary", padding=5)
        summary_frame.pack(fill='x', padx=10, pady=5)

        self.summary_text = tk.Text(summary_frame, height=12, wrap='word', font=('TkDefaultFont', 8))
        self.summary_text.pack(fill='x')

        # Export options
        export_frame = ttk.LabelFrame(self.results_tab, text="Export Options", padding=5)
        export_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(export_frame, text="Export Tracks CSV",
                   command=self.export_tracks).pack(side='left', padx=5)
        ttk.Button(export_frame, text="Export Simple CSV",
                   command=self.export_tracks_simple).pack(side='left', padx=5)
        ttk.Button(export_frame, text="Save Track Visualization",
                   command=self.save_visualization).pack(side='left', padx=5)
        ttk.Button(export_frame, text="Open Track Video Viewer",
                   command=self.launch_track_video_viewer).pack(side='left', padx=5)

        # Visualization options
        viz_frame = ttk.LabelFrame(self.results_tab, text="Visualization Options", padding=5)
        viz_frame.pack(fill='x', padx=10, pady=5)

        self.bg_type_var = tk.StringVar(value="Subtracted")
        ttk.Label(viz_frame, text="Background:").pack(side='left')
        bg_combo = ttk.Combobox(viz_frame, textvariable=self.bg_type_var,
                                values=["Subtracted", "Raw Image"], width=12, state="readonly")
        bg_combo.pack(side='left', padx=(5, 15))
        bg_combo.bind('<<ComboboxSelected>>', lambda e: self.update_visualization())

        self.show_detections_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(viz_frame, text="Show All Detections",
                        variable=self.show_detections_var,
                        command=self.update_visualization).pack(side='left', padx=(0, 15))

        # NEW: Show nose positions option
        self.show_noses_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(viz_frame, text="Show Nose Positions",
                        variable=self.show_noses_var,
                        command=self.update_visualization).pack(side='left', padx=(0, 15))

        ttk.Button(viz_frame, text="Debug Current Sample",
                   command=self.show_debug_detections).pack(side='left')

        ttk.Button(viz_frame, text="Show Frame with Nose Detection",
                   command=self.show_frame_with_nose_detection).pack(side='left', padx=(5, 0))

        # Results display
        results_display_frame = ttk.LabelFrame(self.results_tab, text="Track Visualization", padding=5)
        results_display_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.results_fig = Figure(figsize=(12, 6))
        self.results_canvas = FigureCanvasTkAgg(self.results_fig, results_display_frame)
        self.results_canvas.get_tk_widget().pack(fill='both', expand=True)

    # ------------------------ Setup actions ------------------------

    def select_directory(self):
        """Select directory containing images"""
        directory = filedialog.askdirectory(title="Select directory containing tracking images")
        if not directory:
            return

        self.image_dir = directory
        self.dir_label.config(text=f"Selected: {os.path.basename(directory)}")

        # Find image files (case-insensitive)
        image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp')
        self.image_files = []

        try:
            for file in os.listdir(directory):
                if file.lower().endswith(image_extensions):
                    self.image_files.append(os.path.join(directory, file))

            self.image_files.sort()

            if self.image_files:
                self.bg_button.config(state='normal')
                messagebox.showinfo("Images Found", f"Found {len(self.image_files)} images")
            else:
                messagebox.showerror("No Images", "No images found in selected directory")
        except Exception as e:
            messagebox.showerror("Error", f"Error reading directory: {e}")

    def generate_background(self):
        """Generate background image using memory-efficient sampling"""
        if not self.image_files:
            return

        def bg_worker():
            try:
                total_images = len(self.image_files)
                if total_images <= 50:
                    indices = list(range(total_images))
                elif total_images <= 200:
                    indices = np.linspace(0, total_images - 1, 50, dtype=int)
                else:
                    indices = np.linspace(0, total_images - 1, 75, dtype=int)

                self.root.after(0, lambda: self.bg_progress.config(maximum=len(indices)))
                sampled_files = [self.image_files[i] for i in indices]

                batch_size = 10
                all_backgrounds: List[np.ndarray] = []
                reference_shape = None

                count = 0
                for batch_start in range(0, len(sampled_files), batch_size):
                    batch_files = sampled_files[batch_start:batch_start + batch_size]
                    for img_path in batch_files:
                        try:
                            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            if img is None:
                                continue

                            if reference_shape is None:
                                reference_shape = img.shape
                            elif img.shape != reference_shape:
                                img = cv2.resize(img, (reference_shape[1], reference_shape[0]))

                            all_backgrounds.append(img.astype(np.float32))
                            count += 1
                            self.root.after(0, lambda p=count: self.bg_progress.config(value=p))
                        except Exception as e:
                            print(f"Error loading image {img_path}: {e}")
                            continue

                    gc.collect()

                if all_backgrounds:
                    if len(all_backgrounds) > 30:
                        idx = np.random.choice(len(all_backgrounds), 30, replace=False)
                        sampled_backgrounds = [all_backgrounds[i] for i in idx]
                        self.background = np.median(sampled_backgrounds, axis=0).astype(np.uint8)
                    else:
                        self.background = np.median(all_backgrounds, axis=0).astype(np.uint8)

                    self.root.after(0, self.display_background)
                    self.root.after(0, lambda: self.sample_button2.config(state='normal'))
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Background Complete",
                        f"Background generated from {len(all_backgrounds)} sampled images.\n"
                        f"Next: Go to 'Threshold & QC' tab and click 'Generate Samples'."
                    ))
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", "No valid images could be loaded"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Background generation failed: {e}"))

        self._background_thread = threading.Thread(target=bg_worker, daemon=True)
        self._background_thread.start()

    def display_background(self):
        """Display the generated background"""
        if self.background is None:
            return

        for widget in self.bg_display_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.imshow(self.background, cmap='gray')
        ax.set_title("Generated Background")
        ax.axis('off')

        canvas = FigureCanvasTkAgg(fig, self.bg_display_frame)
        canvas.get_tk_widget().pack(fill='both', expand=True)
        canvas.draw()

    def generate_samples(self):
        """Generate sample images for QC from entire dataset"""
        if not self.image_files or self.background is None:
            return

        total_images = len(self.image_files)
        n_samples = min(5, total_images)
        self.sample_indices = random.sample(range(total_images), n_samples)

        self.sample_images = []
        self.processed_samples = []
        self.threshold_overlays = []

        for idx in self.sample_indices:
            img_path = self.image_files[idx]
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Shape/dtype align with background
            if img.shape != self.background.shape:
                img = cv2.resize(img, (self.background.shape[1], self.background.shape[0]))
            if img.dtype != self.background.dtype:
                img = img.astype(self.background.dtype, copy=False)

            subtracted = cv2.absdiff(self.background, img)

            self.sample_images.append(img)
            self.processed_samples.append(subtracted)

        # Always update threshold overlays after generating samples
        self.update_threshold_overlays()

        # Update GUI
        self.display_samples_and_histogram()
        self.resample_button2.config(state='normal')
        self.start_tracking_button.config(state='normal')
        self.image_scale.config(state='normal', to=len(self.processed_samples) - 1)

        sampled_frames = [f"Frame {idx + 1}" for idx in sorted(self.sample_indices)]
        print(f"Generated samples from: {', '.join(sampled_frames)} (out of {total_images} total frames)")

    # ------------------------ QC & Thresholding ------------------------

    def apply_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply threshold + blob-size filtering"""
        if image is None or image.size == 0:
            return np.zeros_like(image) if image is not None else np.zeros((100, 100), dtype=np.uint8)

        try:
            min_thresh = max(0, min(255, self.threshold_min))
            max_thresh = max(min_thresh, min(255, self.threshold_max))
            min_blob = max(1, self.min_blob_size)
            max_blob = max(min_blob, self.max_blob_size)

            mask = (image >= min_thresh) & (image <= max_thresh)
            thresholded = mask.astype(np.uint8) * 255

            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered = np.zeros_like(thresholded)
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_blob <= area <= max_blob:
                    cv2.fillPoly(filtered, [contour], 255)

            return filtered
        except Exception as e:
            print(f"Error in apply_threshold: {e}")
            return np.zeros_like(image)

    def update_threshold_overlays(self):
        """Update overlays for all samples"""
        if not self.processed_samples:
            return

        self.threshold_overlays = []
        for sample in self.processed_samples:
            thresholded = self.apply_threshold(sample)
            rgb_image = np.stack([sample, sample, sample], axis=2)
            mask = thresholded > 0
            rgb_image[mask] = [255, 100, 100]  # red overlay
            self.threshold_overlays.append(rgb_image)

    def refresh_display(self):
        """Force refresh of all displays"""
        if self.processed_samples:
            self.update_threshold_overlays()
            self.display_histogram()
            self.display_single_image()

    def display_single_image(self):
        """Display one sample image with proper overlay handling"""
        if not self.processed_samples or self.current_image_idx >= len(self.processed_samples):
            return

        self.threshold_fig.clear()
        sample = self.processed_samples[self.current_image_idx]
        ax = self.threshold_fig.add_subplot(1, 1, 1)

        if self.show_threshold_var.get():
            # Show with threshold overlay
            if (self.current_image_idx < len(self.threshold_overlays) and
                self.threshold_overlays[self.current_image_idx] is not None):
                ax.imshow(self.threshold_overlays[self.current_image_idx])
                title = f"With Red Threshold Overlay - Sample {self.current_image_idx + 1}"
            else:
                # Fallback: generate overlay on the fly
                thresholded = self.apply_threshold(sample)
                rgb_image = np.stack([sample, sample, sample], axis=2)
                mask = thresholded > 0
                rgb_image[mask] = [255, 100, 100]
                ax.imshow(rgb_image)
                title = f"With Red Threshold Overlay - Sample {self.current_image_idx + 1}"
        else:
            # Show without overlay
            ax.imshow(sample, cmap='gray')
            title = f"Background Subtracted - Sample {self.current_image_idx + 1}"

        ax.set_title(title, fontsize=14)
        ax.axis('off')
        self.threshold_canvas.draw()

    def update_displayed_image(self, _event=None):
        """Update displayed image when slider changes"""
        if not self.processed_samples:
            return
        self.current_image_idx = self.image_var.get()
        self.image_label.config(text=f"Image {self.current_image_idx + 1} of {len(self.processed_samples)}")
        self.display_single_image()

    def display_histogram(self):
        """Display histogram in histogram panel"""
        if not self.processed_samples:
            return
        self.hist_fig.clear()
        ax = self.hist_fig.add_subplot(1, 1, 1)
        all_pixels = np.concatenate([sample.flatten() for sample in self.processed_samples])
        ax.hist(all_pixels, bins=50, alpha=0.7, color='blue')
        ax.axvline(self.threshold_min, color='red', linestyle='--', label=f'Min: {self.threshold_min}')
        ax.axvline(self.threshold_max, color='red', linestyle='--', label=f'Max: {self.threshold_max}')
        ax.set_title("Pixel Intensity Histogram")
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Frequency")
        ax.legend()
        self.hist_canvas.draw()

    def display_samples_and_histogram(self):
        """Refresh both histogram and current image"""
        if not self.processed_samples:
            return
        self.display_histogram()
        self.display_single_image()

    def update_threshold_from_text(self, _event=None):
        """Sync threshold text boxes → sliders → display"""
        try:
            min_val = int(self.min_entry_var.get())
            max_val = int(self.max_entry_var.get())
            min_val = max(0, min(255, min_val))
            max_val = max(0, min(255, max_val))

            # Update internal values
            self.threshold_min = min_val
            self.threshold_max = max_val

            # Update sliders
            self.min_scale.set(min_val)
            self.max_scale.set(max_val)

            # Force refresh display
            self.refresh_display()
        except ValueError:
            self.min_entry_var.set(str(self.threshold_min))
            self.max_entry_var.set(str(self.threshold_max))

    def adjust_threshold(self, slider_type: str, increment: int):
        """Button-based nudging of thresholds"""
        if slider_type == 'min':
            new_val = max(0, min(255, self.threshold_min + increment))
            self.threshold_min = new_val
            self.min_scale.set(new_val)
            self.min_entry_var.set(str(new_val))
        elif slider_type == 'max':
            new_val = max(0, min(255, self.threshold_max + increment))
            self.threshold_max = new_val
            self.max_scale.set(new_val)
            self.max_entry_var.set(str(new_val))

        # Force refresh display
        self.refresh_display()

    def update_threshold(self, _event=None):
        """Commit threshold slider changes and update display"""
        # Get values from sliders
        self.threshold_min = self.min_scale.get()
        self.threshold_max = self.max_scale.get()

        # Update text boxes
        if self.min_entry_var:
            self.min_entry_var.set(str(self.threshold_min))
        if self.max_entry_var:
            self.max_entry_var.set(str(self.threshold_max))

        # Force refresh display
        self.refresh_display()

    def update_blob_params(self, _event=None):
        """Update blob size params"""
        try:
            self.min_blob_size = int(self.min_blob_var.get())
            self.max_blob_size = int(self.max_blob_var.get())

            if self.min_blob_size < 0:
                self.min_blob_size = 0
                self.min_blob_var.set("0")
            if self.max_blob_size < self.min_blob_size:
                self.max_blob_size = self.min_blob_size + 1
                self.max_blob_var.set(str(self.max_blob_size))

            # Force refresh display
            self.refresh_display()
        except ValueError:
            self.min_blob_var.set(str(self.min_blob_size))
            self.max_blob_var.set(str(self.max_blob_size))

    def update_tracking_params(self, _event=None):
        """Update tracking parameters including trajectory weight"""
        try:
            # Update min track length
            min_track = int(self.min_track_var.get())
            if min_track < 1:
                min_track = 1
            elif min_track > 1000:
                min_track = 1000
            self.min_track_length = min_track
            self.min_track_var.set(str(min_track))

            # Update max distance
            max_dist = int(self.max_distance_var.get())
            if max_dist < 1:
                max_dist = 1
            elif max_dist > 1000:
                max_dist = 1000
            self.max_distance = max_dist
            self.max_distance_var.set(str(max_dist))

            # Update trajectory weight
            traj_weight = float(self.trajectory_weight_var.get())
            if traj_weight < 0.0:
                traj_weight = 0.0
            elif traj_weight > 1.0:
                traj_weight = 1.0
            self.trajectory_weight = traj_weight
            self.trajectory_weight_var.set(str(traj_weight))

        except ValueError:
            self.min_track_var.set(str(self.min_track_length))
            self.max_distance_var.set(str(self.max_distance))
            self.trajectory_weight_var.set(str(self.trajectory_weight))

    def update_nose_params(self, _event=None):
        """Update nose detection parameters with immediate validation and debug output"""
        try:
            # Update nose detection enabled
            self.nose_detection_enabled = self.nose_detection_var.get()

            # Update nose smoothing frames with immediate validation
            smooth_frames = int(self.nose_smoothing_var.get())
            if smooth_frames < 2:
                smooth_frames = 2
                self.nose_smoothing_var.set(str(smooth_frames))
            elif smooth_frames > 10:
                smooth_frames = 10
                self.nose_smoothing_var.set(str(smooth_frames))
            
            # FIXED: Immediately update internal parameter
            self.nose_smoothing_frames = smooth_frames

            # Update min movement threshold with immediate validation
            min_movement = float(self.min_movement_var.get())
            if min_movement < 0.1:
                min_movement = 0.1
                self.min_movement_var.set(str(min_movement))
            elif min_movement > 50.0:
                min_movement = 50.0
                self.min_movement_var.set(str(min_movement))
            
            # FIXED: Immediately update internal parameter
            self.min_movement_threshold = min_movement

            # ENHANCED DEBUG: Show all parameter updates
            print(f"NOSE PARAMS UPDATED - Smoothing frames: {self.nose_smoothing_frames}, Movement threshold: {self.min_movement_threshold}, Enabled: {self.nose_detection_enabled}")

        except ValueError as e:
            print(f"Error updating nose parameters: {e}")
            # Reset to current internal values on error
            self.nose_detection_var.set(self.nose_detection_enabled)
            self.nose_smoothing_var.set(str(self.nose_smoothing_frames))
            self.min_movement_var.set(str(self.min_movement_threshold))

    def show_debug_detections(self):
        """Show all detected blobs on current sample for debugging"""
        if not self.processed_samples or self.current_image_idx >= len(self.processed_samples):
            messagebox.showwarning("No Sample", "Generate samples first in the QC tab")
            return

        sample = self.processed_samples[self.current_image_idx]
        thresholded = self.apply_threshold(sample)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        debug_image = cv2.cvtColor(sample, cv2.COLOR_GRAY2RGB)

        accepted_count = 0
        rejected_small = 0
        rejected_large = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            if area < self.min_blob_size:
                color = (255, 0, 0)  # red
                rejected_small += 1
                label = f"S:{area:.0f}"
            elif area > self.max_blob_size:
                color = (255, 165, 0)  # orange
                rejected_large += 1
                label = f"L:{area:.0f}"
            else:
                color = (0, 255, 0)  # green
                accepted_count += 1
                label = f"OK:{area:.0f}"

            cv2.drawContours(debug_image, [contour], -1, color, 2)
            cv2.circle(debug_image, (int(cx), int(cy)), 3, color, -1)
            cv2.putText(debug_image, label, (int(cx) + 5, int(cy) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        debug_window = tk.Toplevel(self.root)
        debug_window.title(f"Debug Detections - Sample {self.current_image_idx + 1}")
        debug_window.geometry("800x600")

        debug_fig = Figure(figsize=(10, 8))
        debug_ax = debug_fig.add_subplot(1, 1, 1)
        debug_ax.imshow(debug_image)
        debug_ax.set_title(f"All Detections - Sample {self.current_image_idx + 1}\n"
                           f"Accepted: {accepted_count} | Too Small: {rejected_small} | Too Large: {rejected_large}")
        debug_ax.axis('off')

        debug_canvas = FigureCanvasTkAgg(debug_fig, debug_window)
        debug_canvas.get_tk_widget().pack(fill='both', expand=True)
        debug_canvas.draw()

        def close_debug():
            try:
                plt.close(debug_fig)
            except Exception:
                pass
            debug_window.destroy()

        debug_window.protocol("WM_DELETE_WINDOW", close_debug)

        print("DEBUG DETECTION STATS:")
        print(f"  Total contours found: {len(contours)}")
        print(f"  Accepted (green): {accepted_count}")
        print(f"  Rejected small (red): {rejected_small}")
        print(f"  Rejected large (orange): {rejected_large}")
    def show_frame_with_nose_detection(self):
        """Show a specific frame with individual worm blobs, centroids, and nose positions"""
        if not self.tracks or not self.image_files:
            messagebox.showwarning("No Data", "No tracking data available. Run tracking first.")
            return
        
        # Create a dialog to select which frame to display
        frame_window = tk.Toplevel(self.root)
        frame_window.title("Select Frame for Nose Detection View")
        frame_window.geometry("300x150")
        
        ttk.Label(frame_window, text="Enter frame number to display:").pack(pady=10)
        
        frame_var = tk.StringVar(value="50")
        frame_entry = ttk.Entry(frame_window, textvariable=frame_var, width=10)
        frame_entry.pack(pady=5)
        
        ttk.Label(frame_window, text=f"(Range: 1 to {len(self.image_files)})").pack(pady=2)
        
        def show_selected_frame():
            try:
                frame_num = int(frame_var.get()) - 1  # Convert to 0-based index
                if frame_num < 0 or frame_num >= len(self.image_files):
                    messagebox.showerror("Invalid Frame", f"Frame must be between 1 and {len(self.image_files)}")
                    return
                
                self.display_frame_with_nose_detection(frame_num)
                frame_window.destroy()
                
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid frame number")
        
        ttk.Button(frame_window, text="Show Frame", command=show_selected_frame).pack(pady=10)
        
        # Center the window
        frame_window.transient(self.root)
        frame_window.grab_set()
    
    def display_frame_with_nose_detection(self, frame_idx: int):
        """Display a specific frame showing individual blobs with centroids and nose positions"""
        try:
            # Load and process the frame
            img, subtracted = self.load_and_process_frame(frame_idx)
            if img is None or subtracted is None:
                messagebox.showerror("Error", f"Could not load frame {frame_idx + 1}")
                return
            
            # Apply thresholding to get blobs
            thresholded = self.apply_threshold(subtracted)
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find all centroids for this frame from tracking data
            frame_centroids = {}
            frame_noses = {}
            
            for track_id, positions in self.tracks.items():
                for pos in positions:
                    if pos[2] == frame_idx:  # pos = (x, y, frame)
                        frame_centroids[track_id] = (pos[0], pos[1])
                        break
            
            # Find nose positions for this frame
            if self.nose_tracks:
                for track_id, nose_positions in self.nose_tracks.items():
                    for nose_pos in nose_positions:
                        if nose_pos[2] == frame_idx:  # nose_pos = (x, y, frame)
                            frame_noses[track_id] = (nose_pos[0], nose_pos[1])
                            break
            
            # Color palette for tracks
            colors = plt.cm.tab10(np.linspace(0, 1, max(len(frame_centroids), 1)))
            
            # Create display window with zoom functionality and controls
            display_window = tk.Toplevel(self.root)
            display_window.title(f"Frame {frame_idx + 1} - Blobs with Nose Detection (Zoomable)")
            display_window.geometry("1200x900")
            
            # Create control frame at top
            control_frame = ttk.Frame(display_window)
            control_frame.pack(fill='x', padx=10, pady=5)
            
            # Add checkbox to toggle overlays
            show_overlays_var = tk.BooleanVar(value=True)
            overlay_checkbox = ttk.Checkbutton(control_frame, text="Show Centroids & Nose Positions", 
                                               variable=show_overlays_var)
            overlay_checkbox.pack(side='left')
            
            # Create figure with zoom and pan capabilities
            fig = Figure(figsize=(14, 12))
            ax = fig.add_subplot(111)
            
            def update_display():
                """Update the display based on checkbox state"""
                # Create fresh visualization image
                display_image = cv2.cvtColor(subtracted, cv2.COLOR_GRAY2RGB)
                
                # Always draw blob outlines in gray
                for i, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if self.min_blob_size <= area <= self.max_blob_size:
                        # Draw blob outline in white for better visibility
                        cv2.drawContours(display_image, [contour], -1, (200, 200, 200), 2)
                
                # Only draw overlays if checkbox is checked
                if show_overlays_var.get():
                    # Overlay tracked centroids and noses
                    for i, (track_id, centroid_pos) in enumerate(frame_centroids.items()):
                        color = colors[i % len(colors)]
                        color_bgr = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
                        
                        cx, cy = int(centroid_pos[0]), int(centroid_pos[1])
                        
                        # Draw centroid as filled circle
                        cv2.circle(display_image, (cx, cy), 4, color_bgr, -1)
                        cv2.circle(display_image, (cx, cy), 4, (255, 255, 255), 1)  # white outline
                        
                        # Draw track ID
                        cv2.putText(display_image, f"T{track_id}", (cx + 8, cy - 8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Draw nose position if available
                        if track_id in frame_noses:
                            nose_x, nose_y = int(frame_noses[track_id][0]), int(frame_noses[track_id][1])
                            
                            # Draw nose as much larger, highly visible triangle
                            triangle_size = 20  # Much larger
                            triangle_points = np.array([
                                [nose_x, nose_y - triangle_size],
                                [nose_x - triangle_size//2, nose_y + triangle_size//2],
                                [nose_x + triangle_size//2, nose_y + triangle_size//2]
                            ])
                            # Fill triangle with bright color
                            cv2.fillPoly(display_image, [triangle_points], (0, 255, 255))  # Bright cyan
                            cv2.polylines(display_image, [triangle_points], True, (0, 0, 0), 3)  # Thick black outline
                            
                            # Draw thicker line from centroid to nose
                            cv2.line(display_image, (cx, cy), (nose_x, nose_y), (255, 255, 255), 3)  # Thick white line
                            
                            # Label nose with larger, more contrasting text
                            cv2.putText(display_image, f"NOSE", (nose_x + 25, nose_y + 6),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)  # Black text with thick outline
                            cv2.putText(display_image, f"NOSE", (nose_x + 25, nose_y + 6),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)  # White text on top
                
                # Update the image
                ax.clear()
                ax.imshow(display_image)
                
                # Update title based on overlay state
                if show_overlays_var.get():
                    title = f"Frame {frame_idx + 1} - Blobs with Centroids and Nose Positions\n"
                    title += f"Tracked worms: {len(frame_centroids)}, Nose detections: {len(frame_noses)}"
                else:
                    title = f"Frame {frame_idx + 1} - Clean Blob Outlines Only\n"
                    title += f"Detected blobs: {len([c for c in contours if self.min_blob_size <= cv2.contourArea(c) <= self.max_blob_size])}"
                
                if self.nose_detection_enabled:
                    title += f"\nNose parameters: {self.nose_smoothing_frames} smoothing frames, {self.min_movement_threshold} px/frame"
                else:
                    title += "\nNose detection was disabled during tracking"
                title += "\nMouse wheel to zoom, click and drag to pan, double-click to reset view"
                
                ax.set_title(title, fontsize=10)
                ax.axis('off')
                canvas.draw()
            
            # Connect checkbox to update function
            overlay_checkbox.configure(command=update_display)
            
            canvas = FigureCanvasTkAgg(fig, display_window)
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
            # Add navigation toolbar for zoom/pan
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            toolbar_frame = ttk.Frame(display_window)
            toolbar_frame.pack(fill='x', side='bottom')
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            
            # Initial display
            update_display()
            
            # Store initial view limits for reset
            initial_xlim = ax.get_xlim()
            initial_ylim = ax.get_ylim()
            
            # Add mouse wheel zoom functionality
            def on_scroll(event):
                if event.inaxes != ax:
                    return
                
                # Get current axis limits
                cur_xlim = ax.get_xlim()
                cur_ylim = ax.get_ylim()
                cur_xrange = (cur_xlim[1] - cur_xlim[0]) * 0.5
                cur_yrange = (cur_ylim[1] - cur_ylim[0]) * 0.5
                
                # Get mouse position in data coordinates
                xdata = event.xdata
                ydata = event.ydata
                
                if event.button == 'up':
                    # Zoom in
                    scale_factor = 0.8
                elif event.button == 'down':
                    # Zoom out
                    scale_factor = 1.25
                else:
                    return
                
                # Calculate new limits
                new_xrange = cur_xrange * scale_factor
                new_yrange = cur_yrange * scale_factor
                
                ax.set_xlim([xdata - new_xrange, xdata + new_xrange])
                ax.set_ylim([ydata - new_yrange, ydata + new_yrange])
                
                canvas.draw()
            
            # Add double-click to reset view
            def on_double_click(event):
                if event.inaxes == ax and event.dblclick:
                    ax.set_xlim(initial_xlim)
                    ax.set_ylim(initial_ylim)
                    canvas.draw()
            
            # Add click-to-zoom on worms (only when overlays are shown)
            def on_click_zoom(event):
                if event.inaxes != ax or event.dblclick or not show_overlays_var.get():
                    return
                
                # Find nearest centroid
                if not frame_centroids:
                    return
                
                click_x, click_y = event.xdata, event.ydata
                if click_x is None or click_y is None:
                    return
                
                # Find closest centroid
                min_dist = float('inf')
                closest_centroid = None
                
                for track_id, (cx, cy) in frame_centroids.items():
                    dist = np.sqrt((click_x - cx)**2 + (click_y - cy)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_centroid = (cx, cy)
                
                # Zoom in on closest centroid if within reasonable distance
                if closest_centroid and min_dist < 100:  # Within 100 pixels
                    zoom_size = 80  # Show 160x160 pixel area
                    cx, cy = closest_centroid
                    ax.set_xlim([cx - zoom_size, cx + zoom_size])
                    ax.set_ylim([cy + zoom_size, cy - zoom_size])  # Inverted for image coordinates
                    canvas.draw()
            
            # Connect event handlers
            canvas.mpl_connect('scroll_event', on_scroll)
            canvas.mpl_connect('button_press_event', on_double_click)
            canvas.mpl_connect('button_press_event', on_click_zoom)
            
            def close_display():
                try:
                    plt.close(fig)
                except Exception:
                    pass
                display_window.destroy()
            
            display_window.protocol("WM_DELETE_WINDOW", close_display)
            
            print(f"FRAME {frame_idx + 1} ANALYSIS:")
            print(f"  Tracked centroids: {len(frame_centroids)}")
            print(f"  Nose detections: {len(frame_noses)}")
            print(f"  Nose success rate: {len(frame_noses)/len(frame_centroids)*100:.1f}%" if frame_centroids else "  No tracked worms in this frame")
            print("  DISPLAY CONTROLS:")
            print("    - Checkbox: Toggle overlay visibility")
            print("    - Mouse wheel: zoom in/out")
            print("    - Click and drag: pan around")
            print("    - Double-click: reset to full view")
            print("    - Single click near worm: zoom to that worm (when overlays shown)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display frame: {e}")

    # ------------------------ NEW: Nose Detection Methods ------------------------

    def calculate_locomotion_direction(self, track_positions: List[Tuple[float, float, int]]) -> Optional[Tuple[float, float]]:
        """Calculate the direction of locomotion from recent track positions"""
        # DEBUG: Show what smoothing value is actually being used
        if len(track_positions) == 2:  # Only print on first possible nose detection
            print(f"DEBUG: Using nose_smoothing_frames = {self.nose_smoothing_frames}, track length = {len(track_positions)}")
        
        if len(track_positions) < 2:
            return None
        
        # FIXED: Require minimum track length based on smoothing frames setting
        min_required_positions = max(2, self.nose_smoothing_frames)
        if len(track_positions) < min_required_positions:
            return None
        
        # Use the last few positions for direction calculation
        recent_positions = track_positions[-min(self.nose_smoothing_frames, len(track_positions)):]
        
        if len(recent_positions) < 2:
            return None
        
        # Calculate velocity vectors between consecutive points
        velocities = []
        for i in range(1, len(recent_positions)):
            prev_x, prev_y, prev_f = recent_positions[i-1]
            curr_x, curr_y, curr_f = recent_positions[i]
            
            # Skip if frames are too far apart (gap in tracking)
            frame_diff = curr_f - prev_f
            if frame_diff > 5:
                continue
            
            # FIXED: Apply movement filter to raw distance moved per frame
            if frame_diff > 0:
                # Calculate raw distance moved
                raw_distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                distance_per_frame = raw_distance / frame_diff
                
                # Check if movement per frame is significant
                if distance_per_frame >= self.min_movement_threshold:
                    # Calculate velocity for direction
                    vx = (curr_x - prev_x) / frame_diff
                    vy = (curr_y - prev_y) / frame_diff
                    velocities.append((vx, vy))
        
        if not velocities:
            return None
        
        # Calculate average direction
        avg_vx = np.mean([v[0] for v in velocities])
        avg_vy = np.mean([v[1] for v in velocities])
        
        # Normalize the direction vector
        direction_magnitude = np.sqrt(avg_vx**2 + avg_vy**2)
        if direction_magnitude < 0.1:  # Very small movement
            return None
            
        normalized_direction = (avg_vx / direction_magnitude, avg_vy / direction_magnitude)
        return normalized_direction

    def find_nose_position(self, contour: np.ndarray, locomotion_direction: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Find the nose position with sub-pixel precision using centroid of front-most region"""
        if contour is None or len(contour) < 3:
            return None
        
        if locomotion_direction is None:
            return None
        
        try:
            # Get all contour points as float
            points = contour.reshape(-1, 2).astype(np.float64)
            
            # Project each point onto the locomotion direction vector
            direction_vector = np.array(locomotion_direction, dtype=np.float64)
            projections = np.dot(points, direction_vector)
            
            # Find maximum projection
            max_projection = np.max(projections)
            
            # FIXED: Select points within a small range of the maximum projection
            # This creates a "front region" rather than a single point
            projection_tolerance = 1.0  # pixels - adjust this for more/less precision
            front_mask = projections >= (max_projection - projection_tolerance)
            front_points = points[front_mask]
            
            if len(front_points) == 0:
                return None
            
            # Calculate the centroid of the front region points
            # This gives us sub-pixel precision like the main centroid calculation
            nose_x = np.mean(front_points[:, 0])
            nose_y = np.mean(front_points[:, 1])
            
            return (round(float(nose_x), 4), round(float(nose_y), 4))
        
        except Exception as e:
            print(f"Error in find_nose_position: {e}")
            return None

    def detect_nose_for_track(self, track_id: int, active_tracks: Dict, contours: List[np.ndarray], centroids: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """Detect nose position for a specific track"""
        if not self.nose_detection_enabled:
            return None
            
        if track_id not in active_tracks:
            return None
        
        track_data = active_tracks[track_id]
        if 'positions' not in track_data or len(track_data['positions']) < 2:
            return None
        
        # Calculate locomotion direction
        locomotion_direction = self.calculate_locomotion_direction(track_data['positions'])
        if locomotion_direction is None:
            return None
        
        # Find the contour that corresponds to this track's current centroid
        if 'current_centroid_idx' not in track_data:
            return None
        
        centroid_idx = track_data['current_centroid_idx']
        if centroid_idx < 0 or centroid_idx >= len(contours):
            return None
        
        # Get the corresponding contour
        contour = contours[centroid_idx]
        
        # Find nose position
        nose_position = self.find_nose_position(contour, locomotion_direction)
        return nose_position

    # ------------------------ IMPROVED Tracking with Trajectory Awareness and Nose Detection ------------------------

    def predict_next_position(self, track_positions: List[Tuple[float, float, int]], min_frames: int = 3) -> Optional[Tuple[float, float]]:
        """Predict next position based on recent trajectory"""
        if len(track_positions) < min_frames:
            return None

        # Use last few positions for velocity calculation
        recent_positions = track_positions[-min_frames:]

        # Calculate average velocity
        velocities = []
        for i in range(1, len(recent_positions)):
            prev_x, prev_y, prev_f = recent_positions[i-1]
            curr_x, curr_y, curr_f = recent_positions[i]

            # Account for frame gaps
            frame_diff = curr_f - prev_f
            if frame_diff > 0:
                vx = (curr_x - prev_x) / frame_diff
                vy = (curr_y - prev_y) / frame_diff
                velocities.append((vx, vy))

        if not velocities:
            return None

        # Average velocity
        avg_vx = np.mean([v[0] for v in velocities])
        avg_vy = np.mean([v[1] for v in velocities])

        # Predict next position
        last_x, last_y, last_f = track_positions[-1]
        predicted_x = last_x + avg_vx
        predicted_y = last_y + avg_vy

        return (predicted_x, predicted_y)

    def assign_tracks_with_trajectory(self, active_tracks: Dict, centroids: List[Tuple[float, float]]) -> Dict[int, int]:
        """IMPROVED: Enhanced track assignment with trajectory prediction"""
        if not active_tracks or not centroids:
            return {}

        # Get track info
        track_positions = []
        track_ids = []
        track_predictions = []

        for track_id, track_data in active_tracks.items():
            if track_data['positions']:
                last_pos = track_data['positions'][-1]
                track_positions.append([last_pos[0], last_pos[1]])
                track_ids.append(track_id)

                # Get trajectory prediction
                predicted_pos = self.predict_next_position(track_data['positions'])
                track_predictions.append(predicted_pos)

        if not track_positions:
            return {}

        # Create enhanced distance matrix
        track_positions_array = np.array(track_positions)
        centroid_positions_array = np.array(centroids)
        distance_matrix = cdist(track_positions_array, centroid_positions_array)

        # FIXED: Better trajectory integration
        if self.trajectory_weight > 0:
            trajectory_matrix = np.full_like(distance_matrix, np.inf)

            for i, predicted_pos in enumerate(track_predictions):
                if predicted_pos is not None:
                    pred_distances = cdist([predicted_pos], centroid_positions_array)[0]
                    # FIXED: Only use trajectory if it's reasonable
                    for j in range(len(pred_distances)):
                        if pred_distances[j] < self.max_distance * 2:  # Allow some flexibility
                            trajectory_matrix[i, j] = pred_distances[j]

            # FIXED: Combine matrices more carefully
            combined_matrix = np.full_like(distance_matrix, np.inf)
            for i in range(len(track_ids)):
                for j in range(len(centroids)):
                    dist_score = distance_matrix[i, j]
                    traj_score = trajectory_matrix[i, j]

                    # If both are reasonable, combine them
                    if dist_score < self.max_distance and traj_score < np.inf:
                        combined_score = (1 - self.trajectory_weight) * dist_score + \
                                       self.trajectory_weight * traj_score
                        combined_matrix[i, j] = combined_score
                    # If only distance is reasonable, use distance with penalty
                    elif dist_score < self.max_distance:
                        combined_matrix[i, j] = dist_score * (1 + self.trajectory_weight * 0.5)
                    # If only trajectory is reasonable, use trajectory with penalty
                    elif traj_score < self.max_distance * 1.5:
                        combined_matrix[i, j] = traj_score * (1 + (1 - self.trajectory_weight) * 0.5)
        else:
            combined_matrix = distance_matrix.copy()

        # Apply stricter distance threshold to combined matrix
        combined_matrix[combined_matrix > self.max_distance * 1.2] = 1e6

        # Solve assignment
        assignments = {}
        if combined_matrix.size > 0 and np.sum(combined_matrix < 1e6) > 0:
            if self.use_hungarian:
                try:
                    row_indices, col_indices = linear_sum_assignment(combined_matrix)
                    for row_idx, col_idx in zip(row_indices, col_indices):
                        if combined_matrix[row_idx, col_idx] < 1e6:
                            assignments[track_ids[row_idx]] = col_idx
                except Exception as e:
                    print(f"Hungarian assignment error: {e} - falling back to Greedy")
                    # Fallback to greedy
                    assignments = self._greedy_assignment_with_trajectory(track_ids, combined_matrix, centroids)
            else:
                assignments = self._greedy_assignment_with_trajectory(track_ids, combined_matrix, centroids)

        return assignments

    def _greedy_assignment_with_trajectory(self, track_ids: List[int], combined_matrix: np.ndarray, centroids: List[Tuple[float, float]]) -> Dict[int, int]:
        """Greedy assignment fallback with trajectory awareness"""
        assignments = {}
        used_centroids = set()

        for i, track_id in enumerate(track_ids):
            available = [j for j in range(len(centroids)) if j not in used_centroids]
            if not available:
                continue

            distances = [(combined_matrix[i, j], j) for j in available if combined_matrix[i, j] < 1e6]
            if not distances:
                continue

            distances.sort()
            best_dist, best_centroid = distances[0]

            # Use a more flexible distance threshold for greedy
            if best_dist < self.max_distance * 1.2:
                assignments[track_id] = best_centroid
                used_centroids.add(best_centroid)

        return assignments

    def start_tracking(self):
        """Start the improved tracking with trajectory awareness and nose detection"""
        if not self.image_files:
            messagebox.showerror("Error", "No images loaded. Please select a directory first.")
            return
        if self.background is None:
            messagebox.showerror("Error", "No background generated. Please generate background first.")
            return

        try:
            min_track_length = int(self.min_track_var.get())
            max_distance = int(self.max_distance_var.get())
            trajectory_weight = float(self.trajectory_weight_var.get())

            if not (1 <= min_track_length <= 1000):
                messagebox.showerror("Error", "Min track length must be between 1 and 1000 frames.")
                return
            if not (1 <= max_distance <= 1000):
                messagebox.showerror("Error", "Max distance must be between 1 and 1000 pixels.")
                return
            if not (0.0 <= trajectory_weight <= 1.0):
                messagebox.showerror("Error", "Trajectory weight must be between 0.0 and 1.0.")
                return

            self.min_track_length = min_track_length
            self.max_distance = max_distance
            self.trajectory_weight = trajectory_weight
            self.use_hungarian = (self.algo_var.get() == "Hungarian")

            # FIXED: Read nose detection parameters from GUI but don't override internal values
            # Only validate and update if they're different from current internal values
            gui_nose_enabled = self.nose_detection_var.get()
            gui_nose_smoothing = int(self.nose_smoothing_var.get())
            gui_min_movement = float(self.min_movement_var.get())
            
            # Validate GUI values
            if not (2 <= gui_nose_smoothing <= 10):
                messagebox.showerror("Error", "Nose smoothing frames must be between 2 and 10.")
                return
            if not (0.1 <= gui_min_movement <= 50.0):
                messagebox.showerror("Error", "Min movement threshold must be between 0.1 and 50.0.")
                return
            
            # FIXED: Use the current internal values, which should already be updated by update_nose_params
            # But sync with GUI in case there are discrepancies
            if (self.nose_detection_enabled != gui_nose_enabled or 
                self.nose_smoothing_frames != gui_nose_smoothing or 
                abs(self.min_movement_threshold - gui_min_movement) > 0.001):
                
                print(f"SYNCING PARAMETERS:")
                print(f"  Nose enabled: GUI={gui_nose_enabled}, Internal={self.nose_detection_enabled}")
                print(f"  Smoothing frames: GUI={gui_nose_smoothing}, Internal={self.nose_smoothing_frames}")
                print(f"  Min movement: GUI={gui_min_movement}, Internal={self.min_movement_threshold}")
                
                # Use GUI values as the authoritative source when starting tracking
                self.nose_detection_enabled = gui_nose_enabled
                self.nose_smoothing_frames = gui_nose_smoothing
                self.min_movement_threshold = gui_min_movement
            
            print(f"STARTING TRACKING WITH NOSE PARAMETERS:")
            print(f"  Nose detection: {self.nose_detection_enabled}")
            print(f"  Smoothing frames: {self.nose_smoothing_frames}")
            print(f"  Min movement: {self.min_movement_threshold}")

        except ValueError:
            messagebox.showerror("Error", "Invalid parameter values. Please check your inputs.")
            return

        def tracking_worker():
            try:
                self.run_trajectory_aware_tracking_with_nose()
                self.root.after(0, self.display_results)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: messagebox.showerror("Tracking Error", f"Tracking failed: {e}"))
            finally:
                self.root.after(0, lambda: self.start_tracking_button.config(state='normal'))

        self.start_tracking_button.config(state='disabled')
        self._tracking_thread = threading.Thread(target=tracking_worker, daemon=True)
        self._tracking_thread.start()

    def load_and_process_frame(self, frame_idx):
        """Load and background-subtract a frame with shape/dtype alignment"""
        if frame_idx >= len(self.image_files):
            return None, None
        img_path = self.image_files[frame_idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None
        bg = self.background
        if bg is not None:
            if img.shape != bg.shape:
                img = cv2.resize(img, (bg.shape[1], bg.shape[0]))
            if img.dtype != bg.dtype:
                img = img.astype(bg.dtype, copy=False)
            subtracted = cv2.absdiff(bg, img)
        else:
            subtracted = img
        return img, subtracted

    def run_trajectory_aware_tracking_with_nose(self):
        """IMPROVED tracking with trajectory prediction and nose detection"""
        if not self.image_files or self.background is None:
            return

        self.root.after(0, lambda: self.track_progress.config(maximum=len(self.image_files)))

        next_track_id = 1
        active_tracks: Dict[int, Dict[str, object]] = {}
        inactive_tracks: Dict[int, Dict[str, object]] = {}

        MAX_MISSING_FRAMES = 5

        print("\n" + "=" * 70)
        print("STARTING TRAJECTORY-AWARE TRACKING WITH NOSE DETECTION")
        print("=" * 70)
        print(f"Max missing frames: {MAX_MISSING_FRAMES}")
        print(f"Max distance: {self.max_distance} px")
        print(f"Trajectory weight: {self.trajectory_weight}")
        print(f"Min track length filter: {self.min_track_length}")
        print(f"Algorithm: {self.algo_var.get()} with trajectory prediction")
        print(f"Nose detection: {'ENABLED' if self.nose_detection_enabled else 'DISABLED'}")
        if self.nose_detection_enabled:
            print(f"  Smoothing frames: {self.nose_smoothing_frames}")
            print(f"  Min movement: {self.min_movement_threshold} px")

        # DEBUG: Track statistics
        track_debug_info = {}
        trajectory_assignments = 0
        distance_assignments = 0
        nose_detections = 0
        failed_nose_detections = 0

        for frame_idx, _img_path in enumerate(self.image_files):
            try:
                img, subtracted = self.load_and_process_frame(frame_idx)
                if img is None or subtracted is None:
                    continue

                thresholded = self.apply_threshold(subtracted)
                contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                centroids = []
                valid_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if self.min_blob_size <= area <= self.max_blob_size:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = M["m10"] / M["m00"]
                            cy = M["m01"] / M["m00"]
                            centroids.append((cx, cy))
                            valid_contours.append(contour)

                # DEBUG: Track what happens to each track
                frame_debug = {
                    'detections': len(centroids),
                    'active_tracks_before': len(active_tracks),
                    'deactivated_tracks': [],
                    'new_tracks': [],
                    'assigned_tracks': [],
                    'trajectory_influenced': [],
                    'nose_detections_success': 0,
                    'nose_detections_failed': 0
                }

                # Deactivate tracks missing too long
                tracks_to_deactivate = []
                for track_id, track_data in active_tracks.items():
                    frames_missing = frame_idx - track_data['last_frame']
                    if frames_missing > MAX_MISSING_FRAMES:
                        tracks_to_deactivate.append(track_id)
                        frame_debug['deactivated_tracks'].append(track_id)

                for track_id in tracks_to_deactivate:
                    inactive_tracks[track_id] = active_tracks[track_id]
                    del active_tracks[track_id]

                # IMPROVED: Assign detections with trajectory awareness
                assignments = self.assign_tracks_with_trajectory(active_tracks, centroids)

                # Update assigned tracks and detect noses
                assigned_centroids = set()
                for track_id, centroid_idx in assignments.items():
                    cx, cy = centroids[centroid_idx]

                    # Store current centroid index for nose detection
                    active_tracks[track_id]['current_centroid_idx'] = centroid_idx

                    # Check if trajectory prediction influenced this assignment
                    if self.trajectory_weight > 0 and active_tracks[track_id]['positions']:
                        predicted_pos = self.predict_next_position(active_tracks[track_id]['positions'])
                        if predicted_pos is not None:
                            last_pos = active_tracks[track_id]['positions'][-1]
                            dist_to_last = np.sqrt((cx - last_pos[0])**2 + (cy - last_pos[1])**2)
                            dist_to_predicted = np.sqrt((cx - predicted_pos[0])**2 + (cy - predicted_pos[1])**2)

                            if dist_to_predicted < dist_to_last * 0.8:  # Prediction was significantly better
                                frame_debug['trajectory_influenced'].append(track_id)
                                trajectory_assignments += 1
                            else:
                                distance_assignments += 1

                    # Update centroid position
                    active_tracks[track_id]['positions'].append((cx, cy, frame_idx))
                    active_tracks[track_id]['last_frame'] = frame_idx

                    # NEW: Detect nose position
                    nose_position = None
                    if self.nose_detection_enabled:
                        nose_position = self.detect_nose_for_track(track_id, active_tracks, valid_contours, centroids)
                        if nose_position is not None:
                            # Initialize nose_positions if not exists
                            if 'nose_positions' not in active_tracks[track_id]:
                                active_tracks[track_id]['nose_positions'] = []
                            active_tracks[track_id]['nose_positions'].append((nose_position[0], nose_position[1], frame_idx))
                            frame_debug['nose_detections_success'] += 1
                            nose_detections += 1
                        else:
                            frame_debug['nose_detections_failed'] += 1
                            failed_nose_detections += 1

                    assigned_centroids.add(centroid_idx)
                    frame_debug['assigned_tracks'].append(track_id)

                # Start new tracks for unassigned centroids
                for i, (cx, cy) in enumerate(centroids):
                    if i not in assigned_centroids:
                        new_track_id = next_track_id
                        active_tracks[new_track_id] = {
                            'positions': [(cx, cy, frame_idx)],
                            'nose_positions': [],  # Initialize empty nose positions
                            'last_frame': frame_idx
                        }
                        frame_debug['new_tracks'].append(new_track_id)
                        next_track_id += 1

                frame_debug['active_tracks_after'] = len(active_tracks)

                # Store debug info
                if frame_idx not in track_debug_info:
                    track_debug_info[f'frame_{frame_idx}'] = frame_debug

                # Progress update
                self.safe_progress_update(frame_idx + 1)

                # DEBUG: Print periodic status with nose detection info
                if frame_idx % 50 == 0 and frame_idx > 0:
                    print(f"  Frame {frame_idx}: {len(active_tracks)} active, {len(inactive_tracks)} inactive")
                    print(f"    Detections: {len(centroids)}, New tracks: {len(frame_debug['new_tracks'])}")
                    print(f"    Assigned: {len(assignments)}, Deactivated: {len(frame_debug['deactivated_tracks'])}")
                    print(f"    Trajectory influenced: {len(frame_debug['trajectory_influenced'])} assignments")
                    if self.nose_detection_enabled:
                        print(f"    Nose detections: {frame_debug['nose_detections_success']} success, {frame_debug['nose_detections_failed']} failed")

            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                continue

        all_final_tracks = {**active_tracks, **inactive_tracks}

        # DEBUG: Print final statistics
        print("\n" + "=" * 50)
        print("TRAJECTORY & NOSE DETECTION ANALYSIS")
        print("=" * 50)
        total_assignments = trajectory_assignments + distance_assignments
        if total_assignments > 0:
            traj_percent = (trajectory_assignments / total_assignments) * 100
            print(f"Total assignments: {total_assignments}")
            print(f"Trajectory-influenced: {trajectory_assignments} ({traj_percent:.1f}%)")
            print(f"Distance-based: {distance_assignments} ({100-traj_percent:.1f}%)")
            print(f"Trajectory weight setting: {self.trajectory_weight}")
        
        if self.nose_detection_enabled:
            total_nose_attempts = nose_detections + failed_nose_detections
            if total_nose_attempts > 0:
                nose_success_percent = (nose_detections / total_nose_attempts) * 100
                print(f"Nose detection attempts: {total_nose_attempts}")
                print(f"Nose detections successful: {nose_detections} ({nose_success_percent:.1f}%)")
                print(f"Nose detections failed: {failed_nose_detections} ({100-nose_success_percent:.1f}%)")
            else:
                print("No nose detection attempts made")
        else:
            print("Nose detection: DISABLED")

        # Finalize
        self.finalize_tracks_with_nose_data(all_final_tracks, trajectory_assignments, distance_assignments, nose_detections)

    def finalize_tracks_with_nose_data(self, all_tracks, trajectory_assignments, distance_assignments, nose_detections):
        """Finalize tracks with trajectory statistics and nose data"""
        print("\n" + "=" * 40)
        print("FINALIZING IMPROVED TRACKS WITH NOSE DATA")
        print("=" * 40)

        self.tracks = {}
        self.nose_tracks = {}  # NEW: Store nose positions separately
        self.track_data = []
        self.track_statistics = []

        tracks_by_length: Dict[int, int] = {}

        for track_id, track_data in all_tracks.items():
            positions = track_data['positions']
            nose_positions = track_data.get('nose_positions', [])
            track_length = len(positions)

            tracks_by_length[track_length] = tracks_by_length.get(track_length, 0) + 1

            if track_length < 2:
                continue

            track_stats = {
                'track_id': track_id,
                'track_length': track_length,
                'nose_detections': len(nose_positions),
                'nose_success_rate': len(nose_positions) / track_length if track_length > 0 else 0,
                'passed_length_filter': track_length >= self.min_track_length,
                'final_status': 'pending'
            }

            if track_length < self.min_track_length:
                track_stats['final_status'] = 'rejected_short'
                self.track_statistics.append(track_stats)
                continue

            track_stats['final_status'] = 'accepted'
            self.track_statistics.append(track_stats)
            
            # Store centroid positions
            self.tracks[track_id] = positions
            
            # Store nose positions
            if nose_positions:
                self.nose_tracks[track_id] = nose_positions

            # Create track data entries (MODIFIED to include nose positions)
            for pos_idx, pos in enumerate(positions):
                data_entry = {
                    'frame': pos[2],
                    f'worm_{track_id}_x': round(pos[0], 4),
                    f'worm_{track_id}_y': round(pos[1], 4)
                }
                
                # Add nose position if available for this frame
                nose_for_frame = next((n for n in nose_positions if n[2] == pos[2]), None)
                if nose_for_frame is not None:
                    data_entry[f'worm_{track_id}_nose_x'] = round(nose_for_frame[0], 4)
                    data_entry[f'worm_{track_id}_nose_y'] = round(nose_for_frame[1], 4)
                else:
                    data_entry[f'worm_{track_id}_nose_x'] = None
                    data_entry[f'worm_{track_id}_nose_y'] = None
                
                self.track_data.append(data_entry)

        accepted = len([s for s in self.track_statistics if s['final_status'] == 'accepted'])
        rejected = len([s for s in self.track_statistics if s['final_status'] == 'rejected_short'])
        total_nose_points = sum(len(nose_pos) for nose_pos in self.nose_tracks.values())

        print(f"Total tracks processed: {len(all_tracks)}")
        print(f"Accepted tracks: {accepted}")
        print(f"Rejected (too short): {rejected}")
        print(f"Trajectory assignments: {trajectory_assignments}")
        print(f"Distance assignments: {distance_assignments}")
        print(f"Total nose detections: {total_nose_points}")
        print(f"Tracks with nose data: {len(self.nose_tracks)}")

        if accepted > 0:
            avg_nose_success = np.mean([s['nose_success_rate'] for s in self.track_statistics if s['final_status'] == 'accepted'])
            print(f"Average nose detection rate: {avg_nose_success:.2%}")

        print("\nTrack length distribution:")
        for length in sorted(tracks_by_length.keys()):
            count = tracks_by_length[length]
            status = "✅" if length >= self.min_track_length else "❌"
            print(f"  {length:3d} frames: {count:3d} tracks {status}")

    def safe_progress_update(self, value):
        """Thread-safe progress bar update"""
        try:
            if self.root and self.track_progress:
                self.root.after(0, lambda: self.track_progress.config(value=value))
        except Exception:
            pass

    # ------------------------ Results & Visualization ------------------------

    def display_results(self):
        """Display improved results summary & plot with nose detection info"""
        if not self.tracks:
            messagebox.showwarning("No Tracks", "No valid tracks found!")
            return

        accepted_tracks = [s for s in self.track_statistics if s['final_status'] == 'accepted']
        rejected_short = [s for s in self.track_statistics if s['final_status'] == 'rejected_short']

        summary = "=== IMPROVED TRACKING RESULTS WITH NOSE DETECTION ===\n\n"
        summary += f"Final tracks: {len(self.tracks)}\n"
        summary += f"Images processed: {len(self.image_files)}\n\n"

        summary += "=== TRAJECTORY-AWARE PARAMETERS ===\n"
        summary += f"Max distance: {self.max_distance} px\n"
        summary += f"Trajectory weight: {self.trajectory_weight} (Momentum prediction)\n"
        summary += f"Gap tolerance: 5 frames (fixed)\n"
        summary += f"Min track length: {self.min_track_length} frames\n"
        summary += f"Tracking algorithm: {self.algo_var.get()} + Trajectory Prediction\n\n"

        summary += "=== NOSE DETECTION PARAMETERS ===\n"
        summary += f"Nose detection: {'ENABLED' if self.nose_detection_enabled else 'DISABLED'}\n"
        if self.nose_detection_enabled:
            summary += f"Smoothing frames: {self.nose_smoothing_frames}\n"
            summary += f"Min movement threshold: {self.min_movement_threshold} px\n"
            summary += f"Tracks with nose data: {len(self.nose_tracks)}\n"
            total_nose_points = sum(len(nose_pos) for nose_pos in self.nose_tracks.values())
            summary += f"Total nose detections: {total_nose_points}\n"
            if accepted_tracks:
                avg_nose_success = np.mean([s['nose_success_rate'] for s in accepted_tracks])
                summary += f"Average nose detection rate: {avg_nose_success:.1%}\n"
        summary += "\n"

        summary += "=== FILTERING BREAKDOWN ===\n"
        summary += f"Total tracks detected: {len(self.track_statistics)}\n"
        summary += f"✅ Accepted: {len(accepted_tracks)}\n"
        summary += f"❌ Rejected (too short): {len(rejected_short)}\n\n"

        if accepted_tracks:
            track_lengths = [s['track_length'] for s in accepted_tracks]
            summary += "=== ACCEPTED TRACK STATS ===\n"
            summary += f"Track lengths: {np.mean(track_lengths):.1f} ± {np.std(track_lengths):.1f} frames\n"
            summary += f"Longest track: {max(track_lengths)} frames\n"
            summary += f"Shortest track: {min(track_lengths)} frames\n\n"

        summary += "=== ALGORITHM IMPROVEMENTS ===\n"
        summary += f"Tracking algorithm: {self.algo_var.get()} with trajectory prediction\n"
        summary += "Trajectory prediction: ENABLED\n"
        summary += "Position-based assignment: ENHANCED\n"
        summary += "ID swap prevention: IMPROVED\n"
        summary += "Intersection handling: BETTER\n"
        summary += f"Nose detection: {'ENABLED' if self.nose_detection_enabled else 'DISABLED'}\n"
        summary += "Background subtraction: Median-based\n"
        summary += "Gap handling: 5-frame tolerance with track deactivation\n"

        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, summary)

        # Draw visualization
        self.update_visualization()

        # Switch tab
        self.notebook.select(self.results_tab)

        # Re-enable tracking button
        self.start_tracking_button.config(state='normal')

    def update_visualization(self, _event=None):
        """Update the results visualization with nose positions"""
        if not self.tracks:
            return

        self.results_fig.clear()
        ax = self.results_fig.add_subplot(111)

        title_bg = "No Background"
        if self.bg_type_var.get() == "Raw Image" and self.sample_images:
            background_img = self.sample_images[0]
            ax.imshow(background_img, cmap='gray', alpha=0.8)
            title_bg = "Raw Image"
        elif self.processed_samples:
            background_img = self.processed_samples[0]
            ax.imshow(background_img, cmap='gray', alpha=0.7)
            title_bg = "Background Subtracted"

        if self.show_detections_var.get() and self.processed_samples:
            self.overlay_all_detections(ax, self.processed_samples[0])

        if (self._cached_track_colors is None) or (len(self._cached_track_colors) != len(self.tracks)):
            self._cached_track_colors = plt.cm.tab10(np.linspace(0, 1, max(len(self.tracks), 1)))

        colors = self._cached_track_colors

        for i, (track_id, positions) in enumerate(self.tracks.items()):
            color = colors[i % len(colors)]
            xs = [pos[0] for pos in positions]
            ys = [pos[1] for pos in positions]

            # Draw centroid track
            ax.plot(xs, ys, color=color, linewidth=2, alpha=0.8, label=f'Track {track_id}')
            ax.scatter(xs[0], ys[0], color=color, s=100, marker='o')  # Start
            ax.scatter(xs[-1], ys[-1], color=color, s=100, marker='s')  # End
            
            # Add track ID annotation
            ax.annotate(f'{track_id}', (xs[0], ys[0]), xytext=(5, 5),
                        textcoords='offset points', fontsize=12, fontweight='bold',
                        color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor=color))

        # NEW: Show nose positions if enabled and available
        if self.show_noses_var.get() and track_id in self.nose_tracks:
            nose_positions = self.nose_tracks[track_id]
            if nose_positions:
                nose_xs = [pos[0] for pos in nose_positions]
                nose_ys = [pos[1] for pos in nose_positions]
                # Draw nose positions as triangular markers
                ax.scatter(nose_xs, nose_ys, color=color, s=40, marker='^', 
                          alpha=0.9, edgecolors='white', linewidth=0.8)
                
                # Optionally draw lines connecting centroids to noses for same frames
                if len(nose_positions) > 0:
                    # Find centroid-nose pairs for the same frames
                    for nose_pos in nose_positions:
                        nose_frame = nose_pos[2]
                        # Find corresponding centroid for this frame
                        centroid_pos = next((p for p in positions if p[2] == nose_frame), None)
                        if centroid_pos is not None:
                            # Draw thin line from centroid to nose
                            ax.plot([centroid_pos[0], nose_pos[0]], 
                                   [centroid_pos[1], nose_pos[1]], 
                                   color=color, alpha=0.4, linewidth=0.5)

        detection_text = " + All Detections" if self.show_detections_var.get() else ""
        nose_text = " + Noses" if self.show_noses_var.get() else ""
        trajectory_text = f" (Traj. Weight: {self.trajectory_weight})"
        
        ax.set_title(f"Improved Tracking Results ({title_bg}){detection_text}{nose_text}{trajectory_text} - {len(self.tracks)} Tracks")
        ax.set_xlabel("X Position (pixels)")
        ax.set_ylabel("Y Position (pixels)")
        self.results_canvas.draw()

    def overlay_all_detections(self, ax, sample_image):
        """Overlay all detections (accepted and rejected) on the plot"""
        thresholded = self.apply_threshold(sample_image)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            if area < self.min_blob_size:
                ax.scatter(cx, cy, c='red', marker='x', s=30, alpha=0.7)
            elif area > self.max_blob_size:
                ax.scatter(cx, cy, c='orange', marker='x', s=50, alpha=0.7)
            else:
                ax.scatter(cx, cy, c='lime', marker='o', s=20, alpha=0.8, edgecolors='darkgreen', linewidth=1)

    # ------------------------ Export & Viewer ------------------------

    def export_tracks(self):
        """Export tracks to CSV format with nose positions (wide format: frame + per-track x,y + nose_x,nose_y)"""
        if not self.tracks:
            messagebox.showwarning("No Data", "No tracking data to export!")
            return

        try:
            all_frames = sorted({pos[2] for positions in self.tracks.values() for pos in positions})
            track_ids = sorted(self.tracks.keys())
            
            # Build columns: frame + centroid coords + nose coords for each track
            columns = ['frame']
            for tid in track_ids:
                columns.extend([f"worm_{tid}_x", f"worm_{tid}_y", f"worm_{tid}_nose_x", f"worm_{tid}_nose_y"])

            data = []
            for frame in all_frames:
                row = [frame]
                for track_id in track_ids:
                    # Get centroid position for this frame
                    centroid_pos = next(((p[0], p[1]) for p in self.tracks[track_id] if p[2] == frame), None)
                    if centroid_pos:
                        row.extend([round(centroid_pos[0], 4), round(centroid_pos[1], 4)])
                    else:
                        row.extend([None, None])
                    
                    # Get nose position for this frame
                    nose_pos = None
                    if track_id in self.nose_tracks:
                        nose_pos = next(((p[0], p[1]) for p in self.nose_tracks[track_id] if p[2] == frame), None)
                    
                    if nose_pos:
                        row.extend([round(nose_pos[0], 4), round(nose_pos[1], 4)])
                    else:
                        row.extend([None, None])
                
                data.append(row)

            df = pd.DataFrame(data, columns=columns)

            file_path = filedialog.asksaveasfilename(
                title="Save improved tracking data with nose positions",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if not file_path:
                return

            df.to_csv(file_path, index=False)

            summary = f"IMPROVED TRACKING + NOSE DETECTION Export Complete!\n\n"
            summary += f"File: {file_path}\n"
            summary += f"Frames: {len(df)}\n"
            summary += f"Tracks: {len(track_ids)}\n"
            summary += f"Format: One row per frame, columns for each worm's x,y + nose_x,nose_y coordinates\n\n"
            summary += f"Trajectory-aware features:\n"
            summary += f"• Trajectory weight: {self.trajectory_weight}\n"
            summary += f"• ID swap prevention: ENABLED\n"
            summary += f"• Intersection handling: IMPROVED\n"
            summary += f"• Algorithm: {self.algo_var.get()} + Trajectory Prediction\n"
            summary += f"• Nose detection: {'ENABLED' if self.nose_detection_enabled else 'DISABLED'}\n"
            if self.nose_detection_enabled:
                total_nose_detections = sum(len(positions) for positions in self.nose_tracks.values())
                summary += f"• Total nose detections: {total_nose_detections}\n"
            messagebox.showinfo("Export Complete", summary)
        except Exception as e:
            import traceback
            print("Export error details:\n", traceback.format_exc())
            messagebox.showerror("Export Error", f"Failed to export data: {e}\n\nTry 'Export Simple CSV' instead.")

    def export_tracks_simple(self):
        """Backup export: simple long format of collected points with nose data"""
        if not self.track_data:
            messagebox.showwarning("No Data", "No tracking data to export!")
            return

        try:
            df = pd.DataFrame(self.track_data)
            file_path = filedialog.asksaveasfilename(
                title="Save tracking data with nose positions (simple format)",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if not file_path:
                return
            df.to_csv(file_path, index=False)
            messagebox.showinfo("Export Complete",
                                f"Raw tracking data with nose positions saved to:\n{file_path}\n\n"
                                f"Format: One row per track point\n"
                                f"Columns: frame, worm_ID_x, worm_ID_y, worm_ID_nose_x, worm_ID_nose_y\n\n"
                                f"Trajectory-aware with weight: {self.trajectory_weight}!\n"
                                f"Nose detection: {'ENABLED' if self.nose_detection_enabled else 'DISABLED'}!")
        except Exception as e:
            import traceback
            print("Simple export error details:\n", traceback.format_exc())
            messagebox.showerror("Export Error", f"Failed to export data: {e}")

    def save_visualization(self):
        """Save the current results figure"""
        if not self.tracks:
            messagebox.showwarning("No Tracks", "No tracks to visualize!")
            return
        try:
            file_path = filedialog.asksaveasfilename(
                title="Save improved track visualization with nose detection",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
            )
            if not file_path:
                return
            self.results_fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Save Complete", f"Improved tracking visualization with nose detection saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save visualization: {e}")

    def launch_track_video_viewer(self):
        """Launch the Track Video Viewer with current data including nose positions"""
        if not self.tracks:
            messagebox.showwarning("No Tracks", "No tracking data available. Run tracking first.")
            return

        try:
            import tempfile
            temp_csv_path = os.path.join(tempfile.gettempdir(), "improved_tracker_with_nose_temp.csv")
            self.export_tracks_for_viewer(temp_csv_path)

            try:
                from track_editor import TrackEditor
                editor = TrackEditor()

                # Load the CSV data
                editor.tracks_df = pd.read_csv(temp_csv_path)
                if hasattr(editor, "parse_enhanced_tracker_format"):
                    parsed_ok = editor.parse_enhanced_tracker_format()
                else:
                    parsed_ok = True

                if parsed_ok and hasattr(editor, "update_interface_after_load"):
                    editor.update_interface_after_load()

                # Optional: preload images
                if self.image_dir and os.path.exists(self.image_dir):
                    if hasattr(editor, "image_dir"):
                        editor.image_dir = self.image_dir
                    if hasattr(editor, "image_files"):
                        editor.image_files = self.image_files.copy()
                    if hasattr(editor, "image_buffer"):
                        editor.image_buffer = None  # Will be recreated with new files
                    if hasattr(editor, "update_interface_after_load"):
                        editor.update_interface_after_load()
                    nose_status = "ENABLED" if self.nose_detection_enabled else "DISABLED"
                    messagebox.showinfo("Track Editor",
                                        "Launching Track Editor with improved tracking data, nose positions, and images!\n\n"
                                        f"Trajectory weight used: {self.trajectory_weight}\n"
                                        f"Nose detection: {nose_status}")
                else:
                    nose_status = "ENABLED" if self.nose_detection_enabled else "DISABLED"
                    messagebox.showinfo("Track Editor",
                                        "Launching Track Editor with improved tracking data and nose positions.\n\n"
                                        "Load an image directory in the editor to see background images.\n"
                                        f"Trajectory weight used: {self.trajectory_weight}\n"
                                        f"Nose detection: {nose_status}")

                threading.Thread(target=editor.run, daemon=True).start()

            except ImportError:
                # Fallback: run as separate process if script exists nearby
                try:
                    import subprocess
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    editor_script = os.path.join(script_dir, "track_editor.py")
                    if os.path.exists(editor_script):
                        subprocess.Popen([sys.executable, editor_script])
                        nose_status = "ENABLED" if self.nose_detection_enabled else "DISABLED"
                        messagebox.showinfo("Track Editor",
                                            "Launching Track Editor as a separate app.\n\n"
                                            f"Load this CSV in the editor:\n{temp_csv_path}\n"
                                            f"Trajectory weight used: {self.trajectory_weight}\n"
                                            f"Nose detection: {nose_status}")
                    else:
                        messagebox.showinfo("Track Editor",
                                            "track_editor.py not found in the same directory.\n\n"
                                            f"After you add it, load this CSV:\n{temp_csv_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Could not launch Track Editor: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch Track Editor: {e}")

    def export_tracks_for_viewer(self, file_path: str):
        """Export tracks in a wide CSV compatible with Track Editor, including nose positions"""
        if not self.tracks:
            return
        try:
            all_frames = sorted({pos[2] for positions in self.tracks.values() for pos in positions})
            track_ids = sorted(self.tracks.keys())
            
            # Build columns: frame + centroid + nose for each track
            columns = ['frame']
            for tid in track_ids:
                columns.extend([f"worm_{tid}_x", f"worm_{tid}_y", f"worm_{tid}_nose_x", f"worm_{tid}_nose_y"])
            
            rows = []
            for f in all_frames:
                row = [f]
                for tid in track_ids:
                    # Centroid position
                    centroid_pos = next(((p[0], p[1]) for p in self.tracks[tid] if p[2] == f), None)
                    if centroid_pos:
                        row += [round(centroid_pos[0], 4), round(centroid_pos[1], 4)]
                    else:
                        row += [None, None]
                    
                    # Nose position
                    nose_pos = None
                    if tid in self.nose_tracks:
                        nose_pos = next(((p[0], p[1]) for p in self.nose_tracks[tid] if p[2] == f), None)
                    
                    if nose_pos:
                        row += [round(nose_pos[0], 4), round(nose_pos[1], 4)]
                    else:
                        row += [None, None]
                
                rows.append(row)
            
            pd.DataFrame(rows, columns=columns).to_csv(file_path, index=False)
            print(f"Exported improved tracks with nose data for editor: {file_path}")
        except Exception as e:
            print(f"Error exporting tracks for editor: {e}")
            raise

    # ------------------------ App loop ------------------------

    def run(self):
        """Start the GUI application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("Application interrupted by user")
            self.on_closing()
        except Exception as e:
            print(f"Unexpected error in main loop: {e}")
            self.on_closing()


def main():
    print("=== IMPROVED WORM TRACKER - TRAJECTORY-AWARE ASSIGNMENT + NOSE DETECTION ===")
    print("Prevents ID Swapping at Intersections with Trajectory Prediction")
    print("Detects Nose Position Based on Locomotion Direction")
    print("=" * 80)
    print("\nKEY IMPROVEMENTS:")
    print("- Trajectory prediction to prevent ID swapping")
    print("- Velocity-aware assignment scoring")
    print("- Configurable trajectory weight parameter")
    print("- Better handling of worm crossings")
    print("- Enhanced Hungarian and Greedy algorithms")
    print("- FIXED UI scaling for different screen sizes")
    print("- NEW: Nose detection based on locomotion direction")
    print("\nTRAJECTORY WEIGHT PARAMETER:")
    print("  0.0 = Pure distance-based (original behavior)")
    print("  0.7 = Strong momentum prediction (recommended)")
    print("  1.0 = Pure trajectory (may be unstable)")
    print("\nNOSE DETECTION:")
    print("  - Calculates worm locomotion direction from recent positions")
    print("  - Finds front-most point of blob in direction of movement")
    print("  - Smooths direction over configurable number of frames")
    print("  - Exports nose coordinates alongside centroid positions")
    print("\nWORKFLOW:")
    print("1. Select image directory")
    print("2. Generate background")
    print("3. Generate samples and adjust thresholds")
    print("4. Set trajectory weight (0.7 recommended)")
    print("5. Configure nose detection parameters")
    print("6. Use Greedy algorithm (default)")
    print("7. Start trajectory-aware tracking with nose detection")
    print("8. Use Track Editor to fix any remaining issues")
    print("\nSTARTING IMPROVED TRACKER WITH NOSE DETECTION...")

    try:
        tracker = WormTracker()
        tracker.run()
    except Exception as e:
        import traceback
        print(f"Error starting tracker: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()