#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SWT Track Editor

"""

import cv2
import numpy as np
import os
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
from collections import OrderedDict, deque
import time
from typing import Dict, List, Tuple, Optional, Union, Protocol
from dataclasses import dataclass
import argparse
import queue
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc
import weakref
from pathlib import Path
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================== CONFIGURATION ==================

class Config:
    """Application configuration constants"""
    DEFAULT_CACHE_SIZE = 100
    DEFAULT_FPS = 15.0
    DEFAULT_TRAIL_LENGTH = 30
    MAX_DISPLAY_SIZE = (2048, 2048)
    SELECTION_RADIUS = 30
    PREFETCH_RADIUS = 20
    PERFORMANCE_UPDATE_INTERVAL = 2000
    
    # Threading
    MAX_BACKGROUND_WORKERS = 3
    LOAD_QUEUE_SIZE = 10
    
    # Cache tiers
    HOT_CACHE_RATIO = 0.5
    WARM_CACHE_RATIO = 0.5
    
    # Color palette
    TRACK_COLORS = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan', 'orange', 'purple']

# ================== DATA MODELS ==================

@dataclass(frozen=True)
class TrackPosition:
    """Immutable track position data with optional nose coordinates"""
    x: float
    y: float
    frame: int
    nose_x: Optional[float] = None
    nose_y: Optional[float] = None

@dataclass
class ImageMetadata:
    """Image metadata for caching"""
    path: str
    original_size: Tuple[int, int]
    downsample_ratio: float
    load_time: float

@dataclass
class CacheStats:
    """Cache performance statistics"""
    hit_rate: float
    hot_cache_size: int
    warm_cache_size: int
    avg_load_time_ms: float
    queue_size: int

# ================== IMAGE LOADING SYSTEM ==================

class ImageLoader(Protocol):
    """Protocol for image loading strategies"""
    def load_image(self, path: str) -> Optional[np.ndarray]:
        ...

class TiffFileLoader:
    """TIFF loader using tifffile library"""
    def load_image(self, path: str) -> Optional[np.ndarray]:
        try:
            import tifffile
            return tifffile.imread(path)
        except ImportError:
            return None

class PILImageLoader:
    """PIL-based image loader"""
    def load_image(self, path: str) -> Optional[np.ndarray]:
        try:
            from PIL import Image
            with Image.open(path) as pil_img:
                if pil_img.mode in ('RGB', 'RGBA'):
                    pil_img = pil_img.convert('L')
                elif pil_img.mode not in ('L', 'P'):
                    pil_img = pil_img.convert('L')
                return np.array(pil_img)
        except ImportError:
            return None

class SkimageLoader:
    """Scikit-image loader"""
    def load_image(self, path: str) -> Optional[np.ndarray]:
        try:
            from skimage import io, util
            img = io.imread(path, as_gray=True)
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = util.img_as_ubyte(img)
                else:
                    img = img.astype(np.uint8)
            return img
        except ImportError:
            return None

class OpenCVLoader:
    """OpenCV image loader"""
    def load_image(self, path: str) -> Optional[np.ndarray]:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

class MultiFormatImageLoader:
    """Multi-format image loader with fallback strategies"""
    
    def __init__(self):
        self.loaders = self._initialize_loaders()
        logger.info(f"Initialized {len(self.loaders)} image loaders")
    
    def _initialize_loaders(self) -> List[ImageLoader]:
        loaders = []
        
        # Try specialized loaders first
        for loader_class, name in [
            (TiffFileLoader, "Tifffile"),
            (PILImageLoader, "PIL"),
            (SkimageLoader, "Scikit-image"),
            (OpenCVLoader, "OpenCV")
        ]:
            try:
                loader = loader_class()
                # Test if the loader can actually work
                test_result = loader.load_image.__annotations__ if hasattr(loader.load_image, '__annotations__') else True
                loaders.append(loader)
                logger.info(f"✓ {name} loader available")
            except Exception:
                logger.debug(f"✗ {name} loader not available")
        
        return loaders
    
    def load_image(self, path: str) -> Optional[np.ndarray]:
        """Load image with fallback strategies"""
        for i, loader in enumerate(self.loaders):
            try:
                img = loader.load_image(path)
                if img is not None:
                    return self._normalize_image(img)
            except Exception as e:
                if i == 0:  # Only log on first attempt
                    logger.debug(f"Loader {type(loader).__name__} failed for {path}: {e}")
                continue
        
        logger.warning(f"All loaders failed for: {path}")
        return None
    
    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to uint8 grayscale"""
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)
        
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        
        return img.astype(np.uint8)

# ================== CACHING SYSTEM ==================

class LazyImageCache:
    """Ultra-fast lazy image cache with multi-tier caching"""
    
    def __init__(self, image_files: List[str], cache_size: int = None):
        self.image_files = image_files
        self.cache_size = self._calculate_optimal_cache_size(cache_size)
        
        # Multi-tier caching
        self.hot_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.warm_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.metadata_cache: Dict[int, ImageMetadata] = {}
        
        # Image processing
        self.loader = MultiFormatImageLoader()
        self.downsample_ratios: Dict[int, float] = {}
        self.original_image_sizes: Dict[int, Tuple[int, int]] = {}
        
        # Background processing
        self.load_queue = queue.PriorityQueue()
        self.metadata_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=Config.MAX_BACKGROUND_WORKERS)
        self.is_shutdown = False
        
        # Performance tracking
        self.hit_count = 0
        self.miss_count = 0
        self.load_times = deque(maxlen=100)
        
        # Start background workers
        self._start_background_workers()
        
        logger.info(f"LazyImageCache initialized with {len(image_files)} images, cache size: {self.cache_size}")
    
    def _calculate_optimal_cache_size(self, cache_size: int = None) -> int:
        """Calculate optimal cache size based on available memory"""
        if cache_size is not None:
            return cache_size
        
        available_gb = psutil.virtual_memory().available / (1024**3)
        estimated_mb_per_image = 4
        max_memory_mb = available_gb * 1024 * 0.3
        optimal_cache_size = int(max_memory_mb / estimated_mb_per_image)
        
        return max(20, min(optimal_cache_size, len(self.image_files), 500))
    
    def get_image(self, frame_idx: int, priority: int = 1) -> Optional[np.ndarray]:
        """Get image with multi-tier caching"""
        if frame_idx < 0 or frame_idx >= len(self.image_files):
            return None
        
        start_time = time.time()
        
        # Hot cache hit
        if frame_idx in self.hot_cache:
            self.hot_cache.move_to_end(frame_idx)
            self.hit_count += 1
            return self.hot_cache[frame_idx]
        
        # Warm cache hit
        if frame_idx in self.warm_cache:
            img = self.warm_cache.pop(frame_idx)
            self._add_to_hot_cache(frame_idx, img)
            self.hit_count += 1
            return img
        
        # Cache miss - load immediately
        self.miss_count += 1
        img = self._load_image_immediate(frame_idx)
        
        if img is not None:
            self._add_to_hot_cache(frame_idx, img)
            self._queue_surrounding_images(frame_idx, priority)
        
        load_time = time.time() - start_time
        self.load_times.append(load_time)
        
        return img
    
    def _load_image_immediate(self, frame_idx: int) -> Optional[np.ndarray]:
        """Load image immediately with processing"""
        try:
            img_path = self.image_files[frame_idx]
            img = self.loader.load_image(img_path)
            
            if img is None:
                return None
            
            # Store original size
            original_shape = img.shape
            self.original_image_sizes[frame_idx] = original_shape
            
            # Adaptive downsampling
            if max(original_shape) > max(Config.MAX_DISPLAY_SIZE):
                downsample_ratio = min(
                    Config.MAX_DISPLAY_SIZE[0] / original_shape[0],
                    Config.MAX_DISPLAY_SIZE[1] / original_shape[1]
                )
                
                new_height = int(original_shape[0] * downsample_ratio)
                new_width = int(original_shape[1] * downsample_ratio)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                self.downsample_ratios[frame_idx] = downsample_ratio
            else:
                self.downsample_ratios[frame_idx] = 1.0
            
            return img
            
        except Exception as e:
            logger.error(f"Error loading image {frame_idx}: {e}")
            return None
    
    def _add_to_hot_cache(self, frame_idx: int, img: np.ndarray):
        """Add to hot cache with intelligent eviction"""
        self.hot_cache[frame_idx] = img
        
        hot_limit = int(self.cache_size * Config.HOT_CACHE_RATIO)
        while len(self.hot_cache) > hot_limit:
            oldest_idx, oldest_img = self.hot_cache.popitem(last=False)
            warm_limit = self.cache_size - len(self.hot_cache)
            
            if len(self.warm_cache) < warm_limit:
                self.warm_cache[oldest_idx] = oldest_img
    
    def _queue_surrounding_images(self, center_idx: int, priority: int):
        """Queue surrounding images for background loading"""
        if self.load_queue.qsize() > Config.LOAD_QUEUE_SIZE:
            return
        
        for distance in range(1, Config.PREFETCH_RADIUS + 1):
            for offset in [-distance, distance]:
                idx = center_idx + offset
                if (0 <= idx < len(self.image_files) and 
                    idx not in self.hot_cache and 
                    idx not in self.warm_cache):
                    
                    img_priority = priority + distance
                    try:
                        self.load_queue.put((img_priority, idx), block=False)
                    except queue.Full:
                        break
    
    def _start_background_workers(self):
        """Start background processing threads"""
        threading.Thread(target=self._background_loader, daemon=True).start()
    
    def _background_loader(self):
        """Background image loading worker"""
        while not self.is_shutdown:
            try:
                priority, frame_idx = self.load_queue.get(timeout=1)
                
                if (frame_idx in self.hot_cache or 
                    frame_idx in self.warm_cache or
                    self.is_shutdown):
                    continue
                
                img = self._load_image_immediate(frame_idx)
                if img is not None and not self.is_shutdown:
                    warm_limit = self.cache_size - len(self.hot_cache)
                    if len(self.warm_cache) < warm_limit:
                        self.warm_cache[frame_idx] = img
                
            except queue.Empty:
                continue
    
    def get_display_coordinates(self, x: float, y: float, frame_idx: int) -> Tuple[float, float]:
        """Convert original to display coordinates"""
        ratio = self.downsample_ratios.get(frame_idx, 1.0)
        return x * ratio, y * ratio
    
    def get_original_coordinates(self, x: float, y: float, frame_idx: int) -> Tuple[float, float]:
        """Convert display to original coordinates"""
        ratio = self.downsample_ratios.get(frame_idx, 1.0)
        return x / ratio, y / ratio
    
    def get_display_extent(self, frame_idx: int) -> Optional[Tuple[List[float], Tuple[int, int]]]:
        """Get display extent for frame"""
        if frame_idx in self.downsample_ratios and frame_idx in self.original_image_sizes:
            original_h, original_w = self.original_image_sizes[frame_idx]
            ratio = self.downsample_ratios[frame_idx]
            display_w = int(original_w * ratio)
            display_h = int(original_h * ratio)
            return [0, display_w, display_h, 0], (display_h, display_w)
        return None
    
    def prefetch_range(self, start_idx: int, end_idx: int, priority: int = 5):
        """Prefetch range of images"""
        for idx in range(start_idx, end_idx):
            if (0 <= idx < len(self.image_files) and 
                idx not in self.hot_cache and 
                idx not in self.warm_cache):
                try:
                    self.load_queue.put((priority, idx), block=False)
                except queue.Full:
                    break
    
    def get_cache_stats(self) -> CacheStats:
        """Get cache performance statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        avg_load_time = np.mean(self.load_times) if self.load_times else 0
        
        return CacheStats(
            hit_rate=hit_rate,
            hot_cache_size=len(self.hot_cache),
            warm_cache_size=len(self.warm_cache),
            avg_load_time_ms=avg_load_time * 1000,
            queue_size=self.load_queue.qsize()
        )
    
    def clear_cache(self):
        """Clear all caches"""
        self.hot_cache.clear()
        self.warm_cache.clear()
        gc.collect()
    
    def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down image cache...")
        self.is_shutdown = True
        self.executor.shutdown(wait=False)
        self.clear_cache()

# ================== TRACK MANAGEMENT ==================

class TrackManager:
    """Manages track data and operations with nose coordinate support"""
    
    def __init__(self):
        self.tracks: Dict[int, List[TrackPosition]] = {}
        self.deleted_tracks: set = set()
        self.original_tracks: Dict[int, List[TrackPosition]] = {}
        self.frame_count = 0
        self.has_nose_data = False  # Track whether nose coordinates are available
    
    def load_from_dataframe(self, df: pd.DataFrame) -> bool:
        """Load tracks from CSV DataFrame with nose coordinate support"""
        try:
            if 'frame' not in df.columns:
                logger.error("No 'frame' column found in CSV")
                return False
            
            # Look for both regular worm columns and nose columns
            worm_columns = [col for col in df.columns
                           if col.startswith('worm_') and ('_x' in col or '_y' in col)]
            
            nose_columns = [col for col in df.columns
                           if col.startswith('worm_') and ('_nose_x' in col or '_nose_y' in col)]
            
            if not worm_columns:
                logger.error("No worm position columns found")
                return False
            
            # Extract track IDs from regular worm columns
            track_ids = set()
            for col in worm_columns:
                if '_x' in col and '_nose_' not in col:  # Exclude nose columns from main extraction
                    track_id = int(col.replace('_x', '').replace('worm_', ''))
                    track_ids.add(track_id)
            
            # Check for nose data
            nose_track_ids = set()
            for col in nose_columns:
                if '_nose_x' in col:
                    track_id = int(col.replace('_nose_x', '').replace('worm_', ''))
                    nose_track_ids.add(track_id)
            
            self.has_nose_data = len(nose_track_ids) > 0
            if self.has_nose_data:
                logger.info(f"Found nose coordinate data for {len(nose_track_ids)} tracks")
            
            self.tracks = {}
            
            for track_id in sorted(track_ids):
                x_col = f'worm_{track_id}_x'
                y_col = f'worm_{track_id}_y'
                nose_x_col = f'worm_{track_id}_nose_x'
                nose_y_col = f'worm_{track_id}_nose_y'
                
                if x_col in df.columns and y_col in df.columns:
                    positions = []
                    
                    # Process each row individually to maintain frame alignment
                    for idx, row in df.iterrows():
                        frame = int(row['frame'])
                        x = row[x_col]
                        y = row[y_col]
                        
                        # Only add position if centroid coordinates are valid
                        if pd.notna(x) and pd.notna(y):
                            nose_x = None
                            nose_y = None
                            
                            # Get nose coordinates for this exact frame/row
                            if nose_x_col in df.columns and nose_y_col in df.columns:
                                nose_x_val = row[nose_x_col]
                                nose_y_val = row[nose_y_col]
                                
                                if pd.notna(nose_x_val) and pd.notna(nose_y_val):
                                    nose_x = float(nose_x_val)
                                    nose_y = float(nose_y_val)
                            
                            positions.append(TrackPosition(
                                float(x), float(y), frame, nose_x, nose_y
                            ))
                    
                    if positions:
                        # Sort by frame to ensure proper ordering
                        positions.sort(key=lambda p: p.frame)
                        self.tracks[track_id] = positions
            
            self.original_tracks = {k: v.copy() for k, v in self.tracks.items()}
            self.deleted_tracks.clear()
            self.frame_count = int(df['frame'].max()) + 1
            
            # Log diagnostic information about nose data coverage
            if self.has_nose_data:
                total_positions = sum(len(positions) for positions in self.tracks.values())
                positions_with_nose = sum(
                    sum(1 for pos in positions if pos.nose_x is not None and pos.nose_y is not None)
                    for positions in self.tracks.values()
                )
                nose_coverage = (positions_with_nose / total_positions * 100) if total_positions > 0 else 0
                logger.info(f"Nose data coverage: {positions_with_nose}/{total_positions} positions ({nose_coverage:.1f}%)")
            
            logger.info(f"Loaded {len(self.tracks)} tracks, {self.frame_count} frames")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load tracks: {e}")
            return False
    
    def get_tracks_at_frame(self, frame_idx: int, trail_length: int = Config.DEFAULT_TRAIL_LENGTH, 
                           use_nose: bool = False) -> Tuple[Dict[int, List[TrackPosition]], Dict[int, List[TrackPosition]]]:
        """Get active and inactive tracks at frame with nose coordinate option"""
        active_tracks = {}
        inactive_tracks = {}
        
        for track_id, positions in self.tracks.items():
            if track_id in self.deleted_tracks:
                continue
            
            # Get positions up to current frame
            current_positions = [pos for pos in positions if pos.frame <= frame_idx]
            current_frame_positions = [pos for pos in positions if pos.frame == frame_idx]
            
            if current_positions:
                # Limit trail length
                if len(current_positions) > trail_length:
                    current_positions = current_positions[-trail_length:]
                
                # Handle nose coordinate filtering more carefully
                if use_nose and self.has_nose_data:
                    # For trails, we need to be more selective about gaps
                    # Only include positions that have nose data AND are recent enough
                    valid_positions = []
                    for pos in current_positions:
                        if pos.nose_x is not None and pos.nose_y is not None:
                            valid_positions.append(pos)
                    
                    # For current frame, check if there's nose data specifically at this frame
                    valid_current_frame = []
                    for pos in current_frame_positions:
                        if pos.nose_x is not None and pos.nose_y is not None:
                            valid_current_frame.append(pos)
                    
                    # Only use tracks that have some valid nose data
                    if valid_positions:
                        if valid_current_frame:
                            active_tracks[track_id] = valid_positions
                        else:
                            inactive_tracks[track_id] = valid_positions
                else:
                    # Using centroid coordinates - all positions are valid
                    if current_frame_positions:
                        active_tracks[track_id] = current_positions
                    else:
                        inactive_tracks[track_id] = current_positions
        
        return active_tracks, inactive_tracks
    
    def get_track_color(self, track_id: int) -> str:
        """Get consistent color for track"""
        return Config.TRACK_COLORS[track_id % len(Config.TRACK_COLORS)]
    
    def delete_tracks(self, track_ids: set):
        """Mark tracks as deleted"""
        self.deleted_tracks.update(track_ids)
    
    def merge_tracks(self, track_ids: List[int]) -> bool:
        """Merge tracks into first one"""
        if len(track_ids) < 2:
            return False
        
        target_id = track_ids[0]
        all_positions = []
        
        for track_id in track_ids:
            if track_id in self.tracks and track_id not in self.deleted_tracks:
                all_positions.extend(self.tracks[track_id])
        
        if all_positions:
            all_positions.sort(key=lambda p: p.frame)
            unique_positions = []
            used_frames = set()
            
            for pos in all_positions:
                if pos.frame not in used_frames:
                    unique_positions.append(pos)
                    used_frames.add(pos.frame)
            
            self.tracks[target_id] = unique_positions
            
            for track_id in track_ids[1:]:
                self.deleted_tracks.add(track_id)
            
            return True
        
        return False

    def get_next_track_id(self) -> int:
        """Return the next available integer track id."""
        if not self.tracks:
            return 0
        # Avoid deleted ids; we only need a unique new id.
        return max(self.tracks.keys()) + 1

    def split_track(self, track_id: int, split_frame: int) -> Optional[int]:
        """
        Split a single track into two at the given split_frame.
        Left side keeps frames <= split_frame under the same id.
        Right side (frames > split_frame) becomes a NEW track id and is returned.
        Returns None if nothing to split (e.g., no points after split_frame) or invalid id.
        """
        if track_id not in self.tracks or track_id in self.deleted_tracks:
            return None

        positions = self.tracks[track_id]
        if not positions:
            return None

        left = [p for p in positions if p.frame <= split_frame]
        right = [p for p in positions if p.frame > split_frame]

        # Nothing on the right -> no split needed
        if not right:
            return None

        # Keep left under same id; create a new id for the right part
        new_id = self.get_next_track_id()
        # Preserve order
        left.sort(key=lambda p: p.frame)
        right.sort(key=lambda p: p.frame)

        self.tracks[track_id] = left
        self.tracks[new_id] = right

        # Keep book-keeping consistent with original states
        if track_id in self.deleted_tracks:
            self.deleted_tracks.discard(track_id)
        if new_id in self.deleted_tracks:
            self.deleted_tracks.discard(new_id)

        return new_id
    
    def save_to_csv(self, file_path: str) -> bool:
        """Save tracks to CSV with nose coordinates if available"""
        try:
            active_tracks = {k: v for k, v in self.tracks.items()
                           if k not in self.deleted_tracks}
            
            if not active_tracks:
                return False
            
            all_frames = set()
            for positions in active_tracks.values():
                for pos in positions:
                    all_frames.add(pos.frame)
            
            all_frames = sorted(all_frames)
            track_ids = sorted(active_tracks.keys())
            
            # Build columns including nose coordinates if available
            columns = ['frame']
            for tid in track_ids:
                columns.extend([f"worm_{tid}_x", f"worm_{tid}_y"])
                if self.has_nose_data:
                    columns.extend([f"worm_{tid}_nose_x", f"worm_{tid}_nose_y"])
            
            data = []
            for frame in all_frames:
                row = [frame]
                for track_id in track_ids:
                    pos = next((p for p in active_tracks[track_id] if p.frame == frame), None)
                    if pos:
                        row.extend([round(pos.x, 4), round(pos.y, 4)])
                        if self.has_nose_data:
                            nose_x = round(pos.nose_x, 4) if pos.nose_x is not None else None
                            nose_y = round(pos.nose_y, 4) if pos.nose_y is not None else None
                            row.extend([nose_x, nose_y])
                    else:
                        row.extend([None, None])
                        if self.has_nose_data:
                            row.extend([None, None])
                data.append(row)
            
            df = pd.DataFrame(data, columns=columns)
            df.to_csv(file_path, index=False)
            
            logger.info(f"Saved {len(active_tracks)} tracks to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save tracks: {e}")
            return False

    def get_position_coordinates(self, position: TrackPosition, use_nose: bool = False) -> Tuple[float, float]:
        """Get x,y coordinates from position (centroid or nose)"""
        if use_nose and position.nose_x is not None and position.nose_y is not None:
            return position.nose_x, position.nose_y
        return position.x, position.y

# ================== DISPLAY SYSTEM ==================

class BlittedDisplay:
    """High-performance display using matplotlib blitting"""
    
    def __init__(self, ax, canvas):
        self.ax = ax
        self.canvas = canvas
        self.background = None
        self.artists = []
        self.setup_blitting()
    
    def setup_blitting(self):
        """Setup blitting optimization"""
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
    
    def update_background_image(self, img: np.ndarray, extent: List[float]):
        """Update background image"""
        self.ax.clear()
        self.ax.set_facecolor('black')
        
        im = self.ax.imshow(img, cmap='gray', alpha=0.8, 
                           extent=extent, aspect='equal', 
                           interpolation='nearest')
        
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        return im
    
    def fast_update_tracks(self, artists_data: List):
        """Fast track update using blitting"""
        self.canvas.restore_region(self.background)
        
        for artist in self.artists:
            try:
                artist.remove()
            except:
                pass
        self.artists.clear()
        
        for artist_type, data, kwargs in artists_data:
            try:
                if artist_type == 'scatter':
                    artist = self.ax.scatter(*data, **kwargs)
                elif artist_type == 'plot':
                    artist = self.ax.plot(*data, **kwargs)[0]
                elif artist_type == 'text':
                    artist = self.ax.text(*data, **kwargs)
                else:
                    continue
                
                self.artists.append(artist)
                self.ax.draw_artist(artist)
            except:
                pass
        
        self.canvas.blit(self.ax.bbox)
    
    def full_redraw(self):
        """Force full redraw"""
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

# ================== MAIN APPLICATION ==================

class UltraOptimizedTrackEditor:
    """Main application with clean architecture"""
    
    def __init__(self):
        # Core components
        self.image_cache: Optional[LazyImageCache] = None
        self.track_manager = TrackManager()
        self.blitted_display: Optional[BlittedDisplay] = None
        
        # Application state
        self.current_frame = 0
        self.playing = False
        self.fps = Config.DEFAULT_FPS
        self.trail_length = Config.DEFAULT_TRAIL_LENGTH
        self.last_displayed_frame = -1
        self.display_extent = None
        self.update_pending = False
        
        # Display options
        self.show_trails = True
        self.show_labels = True
        self.show_current_positions = True
        self.show_inactive_tracks = False
        self.use_nose_coordinates = False  # Toggle for nose vs centroid display
        
        # Editing state
        self.editing_enabled = False
        self.selection_mode = False
        self.selected_tracks: set = set()
        
        # Performance tracking
        self.frame_times = deque(maxlen=60)
        self.last_stats_update = time.time()
        
        # Initialize GUI
        self.setup_gui()
    
    def setup_gui(self):
        """Initialize GUI"""
        self.root = tk.Tk()
        self.root.title("SWT Track Editor")
        self.root.geometry("1600x1000")
        self.root.minsize(1400, 800)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        plt.ioff()
        plt.rcParams['figure.facecolor'] = 'black'
        plt.rcParams['axes.facecolor'] = 'black'
        
        self.create_interface()

        try:
            self.root.bind('<s>', lambda e: self.split_selected_track())
            self.root.bind('<S>', lambda e: self.split_selected_track())
        except Exception:
            pass    
    def create_interface(self):
        """Create complete interface"""
        # Top controls
        controls_frame = ttk.Frame(self.root)
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        # File loading
        load_frame = ttk.LabelFrame(controls_frame, text="Load Data", padding=5)
        load_frame.pack(fill='x', pady=(0, 5))
        
        ttk.Button(load_frame, text="Load Track CSV", command=self.load_track_csv).pack(side='left', padx=5)
        ttk.Button(load_frame, text="Load Image Directory", command=self.load_image_directory).pack(side='left', padx=5)
        
        self.status_label = ttk.Label(load_frame, text="No data loaded")
        self.status_label.pack(side='left', padx=20)
        
        self.perf_label = ttk.Label(load_frame, text="")
        self.perf_label.pack(side='right', padx=20)
        
        # Playback controls
        self.create_playback_controls(controls_frame)
        
        # Display options
        self.create_display_options(controls_frame)
        
        # Track editing
        self.create_editing_controls(controls_frame)
        
        # Main display
        self.create_display_area()
        
        # Start performance monitoring
        self.update_performance_stats()
    
    def create_playback_controls(self, parent):
        """Create playback control panel"""
        playback_frame = ttk.LabelFrame(parent, text="Playback Controls", padding=5)
        playback_frame.pack(fill='x', pady=5)
        
        # Navigation buttons
        nav_frame = ttk.Frame(playback_frame)
        nav_frame.pack(fill='x')
        
        ttk.Button(nav_frame, text="<<", width=3, command=self.goto_start).pack(side='left', padx=2)
        ttk.Button(nav_frame, text="<", width=3, command=self.step_backward_10).pack(side='left', padx=2)
        ttk.Button(nav_frame, text="[", width=3, command=self.step_backward).pack(side='left', padx=2)
        
        self.play_button = ttk.Button(nav_frame, text=">", width=3, command=self.toggle_playback)
        self.play_button.pack(side='left', padx=5)
        
        ttk.Button(nav_frame, text="]", width=3, command=self.step_forward).pack(side='left', padx=2)
        ttk.Button(nav_frame, text=">", width=3, command=self.step_forward_10).pack(side='left', padx=2)
        ttk.Button(nav_frame, text=">>", width=3, command=self.goto_end).pack(side='left', padx=2)
        
        # Frame slider
        slider_frame = ttk.Frame(playback_frame)
        slider_frame.pack(fill='x', pady=5)
        
        ttk.Label(slider_frame, text="Frame:").pack(side='left')
        self.frame_var = tk.IntVar(value=0)
        self.frame_slider = tk.Scale(slider_frame, from_=0, to=100, orient='horizontal',
                                    variable=self.frame_var, command=self.on_frame_change,
                                    length=400, resolution=1)
        self.frame_slider.pack(side='left', fill='x', expand=True, padx=10)
        
        self.frame_label = ttk.Label(slider_frame, text="Frame 0 / 0")
        self.frame_label.pack(side='right')
    
    def create_display_options(self, parent):
        """Create display options panel"""
        viz_frame = ttk.LabelFrame(parent, text="Display Options", padding=5)
        viz_frame.pack(fill='x', pady=5)
        
        # Performance controls
        perf_frame = ttk.Frame(viz_frame)
        perf_frame.pack(side='left', padx=10)
        
        ttk.Label(perf_frame, text="FPS:").pack(side='left')
        self.fps_var = tk.StringVar(value=str(self.fps))
        fps_entry = ttk.Entry(perf_frame, textvariable=self.fps_var, width=5)
        fps_entry.pack(side='left', padx=5)
        fps_entry.bind('<Return>', self.update_fps)
        
        ttk.Label(perf_frame, text="Trail:").pack(side='left', padx=(10, 0))
        self.trail_var = tk.StringVar(value=str(self.trail_length))
        trail_entry = ttk.Entry(perf_frame, textvariable=self.trail_var, width=5)
        trail_entry.pack(side='left', padx=5)
        trail_entry.bind('<Return>', self.update_trail_length)
        
        # Display options
        options_frame = ttk.Frame(viz_frame)
        options_frame.pack(side='left', padx=20)
        
        self.show_trails_var = tk.BooleanVar(value=self.show_trails)
        ttk.Checkbutton(options_frame, text="Trails", variable=self.show_trails_var,
                       command=self.update_display_options).pack(side='left', padx=5)
        
        self.show_labels_var = tk.BooleanVar(value=self.show_labels)
        ttk.Checkbutton(options_frame, text="Labels", variable=self.show_labels_var,
                       command=self.update_display_options).pack(side='left', padx=5)
        
        self.show_positions_var = tk.BooleanVar(value=self.show_current_positions)
        ttk.Checkbutton(options_frame, text="Positions", variable=self.show_positions_var,
                       command=self.update_display_options).pack(side='left', padx=5)
        
        # Nose coordinate toggle
        self.use_nose_var = tk.BooleanVar(value=self.use_nose_coordinates)
        self.nose_checkbox = ttk.Checkbutton(options_frame, text="Use Nose", 
                                           variable=self.use_nose_var,
                                           command=self.update_nose_option, state='disabled')
        self.nose_checkbox.pack(side='left', padx=5)

        # Performance options
        perf_options_frame = ttk.Frame(viz_frame)
        perf_options_frame.pack(side='right', padx=20)
        
        ttk.Button(perf_options_frame, text="Clear Cache",
                  command=self.clear_cache).pack(side='left', padx=5)
        ttk.Button(perf_options_frame, text="Stats",
                  command=self.show_performance_stats).pack(side='left', padx=5)
    
    def create_editing_controls(self, parent):
        """Create track editing control panel"""
        editing_frame = ttk.LabelFrame(parent, text="TRACK EDITING", padding=5)
        editing_frame.pack(fill='x', pady=5)
        
        # Enable editing
        edit_toggle_frame = ttk.Frame(editing_frame)
        edit_toggle_frame.pack(fill='x', pady=2)
        
        self.editing_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(edit_toggle_frame, text="Enable Track Editing",
                       variable=self.editing_var, command=self.toggle_editing_mode).pack(side='left')
        
        # Selection controls
        selection_frame = ttk.Frame(editing_frame)
        selection_frame.pack(fill='x', pady=2)
        
        self.selection_button = ttk.Button(selection_frame, text="Select Tracks",
                                          command=self.toggle_selection_mode, state='disabled')
        self.selection_button.pack(side='left', padx=2)
        
        self.clear_selection_button = ttk.Button(selection_frame, text="Clear Selection",
                                                command=self.clear_selection, state='disabled')
        self.clear_selection_button.pack(side='left', padx=2)
        
        self.selected_tracks_label = ttk.Label(selection_frame, text="Selected: None")
        self.selected_tracks_label.pack(side='left', padx=(10, 0))
        
        # Action controls
        action_frame = ttk.Frame(editing_frame)
        action_frame.pack(fill='x', pady=2)
        
        self.delete_button = ttk.Button(action_frame, text="Delete Selected",
                                       command=self.delete_selected_tracks, state='disabled')
        self.delete_button.pack(side='left', padx=2)
        
        self.keep_button = ttk.Button(action_frame, text="Keep Selected",
                                     command=self.keep_selected_tracks, state='disabled')
        self.keep_button.pack(side='left', padx=2)
        
        self.merge_button = ttk.Button(action_frame, text="Merge Selected",
                                      command=self.merge_selected_tracks, state='disabled')
        self.merge_button.pack(side='left', padx=2)
        
        self.split_button = ttk.Button(action_frame, text="Split @ Frame",
                                       command=self.split_selected_track, state='disabled')
        self.split_button.pack(side='left', padx=2)
        self.save_edits_button = ttk.Button(action_frame, text="Save Edits",
                                           command=self.save_edited_tracks, state='disabled')
        self.save_edits_button.pack(side='left', padx=2)
    
    def create_display_area(self):
        """Create matplotlib display area"""
        display_frame = ttk.Frame(self.root)
        display_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.viewer_fig = Figure(figsize=(12, 8), tight_layout=True, facecolor='black')
        self.viewer_canvas = FigureCanvasTkAgg(self.viewer_fig, display_frame)
        self.viewer_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.viewer_ax = self.viewer_fig.add_subplot(111)
        self.viewer_ax.set_facecolor('black')
        self.blitted_display = BlittedDisplay(self.viewer_ax, self.viewer_canvas)
        
        self.viewer_canvas.mpl_connect('button_press_event', self.on_canvas_click)

    def update_nose_option(self):
        """Handle nose coordinate toggle"""
        self.use_nose_coordinates = self.use_nose_var.get()
        logger.info(f"Switched to {'nose' if self.use_nose_coordinates else 'centroid'} coordinates")
        self.update_frame_display(force_update=True, use_blitting=False)
    
    # ================== FILE LOADING ==================
    
    def load_track_csv(self):
        """Load track CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select Track CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        self.load_csv_from_path(file_path)
    
    def load_csv_from_path(self, file_path: str) -> bool:
        """Load CSV from path"""
        try:
            logger.info(f"Loading CSV: {file_path}")
            df = pd.read_csv(file_path)
            
            if self.track_manager.load_from_dataframe(df):
                self.update_interface_after_load()
                nose_status = " (with nose data)" if self.track_manager.has_nose_data else ""
                self.status_label.config(text=f"Tracks: {len(self.track_manager.tracks)} loaded{nose_status}")
                
                # Enable nose toggle if nose data is available
                if self.track_manager.has_nose_data:
                    self.nose_checkbox.config(state='normal')
                else:
                    self.nose_checkbox.config(state='disabled')
                    self.use_nose_coordinates = False
                    self.use_nose_var.set(False)
                
                return True
            else:
                messagebox.showerror("Error", "Could not parse CSV format")
                return False
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")
            return False
    
    def load_image_directory(self):
        """Load image directory"""
        directory = filedialog.askdirectory(title="Select directory containing images")
        if not directory:
            return
        
        self.load_images_from_dir(directory)
    
    def load_images_from_dir(self, directory: str) -> bool:
        """Load images from directory"""
        try:
            if not os.path.isdir(directory):
                messagebox.showerror("Error", f"Not a directory: {directory}")
                return False
            
            logger.info("Scanning for images...")
            image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}
            
            image_files = [
                os.path.join(directory, f)
                for f in sorted(os.listdir(directory))
                if Path(f).suffix.lower() in image_extensions
            ]
            
            if image_files:
                logger.info(f"Creating cache for {len(image_files)} images...")
                
                if self.image_cache:
                    self.image_cache.shutdown()
                
                self.image_cache = LazyImageCache(image_files)
                self.image_cache.prefetch_range(0, min(10, len(image_files)), priority=1)
                
                self.update_interface_after_load()
                self.status_label.config(text=f"Images: {len(image_files)} loaded")
                return True
            else:
                messagebox.showerror("Error", "No images found")
                return False
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load images: {str(e)}")
            return False
    
    def update_interface_after_load(self):
        """Update interface after loading data"""
        if self.track_manager.tracks and self.image_cache:
            max_frame = min(self.track_manager.frame_count - 1, len(self.image_cache.image_files) - 1)
        elif self.track_manager.tracks:
            max_frame = self.track_manager.frame_count - 1
        elif self.image_cache:
            max_frame = len(self.image_cache.image_files) - 1
        else:
            return
        
        self.frame_slider.config(to=max_frame)
        self.current_frame = 0
        self.frame_var.set(0)
        self.update_frame_display(force_update=True)
    
    # ================== DISPLAY UPDATES ==================
    
    def update_frame_display(self, force_update=False, use_blitting=True):
        """Update frame display with optimization"""
        frame_start_time = time.time()
        
        if not force_update and self.current_frame == self.last_displayed_frame:
            return
        
        if self.update_pending and not force_update:
            return
        
        self.update_pending = True
        
        try:
            background_img = None
            if self.image_cache and self.current_frame < len(self.image_cache.image_files):
                background_img = self.image_cache.get_image(self.current_frame, priority=1)
            
            # Check if we jumped to a significantly different frame
            frame_jump = abs(self.current_frame - self.last_displayed_frame) > 5
            
            # Determine if we can use blitting
            can_use_blitting = (use_blitting and
                               background_img is not None and
                               self.display_extent is not None and
                               self.last_displayed_frame >= 0 and
                               not frame_jump)
            
            if not can_use_blitting or background_img is None or frame_jump:
                self._full_frame_redraw(background_img)
            else:
                self._blitted_track_update()
            
            # Update frame info
            total_frames = 0
            if self.track_manager.tracks:
                total_frames = max(total_frames, self.track_manager.frame_count)
            if self.image_cache:
                total_frames = max(total_frames, len(self.image_cache.image_files))
            
            self.frame_label.config(text=f"Frame {self.current_frame} / {total_frames - 1}")
            self.last_displayed_frame = self.current_frame
            
            frame_time = time.time() - frame_start_time
            self.frame_times.append(frame_time)
            
        finally:
            self.update_pending = False
    
    def _full_frame_redraw(self, background_img):
        """Full frame redraw"""
        self.viewer_ax.clear()
        self.viewer_ax.set_facecolor('black')
        
        if background_img is not None:
            display_extent_data = self.image_cache.get_display_extent(self.current_frame)
            
            if display_extent_data:
                display_extent, display_shape = display_extent_data
                self.display_extent = display_extent
                
                self.viewer_ax.imshow(background_img, cmap='gray', alpha=0.8,
                                    extent=self.display_extent, aspect='equal',
                                    interpolation='nearest', origin='upper')
                
                self.viewer_ax.set_xlim(display_extent[0], display_extent[1])
                self.viewer_ax.set_ylim(display_extent[2], display_extent[3])
            else:
                h, w = background_img.shape
                self.display_extent = [0, w, h, 0]
                self.viewer_ax.imshow(background_img, cmap='gray', alpha=0.8,
                                    extent=self.display_extent, aspect='equal',
                                    interpolation='nearest', origin='upper')
                self.viewer_ax.set_xlim(0, w)
                self.viewer_ax.set_ylim(h, 0)
        else:
            self._set_default_limits()
        
        if self.track_manager.tracks:
            self._draw_tracks_on_axis()
        
        title = f"SWT Track Editor - Frame {self.current_frame}"
        if self.use_nose_coordinates:
            title += " (Nose Coordinates)"
        
        if background_img is not None and self.image_cache:
            if self.current_frame in self.image_cache.original_image_sizes:
                orig_h, orig_w = self.image_cache.original_image_sizes[self.current_frame]
                ratio = self.image_cache.downsample_ratios.get(self.current_frame, 1.0)
                title += f" (Original: {orig_w}x{orig_h}, Display: {ratio:.1%})"
        
        self.viewer_ax.set_title(title, fontsize=14, color='white')
        self.viewer_ax.set_xlabel("X Position (pixels)", color='white')
        self.viewer_ax.set_ylabel("Y Position (pixels)", color='white')
        
        self.viewer_canvas.draw()
        
        if background_img is not None:
            self.blitted_display.background = self.viewer_canvas.copy_from_bbox(self.viewer_ax.bbox)
    
    def _blitted_track_update(self):
        """Fast track update using blitting"""
        if not self.track_manager.tracks:
            return
        
        artists_data = []
        active_tracks, inactive_tracks = self.track_manager.get_tracks_at_frame(
            self.current_frame, self.trail_length, use_nose=self.use_nose_coordinates)
        
        for track_id, positions in active_tracks.items():
            if not positions:
                continue
            
            color = self.track_manager.get_track_color(track_id)
            is_selected = track_id in self.selected_tracks
            
            # Get display coordinates for all positions
            valid_display_positions = []
            
            for pos in positions:
                # Use nose coordinates if requested and available
                orig_x, orig_y = self.track_manager.get_position_coordinates(pos, self.use_nose_coordinates)
                
                # Skip positions where we don't have the requested coordinate type
                if self.use_nose_coordinates and (pos.nose_x is None or pos.nose_y is None):
                    continue
                
                if self.image_cache:
                    if pos.frame in self.image_cache.downsample_ratios:
                        x, y = self.image_cache.get_display_coordinates(orig_x, orig_y, pos.frame)
                    else:
                        current_frame_ratio = self.image_cache.downsample_ratios.get(self.current_frame, 1.0)
                        x, y = orig_x * current_frame_ratio, orig_y * current_frame_ratio
                else:
                    x, y = orig_x, orig_y
                
                valid_display_positions.append((x, y, pos.frame))
            
            if not valid_display_positions:
                continue
            
            # For nose coordinates, create segmented trails to handle gaps
            if self.use_nose_coordinates and self.show_trails_var.get() and len(valid_display_positions) > 1:
                # Group consecutive positions to avoid connecting across gaps
                segments = []
                current_segment = [valid_display_positions[0]]
                
                for i in range(1, len(valid_display_positions)):
                    prev_frame = valid_display_positions[i-1][2]
                    curr_frame = valid_display_positions[i][2]
                    
                    # If frames are consecutive or close, continue segment
                    if curr_frame - prev_frame <= 3:  # Allow small gaps
                        current_segment.append(valid_display_positions[i])
                    else:
                        # Gap too large, start new segment
                        if len(current_segment) > 1:
                            segments.append(current_segment)
                        current_segment = [valid_display_positions[i]]
                
                # Add final segment
                if len(current_segment) > 1:
                    segments.append(current_segment)
                
                # Draw each segment
                for segment in segments:
                    if len(segment) > 1:
                        xs, ys = zip(*[(pos[0], pos[1]) for pos in segment])
                        artists_data.append(('plot', (xs, ys), {
                            'color': color, 'alpha': 0.6, 'linewidth': 1.5
                        }))
            
            # For centroid coordinates, draw normal continuous trail
            elif not self.use_nose_coordinates and self.show_trails_var.get() and len(valid_display_positions) > 1:
                xs, ys = zip(*[(pos[0], pos[1]) for pos in valid_display_positions])
                artists_data.append(('plot', (xs, ys), {
                    'color': color, 'alpha': 0.6, 'linewidth': 1.5
                }))
            
            # Current position - use the most recent valid position
            if self.show_current_positions and valid_display_positions:
                x, y, _ = valid_display_positions[-1]
                size = 150 if is_selected else 120
                artists_data.append(('scatter', ([x], [y]), {
                    'c': [color], 's': [size],
                    'edgecolors': 'white', 'linewidth': 2, 'zorder': 5
                }))
                
            if self.show_labels_var.get():
                label_text = f'{track_id}' + ('*' if is_selected else '')
                label_color = 'yellow' if is_selected else 'white'
                artists_data.append(('text', (x, y, label_text), {
                    'fontsize': 10, 'fontweight': 'bold',
                    'color': label_color, 'xytext': (5, 5),
                    'textcoords': 'offset points'
                }))
        
        # Add coordinate type indicator
        coord_type = "NOSE" if self.use_nose_coordinates else "CENTROID"
        artists_data.append(('text', (0.98, 0.02, f"Mode: {coord_type}"), {
            'transform': self.viewer_ax.transAxes, 'fontsize': 10, 'fontweight': 'bold',
            'bbox': dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7, edgecolor='white'),
            'color': 'white', 'horizontalalignment': 'right', 'verticalalignment': 'bottom'
        }))
        
        if self.editing_enabled and self.selection_mode:
            artists_data.append(('text', (0.02, 0.98, "SELECTION MODE: Click tracks to select"), {
                'transform': self.viewer_ax.transAxes, 'fontsize': 12, 'fontweight': 'bold',
                'bbox': dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                'verticalalignment': 'top'
            }))
        
        self.blitted_display.fast_update_tracks(artists_data)
    
    def _draw_tracks_on_axis(self):
        """Draw tracks on axis for full redraws"""
        active_tracks, inactive_tracks = self.track_manager.get_tracks_at_frame(
            self.current_frame, self.trail_length, use_nose=self.use_nose_coordinates)
        
        all_xs, all_ys, all_colors, all_sizes = [], [], [], []
        
        for track_id, positions in active_tracks.items():
            if not positions:
                continue
            
            color = self.track_manager.get_track_color(track_id)
            is_selected = track_id in self.selected_tracks
            
            # Get display coordinates for all positions
            valid_display_positions = []
            
            for pos in positions:
                # Use nose coordinates if requested and available
                orig_x, orig_y = self.track_manager.get_position_coordinates(pos, self.use_nose_coordinates)
                
                # Skip positions where we don't have the requested coordinate type
                if self.use_nose_coordinates and (pos.nose_x is None or pos.nose_y is None):
                    continue
                
                if self.image_cache:
                    if pos.frame in self.image_cache.downsample_ratios:
                        x, y = self.image_cache.get_display_coordinates(orig_x, orig_y, pos.frame)
                    else:
                        current_frame_ratio = self.image_cache.downsample_ratios.get(self.current_frame, 1.0)
                        x, y = orig_x * current_frame_ratio, orig_y * current_frame_ratio
                else:
                    x, y = orig_x, orig_y
                
                valid_display_positions.append((x, y, pos.frame))
            
            if not valid_display_positions:
                continue
            
            # Handle trails with gap awareness for nose coordinates
            if self.show_trails_var.get() and len(valid_display_positions) > 1:
                if self.use_nose_coordinates:
                    # Group consecutive positions to avoid connecting across gaps
                    segments = []
                    current_segment = [valid_display_positions[0]]
                    
                    for i in range(1, len(valid_display_positions)):
                        prev_frame = valid_display_positions[i-1][2]
                        curr_frame = valid_display_positions[i][2]
                        
                        # If frames are consecutive or close, continue segment
                        if curr_frame - prev_frame <= 3:  # Allow small gaps
                            current_segment.append(valid_display_positions[i])
                        else:
                            # Gap too large, start new segment
                            if len(current_segment) > 1:
                                segments.append(current_segment)
                            current_segment = [valid_display_positions[i]]
                    
                    # Add final segment
                    if len(current_segment) > 1:
                        segments.append(current_segment)
                    
                    # Draw each segment
                    for segment in segments:
                        if len(segment) > 1:
                            xs, ys = zip(*[(pos[0], pos[1]) for pos in segment])
                            self.viewer_ax.plot(xs, ys, color=color, alpha=0.6, linewidth=1.5)
                else:
                    # For centroid coordinates, draw normal continuous trail
                    xs, ys = zip(*[(pos[0], pos[1]) for pos in valid_display_positions])
                    self.viewer_ax.plot(xs, ys, color=color, alpha=0.6, linewidth=1.5)
            
            # Current position - use the most recent valid position
            if self.show_positions_var.get() and valid_display_positions:
                x, y, _ = valid_display_positions[-1]
                all_xs.append(x)
                all_ys.append(y)
                all_colors.append(color)
                all_sizes.append(150 if is_selected else 120)
                
                if self.show_labels_var.get():
                    label_text = f'{track_id}' + ('*' if is_selected else '')
                    label_color = 'yellow' if is_selected else 'white'
                    self.viewer_ax.annotate(label_text, (x, y), xytext=(5, 5),
                                          textcoords='offset points', fontsize=10,
                                          fontweight='bold', color=label_color)
        
        if all_xs:
            self.viewer_ax.scatter(all_xs, all_ys, c=all_colors, s=all_sizes,
                                 edgecolors='white', linewidth=2, zorder=5)
        
        # Add coordinate type indicator
        coord_type = "NOSE" if self.use_nose_coordinates else "CENTROID"
        self.viewer_ax.text(0.98, 0.02, f"Mode: {coord_type}",
                          transform=self.viewer_ax.transAxes, fontsize=10, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7, edgecolor='white'),
                          color='white', horizontalalignment='right', verticalalignment='bottom')
        
        if self.editing_enabled and self.selection_mode:
            self.viewer_ax.text(0.02, 0.98, "SELECTION MODE: Click tracks to select",
                              transform=self.viewer_ax.transAxes, fontsize=12, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                              verticalalignment='top')
                
        if all_xs:
            self.viewer_ax.scatter(all_xs, all_ys, c=all_colors, s=all_sizes,
                                 edgecolors='white', linewidth=2, zorder=5)
        
        # Add coordinate type indicator
        coord_type = "NOSE" if self.use_nose_coordinates else "CENTROID"
        self.viewer_ax.text(0.98, 0.02, f"Mode: {coord_type}",
                          transform=self.viewer_ax.transAxes, fontsize=10, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7, edgecolor='white'),
                          color='white', horizontalalignment='right', verticalalignment='bottom')
        
        if self.editing_enabled and self.selection_mode:
            self.viewer_ax.text(0.02, 0.98, "SELECTION MODE: Click tracks to select",
                              transform=self.viewer_ax.transAxes, fontsize=12, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                              verticalalignment='top')
                
        if all_xs:
            self.viewer_ax.scatter(all_xs, all_ys, c=all_colors, s=all_sizes,
                                 edgecolors='white', linewidth=2, zorder=5)
        
        if self.editing_enabled and self.selection_mode:
            self.viewer_ax.text(0.02, 0.98, "SELECTION MODE: Click tracks to select",
                              transform=self.viewer_ax.transAxes, fontsize=12, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                              verticalalignment='top')
    
    def _set_default_limits(self):
        """Set default axis limits"""
        if self.track_manager.tracks:
            all_xs, all_ys = [], []
            for positions in self.track_manager.tracks.values():
                for pos in positions:
                    if self.use_nose_coordinates and pos.nose_x is not None and pos.nose_y is not None:
                        all_xs.append(pos.nose_x)
                        all_ys.append(pos.nose_y)
                    else:
                        all_xs.append(pos.x)
                        all_ys.append(pos.y)
            
            if all_xs and all_ys:
                margin = 50
                self.viewer_ax.set_xlim(min(all_xs) - margin, max(all_xs) + margin)
                self.viewer_ax.set_ylim(max(all_ys) + margin, min(all_ys) - margin)
    
    # ================== PLAYBACK CONTROLS ==================
    
    def toggle_playback(self):
        """Toggle playback"""
        if self.playing:
            self.stop_playback()
        else:
            self.start_playback()
    
    def start_playback(self):
        """Start playback with prefetching"""
        if not (self.track_manager.tracks or self.image_cache):
            return
        
        self.playing = True
        self.play_button.config(text="||")
        
        if self.image_cache:
            end_frame = min(self.current_frame + 50, len(self.image_cache.image_files))
            self.image_cache.prefetch_range(self.current_frame, end_frame, priority=2)
        
        self.schedule_next_frame()
    
    def schedule_next_frame(self):
        """Schedule next frame"""
        if not self.playing:
            return
        
        max_frame = 0
        if self.track_manager.tracks:
            max_frame = max(max_frame, self.track_manager.frame_count - 1)
        if self.image_cache:
            max_frame = max(max_frame, len(self.image_cache.image_files) - 1)
        
        if self.current_frame >= max_frame:
            self.stop_playback()
            return
        
        self.current_frame += 1
        self.frame_var.set(self.current_frame)
        
        self.update_frame_display(force_update=True, use_blitting=False)
        
        target_delay = int(1000 / self.fps)
        avg_frame_time = np.mean(self.frame_times) if self.frame_times else 0
        actual_delay = max(10, target_delay - int(avg_frame_time * 1000))
        
        self.root.after(actual_delay, self.schedule_next_frame)
    
    def stop_playback(self):
        """Stop playback"""
        self.playing = False
        self.play_button.config(text=">")
    
    def goto_frame(self, frame_idx):
        """Navigate to specific frame"""
        max_frame = 0
        if self.track_manager.tracks:
            max_frame = max(max_frame, self.track_manager.frame_count - 1)
        if self.image_cache:
            max_frame = max(max_frame, len(self.image_cache.image_files) - 1)
        
        new_frame = max(0, min(frame_idx, max_frame))
        
        if new_frame != self.current_frame:
            frame_jump = abs(new_frame - self.current_frame) > 5
            
            self.current_frame = new_frame
            self.frame_var.set(self.current_frame)
            
            if self.image_cache:
                self.image_cache.prefetch_range(
                    max(0, new_frame - 10),
                    min(len(self.image_cache.image_files), new_frame + 20),
                    priority=1
                )
            
            self.update_frame_display(force_update=True, use_blitting=not frame_jump)
    
    def on_frame_change(self, value):
        """Handle frame slider changes"""
        if not self.playing:
            new_frame = int(value)
            if new_frame != self.current_frame:
                if hasattr(self, '_slider_timer'):
                    self.root.after_cancel(self._slider_timer)
                
                self._slider_timer = self.root.after(30, lambda: self.goto_frame(new_frame))
    
    # Navigation methods
    def goto_start(self):
        self.goto_frame(0)
    
    def goto_end(self):
        max_frame = 0
        if self.track_manager.tracks:
            max_frame = max(max_frame, self.track_manager.frame_count - 1)
        if self.image_cache:
            max_frame = max(max_frame, len(self.image_cache.image_files) - 1)
        self.goto_frame(max_frame)
    
    def step_forward(self):
        self.goto_frame(self.current_frame + 1)
    
    def step_backward(self):
        self.goto_frame(self.current_frame - 1)
    
    def step_forward_10(self):
        self.goto_frame(self.current_frame + 10)
    
    def step_backward_10(self):
        self.goto_frame(self.current_frame - 10)
    
    # ================== TRACK EDITING ==================
    
    def toggle_editing_mode(self):
        """Toggle editing mode"""
        self.editing_enabled = self.editing_var.get()
        
        if self.editing_enabled:
            if self.track_manager.tracks:
                self.track_manager.original_tracks = {k: v.copy() for k, v in self.track_manager.tracks.items()}
                self.track_manager.deleted_tracks.clear()
                self.selected_tracks.clear()
            
            for btn in [self.selection_button, self.clear_selection_button,
                       self.delete_button, self.keep_button, self.merge_button,
                       self.split_button,self.save_edits_button]:
                btn.config(state='normal')
        else:
            self.selection_mode = False
            self.selected_tracks.clear()
            
            for btn in [self.selection_button, self.clear_selection_button,
                       self.delete_button, self.keep_button, self.merge_button,
                       self.split_button,self.save_edits_button]:
                btn.config(state='disabled')
            
            self.selection_button.config(text="Select Tracks")
        
        self.update_selection_display()
        self.update_frame_display(force_update=True)
    
    def toggle_selection_mode(self):
        """Toggle selection mode"""
        self.selection_mode = not self.selection_mode
        
        if self.selection_mode:
            self.selection_button.config(text="Selection ON")
        else:
            self.selection_button.config(text="Select Tracks")
        
        self.update_frame_display(force_update=True)
    
    def on_canvas_click(self, event):
        """Handle canvas clicks for selection"""
        if not self.editing_enabled or not self.selection_mode or event.inaxes is None:
            return
        
        click_x, click_y = event.xdata, event.ydata
        if click_x is None or click_y is None:
            return
        
        closest_track = None
        min_distance = float('inf')
        
        active_tracks, inactive_tracks = self.track_manager.get_tracks_at_frame(
            self.current_frame, use_nose=self.use_nose_coordinates)
        all_tracks = {**active_tracks, **inactive_tracks}
        
        for track_id, positions in all_tracks.items():
            if not positions:
                continue
            
            for pos in positions[-5:]:
                if abs(pos.frame - self.current_frame) <= 5:
                    orig_x, orig_y = self.track_manager.get_position_coordinates(pos, self.use_nose_coordinates)
                    
                    if self.image_cache:
                        x, y = self.image_cache.get_display_coordinates(orig_x, orig_y, pos.frame)
                    else:
                        x, y = orig_x, orig_y
                    
                    distance = np.sqrt((x - click_x)**2 + (y - click_y)**2)
                    if distance < min_distance and distance < Config.SELECTION_RADIUS:
                        min_distance = distance
                        closest_track = track_id
        
        if closest_track is not None:
            if closest_track in self.selected_tracks:
                self.selected_tracks.remove(closest_track)
            else:
                self.selected_tracks.add(closest_track)
            
            self.update_selection_display()
            self.update_frame_display(force_update=True)
    
    def clear_selection(self):
        """Clear selection"""
        self.selected_tracks.clear()
        self.update_selection_display()
        self.update_frame_display(force_update=True)
    
    def update_selection_display(self):
        """Update selection display"""
        if self.selected_tracks:
            selected_list = sorted(list(self.selected_tracks))
            self.selected_tracks_label.config(
                text=f"Selected: {selected_list[:5]}{'...' if len(selected_list) > 5 else ''} ({len(selected_list)} tracks)")
        else:
            self.selected_tracks_label.config(text="Selected: None")

        # Enable split only when editing is on and exactly one track is selected
        if hasattr(self, "split_button"):
            if self.editing_enabled and len(self.selected_tracks) == 1:
                self.split_button.config(state='normal')
            else:
                self.split_button.config(state='disabled')
    
    def delete_selected_tracks(self):
        """Delete selected tracks"""
        if not self.selected_tracks:
            messagebox.showwarning("No Selection", "No tracks selected!")
            return
        
        confirm = messagebox.askyesno("Confirm Deletion",
                                     f"Delete {len(self.selected_tracks)} tracks?")
        
        if confirm:
            self.track_manager.delete_tracks(self.selected_tracks)
            self.selected_tracks.clear()
            self.update_selection_display()
            self.update_frame_display(force_update=True)
    
    def keep_selected_tracks(self):
        """Keep only selected tracks"""
        if not self.selected_tracks:
            messagebox.showwarning("No Selection", "No tracks selected!")
            return
        
        all_track_ids = set(self.track_manager.tracks.keys()) - self.track_manager.deleted_tracks
        tracks_to_delete = all_track_ids - self.selected_tracks
        
        if not tracks_to_delete:
            messagebox.showinfo("No Action", "All existing tracks are already selected!")
            return
        
        confirm = messagebox.askyesno("Confirm Keep Selected",
                                     f"Keep {len(self.selected_tracks)} selected tracks and "
                                     f"delete {len(tracks_to_delete)} others?")
        
        if confirm:
            self.track_manager.delete_tracks(tracks_to_delete)
            self.selected_tracks.clear()
            self.update_selection_display()
            self.update_frame_display(force_update=True)
    
    def merge_selected_tracks(self):
        """Merge selected tracks"""
        if len(self.selected_tracks) < 2:
            messagebox.showwarning("Insufficient Selection", "Select at least 2 tracks!")
            return
        
        selected_list = sorted(list(self.selected_tracks))
        confirm = messagebox.askyesno("Confirm Merge",
                                     f"Merge {len(selected_list)} tracks into track {selected_list[0]}?")
        
        if confirm:
            if self.track_manager.merge_tracks(selected_list):
                self.selected_tracks.clear()
                self.update_selection_display()
                self.update_frame_display(force_update=True)

    def split_selected_track(self):
        """Split the single selected track at the current frame into two tracks."""
        # Guardrails
        if not self.selected_tracks or len(self.selected_tracks) != 1:
            messagebox.showwarning("Split", "Select exactly one track to split.")
            return

        track_id = next(iter(self.selected_tracks))

        # Perform split in the data model
        new_id = self.track_manager.split_track(track_id, self.current_frame)
        if new_id is None:
            messagebox.showinfo("Split", "Nothing to split at this frame (no future positions).")
            return

        # Make the effect clear — keep both parts selected
        self.selected_tracks = {track_id, new_id}

        # Refresh UI
        self.update_selection_display()
        self.update_frame_display(force_update=True)    
    def save_edited_tracks(self):
        """Save edited tracks"""
        if not self.track_manager.tracks:
            messagebox.showwarning("No Data", "No tracks to save!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save edited tracks",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            if self.track_manager.save_to_csv(file_path):
                active_tracks = {k: v for k, v in self.track_manager.tracks.items()
                               if k not in self.track_manager.deleted_tracks}
                messagebox.showinfo("Save Complete",
                                   f"Saved {len(active_tracks)} tracks to:\n{os.path.basename(file_path)}")
            else:
                messagebox.showerror("Save Error", "Failed to save tracks")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save: {e}")
    
    # ================== SETTINGS AND OPTIONS ==================
    
    def update_fps(self, event=None):
        """Update FPS setting"""
        try:
            fps = float(self.fps_var.get())
            if 0.1 <= fps <= 120:
                self.fps = fps
            else:
                self.fps_var.set(str(self.fps))
        except ValueError:
            self.fps_var.set(str(self.fps))
    
    def update_trail_length(self, event=None):
        """Update trail length"""
        try:
            length = int(self.trail_var.get())
            if 1 <= length <= 200:
                self.trail_length = length
                self.update_frame_display(force_update=True)
            else:
                self.trail_var.set(str(self.trail_length))
        except ValueError:
            self.trail_var.set(str(self.trail_length))
    
    def update_display_options(self):
        """Update display options"""
        self.show_trails = self.show_trails_var.get()
        self.show_labels = self.show_labels_var.get()
        self.show_current_positions = self.show_positions_var.get()
        self.update_frame_display(force_update=True, use_blitting=False)
    
    def clear_cache(self):
        """Clear image cache"""
        if self.image_cache:
            self.image_cache.clear_cache()
            messagebox.showinfo("Cache Cleared", "Image cache has been cleared")
    
    def show_performance_stats(self):
        """Show detailed performance statistics"""
        if not self.image_cache:
            messagebox.showinfo("Performance Stats", "No image cache active")
            return
        
        stats = self.image_cache.get_cache_stats()
        
        if len(self.frame_times) > 0:
            avg_frame_time = np.mean(self.frame_times) * 1000
            max_frame_time = np.max(self.frame_times) * 1000
            min_frame_time = np.min(self.frame_times) * 1000
        else:
            avg_frame_time = max_frame_time = min_frame_time = 0
        
        memory_usage = psutil.Process().memory_info().rss / (1024**2)
        
        stats_text = f"""Performance Statistics:

Cache Performance:
• Hit Rate: {stats.hit_rate:.1f}%
• Hot Cache: {stats.hot_cache_size} images
• Warm Cache: {stats.warm_cache_size} images
• Load Queue: {stats.queue_size} pending

Frame Rendering:
• Average: {avg_frame_time:.1f}ms
• Min: {min_frame_time:.1f}ms  
• Max: {max_frame_time:.1f}ms
• Target FPS: {self.fps}

System:
• Memory Usage: {memory_usage:.1f}MB
• Total Images: {len(self.image_cache.image_files) if self.image_cache else 0}
• Total Tracks: {len(self.track_manager.tracks) if self.track_manager.tracks else 0}"""
        
        messagebox.showinfo("Performance Statistics", stats_text)
    
    # ================== PERFORMANCE MONITORING ==================
    
    def update_performance_stats(self):
        """Update performance display"""
        if self.image_cache and time.time() - self.last_stats_update > 2.0:
            stats = self.image_cache.get_cache_stats()
            
            if len(self.frame_times) > 10:
                avg_frame_time = np.mean(self.frame_times)
                actual_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            else:
                actual_fps = 0
            
            perf_text = (f"Cache: {stats.hit_rate:.1f}% hit | "
                        f"FPS: {actual_fps:.1f} | "
                        f"Load: {stats.avg_load_time_ms:.1f}ms")
            
            self.perf_label.config(text=perf_text)
            self.last_stats_update = time.time()
        
        self.root.after(Config.PERFORMANCE_UPDATE_INTERVAL, self.update_performance_stats)
    
    # ================== CLEANUP ==================
    
    def on_closing(self):
        """Clean shutdown"""
        logger.info("Shutting down SWT Track Editor...")
        
        self.playing = False
        
        if self.image_cache:
            self.image_cache.shutdown()
        
        if self.viewer_fig:
            try:
                plt.close(self.viewer_fig)
            except:
                pass
        
        gc.collect()
        
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass
    
    def run(self):
        """Start the application"""
        try:
            logger.info("Starting SWT Track Editor...")
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()
        except Exception as e:
            logger.error(f"Application error: {e}")
            self.on_closing()


# ================== ENTRY POINT ==================

def main():
    """Main entry point with command line support"""
    parser = argparse.ArgumentParser(description="SWT Track Editor")
    parser.add_argument("--csv", help="Path to track CSV to auto-load")
    parser.add_argument("--images", help="Path to image directory to auto-load")
    parser.add_argument("--cache-size", type=int, help="Override cache size")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help="Set logging level")
    args = parser.parse_args()
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info("=== SWT Track Editor - Production Ready ===")
    logger.info("Architecture Improvements:")
    logger.info("✓ Modular design with separated concerns")
    logger.info("✓ Protocol-based image loading with fallbacks")
    logger.info("✓ Immutable data models with type safety")
    logger.info("✓ Clean separation of GUI, data, and display logic")
    logger.info("✓ Proper logging instead of print statements")
    logger.info("✓ Configuration constants instead of magic numbers")
    logger.info("✓ Enhanced error handling and resource management")
    logger.info("✓ Full support for nose and centroid coordinates")
    logger.info("=" * 60)
    
    # Check for dependencies
    dependencies_info = []
    
    try:
        import tifffile
        dependencies_info.append("✓ Tifffile available for LZW/compressed TIFF support")
    except ImportError:
        dependencies_info.append("⚠ Tifffile not available - LZW compressed TIFFs may fail")
    
    try:
        from PIL import Image
        dependencies_info.append("✓ PIL/Pillow available as TIFF fallback")
    except ImportError:
        dependencies_info.append("⚠ PIL/Pillow not available")
    
    try:
        from skimage import io
        dependencies_info.append("✓ Scikit-image available for optimized processing")
    except ImportError:
        dependencies_info.append("⚠ Scikit-image not available, using OpenCV fallback")
    
    for info in dependencies_info:
        logger.info(info)
    
    # Create and configure editor
    editor = UltraOptimizedTrackEditor()
    
    # Auto-load data if provided
    if args.csv:
        logger.info(f"Auto-loading CSV: {args.csv}")
        editor.load_csv_from_path(args.csv)
    
    if args.images:
        logger.info(f"Auto-loading images: {args.images}")
        editor.load_images_from_dir(args.images)
    
    # Override cache size if specified
    if args.cache_size and editor.image_cache:
        logger.info(f"Overriding cache size to: {args.cache_size}")
        editor.image_cache.cache_size = args.cache_size
    
    # Start the application
    editor.run()


if __name__ == "__main__":
    main()