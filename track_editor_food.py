#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SWT Track Editor with Persistent Object Optimization

Key optimization: Persistent imshow object and track collections that never get cleared,
using set_data() and set_segments() for ultra-fast updates.
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
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors

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

@dataclass
class PrerenderedTrackData:
    """Pre-rendered track data for fast display"""
    track_id: int
    original_positions: List[TrackPosition]
    display_positions: List[Tuple[float, float, int]]
    line_segments: List[np.ndarray]
    segment_frames: List[int]
    color: str

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
        self.global_downsample_ratio: float = 1.0
        self.original_image_size: Optional[Tuple[int, int]] = None

        # Background processing
        self.load_queue = queue.PriorityQueue()
        self.metadata_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=Config.MAX_BACKGROUND_WORKERS)
        self.is_shutdown = False

        # Performance tracking
        self.hit_count = 0
        self.miss_count = 0
        self.load_times = deque(maxlen=100)

        # Calculate global downsample ratio from first image
        self._determine_global_scaling()

        # Start background workers
        self._start_background_workers()

        logger.info(f"LazyImageCache initialized with {len(image_files)} images, cache size: {self.cache_size}")
        logger.info(f"Global downsample ratio: {self.global_downsample_ratio}")

    def _determine_global_scaling(self):
        """Determine global downsample ratio from first image"""
        if not self.image_files:
            return

        try:
            # Load first image to determine scaling
            first_img = self.loader.load_image(self.image_files[0])
            if first_img is not None:
                self.original_image_size = first_img.shape

                if max(self.original_image_size) > max(Config.MAX_DISPLAY_SIZE):
                    self.global_downsample_ratio = min(
                        Config.MAX_DISPLAY_SIZE[0] / self.original_image_size[0],
                        Config.MAX_DISPLAY_SIZE[1] / self.original_image_size[1]
                    )
                else:
                    self.global_downsample_ratio = 1.0

                logger.info(f"Original image size: {self.original_image_size}")
                logger.info(f"Global downsample ratio: {self.global_downsample_ratio}")

        except Exception as e:
            logger.warning(f"Could not determine scaling from first image: {e}")
            self.global_downsample_ratio = 1.0

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

            # Apply global downsampling
            if self.global_downsample_ratio < 1.0:
                new_height = int(img.shape[0] * self.global_downsample_ratio)
                new_width = int(img.shape[1] * self.global_downsample_ratio)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

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

    def get_display_extent(self) -> Optional[Tuple[List[float], Tuple[int, int]]]:
        """Get display extent using global scaling"""
        if self.original_image_size:
            original_h, original_w = self.original_image_size
            display_w = int(original_w * self.global_downsample_ratio)
            display_h = int(original_h * self.global_downsample_ratio)
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
        self.has_nose_data = False

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
                if '_x' in col and '_nose_' not in col:
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

                    for idx, row in df.iterrows():
                        frame = int(row['frame'])
                        x = row[x_col]
                        y = row[y_col]

                        if pd.notna(x) and pd.notna(y):
                            nose_x = None
                            nose_y = None

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
                        positions.sort(key=lambda p: p.frame)
                        self.tracks[track_id] = positions

            self.original_tracks = {k: v.copy() for k, v in self.tracks.items()}
            self.deleted_tracks.clear()
            self.frame_count = int(df['frame'].max()) + 1

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

            current_positions = [pos for pos in positions if pos.frame <= frame_idx]
            current_frame_positions = [pos for pos in positions if pos.frame == frame_idx]

            if current_positions:
                if len(current_positions) > trail_length:
                    current_positions = current_positions[-trail_length:]

                if use_nose and self.has_nose_data:
                    valid_positions = []
                    for pos in current_positions:
                        if pos.nose_x is not None and pos.nose_y is not None:
                            valid_positions.append(pos)

                    valid_current_frame = []
                    for pos in current_frame_positions:
                        if pos.nose_x is not None and pos.nose_y is not None:
                            valid_current_frame.append(pos)

                    if valid_positions:
                        if valid_current_frame:
                            active_tracks[track_id] = valid_positions
                        else:
                            inactive_tracks[track_id] = valid_positions
                else:
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

    def _analyze_track_conflicts(self, track_ids: List[int]) -> Dict:
        """Analyze potential conflicts between tracks before merging"""
        analysis = {
            'total_positions': 0,
            'frame_conflicts': {},
            'temporal_gaps': [],
            'track_summaries': {},
            'merge_viable': True,
            'warnings': []
        }
        
        # Collect all positions and analyze each track
        all_positions_by_frame = {}
        
        for track_id in track_ids:
            if track_id not in self.tracks or track_id in self.deleted_tracks:
                analysis['warnings'].append(f"Track {track_id} does not exist or is deleted")
                continue
                
            positions = self.tracks[track_id]
            if not positions:
                analysis['warnings'].append(f"Track {track_id} is empty")
                continue
                
            # Track summary
            frames = [pos.frame for pos in positions]
            analysis['track_summaries'][track_id] = {
                'count': len(positions),
                'start_frame': min(frames),
                'end_frame': max(frames),
                'frame_range': f"{min(frames)}-{max(frames)}",
                'avg_position': (
                    sum(pos.x for pos in positions) / len(positions),
                    sum(pos.y for pos in positions) / len(positions)
                )
            }
            
            analysis['total_positions'] += len(positions)
            
            # Check for frame conflicts
            for pos in positions:
                frame = pos.frame
                if frame not in all_positions_by_frame:
                    all_positions_by_frame[frame] = []
                all_positions_by_frame[frame].append((track_id, pos))
        
        # Identify conflicts
        for frame, positions in all_positions_by_frame.items():
            if len(positions) > 1:
                analysis['frame_conflicts'][frame] = positions
        
        # Check for temporal gaps
        if all_positions_by_frame:
            all_frames = sorted(all_positions_by_frame.keys())
            for i in range(len(all_frames) - 1):
                gap = all_frames[i + 1] - all_frames[i]
                if gap > 1:
                    analysis['temporal_gaps'].append((all_frames[i], all_frames[i + 1], gap - 1))
        
        # Determine if merge is viable
        if len(analysis['frame_conflicts']) > len(all_positions_by_frame) * 0.5:
            analysis['merge_viable'] = False
            analysis['warnings'].append("Too many frame conflicts - over 50% of frames have overlapping tracks")
        
        return analysis

    def _resolve_position_conflict(self, conflicted_positions: List[Tuple[int, TrackPosition]], 
                                 previous_positions: List[TrackPosition]) -> TrackPosition:
        """Resolve conflicts when multiple tracks have positions at the same frame"""
        
        if len(conflicted_positions) == 1:
            return conflicted_positions[0][1]
        
        # Strategy 1: If we have previous track history, choose position closest to trajectory
        if len(previous_positions) >= 2:
            # Calculate expected position based on velocity
            prev_pos = previous_positions[-1]
            prev_prev_pos = previous_positions[-2]
            
            velocity_x = prev_pos.x - prev_prev_pos.x
            velocity_y = prev_pos.y - prev_prev_pos.y
            
            expected_x = prev_pos.x + velocity_x
            expected_y = prev_pos.y + velocity_y
            
            best_distance = float('inf')
            best_position = conflicted_positions[0][1]
            
            for track_id, pos in conflicted_positions:
                distance = np.sqrt((pos.x - expected_x)**2 + (pos.y - expected_y)**2)
                if distance < best_distance:
                    best_distance = distance
                    best_position = pos
            
            logger.debug(f"Conflict resolution: chose position based on trajectory (distance: {best_distance:.1f})")
            return best_position
        
        # Strategy 2: If no trajectory history, choose position closest to previous position
        elif len(previous_positions) >= 1:
            prev_pos = previous_positions[-1]
            
            best_distance = float('inf')
            best_position = conflicted_positions[0][1]
            
            for track_id, pos in conflicted_positions:
                distance = np.sqrt((pos.x - prev_pos.x)**2 + (pos.y - prev_pos.y)**2)
                if distance < best_distance:
                    best_distance = distance
                    best_position = pos
            
            logger.debug(f"Conflict resolution: chose closest position (distance: {best_distance:.1f})")
            return best_position
        
        # Strategy 3: No history - choose from lowest track ID (most predictable)
        conflicted_positions.sort(key=lambda x: x[0])  # Sort by track ID
        chosen = conflicted_positions[0]
        logger.debug(f"Conflict resolution: chose from lowest track ID {chosen[0]}")
        return chosen[1]

    def merge_tracks(self, track_ids: List[int]) -> bool:
        """Merge tracks with robust conflict resolution and detailed debugging"""
        if len(track_ids) < 2:
            logger.warning("MERGE: Need at least 2 tracks to merge")
            return False

        logger.info(f"MERGE: Starting merge of tracks {track_ids}")
        
        # Analyze tracks before merging
        analysis = self._analyze_track_conflicts(track_ids)
        
        # Print detailed analysis
        logger.info("MERGE ANALYSIS:")
        logger.info(f"  Total positions to merge: {analysis['total_positions']}")
        logger.info(f"  Frame conflicts detected: {len(analysis['frame_conflicts'])}")
        logger.info(f"  Temporal gaps: {len(analysis['temporal_gaps'])}")
        
        for track_id, summary in analysis['track_summaries'].items():
            logger.info(f"  Track {track_id}: {summary['count']} positions, "
                       f"frames {summary['frame_range']}, "
                       f"avg pos ({summary['avg_position'][0]:.1f}, {summary['avg_position'][1]:.1f})")
        
        if analysis['warnings']:
            for warning in analysis['warnings']:
                logger.warning(f"  WARNING: {warning}")
        
        if not analysis['merge_viable']:
            logger.error("MERGE: Merge not viable due to excessive conflicts")
            return False
        
        # Proceed with merge
        target_id = track_ids[0]
        logger.info(f"MERGE: Target track ID: {target_id}")
        
        # Collect all positions grouped by frame
        frame_positions = {}
        for track_id in track_ids:
            if track_id in self.tracks and track_id not in self.deleted_tracks:
                for pos in self.tracks[track_id]:
                    frame = pos.frame
                    if frame not in frame_positions:
                        frame_positions[frame] = []
                    frame_positions[frame].append((track_id, pos))
        
        # Resolve conflicts and build final track
        final_positions = []
        conflicts_resolved = 0
        
        for frame in sorted(frame_positions.keys()):
            positions = frame_positions[frame]
            
            if len(positions) == 1:
                # No conflict
                final_positions.append(positions[0][1])
            else:
                # Conflict - resolve intelligently
                chosen_position = self._resolve_position_conflict(positions, final_positions)
                final_positions.append(chosen_position)
                conflicts_resolved += 1
                
                # Log conflict details
                conflict_details = [(tid, f"({pos.x:.1f},{pos.y:.1f})") for tid, pos in positions]
                logger.debug(f"MERGE: Frame {frame} conflict - chose from {conflict_details}")
        
        # Validate merge result
        if not final_positions:
            logger.error("MERGE: No positions to merge")
            return False
        
        logger.info(f"MERGE: Final track has {len(final_positions)} positions")
        logger.info(f"MERGE: Resolved {conflicts_resolved} frame conflicts")
        logger.info(f"MERGE: Frame range: {final_positions[0].frame}-{final_positions[-1].frame}")
        
        # Set the merged track
        self.tracks[target_id] = final_positions
        
        # Mark other tracks as deleted
        for track_id in track_ids[1:]:
            self.deleted_tracks.add(track_id)
            logger.info(f"MERGE: Marked track {track_id} as deleted")
        
        # Final validation
        gaps = []
        for i in range(len(final_positions) - 1):
            gap = final_positions[i + 1].frame - final_positions[i].frame
            if gap > 1:
                gaps.append(gap - 1)
        
        if gaps:
            total_missing = sum(gaps)
            logger.info(f"MERGE: Track has {len(gaps)} gaps totaling {total_missing} missing frames")
        
        logger.info(f"MERGE: Successfully merged {len(track_ids)} tracks into track {target_id}")
        return True

    def get_next_track_id(self) -> int:
        """Return the next available integer track id."""
        if not self.tracks:
            return 0
        return max(self.tracks.keys()) + 1

    def split_track(self, track_id: int, split_frame: int) -> Optional[int]:
        """Split a single track into two at the given split_frame."""
        if track_id not in self.tracks or track_id in self.deleted_tracks:
            return None

        positions = self.tracks[track_id]
        if not positions:
            return None

        left = [p for p in positions if p.frame <= split_frame]
        right = [p for p in positions if p.frame > split_frame]

        if not right:
            return None

        new_id = self.get_next_track_id()
        left.sort(key=lambda p: p.frame)
        right.sort(key=lambda p: p.frame)

        self.tracks[track_id] = left
        self.tracks[new_id] = right

        if track_id in self.deleted_tracks:
            self.deleted_tracks.discard(track_id)
        if new_id in self.deleted_tracks:
            self.deleted_tracks.discard(new_id)

        logger.info(f"SPLIT: Track {track_id} split at frame {split_frame}, created track {new_id}")
        logger.info(f"SPLIT: Original track now has {len(left)} positions (frames {left[0].frame}-{left[-1].frame})")
        logger.info(f"SPLIT: New track has {len(right)} positions (frames {right[0].frame}-{right[-1].frame})")

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

    def get_all_track_ids(self) -> List[int]:
        """Get all track IDs that aren't deleted"""
        return [tid for tid in self.tracks.keys() if tid not in self.deleted_tracks]

# ================== PERSISTENT TRACK OVERLAY SYSTEM ==================

class PersistentTrackOverlay:
    """Persistent track overlay system with set_data() and set_segments() optimization"""

    def __init__(self, ax, canvas):
        self.ax = ax
        self.canvas = canvas
        self.prerendered_tracks: Dict[int, PrerenderedTrackData] = {}
        self.line_collections: Dict[int, LineCollection] = {}
        self.full_track_collections: Dict[int, LineCollection] = {}
        self.position_scatter = None
        self.label_texts: List = []
        self.coordinate_mode_text = None
        self.selection_mode_text = None

        # Track state for updates
        self.tracks_need_update = True

    def update_prerendered_tracks(self, track_manager: TrackManager, image_cache: Optional[LazyImageCache],
                                 use_nose: bool = False):
        """Pre-render complete tracks with global scaling applied once"""
        logger.info("Pre-rendering tracks with global scaling...")
        self.prerendered_tracks.clear()
        self.line_collections.clear()
        self.full_track_collections.clear()

        active_tracks = {k: v for k, v in track_manager.tracks.items()
                        if k not in track_manager.deleted_tracks}

        # Get global downsample ratio
        downsample_ratio = image_cache.global_downsample_ratio if image_cache else 1.0

        for track_id, positions in active_tracks.items():
            if not positions:
                continue

            # Filter positions based on coordinate type
            valid_positions = []
            for pos in positions:
                if use_nose and track_manager.has_nose_data:
                    if pos.nose_x is not None and pos.nose_y is not None:
                        valid_positions.append(pos)
                else:
                    valid_positions.append(pos)

            if not valid_positions:
                continue

            # Convert to pre-scaled display coordinates
            display_positions = []
            for pos in valid_positions:
                orig_x, orig_y = track_manager.get_position_coordinates(pos, use_nose)
                display_x = orig_x * downsample_ratio
                display_y = orig_y * downsample_ratio
                display_positions.append((display_x, display_y, pos.frame))

            # Create line segments for LineCollection
            line_segments = []
            segment_frames = []

            if len(display_positions) > 1:
                for i in range(len(display_positions) - 1):
                    x1, y1, frame1 = display_positions[i]
                    x2, y2, frame2 = display_positions[i + 1]

                    # Handle gaps in nose data by not connecting distant frames
                    if use_nose and track_manager.has_nose_data:
                        if frame2 - frame1 <= 3:  # Allow small gaps
                            line_segments.append([(x1, y1), (x2, y2)])
                            segment_frames.append(frame1)
                    else:
                        line_segments.append([(x1, y1), (x2, y2)])
                        segment_frames.append(frame1)

            if line_segments:
                prerendered_track = PrerenderedTrackData(
                    track_id=track_id,
                    original_positions=valid_positions,
                    display_positions=display_positions,
                    line_segments=line_segments,
                    segment_frames=segment_frames,
                    color=track_manager.get_track_color(track_id)
                )

                self.prerendered_tracks[track_id] = prerendered_track

        logger.info(f"Pre-rendered {len(self.prerendered_tracks)} tracks with global scaling")
        self.tracks_need_update = False

    def add_persistent_collections_to_axis(self):
        """Add all pre-rendered track collections to axis once"""
        # Clear any existing collections
        self.clear_all()

        # Add all track collections to axis
        for track_id, prerendered_track in self.prerendered_tracks.items():
            if prerendered_track.line_segments:
                # Create complete LineCollection for this track (for trails)
                line_collection = LineCollection(
                    prerendered_track.line_segments,
                    colors=[prerendered_track.color],
                    alpha=0.6,
                    linewidths=1.5,
                    zorder=1
                )

                # Create full track LineCollection (for full track display)
                full_track_collection = LineCollection(
                    prerendered_track.line_segments,
                    colors=[prerendered_track.color],
                    alpha=0.3,
                    linewidths=1.0,
                    zorder=0
                )

                # Add to axis and store references
                self.ax.add_collection(line_collection)
                self.ax.add_collection(full_track_collection)
                self.line_collections[track_id] = line_collection
                self.full_track_collections[track_id] = full_track_collection

    def update_visibility_only(self, current_frame: int, trail_length: int, use_nose: bool,
                              selected_tracks: set, show_trails: bool, show_labels: bool,
                              show_positions: bool, editing_enabled: bool, selection_mode: bool,
                              show_full_tracks: bool = False):
        """Update visibility of existing collections without recreation"""

        # Store use_nose for mode indicator
        self._use_nose = use_nose

        # Clear previous position scatter and labels only
        if self.position_scatter:
            try:
                self.position_scatter.remove()
            except:
                pass
            self.position_scatter = None

        for text in self.label_texts:
            try:
                text.remove()
            except:
                pass
        self.label_texts.clear()

        if self.coordinate_mode_text:
            try:
                self.coordinate_mode_text.remove()
            except:
                pass
        if self.selection_mode_text:
            try:
                self.selection_mode_text.remove()
            except:
                pass

        # Update track collection visibility with optimized full tracks handling
        for track_id, line_collection in self.line_collections.items():
            if track_id in self.prerendered_tracks:
                prerendered_track = self.prerendered_tracks[track_id]

                if show_trails:
                    # Determine which segments should be visible
                    visible_segments = []
                    for i, segment_frame in enumerate(prerendered_track.segment_frames):
                        if (segment_frame <= current_frame and
                            segment_frame >= current_frame - trail_length):
                            visible_segments.append(prerendered_track.line_segments[i])

                    if visible_segments:
                        # Update the collection with visible segments
                        line_collection.set_segments(visible_segments)
                        line_collection.set_visible(True)
                    else:
                        line_collection.set_visible(False)
                else:
                    line_collection.set_visible(False)

        # Handle full tracks more efficiently
        if show_full_tracks:
            # Only update full tracks if they're not already visible
            for track_id, full_track_collection in self.full_track_collections.items():
                if not full_track_collection.get_visible():
                    if track_id in self.prerendered_tracks:
                        prerendered_track = self.prerendered_tracks[track_id]
                        full_track_collection.set_segments(prerendered_track.line_segments)
                        full_track_collection.set_visible(True)
        else:
            # Quickly hide all full tracks
            for full_track_collection in self.full_track_collections.values():
                if full_track_collection.get_visible():
                    full_track_collection.set_visible(False)

        # Handle current positions (these still need recreation but are fast)
        position_data = {'x': [], 'y': [], 'colors': [], 'sizes': []}

        for track_id, prerendered_track in self.prerendered_tracks.items():
            # Find current position
            current_position = None
            for x, y, frame in prerendered_track.display_positions:
                if frame == current_frame:
                    current_position = (x, y)
                    break

            if show_positions and current_position:
                x, y = current_position
                position_data['x'].append(x)
                position_data['y'].append(y)
                position_data['colors'].append(prerendered_track.color)

                is_selected = track_id in selected_tracks
                position_data['sizes'].append(150 if is_selected else 120)

                # Labels
                if show_labels:
                    label_text = f'{track_id}' + ('*' if is_selected else '')
                    label_color = 'yellow' if is_selected else 'white'

                    text = self.ax.annotate(
                        label_text, (x, y),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold', color=label_color,
                        zorder=6
                    )
                    self.label_texts.append(text)

        # Add current positions
        if position_data['x']:
            self.position_scatter = self.ax.scatter(
                position_data['x'], position_data['y'],
                c=position_data['colors'], s=position_data['sizes'],
                edgecolors='white', linewidth=2, zorder=5
            )

        # Add mode indicators (these are fast)
        coord_type = "NOSE" if use_nose else "CENTROID"
        self.coordinate_mode_text = self.ax.text(
            0.98, 0.02, f"Mode: {coord_type}",
            transform=self.ax.transAxes, fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7, edgecolor='white'),
            color='white', horizontalalignment='right', verticalalignment='bottom',
            zorder=10
        )

        if editing_enabled and selection_mode:
            self.selection_mode_text = self.ax.text(
                0.02, 0.98, "SELECTION MODE: Click tracks or use list",
                transform=self.ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                verticalalignment='top', zorder=10
            )

    def find_track_at_point(self, click_x: float, click_y: float, selection_radius: float = Config.SELECTION_RADIUS) -> Optional[int]:
        """Find track at a given point by checking both current positions and track segments"""
        closest_track = None
        min_distance = float('inf')

        for track_id, prerendered_track in self.prerendered_tracks.items():
            # Check distance to any part of the track segments
            for segment in prerendered_track.line_segments:
                if len(segment) == 2:
                    x1, y1 = segment[0]
                    x2, y2 = segment[1]
                    
                    # Calculate distance from point to line segment
                    distance = self._point_to_line_distance(click_x, click_y, x1, y1, x2, y2)
                    if distance < min_distance and distance < selection_radius:
                        min_distance = distance
                        closest_track = track_id

        return closest_track

    def _point_to_line_distance(self, px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate shortest distance from point to line segment"""
        # Vector from line start to end
        line_vec = np.array([x2 - x1, y2 - y1])
        # Vector from line start to point
        point_vec = np.array([px - x1, py - y1])
        
        # Length squared of line segment
        line_len_sq = np.dot(line_vec, line_vec)
        
        if line_len_sq == 0:
            # Degenerate case: line is actually a point
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        
        # Project point onto line, clamped to segment
        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
        
        # Find closest point on segment
        closest_x = x1 + t * line_vec[0]
        closest_y = y1 + t * line_vec[1]
        
        # Return distance to closest point
        return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

    def clear_all(self):
        """Clear all rendered elements"""
        for collection in self.line_collections.values():
            try:
                collection.remove()
            except:
                pass
        self.line_collections.clear()

        for collection in self.full_track_collections.values():
            try:
                collection.remove()
            except:
                pass
        self.full_track_collections.clear()

        if self.position_scatter:
            try:
                self.position_scatter.remove()
            except:
                pass
            self.position_scatter = None

        for text in self.label_texts:
            try:
                text.remove()
            except:
                pass
        self.label_texts.clear()

        if self.coordinate_mode_text:
            try:
                self.coordinate_mode_text.remove()
            except:
                pass
            self.coordinate_mode_text = None

        if self.selection_mode_text:
            try:
                self.selection_mode_text.remove()
            except:
                pass
            self.selection_mode_text = None

# ================== MAIN APPLICATION ==================

class UltraOptimizedTrackEditor:
    """Main application with persistent object optimization and robust update handling"""

    def __init__(self):
        # Core components
        self.image_cache: Optional[LazyImageCache] = None
        self.track_manager = TrackManager()
        self.track_overlay: Optional[PersistentTrackOverlay] = None

        # Application state
        self.current_frame = 0
        self.playing = False
        self.fps = Config.DEFAULT_FPS
        self.trail_length = Config.DEFAULT_TRAIL_LENGTH
        self.last_displayed_frame = -1
        self.display_extent = None
        self.tracks_need_update = True

        # Robust timer management for playback control
        self._playback_timer = None
        self._slider_timer = None
        self._timer_generation = 0
        
        # Race condition protection
        self._frame_update_pending = False
        self._draw_pending = False
        self._selection_update_pending = False  # NEW: Prevent selection race conditions

        # Display options
        self.show_trails = True
        self.show_labels = True
        self.show_current_positions = True
        self.show_inactive_tracks = False
        self.use_nose_coordinates = False
        self.show_full_tracks = False

        # Editing state
        self.editing_enabled = False
        self.selection_mode = False
        self.selected_tracks: set = set()

        # Performance tracking
        self.frame_times = deque(maxlen=60)
        self.last_stats_update = time.time()

        # Persistent display objects
        self._background_imshow = None

        # Initialize GUI
        self.setup_gui()

    def setup_gui(self):
        """Initialize GUI"""
        self.root = tk.Tk()
        self.root.title("SWT Track Editor - Persistent Object System")
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
        """Create complete interface with compact layout"""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill='both', expand=True, padx=5, pady=5)

        # Top controls (compact)
        controls_frame = ttk.Frame(main_container)
        controls_frame.pack(fill='x', pady=(0, 3))

        # File loading - single row
        load_frame = ttk.LabelFrame(controls_frame, text="Load Data", padding=3)
        load_frame.pack(fill='x', pady=(0, 2))

        ttk.Button(load_frame, text="Load CSV", command=self.load_track_csv).pack(side='left', padx=2)
        ttk.Button(load_frame, text="Load Images", command=self.load_image_directory).pack(side='left', padx=2)

        self.status_label = ttk.Label(load_frame, text="No data loaded")
        self.status_label.pack(side='left', padx=10)

        self.perf_label = ttk.Label(load_frame, text="")
        self.perf_label.pack(side='right', padx=10)

        # Playback controls - single row
        self.create_playback_controls(controls_frame)

        # Display and editing options - single row
        options_frame = ttk.LabelFrame(controls_frame, text="Options & Editing", padding=3)
        options_frame.pack(fill='x', pady=2)

        # Display options
        display_opts = ttk.Frame(options_frame)
        display_opts.pack(side='left', padx=5)

        ttk.Label(display_opts, text="FPS:").pack(side='left')
        self.fps_var = tk.StringVar(value=str(self.fps))
        fps_entry = ttk.Entry(display_opts, textvariable=self.fps_var, width=4)
        fps_entry.pack(side='left', padx=2)
        fps_entry.bind('<Return>', self.update_fps)

        ttk.Label(display_opts, text="Trail:").pack(side='left', padx=(5, 0))
        self.trail_var = tk.StringVar(value=str(self.trail_length))
        trail_entry = ttk.Entry(display_opts, textvariable=self.trail_var, width=4)
        trail_entry.pack(side='left', padx=2)
        trail_entry.bind('<Return>', self.update_trail_length)

        # Checkboxes
        checks_frame = ttk.Frame(options_frame)
        checks_frame.pack(side='left', padx=10)

        self.show_trails_var = tk.BooleanVar(value=self.show_trails)
        ttk.Checkbutton(checks_frame, text="Trails", variable=self.show_trails_var,
                       command=self.update_display_options).pack(side='left', padx=2)

        self.show_labels_var = tk.BooleanVar(value=self.show_labels)
        ttk.Checkbutton(checks_frame, text="Labels", variable=self.show_labels_var,
                       command=self.update_display_options).pack(side='left', padx=2)

        self.show_positions_var = tk.BooleanVar(value=self.show_current_positions)
        ttk.Checkbutton(checks_frame, text="Positions", variable=self.show_positions_var,
                       command=self.update_display_options).pack(side='left', padx=2)

        self.show_full_tracks_var = tk.BooleanVar(value=self.show_full_tracks)
        ttk.Checkbutton(checks_frame, text="Full Tracks", variable=self.show_full_tracks_var,
                       command=self.update_display_options).pack(side='left', padx=2)

        self.use_nose_var = tk.BooleanVar(value=self.use_nose_coordinates)
        self.nose_checkbox = ttk.Checkbutton(checks_frame, text="Use Nose",
                                           variable=self.use_nose_var,
                                           command=self.update_nose_option, state='disabled')
        self.nose_checkbox.pack(side='left', padx=2)

        # Editing controls - all in one line
        self.create_editing_controls(options_frame)

        # Main display area with track list
        display_container = ttk.Frame(main_container)
        display_container.pack(fill='both', expand=True)

        # Configure grid weights for 80% width to image, 20% to track list
        display_container.grid_columnconfigure(0, weight=4)  # Image area gets 80%
        display_container.grid_columnconfigure(1, weight=1)  # Track list gets 20%
        display_container.grid_rowconfigure(0, weight=1)

        # Image display area (80% width)
        self.create_display_area(display_container)

        # Track list area (20% width)
        self.create_track_list(display_container)

        # Start performance monitoring
        self.update_performance_stats()

    def create_playback_controls(self, parent):
        """Create compact playback control panel"""
        playback_frame = ttk.LabelFrame(parent, text="Playback", padding=3)
        playback_frame.pack(fill='x', pady=2)

        # Navigation buttons
        nav_frame = ttk.Frame(playback_frame)
        nav_frame.pack(side='left')

        ttk.Button(nav_frame, text="<<", width=3, command=self.goto_start).pack(side='left', padx=1)
        ttk.Button(nav_frame, text="<10", width=3, command=self.step_backward_10).pack(side='left', padx=1)
        ttk.Button(nav_frame, text="<", width=3, command=self.step_backward).pack(side='left', padx=1)

        self.play_button = ttk.Button(nav_frame, text=">", width=3, command=self.toggle_playback)
        self.play_button.pack(side='left', padx=2)

        ttk.Button(nav_frame, text=">", width=3, command=self.step_forward).pack(side='left', padx=1)
        ttk.Button(nav_frame, text="10>", width=3, command=self.step_forward_10).pack(side='left', padx=1)
        ttk.Button(nav_frame, text=">>", width=3, command=self.goto_end).pack(side='left', padx=1)

        # Frame slider
        slider_frame = ttk.Frame(playback_frame)
        slider_frame.pack(side='left', fill='x', expand=True, padx=10)

        ttk.Label(slider_frame, text="Frame:").pack(side='left')
        self.frame_var = tk.IntVar(value=0)
        self.frame_slider = tk.Scale(slider_frame, from_=0, to=100, orient='horizontal',
                                    variable=self.frame_var, command=self.on_frame_change,
                                    length=300, resolution=1)
        self.frame_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.frame_label = ttk.Label(slider_frame, text="Frame 0 / 0")
        self.frame_label.pack(side='right')

    def create_editing_controls(self, parent):
        """Create compact track editing controls"""
        editing_frame = ttk.Frame(parent)
        editing_frame.pack(side='right', padx=10)

        # Enable editing
        self.editing_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(editing_frame, text="Edit Mode",
                       variable=self.editing_var, command=self.toggle_editing_mode).pack(side='left', padx=2)

        # Action buttons
        self.delete_button = ttk.Button(editing_frame, text="Delete",
                                       command=self.delete_selected_tracks, state='disabled')
        self.delete_button.pack(side='left', padx=1)

        self.keep_button = ttk.Button(editing_frame, text="Keep",
                                     command=self.keep_selected_tracks, state='disabled')
        self.keep_button.pack(side='left', padx=1)

        self.merge_button = ttk.Button(editing_frame, text="Merge",
                                      command=self.merge_selected_tracks, state='disabled')
        self.merge_button.pack(side='left', padx=1)

        self.split_button = ttk.Button(editing_frame, text="Split",
                                       command=self.split_selected_track, state='disabled')
        self.split_button.pack(side='left', padx=1)

        self.save_edits_button = ttk.Button(editing_frame, text="Save",
                                           command=self.save_edited_tracks, state='disabled')
        self.save_edits_button.pack(side='left', padx=1)

    def create_display_area(self, parent):
        """Create matplotlib display area - 80% width"""
        display_frame = ttk.Frame(parent)
        display_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 3))

        self.viewer_fig = Figure(figsize=(10, 8), tight_layout=True, facecolor='black')
        self.viewer_canvas = FigureCanvasTkAgg(self.viewer_fig, display_frame)
        self.viewer_canvas.get_tk_widget().pack(fill='both', expand=True)

        self.viewer_ax = self.viewer_fig.add_subplot(111)
        self.viewer_ax.set_facecolor('black')

        # Initialize track overlay system
        self.track_overlay = PersistentTrackOverlay(self.viewer_ax, self.viewer_canvas)

        # RESTORED: Canvas click handling for track selection
        self.viewer_canvas.mpl_connect('button_press_event', self.on_canvas_click)

    def create_track_list(self, parent):
        """Create scrollable track list for selection - 20% width"""
        track_frame = ttk.LabelFrame(parent, text="Track List", padding=3)
        track_frame.grid(row=0, column=1, sticky='nsew')

        # Selection info
        self.selected_tracks_label = ttk.Label(track_frame, text="Selected: None")
        self.selected_tracks_label.pack(fill='x', pady=(0, 1))

        # ADD THIS: Multi-select instruction
        instruction_label = ttk.Label(track_frame, text="Ctrl+click for multi-select", 
                                 font=('TkDefaultFont', 8), foreground='gray')
        instruction_label.pack(fill='x', pady=(0, 3))

        # Scrollable listbox
        list_frame = ttk.Frame(track_frame)
        list_frame.pack(fill='both', expand=True)
    
        # ... rest of the method stays the same

        # Create listbox with scrollbar - FIXED: Added exportselection=False
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')

        self.track_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, 
                                       selectmode='extended', height=20,
                                       exportselection=False)  # KEY FIX: Prevents selection loss
        self.track_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.track_listbox.yview)

        # Bind selection events
        self.track_listbox.bind('<<ListboxSelect>>', self.on_track_list_select)

        # Selection buttons
        button_frame = ttk.Frame(track_frame)
        button_frame.pack(fill='x', pady=(3, 0))

        ttk.Button(button_frame, text="Select All", 
                  command=self.select_all_tracks).pack(side='left', padx=1)
        ttk.Button(button_frame, text="Clear", 
                  command=self.clear_selection).pack(side='left', padx=1)

    def update_track_list(self):
        """Update the track listbox with current tracks"""
        self.track_listbox.delete(0, tk.END)
        
        if self.track_manager.tracks:
            track_ids = self.track_manager.get_all_track_ids()
            for track_id in sorted(track_ids):
                color = self.track_manager.get_track_color(track_id)
                positions_count = len(self.track_manager.tracks[track_id])
                selected_indicator = "★" if track_id in self.selected_tracks else ""
                
                # Format: "Track 5 (120 pts) ★"
                display_text = f"Track {track_id} ({positions_count} pts) {selected_indicator}"
                self.track_listbox.insert(tk.END, display_text)

    # ROBUST SELECTION HANDLING WITH RACE CONDITION PROTECTION
    def _update_selection_safe(self, update_func, *args, **kwargs):
        """Safely update selection with race condition protection"""
        if self._selection_update_pending:
            return
        
        self._selection_update_pending = True
        try:
            update_func(*args, **kwargs)
        finally:
            self._selection_update_pending = False

    def on_track_list_select(self, event):
        """Handle track list selection with synchronization"""
        if not self.editing_enabled:
            return
        
        def _handle_list_selection():
            # Get selected indices
            selected_indices = self.track_listbox.curselection()
            
            # Clear current selection
            new_selection = set()
            
            # Add newly selected tracks
            if self.track_manager.tracks:
                track_ids = sorted(self.track_manager.get_all_track_ids())
                for idx in selected_indices:
                    if idx < len(track_ids):
                        track_id = track_ids[idx]
                        new_selection.add(track_id)
            
            # Only update if selection actually changed
            if new_selection != self.selected_tracks:
                self.selected_tracks = new_selection
                self.update_selection_display()
                self.update_tracks_only()
                # DON'T update track list here to avoid recursion
        
        self._update_selection_safe(_handle_list_selection)

    def on_canvas_click(self, event):
        """RESTORED: Handle canvas clicks for track selection with enhanced detection"""
        if not self.editing_enabled or not self.selection_mode or event.inaxes is None:
            return

        click_x, click_y = event.xdata, event.ydata
        if click_x is None or click_y is None:
            return

        def _handle_canvas_selection():
            # Use the enhanced track finding method
            closest_track = self.track_overlay.find_track_at_point(click_x, click_y)

            # Also check current position detection
            if closest_track is None and self.track_overlay and self.track_overlay.prerendered_tracks:
                min_distance = float('inf')
                for track_id, prerendered_track in self.track_overlay.prerendered_tracks.items():
                    # Find current position for this track
                    current_position = None
                    for x, y, frame in prerendered_track.display_positions:
                        if frame == self.current_frame:
                            current_position = (x, y)
                            break

                    if current_position:
                        x, y = current_position
                        distance = np.sqrt((x - click_x)**2 + (y - click_y)**2)
                        if distance < min_distance and distance < Config.SELECTION_RADIUS:
                            min_distance = distance
                            closest_track = track_id

            if closest_track is not None:
                # Handle Ctrl+click for multiple selection
                if event.key == 'control':
                    if closest_track in self.selected_tracks:
                        self.selected_tracks.remove(closest_track)
                    else:
                        self.selected_tracks.add(closest_track)
                else:
                    # Single selection
                    self.selected_tracks = {closest_track}

                self.update_selection_display()
                self.update_tracks_only()
                self._sync_listbox_selection()  # Sync with listbox

        self._update_selection_safe(_handle_canvas_selection)

    def _sync_listbox_selection(self):
        """Synchronize listbox selection with canvas selection"""
        if not self.track_manager.tracks:
            return
            
        # Clear listbox selection
        self.track_listbox.selection_clear(0, tk.END)
        
        # Set listbox selection to match current selection
        track_ids = sorted(self.track_manager.get_all_track_ids())
        for i, track_id in enumerate(track_ids):
            if track_id in self.selected_tracks:
                self.track_listbox.selection_set(i)
        
        # Update track list display to show stars
        self.update_track_list()

    def select_all_tracks(self):
        """Select all tracks in the list"""
        if not self.editing_enabled or not self.track_manager.tracks:
            return
        
        def _select_all():
            self.selected_tracks = set(self.track_manager.get_all_track_ids())
            
            # Update listbox selection
            self.track_listbox.select_set(0, tk.END)
            
            self.update_selection_display()
            self.update_tracks_only()
            self.update_track_list()
        
        self._update_selection_safe(_select_all)

    def clear_selection(self):
        """Clear selection with synchronization"""
        def _clear_selection():
            self.selected_tracks.clear()
            self.track_listbox.selection_clear(0, tk.END)
            self.update_selection_display()
            self.update_tracks_only()
            self.update_track_list()
        
        self._update_selection_safe(_clear_selection)

    def update_nose_option(self):
        """Handle nose coordinate toggle"""
        self.use_nose_coordinates = self.use_nose_var.get()
        logger.info(f"Switched to {'nose' if self.use_nose_coordinates else 'centroid'} coordinates")
        self.tracks_need_update = True
        self.update_frame_display(force_update=True)

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
                self.tracks_need_update = True
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

                # Update track list
                self.update_track_list()
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

                self.tracks_need_update = True
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

        # Force persistent axis setup
        self._background_imshow = None
        self.update_frame_display(force_update=True)

    # ================== ROBUST UPDATE HANDLING ==================

    def update_frame_display(self, force_update=False):
        """Update frame display with improved race condition protection"""
        # Allow force updates to bypass pending check, but not during playback updates
        if self._frame_update_pending and not force_update:
            return
        
        # For playback updates, check if we should abort early
        if not force_update and not self.playing:
            return
        
        self._frame_update_pending = True
        
        try:
            # Check playing state again after acquiring the lock
            if not force_update and not self.playing:
                return
                
            frame_start_time = time.time()

            # Update pre-rendered tracks if needed (only on track edits)
            if self.tracks_need_update and self.track_overlay and self.track_manager.tracks:
                self.track_overlay.update_prerendered_tracks(
                    self.track_manager, self.image_cache, self.use_nose_coordinates
                )
                self.tracks_need_update = False
                # Force axis setup since tracks changed
                self._setup_persistent_axis()

            # Check if we need to update display
            if not force_update and self.current_frame == self.last_displayed_frame:
                return

            # Initialize persistent objects if needed
            if not hasattr(self, '_background_imshow') or self._background_imshow is None:
                self._setup_persistent_axis()

            # Update background image data only
            if self.image_cache and self.current_frame < len(self.image_cache.image_files):
                background_img = self.image_cache.get_image(self.current_frame, priority=1)

                if background_img is not None and hasattr(self, '_background_imshow') and self._background_imshow is not None:
                    # Just update the image data - no recreation
                    self._background_imshow.set_data(background_img)

            # Update track visibility (no recreation)
            if self.track_overlay:
                self.track_overlay.update_visibility_only(
                    current_frame=self.current_frame,
                    trail_length=self.trail_length,
                    use_nose=self.use_nose_coordinates,
                    selected_tracks=self.selected_tracks,
                    show_trails=self.show_trails_var.get(),
                    show_labels=self.show_labels_var.get(),
                    show_positions=self.show_positions_var.get(),
                    editing_enabled=self.editing_enabled,
                    selection_mode=self.selection_mode,
                    show_full_tracks=self.show_full_tracks_var.get()
                )

            # Critical: Check playing state before any timer scheduling
            if not force_update and not self.playing:
                return

            # Minimal canvas update
            self.viewer_canvas.draw_idle()

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

        except Exception as e:
            logger.error(f"Error updating frame display: {e}")
        finally:
            self._frame_update_pending = False

    def update_tracks_only(self, force_redraw=True):
        """Ultra-lightweight track-only update for user interactions"""
        if not self.track_overlay:
            return
            
        start_time = time.perf_counter()
        
        # Only update track visibility - no image operations
        self.track_overlay.update_visibility_only(
            current_frame=self.current_frame,
            trail_length=self.trail_length,
            use_nose=self.use_nose_coordinates,
            selected_tracks=self.selected_tracks,
            show_trails=self.show_trails_var.get(),
            show_labels=self.show_labels_var.get(),
            show_positions=self.show_positions_var.get(),
            editing_enabled=self.editing_enabled,
            selection_mode=self.selection_mode,
            show_full_tracks=self.show_full_tracks_var.get()
        )
        
        # Only redraw if requested and no other draw is pending
        if force_redraw:
            self._execute_batched_draw()
        
        # Log if update takes too long
        update_time = time.perf_counter() - start_time
        if update_time > 0.005:  # 5ms threshold
            logger.debug(f"Slow track update: {update_time*1000:.1f}ms")

    def _execute_batched_draw(self):
        """Execute batched canvas draw to prevent multiple rapid redraws"""
        if not self._draw_pending:
            self._draw_pending = True
            # Use after_idle to batch multiple rapid updates
            self.root.after_idle(self._perform_canvas_draw)

    def _perform_canvas_draw(self):
        """Perform the actual canvas draw"""
        if self._draw_pending:
            self._draw_pending = False
            self.viewer_canvas.draw_idle()

    def _setup_persistent_axis(self):
        """Setup persistent axis objects that don't get cleared each frame"""
        # Clear only once during setup
        self.viewer_ax.clear()
        self.viewer_ax.set_facecolor('black')

        # Create persistent background image object
        if self.image_cache and len(self.image_cache.image_files) > 0:
            # Load first image to establish size
            first_img = self.image_cache.get_image(0, priority=1)

            if first_img is not None:
                display_extent_data = self.image_cache.get_display_extent()

                if display_extent_data:
                    display_extent, display_shape = display_extent_data
                    self.display_extent = display_extent
                else:
                    h, w = first_img.shape
                    self.display_extent = [0, w, h, 0]

                # Create persistent imshow object
                self._background_imshow = self.viewer_ax.imshow(
                    first_img, cmap='gray', alpha=0.8,
                    extent=self.display_extent, aspect='equal',
                    interpolation='nearest', origin='upper'
                )

                self.viewer_ax.set_xlim(self.display_extent[0], self.display_extent[1])
                self.viewer_ax.set_ylim(self.display_extent[2], self.display_extent[3])
            else:
                self._background_imshow = None
        else:
            self._background_imshow = None
            self._set_default_view()

        # Add persistent track collections to axis
        if self.track_overlay:
            self.track_overlay.add_persistent_collections_to_axis()

        # Set title and labels
        title = f"SWT Track Editor - Persistent Object System"
        if self.use_nose_coordinates:
            title += " (Nose Coordinates)"

        if self.image_cache and self.image_cache.original_image_size:
            orig_h, orig_w = self.image_cache.original_image_size
            ratio = self.image_cache.global_downsample_ratio
            title += f" (Original: {orig_w}x{orig_h}, Display: {ratio:.1%})"

        self.viewer_ax.set_title(title, fontsize=14, color='white')
        self.viewer_ax.set_xlabel("X Position (pixels)", color='white')
        self.viewer_ax.set_ylabel("Y Position (pixels)", color='white')

        # Force initial draw
        self.viewer_canvas.draw()

    def _set_default_view(self):
        """Set default view when no images available"""
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

    # ================== ROBUST PLAYBACK CONTROLS ==================

    def toggle_playback(self):
        """Toggle playback with state verification and cleanup"""
        if not (self.track_manager.tracks or self.image_cache):
            return

        # Force stop first, then start if needed
        if self.playing:
            self.stop_playback()
            logger.debug("Playback stopped by user")
        else:
            # Ensure we're really stopped before starting
            self.stop_playback()
            self.start_playback()
            logger.debug("Playback started by user")

    def start_playback(self):
        """Start playback with prefetching and state cleanup"""
        if not (self.track_manager.tracks or self.image_cache):
            return

        # Ensure clean state
        self.stop_playback()

        self.playing = True
        self.play_button.config(text="||")

        if self.image_cache:
            end_frame = min(self.current_frame + 50, len(self.image_cache.image_files))
            self.image_cache.prefetch_range(self.current_frame, end_frame, priority=2)

        self.schedule_next_frame()

    def schedule_next_frame(self):
        """Schedule next frame with generation tracking and robust state checking"""
        # Increment generation to invalidate any pending timers
        self._timer_generation += 1
        current_generation = self._timer_generation
        
        # Clear any existing timer first
        if self._playback_timer:
            self.root.after_cancel(self._playback_timer)
            self._playback_timer = None
        
        # Double-check playing state before proceeding
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

        self.update_frame_display()

        # Check again before scheduling - state might have changed during update
        if not self.playing:
            return

        # Calculate delay and schedule next frame
        target_delay = int(1000 / self.fps)
        avg_frame_time = np.mean(self.frame_times) if self.frame_times else 0
        actual_delay = max(10, target_delay - int(avg_frame_time * 1000))

        # Schedule with generation check
        def next_frame_with_check():
            if self.playing and self._timer_generation == current_generation:
                self.schedule_next_frame()

        self._playback_timer = self.root.after(actual_delay, next_frame_with_check)

    def stop_playback(self):
        """Stop playback with generation invalidation"""
        was_playing = self.playing
        self.playing = False
        self._timer_generation += 1  # Invalidate all pending timers
        
        # Cancel any pending playback timer
        if self._playback_timer:
            self.root.after_cancel(self._playback_timer)
            self._playback_timer = None
        
        # Update button state
        self.play_button.config(text=">")
        
        # Log for debugging
        if was_playing:
            logger.debug("Playback stopped - timer canceled")

    def goto_frame(self, frame_idx):
        """Navigate to specific frame with bounds checking"""
        max_frame = 0
        if self.track_manager.tracks:
            max_frame = max(max_frame, self.track_manager.frame_count - 1)
        if self.image_cache:
            max_frame = max(max_frame, len(self.image_cache.image_files) - 1)

        # Ensure frame is within bounds
        new_frame = max(0, min(frame_idx, max_frame))

        # Only update if frame actually changed
        if new_frame != self.current_frame:
            self.current_frame = new_frame

            # Update slider to match (prevent recursion)
            if self.frame_var.get() != self.current_frame:
                self.frame_var.set(self.current_frame)

            if self.image_cache:
                self.image_cache.prefetch_range(
                    max(0, new_frame - 10),
                    min(len(self.image_cache.image_files), new_frame + 20),
                    priority=1
                )

            self.update_frame_display(force_update=True)

    def on_frame_change(self, value):
        """Handle frame slider changes with playback protection"""
        new_frame = int(value)

        # If playing, stop playback to avoid conflicts
        if self.playing:
            self.stop_playback()

        if new_frame != self.current_frame:
            # Cancel any pending slider timer
            if self._slider_timer:
                self.root.after_cancel(self._slider_timer)
                self._slider_timer = None

            # Debounce rapid slider changes
            self._slider_timer = self.root.after(50, lambda: self._handle_slider_change(new_frame))

    def _handle_slider_change(self, new_frame):
        """Handle delayed slider change"""
        self._slider_timer = None
        self.goto_frame(new_frame)

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

            for btn in [self.delete_button, self.keep_button, self.merge_button,
                       self.split_button, self.save_edits_button]:
                btn.config(state='normal')
                
            self.selection_mode = True
        else:
            self.selection_mode = False
            self.selected_tracks.clear()

            for btn in [self.delete_button, self.keep_button, self.merge_button,
                       self.split_button, self.save_edits_button]:
                btn.config(state='disabled')

        self.update_selection_display()
        self.update_tracks_only()
        self.update_track_list()

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
            self.tracks_need_update = True
            self.update_selection_display()
            self.update_frame_display(force_update=True)
            self.update_track_list()

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
            self.tracks_need_update = True
            self.update_selection_display()
            self.update_frame_display(force_update=True)
            self.update_track_list()

    def merge_selected_tracks(self):
        """Merge selected tracks with robust conflict resolution"""
        if len(self.selected_tracks) < 2:
            messagebox.showwarning("Insufficient Selection", "Select at least 2 tracks!")
            return

        selected_list = sorted(list(self.selected_tracks))
        
        # Show merge preview in the log
        logger.info(f"USER: Attempting to merge tracks {selected_list}")
        
        confirm = messagebox.askyesno("Confirm Merge",
                                     f"Merge {len(selected_list)} tracks into track {selected_list[0]}?\n\n"
                                     f"Check the console/log for detailed merge analysis.")

        if confirm:
            if self.track_manager.merge_tracks(selected_list):
                self.selected_tracks.clear()
                self.tracks_need_update = True
                self.update_selection_display()
                self.update_frame_display(force_update=True)
                self.update_track_list()
                messagebox.showinfo("Merge Complete", 
                                   f"Successfully merged tracks. Check console for detailed merge report.")
            else:
                messagebox.showerror("Merge Failed", 
                                   "Merge failed. Check console for details.")

    def split_selected_track(self):
        """Split the single selected track at the current frame into two tracks."""
        if not self.selected_tracks or len(self.selected_tracks) != 1:
            messagebox.showwarning("Split", "Select exactly one track to split.")
            return

        track_id = next(iter(self.selected_tracks))

        new_id = self.track_manager.split_track(track_id, self.current_frame)
        if new_id is None:
            messagebox.showinfo("Split", "Nothing to split at this frame (no future positions).")
            return

        # Keep both parts selected
        self.selected_tracks = {track_id, new_id}
        self.tracks_need_update = True

        self.update_selection_display()
        self.update_frame_display(force_update=True)
        self.update_track_list()
        self._sync_listbox_selection()

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
                self.update_tracks_only()
            else:
                self.trail_var.set(str(self.trail_length))
        except ValueError:
            self.trail_var.set(str(self.trail_length))

    def update_display_options(self):
        """Update display options"""
        self.show_trails = self.show_trails_var.get()
        self.show_labels = self.show_labels_var.get()
        self.show_current_positions = self.show_positions_var.get()
        self.show_full_tracks = self.show_full_tracks_var.get()
        self.update_tracks_only()

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
        prerendered_tracks = len(self.track_overlay.prerendered_tracks) if self.track_overlay else 0

        stats_text = f"""Performance Statistics:

Persistent Object System:
- Pre-rendered tracks: {prerendered_tracks}
- Ultra-smooth scrolling with set_data() optimization
- Global coordinate scaling: {self.image_cache.global_downsample_ratio:.3f}
- Robust timer management with generation tracking

Cache Performance:
- Hit Rate: {stats.hit_rate:.1f}%
- Hot Cache: {stats.hot_cache_size} images
- Warm Cache: {stats.warm_cache_size} images
- Load Queue: {stats.queue_size} pending

Frame Rendering:
- Average: {avg_frame_time:.1f}ms
- Min: {min_frame_time:.1f}ms
- Max: {max_frame_time:.1f}ms
- Target FPS: {self.fps}

Merge System:
- Intelligent conflict resolution enabled
- Trajectory-based position selection
- Detailed merge analysis and logging

System:
- Memory Usage: {memory_usage:.1f}MB
- Total Images: {len(self.image_cache.image_files) if self.image_cache else 0}
- Total Tracks: {len(self.track_manager.tracks) if self.track_manager.tracks else 0}"""

        messagebox.showinfo("Performance Statistics", stats_text)

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
                        f"Persistent: {len(self.track_overlay.prerendered_tracks) if self.track_overlay else 0}")

            self.perf_label.config(text=perf_text)
            self.last_stats_update = time.time()

        self.root.after(Config.PERFORMANCE_UPDATE_INTERVAL, self.update_performance_stats)

    def on_closing(self):
        """Clean shutdown with timer cleanup"""
        logger.info("Shutting down SWT Track Editor...")

        self.playing = False
        self._timer_generation += 1  # Invalidate all timers

        # Cancel all pending timers
        if self._playback_timer:
            self.root.after_cancel(self._playback_timer)
            self._playback_timer = None

        if self._slider_timer:
            self.root.after_cancel(self._slider_timer)
            self._slider_timer = None

        if self.track_overlay:
            self.track_overlay.clear_all()

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
            logger.info("Starting SWT Track Editor with Persistent Object System...")
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()
        except Exception as e:
            logger.error(f"Application error: {e}")
            self.on_closing()


# ================== ENTRY POINT ==================

def main():
    """Main entry point with command line support"""
    parser = argparse.ArgumentParser(description="SWT Track Editor - Persistent Object System")
    parser.add_argument("--csv", help="Path to track CSV to auto-load")
    parser.add_argument("--images", help="Path to image directory to auto-load")
    parser.add_argument("--cache-size", type=int, help="Override cache size")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help="Set logging level")
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("=== SWT Track Editor - Persistent Object System ===")
    logger.info("Performance Optimizations:")
    logger.info("✓ Persistent imshow object with set_data() updates")
    logger.info("✓ Persistent track LineCollections with set_segments() updates")
    logger.info("✓ No axis clearing during playback")
    logger.info("✓ Global coordinate scaling applied once at load time")
    logger.info("✓ LineCollection rebuild only on track edits")
    logger.info("✓ All editing functionality preserved")
    logger.info("✓ Dual selection: Canvas clicking AND scrollable list")
    logger.info("✓ Robust timer management with generation tracking")
    logger.info("✓ Race condition protection for selections")
    logger.info("✓ Synchronized selection between canvas and list")
    logger.info("✓ Ctrl+click support for multiple selection")
    logger.info("✓ Optimized layout: 80% width for images/tracks")
    logger.info("=" * 70)

    # Check for dependencies
    dependencies_info = []

    try:
        import tifffile
        dependencies_info.append("✓ Tifffile available")
    except ImportError:
        dependencies_info.append("⚠ Tifffile not available")

    try:
        from PIL import Image
        dependencies_info.append("✓ PIL/Pillow available")
    except ImportError:
        dependencies_info.append("⚠ PIL/Pillow not available")

    try:
        from skimage import io
        dependencies_info.append("✓ Scikit-image available")
    except ImportError:
        dependencies_info.append("⚠ Scikit-image not available")

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