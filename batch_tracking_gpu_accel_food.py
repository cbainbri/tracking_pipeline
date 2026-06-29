#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Batch Worm Tracker — GPU-accelerated (platform-aware cascade)

Identical analytical/tracking logic to batch_tracking_gpu.py.  The only
difference is how three image operations are executed:
  - median (background generation)
  - absdiff (background subtraction)
  - inRange (thresholding)

Backend selection at startup:
  Windows + NVIDIA  →  CuPy (CUDA)
  Windows + AMD     →  PyTorch DirectML
  Linux   + NVIDIA  →  CuPy (CUDA)
  Linux   + AMD     →  PyTorch ROCm
  Anything else     →  NumPy CPU fallback
"""

import os
import sys
import gc
import platform
import queue
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from pathlib import Path
import logging
from dataclasses import dataclass

N_WORKERS = max(1, (os.cpu_count() or 4) * 3 // 4)

# ---------------------------------------------------------------------------
# GPU backend detection — sets gpu_median, gpu_absdiff, gpu_inrange
# ---------------------------------------------------------------------------

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

GPU_BACKEND = "cpu"
GPU_DEVICE_NAME = "CPU (NumPy)"


def _detect_gpu_vendor() -> str:
    """Probe OpenCL for GPU vendor. Returns 'nvidia', 'amd', or 'unknown'."""
    try:
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            if cv2.ocl.useOpenCL():
                device = cv2.ocl.Device.getDefault()
                vendor = (device.vendorName() or "").lower()
                name = (device.name() or "").strip()
                if "nvidia" in vendor:
                    return "nvidia"
                elif "amd" in vendor or "advanced micro" in vendor:
                    return "amd"
    except Exception:
        pass
    return "unknown"


def _try_cupy() -> bool:
    """Attempt to initialise CuPy (NVIDIA CUDA)."""
    global GPU_BACKEND, GPU_DEVICE_NAME
    try:
        import cupy as cp
        a = cp.zeros((4, 4), dtype=cp.float32)
        cp.median(a, axis=0)
        try:
            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props.get("name", "Unknown CUDA device")
            if isinstance(name, bytes):
                name = name.decode()
        except Exception:
            name = "CUDA device"
        GPU_BACKEND = "cupy"
        GPU_DEVICE_NAME = f"{name} (CuPy/CUDA)"
        return True
    except Exception:
        return False


def _try_torch_cuda() -> bool:
    """Attempt to initialise PyTorch with CUDA."""
    global GPU_BACKEND, GPU_DEVICE_NAME
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        # smoke test
        a = torch.zeros((4, 4), dtype=torch.float32, device="cuda")
        torch.median(a, dim=0)
        name = torch.cuda.get_device_name(0)
        GPU_BACKEND = "torch_cuda"
        GPU_DEVICE_NAME = f"{name} (PyTorch/CUDA)"
        return True
    except Exception:
        return False


def _try_torch_rocm() -> bool:
    """Attempt to initialise PyTorch with ROCm (AMD Linux)."""
    global GPU_BACKEND, GPU_DEVICE_NAME
    try:
        import torch
        # ROCm exposes itself as torch.cuda in PyTorch's HIP build
        if not torch.cuda.is_available():
            return False
        # Verify this is actually ROCm, not CUDA
        name = torch.cuda.get_device_name(0)
        # ROCm devices typically have AMD in the name; CUDA devices have NVIDIA
        if "nvidia" in name.lower():
            return False
        a = torch.zeros((4, 4), dtype=torch.float32, device="cuda")
        torch.median(a, dim=0)
        GPU_BACKEND = "torch_rocm"
        GPU_DEVICE_NAME = f"{name} (PyTorch/ROCm)"
        return True
    except Exception:
        return False


def _try_torch_directml() -> bool:
    """Attempt to initialise PyTorch with DirectML (Windows AMD/Intel)."""
    global GPU_BACKEND, GPU_DEVICE_NAME
    try:
        import torch
        import torch_directml
        dml_device = torch_directml.device()
        # smoke test — median along an axis
        a = torch.zeros((4, 4), dtype=torch.float32, device=dml_device)
        torch.median(a, dim=0)
        name = torch_directml.device_name(0)
        GPU_BACKEND = "torch_dml"
        GPU_DEVICE_NAME = f"{name} (PyTorch/DirectML)"
        return True
    except Exception:
        return False


# --- Run detection in priority order ---

_gpu_vendor = _detect_gpu_vendor()

if IS_WINDOWS and _gpu_vendor == "nvidia":
    _try_cupy() or _try_torch_cuda()
elif IS_WINDOWS and _gpu_vendor == "amd":
    _try_torch_directml()
elif IS_LINUX and _gpu_vendor == "nvidia":
    _try_cupy() or _try_torch_cuda()
elif IS_LINUX and _gpu_vendor == "amd":
    _try_torch_rocm()
elif IS_LINUX and _gpu_vendor == "unknown":
    # OpenCL ICD unavailable (common in conda envs) — try PyTorch directly.
    # If launched via launcher.py the HSA override is already inherited.
    # If run directly, probe for the right GFX version before torch is imported
    # (HSA locks in the version at first torch import; subprocess gives a fresh
    # context for each candidate).
    if 'HSA_OVERRIDE_GFX_VERSION' not in os.environ:
        _GFX_PROBE = (
            'import torch; t=torch.zeros(1,device="cuda"); '
            'assert (t+1).item()==1.0'
        )
        _GFX_CANDIDATES = ['11.0.0', '10.3.0', '9.4.0', '9.0.10', '9.0.6']
        for _ver in _GFX_CANDIDATES:
            _env = {**os.environ, 'HSA_OVERRIDE_GFX_VERSION': _ver}
            _r = __import__('subprocess').run(
                [sys.executable, '-c', _GFX_PROBE],
                env=_env, capture_output=True, timeout=20
            )
            if _r.returncode == 0:
                os.environ['HSA_OVERRIDE_GFX_VERSION'] = _ver
                break
    _try_torch_rocm() or _try_cupy() or _try_torch_cuda()
# else: stays "cpu"


# ---------------------------------------------------------------------------
# Backend wrapper functions — these are the only GPU-touching code paths
# ---------------------------------------------------------------------------

def _numpy_median(stack: np.ndarray) -> np.ndarray:
    return np.median(stack, axis=0).astype(np.uint8)

def _numpy_absdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cv2.absdiff(a, b)

def _numpy_inrange(img: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return cv2.inRange(img, lo, hi)


def _cupy_median(stack: np.ndarray) -> np.ndarray:
    import cupy as cp
    logger = logging.getLogger('BatchWormTrackerAccel')
    n_frames, h, w = stack.shape
    free_vram = cp.cuda.Device(0).mem_info[0]
    # each tile row: n_frames * w * 4 bytes (float32), median needs ~2.5x that
    bytes_per_row = n_frames * w * 4
    usable_vram = int(free_vram * 0.7)
    rows_per_tile = max(1, usable_vram // int(bytes_per_row * 2.5))
    rows_per_tile = min(rows_per_tile, h)
    n_tiles = (h + rows_per_tile - 1) // rows_per_tile

    if n_tiles == 1:
        logger.info(f"GPU median: full image fits in VRAM ({free_vram/1e9:.1f}GB free)")
    else:
        logger.info(f"GPU median: tiling {h} rows into {n_tiles} strips of ~{rows_per_tile} rows "
                    f"({free_vram/1e9:.1f}GB free)")

    result = np.empty((h, w), dtype=np.uint8)
    for tile_idx in range(n_tiles):
        r_start = tile_idx * rows_per_tile
        r_end = min(r_start + rows_per_tile, h)
        tile = cp.asarray(stack[:, r_start:r_end, :])
        median_tile = cp.median(tile, axis=0).astype(cp.uint8)
        result[r_start:r_end, :] = cp.asnumpy(median_tile)
        del tile, median_tile
        cp.get_default_memory_pool().free_all_blocks()

    return result

def _cupy_absdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    import cupy as cp
    ga, gb = cp.asarray(a), cp.asarray(b)
    result = cp.asnumpy(cp.abs(ga.astype(cp.int16) - gb.astype(cp.int16)).astype(cp.uint8))
    del ga, gb
    return result

def _cupy_inrange(img: np.ndarray, lo: int, hi: int) -> np.ndarray:
    import cupy as cp
    gi = cp.asarray(img)
    mask = ((gi >= lo) & (gi <= hi)).astype(cp.uint8) * 255
    result = cp.asnumpy(mask)
    del gi, mask
    return result


def _torch_median(stack: np.ndarray, device: str) -> np.ndarray:
    import torch
    logger = logging.getLogger('BatchWormTrackerAccel')
    n_frames, h, w = stack.shape

    if "cuda" in device:
        free_vram = torch.cuda.mem_get_info(0)[0]
    else:
        free_vram = 2 * 1024**3  # conservative 2GB assumption for DirectML

    bytes_per_row = n_frames * w * 4
    usable_vram = int(free_vram * 0.7)
    rows_per_tile = max(1, usable_vram // int(bytes_per_row * 2.5))
    rows_per_tile = min(rows_per_tile, h)
    n_tiles = (h + rows_per_tile - 1) // rows_per_tile

    if n_tiles == 1:
        logger.info(f"GPU median: full image fits in VRAM ({free_vram/1e9:.1f}GB free)")
    else:
        logger.info(f"GPU median: tiling {h} rows into {n_tiles} strips of ~{rows_per_tile} rows "
                    f"({free_vram/1e9:.1f}GB free)")

    result = np.empty((h, w), dtype=np.uint8)
    for tile_idx in range(n_tiles):
        r_start = tile_idx * rows_per_tile
        r_end = min(r_start + rows_per_tile, h)
        t = torch.from_numpy(stack[:, r_start:r_end, :].copy()).to(device=device, dtype=torch.float32)
        median_tile = torch.median(t, dim=0).values.to(dtype=torch.uint8)
        result[r_start:r_end, :] = median_tile.cpu().numpy()
        del t, median_tile
        if "cuda" in device:
            torch.cuda.empty_cache()

    return result

def _torch_absdiff(a: np.ndarray, b: np.ndarray, device: str) -> np.ndarray:
    import torch
    ta = torch.from_numpy(a).to(device=device, dtype=torch.int16)
    tb = torch.from_numpy(b).to(device=device, dtype=torch.int16)
    result = torch.abs(ta - tb).to(dtype=torch.uint8).cpu().numpy()
    del ta, tb
    return result

def _torch_inrange(img: np.ndarray, lo: int, hi: int, device: str) -> np.ndarray:
    import torch
    t = torch.from_numpy(img).to(device=device)
    mask = ((t >= lo) & (t <= hi)).to(dtype=torch.uint8) * 255
    result = mask.cpu().numpy()
    del t, mask
    return result


# --- Bind the right implementation ---
# Median: GPU when available (the heavy compute that benefits from acceleration)
if GPU_BACKEND == "cupy":
    gpu_median = _cupy_median
elif GPU_BACKEND == "torch_cuda":
    gpu_median = lambda stack: _torch_median(stack, "cuda")
elif GPU_BACKEND == "torch_rocm":
    gpu_median = lambda stack: _torch_median(stack, "cuda")
elif GPU_BACKEND == "torch_dml":
    def _dml_device():
        import torch_directml
        return torch_directml.device()
    gpu_median = lambda stack: _torch_median(stack, str(_dml_device()))
else:
    gpu_median = _numpy_median

# Per-frame ops: always CPU. GPU transfer overhead per frame exceeds compute
# savings, especially with parallel directory processing.
gpu_absdiff = _numpy_absdiff
gpu_inrange = _numpy_inrange

GPU_ACCELERATED = GPU_BACKEND != "cpu"


# ---------------------------------------------------------------------------
# Everything below is identical tracking logic from batch_tracking_gpu.py
# with only generate_background, load_and_process_frame, apply_threshold
# calling the gpu_* wrappers above instead of raw numpy/UMat.
# ---------------------------------------------------------------------------

class ImagePrefetcher:
    """Reads images from disk in a background thread, buffering ahead."""

    def __init__(self, image_files: List[str], buffer_size: int = 12):
        self._queue = queue.Queue(maxsize=buffer_size)
        self._thread = threading.Thread(target=self._worker, args=(image_files,), daemon=True)
        self._thread.start()

    def _worker(self, image_files):
        for path in image_files:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self._queue.put((path, img))
        self._queue.put(None)

    def __iter__(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            yield item


@dataclass
class BatchConfig:
    """Configuration for batch processing with same defaults as main tracker"""
    threshold_min: int = 12
    threshold_max: int = 255
    min_blob_size: int = 120
    max_blob_size: int = 2400
    max_distance: int = 75
    trajectory_weight: float = 0.7
    min_track_length: int = 50
    use_hungarian: bool = False
    nose_detection_enabled: bool = True
    nose_smoothing_frames: int = 2
    min_movement_threshold: float = 2.0
    filter_stationary_tracks: bool = True
    min_displacement_distance: float = 75.0


@dataclass
class ProcessingResult:
    """Result of processing a single directory"""
    directory: str
    success: bool
    num_images: int
    num_tracks: int
    num_accepted_tracks: int
    processing_time: float
    error_message: Optional[str] = None
    quality_flag: str = "normal"


class BatchWormTracker:
    """Core tracking functionality extracted for batch processing"""

    def __init__(self, config: BatchConfig):
        self.config = config
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('BatchWormTrackerAccel')
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger

    def find_image_files(self, directory: str) -> List[str]:
        image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp')
        image_files = []
        try:
            for file in os.listdir(directory):
                if file.lower().endswith(image_extensions):
                    image_files.append(os.path.join(directory, file))
            image_files.sort()
        except Exception as e:
            self.logger.error(f"Error reading directory {directory}: {e}")
        return image_files

    def generate_background(self, image_files: List[str]) -> Optional[np.ndarray]:
        """Generate background image — GPU-accelerated median when available."""
        if not image_files:
            return None

        total_images = len(image_files)
        if total_images <= 50:
            indices = list(range(total_images))
        elif total_images <= 200:
            indices = np.linspace(0, total_images - 1, 50, dtype=int)
        else:
            indices = np.linspace(0, total_images - 1, 75, dtype=int)

        sampled_files = [image_files[i] for i in indices]

        # Read first image to get reference shape
        reference_shape = None
        for path in sampled_files:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                reference_shape = img.shape
                break
        if reference_shape is None:
            self.logger.error("No valid images found for background generation")
            return None

        def _load_one(img_path):
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    return None
                if img.shape != reference_shape:
                    img = cv2.resize(img, (reference_shape[1], reference_shape[0]))
                return img.astype(np.float32)
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=min(4, N_WORKERS)) as pool:
            loaded = list(pool.map(_load_one, sampled_files))
        all_backgrounds = [img for img in loaded if img is not None]
        del loaded
        gc.collect()

        if not all_backgrounds:
            self.logger.error("No valid images found for background generation")
            return None

        stack = np.array(all_backgrounds)
        del all_backgrounds
        gc.collect()

        background = gpu_median(stack)
        del stack
        gc.collect()

        self.logger.info(f"Background generated (median, {GPU_BACKEND}) from {len(sampled_files)} images")
        return background

    def apply_threshold(self, image: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Apply threshold + blob-size filtering via gpu_inrange."""
        if image is None or image.size == 0:
            return np.zeros((100, 100), dtype=np.uint8), []

        try:
            min_thresh = max(0, min(255, self.config.threshold_min))
            max_thresh = max(min_thresh, min(255, self.config.threshold_max))
            min_blob = max(1, self.config.min_blob_size)
            max_blob = max(min_blob, self.config.max_blob_size)

            thresholded = gpu_inrange(image, min_thresh, max_thresh)

            raw_contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered = np.zeros_like(thresholded)
            kept_contours = []
            for contour in raw_contours:
                area = cv2.contourArea(contour)
                if min_blob <= area <= max_blob:
                    cv2.fillPoly(filtered, [contour], 255)
                    kept_contours.append(contour)

            return filtered, kept_contours
        except Exception as e:
            self.logger.error(f"Error in apply_threshold: {e}")
            return np.zeros_like(image), []

    def load_and_process_frame(self, image_path: str, background: np.ndarray, preloaded_img: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load and background-subtract a frame via gpu_absdiff."""
        img = preloaded_img if preloaded_img is not None else cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None

        if background is not None:
            if img.shape != background.shape:
                img = cv2.resize(img, (background.shape[1], background.shape[0]))
            if img.dtype != background.dtype:
                img = img.astype(background.dtype, copy=False)
            subtracted = gpu_absdiff(background, img)
        else:
            subtracted = img

        return img, subtracted

    # --- Everything below is IDENTICAL to batch_tracking_gpu.py ---

    def predict_next_position(self, track_positions: List[Tuple[float, float, int]], min_frames: int = 3) -> Optional[Tuple[float, float]]:
        if len(track_positions) < min_frames:
            return None

        recent_positions = track_positions[-min_frames:]
        velocities = []
        for i in range(1, len(recent_positions)):
            prev_x, prev_y, prev_f = recent_positions[i-1]
            curr_x, curr_y, curr_f = recent_positions[i]
            frame_diff = curr_f - prev_f
            if frame_diff > 0:
                vx = (curr_x - prev_x) / frame_diff
                vy = (curr_y - prev_y) / frame_diff
                velocities.append((vx, vy))

        if not velocities:
            return None

        avg_vx = np.mean([v[0] for v in velocities])
        avg_vy = np.mean([v[1] for v in velocities])
        last_x, last_y, last_f = track_positions[-1]
        return (last_x + avg_vx, last_y + avg_vy)

    def assign_tracks_with_trajectory(self, active_tracks: Dict, centroids: List[Tuple[float, float]]) -> Dict[int, int]:
        if not active_tracks or not centroids:
            return {}

        track_positions = []
        track_ids = []
        track_predictions = []

        for track_id, track_data in active_tracks.items():
            if track_data['positions']:
                last_pos = track_data['positions'][-1]
                track_positions.append([last_pos[0], last_pos[1]])
                track_ids.append(track_id)
                track_predictions.append(self.predict_next_position(track_data['positions']))

        if not track_positions:
            return {}

        track_positions_array = np.array(track_positions)
        centroid_positions_array = np.array(centroids)
        distance_matrix = cdist(track_positions_array, centroid_positions_array)

        if self.config.trajectory_weight > 0:
            trajectory_matrix = np.full_like(distance_matrix, np.inf)
            for i, predicted_pos in enumerate(track_predictions):
                if predicted_pos is not None:
                    pred_distances = cdist([predicted_pos], centroid_positions_array)[0]
                    for j in range(len(pred_distances)):
                        if pred_distances[j] < self.config.max_distance * 2:
                            trajectory_matrix[i, j] = pred_distances[j]

            combined_matrix = np.full_like(distance_matrix, np.inf)
            for i in range(len(track_ids)):
                for j in range(len(centroids)):
                    dist_score = distance_matrix[i, j]
                    traj_score = trajectory_matrix[i, j]
                    if dist_score < self.config.max_distance and traj_score < np.inf:
                        combined_score = (1 - self.config.trajectory_weight) * dist_score + \
                                       self.config.trajectory_weight * traj_score
                        combined_matrix[i, j] = combined_score
                    elif dist_score < self.config.max_distance:
                        combined_matrix[i, j] = dist_score * (1 + self.config.trajectory_weight * 0.5)
                    elif traj_score < self.config.max_distance * 1.5:
                        combined_matrix[i, j] = traj_score * (1 + (1 - self.config.trajectory_weight) * 0.5)
        else:
            combined_matrix = distance_matrix.copy()

        combined_matrix[combined_matrix > self.config.max_distance * 1.2] = 1e6

        assignments = {}
        if combined_matrix.size > 0 and np.sum(combined_matrix < 1e6) > 0:
            if self.config.use_hungarian:
                try:
                    row_indices, col_indices = linear_sum_assignment(combined_matrix)
                    for row_idx, col_idx in zip(row_indices, col_indices):
                        if combined_matrix[row_idx, col_idx] < 1e6:
                            assignments[track_ids[row_idx]] = col_idx
                except Exception as e:
                    self.logger.warning(f"Hungarian assignment error: {e} - falling back to Greedy")
                    assignments = self._greedy_assignment(track_ids, combined_matrix, centroids)
            else:
                assignments = self._greedy_assignment(track_ids, combined_matrix, centroids)

        return assignments

    def _greedy_assignment(self, track_ids: List[int], combined_matrix: np.ndarray, centroids: List[Tuple[float, float]]) -> Dict[int, int]:
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
            if best_dist < self.config.max_distance * 1.2:
                assignments[track_id] = best_centroid
                used_centroids.add(best_centroid)
        return assignments

    def calculate_locomotion_direction(self, track_positions: List[Tuple[float, float, int]]) -> Optional[Tuple[float, float]]:
        min_required_positions = max(2, self.config.nose_smoothing_frames)
        if len(track_positions) < min_required_positions:
            return None

        recent_positions = track_positions[-min(self.config.nose_smoothing_frames, len(track_positions)):]
        if len(recent_positions) < 2:
            return None

        velocities = []
        for i in range(1, len(recent_positions)):
            prev_x, prev_y, prev_f = recent_positions[i-1]
            curr_x, curr_y, curr_f = recent_positions[i]
            frame_diff = curr_f - prev_f
            if frame_diff > 5:
                continue
            if frame_diff > 0:
                raw_distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                distance_per_frame = raw_distance / frame_diff
                if distance_per_frame >= self.config.min_movement_threshold:
                    vx = (curr_x - prev_x) / frame_diff
                    vy = (curr_y - prev_y) / frame_diff
                    velocities.append((vx, vy))

        if not velocities:
            return None

        avg_vx = np.mean([v[0] for v in velocities])
        avg_vy = np.mean([v[1] for v in velocities])
        direction_magnitude = np.sqrt(avg_vx**2 + avg_vy**2)
        if direction_magnitude < 0.1:
            return None
        return (avg_vx / direction_magnitude, avg_vy / direction_magnitude)

    def find_nose_position(self, contour: np.ndarray, locomotion_direction: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        if contour is None or len(contour) < 3 or locomotion_direction is None:
            return None
        try:
            points = contour.reshape(-1, 2).astype(np.float64)
            direction_vector = np.array(locomotion_direction, dtype=np.float64)
            projections = np.dot(points, direction_vector)
            max_projection = np.max(projections)
            projection_tolerance = 1.0
            front_mask = projections >= (max_projection - projection_tolerance)
            front_points = points[front_mask]
            if len(front_points) == 0:
                return None
            nose_x = np.mean(front_points[:, 0])
            nose_y = np.mean(front_points[:, 1])
            return (round(float(nose_x), 4), round(float(nose_y), 4))
        except Exception as e:
            self.logger.error(f"Error in find_nose_position: {e}")
            return None

    def detect_nose_for_track(self, track_id: int, active_tracks: Dict, contours: List[np.ndarray], centroids: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if not self.config.nose_detection_enabled:
            return None
        if track_id not in active_tracks:
            return None
        track_data = active_tracks[track_id]
        if 'positions' not in track_data or len(track_data['positions']) < 2:
            return None
        locomotion_direction = self.calculate_locomotion_direction(track_data['positions'])
        if locomotion_direction is None:
            return None
        if 'current_centroid_idx' not in track_data:
            return None
        centroid_idx = track_data['current_centroid_idx']
        if centroid_idx < 0 or centroid_idx >= len(contours):
            return None
        return self.find_nose_position(contours[centroid_idx], locomotion_direction)

    def calculate_track_displacement(self, track_positions: List[Tuple[float, float, int]]) -> float:
        if len(track_positions) < 2:
            return 0.0
        first_pos = track_positions[0]
        xs = np.array([p[0] for p in track_positions[1:]])
        ys = np.array([p[1] for p in track_positions[1:]])
        distances = np.sqrt((xs - first_pos[0])**2 + (ys - first_pos[1])**2)
        return float(distances.max())

    def process_directory(self, image_directory: str, progress_callback=None) -> ProcessingResult:
        start_time = time.time()
        result = ProcessingResult(
            directory=image_directory, success=False,
            num_images=0, num_tracks=0, num_accepted_tracks=0, processing_time=0.0
        )
        try:
            self.logger.info(f"Processing directory: {image_directory}")
            image_files = self.find_image_files(image_directory)
            if not image_files:
                result.error_message = "No image files found"
                return result

            result.num_images = len(image_files)
            self.logger.info(f"Found {len(image_files)} images")

            background = self.generate_background(image_files)
            if background is None:
                result.error_message = "Failed to generate background"
                return result

            tracks, nose_tracks, track_statistics = self._run_tracking(image_files, background, progress_callback)
            if not tracks:
                result.error_message = "No tracks generated"
                return result

            result.num_tracks = len(tracks)
            result.num_accepted_tracks = len([s for s in track_statistics if s['final_status'] == 'accepted'])
            result.quality_flag = self._assess_quality(result.num_accepted_tracks, result.num_images)

            csv_path = os.path.join(image_directory, "tracks_debug.csv")
            self._export_tracks_csv(tracks, nose_tracks, csv_path)

            result.success = True
            result.processing_time = time.time() - start_time
            self.logger.info(f"Successfully processed {image_directory}")
            self.logger.info(f"Generated {result.num_accepted_tracks} tracks in {result.processing_time:.1f}s")
        except Exception as e:
            result.error_message = str(e)
            result.processing_time = time.time() - start_time
            self.logger.error(f"Error processing {image_directory}: {e}")
        return result

    def _assess_quality(self, num_tracks: int, num_images: int) -> str:
        if num_tracks == 0:
            return "empty"
        tracks_per_image = num_tracks / max(num_images, 1)
        if tracks_per_image > 2.0:
            return "noisy"
        elif num_tracks > 100:
            return "noisy"
        return "normal"

    def _run_tracking(self, image_files: List[str], background: np.ndarray, progress_callback=None) -> Tuple[Dict, Dict, List]:
        next_track_id = 1
        active_tracks = {}
        inactive_tracks = {}
        used_track_ids = set()
        MAX_MISSING_FRAMES = 5
        total_frames = len(image_files)

        prefetcher = ImagePrefetcher(image_files)
        for frame_idx, (img_path, preloaded_img) in enumerate(prefetcher):
            try:
                if progress_callback:
                    progress_callback(frame_idx + 1, total_frames)

                img, subtracted = self.load_and_process_frame(img_path, background, preloaded_img=preloaded_img)
                if img is None or subtracted is None:
                    continue

                thresholded, valid_contours = self.apply_threshold(subtracted)

                centroids = []
                kept_contours = []
                for contour in valid_contours:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                        centroids.append((cx, cy))
                        kept_contours.append(contour)
                valid_contours = kept_contours

                tracks_to_deactivate = []
                for track_id, track_data in active_tracks.items():
                    frames_missing = frame_idx - track_data['last_frame']
                    if frames_missing > MAX_MISSING_FRAMES:
                        tracks_to_deactivate.append(track_id)

                for track_id in tracks_to_deactivate:
                    inactive_tracks[track_id] = active_tracks[track_id]
                    del active_tracks[track_id]

                assignments = self.assign_tracks_with_trajectory(active_tracks, centroids)

                assigned_centroids = set()
                for track_id, centroid_idx in assignments.items():
                    cx, cy = centroids[centroid_idx]
                    active_tracks[track_id]['current_centroid_idx'] = centroid_idx
                    active_tracks[track_id]['positions'].append((cx, cy, frame_idx))
                    active_tracks[track_id]['last_frame'] = frame_idx

                    if self.config.nose_detection_enabled:
                        nose_position = self.detect_nose_for_track(track_id, active_tracks, valid_contours, centroids)
                        if nose_position is not None:
                            if 'nose_positions' not in active_tracks[track_id]:
                                active_tracks[track_id]['nose_positions'] = []
                            active_tracks[track_id]['nose_positions'].append((nose_position[0], nose_position[1], frame_idx))

                    assigned_centroids.add(centroid_idx)

                for i, (cx, cy) in enumerate(centroids):
                    if i not in assigned_centroids:
                        while next_track_id in used_track_ids:
                            next_track_id += 1
                        used_track_ids.add(next_track_id)
                        active_tracks[next_track_id] = {
                            'positions': [(cx, cy, frame_idx)],
                            'nose_positions': [],
                            'last_frame': frame_idx
                        }
                        next_track_id += 1

            except Exception as e:
                self.logger.warning(f"Error processing frame {frame_idx}: {e}")
                continue

        all_final_tracks = {**active_tracks, **inactive_tracks}
        tracks = {}
        nose_tracks = {}
        track_statistics = []
        sorted_track_ids = sorted(all_final_tracks.keys())
        new_track_id = 1

        for original_track_id in sorted_track_ids:
            track_data = all_final_tracks[original_track_id]
            positions = track_data['positions']
            nose_positions = track_data.get('nose_positions', [])
            track_length = len(positions)
            displacement = self.calculate_track_displacement(positions)

            track_stats = {
                'original_track_id': original_track_id,
                'track_length': track_length,
                'displacement_distance': round(displacement, 2),
                'nose_detections': len(nose_positions),
                'nose_success_rate': len(nose_positions) / track_length if track_length > 0 else 0,
                'passed_length_filter': track_length >= self.config.min_track_length,
                'passed_movement_filter': not self.config.filter_stationary_tracks or displacement >= self.config.min_displacement_distance,
                'final_status': 'pending'
            }

            if track_length < self.config.min_track_length:
                track_stats['final_status'] = 'rejected_short'
                track_stats['final_track_id'] = None
            elif self.config.filter_stationary_tracks and displacement < self.config.min_displacement_distance:
                track_stats['final_status'] = 'rejected_stationary'
                track_stats['final_track_id'] = None
            else:
                track_stats['final_status'] = 'accepted'
                track_stats['final_track_id'] = new_track_id
                tracks[new_track_id] = positions
                if nose_positions:
                    nose_tracks[new_track_id] = nose_positions
                new_track_id += 1

            track_statistics.append(track_stats)

        return tracks, nose_tracks, track_statistics

    def _export_tracks_csv(self, tracks: Dict, nose_tracks: Dict, csv_path: str):
        all_frames = sorted({pos[2] for positions in tracks.values() for pos in positions})
        track_ids = sorted(tracks.keys())

        columns = ['frame']
        for tid in track_ids:
            columns.extend([f"worm_{tid}_x", f"worm_{tid}_y", f"worm_{tid}_nose_x", f"worm_{tid}_nose_y"])

        data = []
        for frame in all_frames:
            row = [frame]
            for track_id in track_ids:
                centroid_pos = next(((p[0], p[1]) for p in tracks[track_id] if p[2] == frame), None)
                if centroid_pos:
                    row.extend([round(centroid_pos[0], 4), round(centroid_pos[1], 4)])
                else:
                    row.extend([None, None])

                nose_pos = None
                if track_id in nose_tracks:
                    nose_pos = next(((p[0], p[1]) for p in nose_tracks[track_id] if p[2] == frame), None)
                if nose_pos:
                    row.extend([round(nose_pos[0], 4), round(nose_pos[1], 4)])
                else:
                    row.extend([None, None])
            data.append(row)

        df = pd.DataFrame(data, columns=columns)
        if os.path.exists(csv_path):
            self.logger.info(f"Overwriting existing tracks_debug.csv at {csv_path}")
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Exported tracks to {csv_path}")


# ---------------------------------------------------------------------------
# GUI — identical to batch_tracking_gpu.py with updated status display
# ---------------------------------------------------------------------------

class SmartDirectoryDialog:
    def __init__(self, parent):
        self.parent = parent
        self.selected_directories = []
        self.result = None

    def show(self):
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Add Directories")
        self.dialog.geometry("800x600")
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - 400
        y = (self.dialog.winfo_screenheight() // 2) - 300
        self.dialog.geometry(f"800x600+{x}+{y}")
        self.create_widgets()
        self.dialog.wait_window()
        return self.result

    def create_widgets(self):
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill='both', expand=True)
        ttk.Label(main_frame, text="Add Directories for Batch Processing",
                 font=('TkDefaultFont', 12, 'bold')).pack(pady=(0, 10))
        instructions = ("• Click 'Browse & Add Directory' to add individual directories\n"
                       "• Click 'Add All Subdirectories' to add all subdirs from a parent folder\n"
                       "• Use the list below to manage your selection\n"
                       "• Click OK when finished")
        ttk.Label(main_frame, text=instructions, justify='left').pack(pady=(0, 10), anchor='w')

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(0, 10))
        ttk.Button(button_frame, text="Browse & Add Directory", command=self.add_single_directory).pack(side='left', padx=(0, 5))
        ttk.Button(button_frame, text="Add All Subdirectories", command=self.add_subdirectories).pack(side='left', padx=(0, 5))
        ttk.Button(button_frame, text="Remove Selected", command=self.remove_selected).pack(side='left', padx=(0, 5))
        ttk.Button(button_frame, text="Clear All", command=self.clear_all).pack(side='left', padx=(0, 5))

        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill='both', expand=True, pady=(0, 10))
        self.dir_listbox = tk.Listbox(list_frame, selectmode='extended', height=15)
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.dir_listbox.yview)
        self.dir_listbox.configure(yscrollcommand=scrollbar.set)
        self.dir_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        self.status_label = ttk.Label(main_frame, text="No directories selected")
        self.status_label.pack(pady=(0, 10))
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill='x')
        ttk.Button(bottom_frame, text="Cancel", command=self.cancel).pack(side='right', padx=(5, 0))
        ttk.Button(bottom_frame, text="OK", command=self.ok).pack(side='right')

    def add_single_directory(self):
        directory = filedialog.askdirectory(parent=self.dialog, title="Select directory containing images")
        if directory and directory not in self.selected_directories:
            self.selected_directories.append(directory)
            self.dir_listbox.insert(tk.END, directory)
            self.update_status()

    def add_subdirectories(self):
        parent_dir = filedialog.askdirectory(parent=self.dialog, title="Select parent directory")
        if not parent_dir:
            return
        try:
            subdirs = [os.path.join(parent_dir, item) for item in os.listdir(parent_dir)
                      if os.path.isdir(os.path.join(parent_dir, item))]
            if not subdirs:
                messagebox.showinfo("No Subdirectories", "No subdirectories found.", parent=self.dialog)
                return
            if messagebox.askyesno("Confirm", f"Found {len(subdirs)} subdirectories. Add all?", parent=self.dialog):
                added = 0
                for subdir in subdirs:
                    if subdir not in self.selected_directories:
                        self.selected_directories.append(subdir)
                        self.dir_listbox.insert(tk.END, subdir)
                        added += 1
                messagebox.showinfo("Added", f"Added {added}, skipped {len(subdirs) - added} duplicates.", parent=self.dialog)
                self.update_status()
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}", parent=self.dialog)

    def remove_selected(self):
        for index in reversed(self.dir_listbox.curselection()):
            self.selected_directories.pop(index)
            self.dir_listbox.delete(index)
        self.update_status()

    def clear_all(self):
        if self.selected_directories and messagebox.askyesno("Clear All", "Remove all?", parent=self.dialog):
            self.selected_directories.clear()
            self.dir_listbox.delete(0, tk.END)
            self.update_status()

    def update_status(self):
        count = len(self.selected_directories)
        self.status_label.config(text=f"{count} director{'y' if count == 1 else 'ies'} selected" if count else "No directories selected")

    def ok(self):
        self.result = self.selected_directories.copy()
        self.dialog.destroy()

    def cancel(self):
        self.result = None
        self.dialog.destroy()


class BatchWormTrackerGUI:
    def __init__(self):
        self.root = tk.Tk()
        tag = f"[{GPU_BACKEND.upper()}]" if GPU_ACCELERATED else "[CPU]"
        self.root.title(f"Batch Worm Tracker {tag} — GPU-Accelerated Median")
        self.root.geometry("1200x800")

        self.config = BatchConfig()
        self.tracker = BatchWormTracker(self.config)
        self.directories = []
        self.results = []
        self.processing_thread = None
        self.setup_gui()

    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # GPU status bar
        gpu_frame = ttk.Frame(main_frame)
        gpu_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        if GPU_ACCELERATED:
            gpu_text = f"Compute: {GPU_BACKEND.upper()} — {GPU_DEVICE_NAME}"
            gpu_color = 'green'
        else:
            gpu_text = f"Compute: CPU (NumPy) — no GPU backend available"
            gpu_color = 'red'
        ttk.Label(gpu_frame, text=gpu_text,
                 font=('TkDefaultFont', 9, 'bold'), foreground=gpu_color).pack(side='left')

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.config_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.config_tab, text="1. Configuration")
        self.create_config_tab()

        self.batch_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.batch_tab, text="2. Batch Processing")
        self.create_batch_tab()

        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="3. Results")
        self.create_results_tab()

    def create_config_tab(self):
        canvas = tk.Canvas(self.config_tab)
        scrollbar = ttk.Scrollbar(self.config_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        self.create_threshold_config(scrollable_frame)
        self.create_blob_config(scrollable_frame)
        self.create_tracking_config(scrollable_frame)
        self.create_nose_config(scrollable_frame)
        self.create_action_buttons(scrollable_frame)

    def create_threshold_config(self, parent):
        frame = ttk.LabelFrame(parent, text="Threshold Parameters", padding="10")
        frame.pack(fill='x', padx=5, pady=5)
        r1 = ttk.Frame(frame); r1.pack(fill='x', pady=2)
        ttk.Label(r1, text="Minimum Threshold:").pack(side='left')
        self.min_thresh_var = tk.StringVar(value=str(self.config.threshold_min))
        ttk.Entry(r1, textvariable=self.min_thresh_var, width=10).pack(side='right')
        r2 = ttk.Frame(frame); r2.pack(fill='x', pady=2)
        ttk.Label(r2, text="Maximum Threshold:").pack(side='left')
        self.max_thresh_var = tk.StringVar(value=str(self.config.threshold_max))
        ttk.Entry(r2, textvariable=self.max_thresh_var, width=10).pack(side='right')
        ttk.Label(frame, text="Range: 0-255. Higher values detect brighter objects.",
                 font=('TkDefaultFont', 8), foreground='gray').pack(anchor='w', pady=(5, 0))

    def create_blob_config(self, parent):
        frame = ttk.LabelFrame(parent, text="Blob Size Filters", padding="10")
        frame.pack(fill='x', padx=5, pady=5)
        r1 = ttk.Frame(frame); r1.pack(fill='x', pady=2)
        ttk.Label(r1, text="Minimum Blob Size (pixels):").pack(side='left')
        self.min_blob_var = tk.StringVar(value=str(self.config.min_blob_size))
        ttk.Entry(r1, textvariable=self.min_blob_var, width=10).pack(side='right')
        r2 = ttk.Frame(frame); r2.pack(fill='x', pady=2)
        ttk.Label(r2, text="Maximum Blob Size (pixels):").pack(side='left')
        self.max_blob_var = tk.StringVar(value=str(self.config.max_blob_size))
        ttk.Entry(r2, textvariable=self.max_blob_var, width=10).pack(side='right')
        ttk.Label(frame, text="Filters detected objects by area. Adjust for worm size.",
                 font=('TkDefaultFont', 8), foreground='gray').pack(anchor='w', pady=(5, 0))

    def create_tracking_config(self, parent):
        frame = ttk.LabelFrame(parent, text="Trajectory-Aware Tracking Parameters", padding="10")
        frame.pack(fill='x', padx=5, pady=5)

        for label, var_name, default in [
            ("Max Distance Between Frames (pixels):", "max_dist_var", self.config.max_distance),
            ("Trajectory Weight (0.0-1.0):", "traj_weight_var", self.config.trajectory_weight),
            ("Minimum Track Length (frames):", "min_track_var", self.config.min_track_length),
        ]:
            r = ttk.Frame(frame); r.pack(fill='x', pady=2)
            ttk.Label(r, text=label).pack(side='left')
            var = tk.StringVar(value=str(default))
            setattr(self, var_name, var)
            ttk.Entry(r, textvariable=var, width=10).pack(side='right')

        algo_frame = ttk.Frame(frame); algo_frame.pack(fill='x', pady=2)
        ttk.Label(algo_frame, text="Assignment Algorithm:").pack(side='left')
        self.algorithm_var = tk.StringVar(value="Greedy" if not self.config.use_hungarian else "Hungarian")
        combo = ttk.Combobox(algo_frame, textvariable=self.algorithm_var,
                            values=["Greedy", "Hungarian"], width=12, state="readonly")
        combo.pack(side='right')
        combo.bind('<<ComboboxSelected>>', self.update_algorithm)

        ttk.Label(frame, text="Trajectory weight: 0.7 recommended for ID swap prevention",
                 font=('TkDefaultFont', 8), foreground='gray').pack(anchor='w', pady=(5, 0))
        ttk.Label(frame, text="Greedy algorithm recommended for worm tracking",
                 font=('TkDefaultFont', 8), foreground='gray').pack(anchor='w')

        mf = ttk.Frame(frame); mf.pack(fill='x', pady=(10, 2))
        self.filter_stationary_var = tk.BooleanVar(value=self.config.filter_stationary_tracks)
        ttk.Checkbutton(mf, text="Filter Stationary Tracks", variable=self.filter_stationary_var,
                       command=self.update_movement_filter).pack(side='left')

        df = ttk.Frame(frame); df.pack(fill='x', pady=2)
        ttk.Label(df, text="Min Displacement Distance (pixels):").pack(side='left')
        self.min_displacement_var = tk.StringVar(value=str(self.config.min_displacement_distance))
        ttk.Entry(df, textvariable=self.min_displacement_var, width=10).pack(side='right')
        ttk.Label(frame, text="Removes tracks that don't move significantly from start to end position",
                 font=('TkDefaultFont', 8), foreground='gray').pack(anchor='w')

    def create_nose_config(self, parent):
        frame = ttk.LabelFrame(parent, text="Nose Detection Parameters", padding="10")
        frame.pack(fill='x', padx=5, pady=5)

        ef = ttk.Frame(frame); ef.pack(fill='x', pady=2)
        self.nose_enabled_var = tk.BooleanVar(value=self.config.nose_detection_enabled)
        ttk.Checkbutton(ef, text="Enable Nose Detection", variable=self.nose_enabled_var,
                       command=self.update_nose_detection).pack(side='left')

        sf = ttk.Frame(frame); sf.pack(fill='x', pady=2)
        ttk.Label(sf, text="Smoothing Frames (2-10):").pack(side='left')
        self.nose_smooth_var = tk.StringVar(value=str(self.config.nose_smoothing_frames))
        ttk.Entry(sf, textvariable=self.nose_smooth_var, width=10).pack(side='right')

        mvf = ttk.Frame(frame); mvf.pack(fill='x', pady=2)
        ttk.Label(mvf, text="Min Movement Threshold (pixels/frame):").pack(side='left')
        self.nose_movement_var = tk.StringVar(value=str(self.config.min_movement_threshold))
        ttk.Entry(mvf, textvariable=self.nose_movement_var, width=10).pack(side='right')

        ttk.Label(frame, text="Detects worm front based on locomotion direction",
                 font=('TkDefaultFont', 8), foreground='gray').pack(anchor='w', pady=(5, 0))

    def create_action_buttons(self, parent):
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill='x', padx=5, pady=10)
        ttk.Button(frame, text="Reset to Defaults", command=self.reset_to_defaults).pack(side='left', padx=5)
        ttk.Button(frame, text="Validate All Parameters", command=self.validate_all_parameters).pack(side='left', padx=5)
        ttk.Button(frame, text="Save Configuration", command=self.save_configuration).pack(side='left', padx=5)
        ttk.Button(frame, text="Load Configuration", command=self.load_configuration).pack(side='left', padx=5)
        self.config_status_label = ttk.Label(frame, text="Parameters valid", foreground='green')
        self.config_status_label.pack(side='right', padx=10)

    def create_batch_tab(self):
        batch_frame = ttk.Frame(self.batch_tab, padding="10")
        batch_frame.pack(fill='both', expand=True)
        batch_frame.columnconfigure(1, weight=1)
        batch_frame.rowconfigure(2, weight=1)

        sf = ttk.LabelFrame(batch_frame, text="Current Configuration Summary", padding="10")
        sf.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        self.config_summary_text = tk.Text(sf, height=6, wrap='word', font=('TkDefaultFont', 8))
        self.config_summary_text.pack(fill='x')
        self.update_config_summary()

        dir_frame = ttk.LabelFrame(batch_frame, text="Directory Selection", padding="10")
        dir_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        bf = ttk.Frame(dir_frame); bf.pack(fill='x', pady=(0, 10))
        ttk.Button(bf, text="Add Directories", command=self.add_directories).pack(side='left', padx=(0, 5))
        ttk.Button(bf, text="Remove Selected", command=self.remove_directory).pack(side='left', padx=(0, 5))
        ttk.Button(bf, text="Clear All", command=self.clear_directories).pack(side='left', padx=(0, 5))
        ttk.Button(bf, text="Start Batch Processing", command=self.start_batch_processing).pack(side='right')

        lf = ttk.Frame(dir_frame); lf.pack(fill='both', expand=True)
        self.dir_listbox = tk.Listbox(lf, height=8, selectmode='extended')
        dsb = ttk.Scrollbar(lf, orient='vertical', command=self.dir_listbox.yview)
        self.dir_listbox.configure(yscrollcommand=dsb.set)
        self.dir_listbox.pack(side='left', fill='both', expand=True)
        dsb.pack(side='right', fill='y')

        pf = ttk.LabelFrame(batch_frame, text="Processing Progress", padding="10")
        pf.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        ttk.Label(pf, text="Overall Progress:").pack(anchor='w')
        self.overall_progress = ttk.Progressbar(pf, mode='determinate')
        self.overall_progress.pack(fill='x', pady=(2, 10))
        ttk.Label(pf, text="Current Directory:").pack(anchor='w')
        self.current_progress = ttk.Progressbar(pf, mode='determinate')
        self.current_progress.pack(fill='x', pady=(2, 10))
        self.status_label = ttk.Label(pf, text=f"Ready — {N_WORKERS} threads for I/O ({os.cpu_count()} cores detected)")
        self.status_label.pack(anchor='w')
        self.current_dir_label = ttk.Label(pf, text="")
        self.current_dir_label.pack(anchor='w')

    def create_results_tab(self):
        rf = ttk.Frame(self.results_tab, padding="10")
        rf.pack(fill='both', expand=True)
        rf.columnconfigure(0, weight=1)
        rf.rowconfigure(1, weight=1)

        sf = ttk.LabelFrame(rf, text="Processing Summary", padding="10")
        sf.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.results_summary_label = ttk.Label(sf, text="No processing results yet")
        self.results_summary_label.pack(anchor='w')

        rtf = ttk.LabelFrame(rf, text="Detailed Results", padding="10")
        rtf.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.results_text = scrolledtext.ScrolledText(rtf, height=20, wrap='word')
        self.results_text.pack(fill='both', expand=True)

        bf = ttk.Frame(rf)
        bf.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        ttk.Button(bf, text="Export Summary Report", command=self.export_summary).pack(side='left')
        ttk.Button(bf, text="Clear Results", command=self.clear_results).pack(side='left', padx=(10, 0))
        ttk.Button(bf, text="Exit", command=self.root.quit).pack(side='right')

    # --- validation ---

    def validate_threshold_params(self, event=None):
        try:
            mn, mx = int(self.min_thresh_var.get()), int(self.max_thresh_var.get())
            if not (0 <= mn <= 255) or not (0 <= mx <= 255) or mn >= mx:
                self.config_status_label.config(text="Invalid threshold range", foreground='red'); return False
            self.config.threshold_min, self.config.threshold_max = mn, mx
            self.update_config_summary(); return True
        except ValueError:
            self.config_status_label.config(text="Invalid threshold values", foreground='red'); return False

    def validate_blob_params(self, event=None):
        try:
            mn, mx = int(self.min_blob_var.get()), int(self.max_blob_var.get())
            if mn < 1 or mx < mn:
                self.config_status_label.config(text="Invalid blob size range", foreground='red'); return False
            self.config.min_blob_size, self.config.max_blob_size = mn, mx
            self.update_config_summary(); return True
        except ValueError:
            self.config_status_label.config(text="Invalid blob size values", foreground='red'); return False

    def validate_tracking_params(self, event=None):
        try:
            md = int(self.max_dist_var.get())
            tw = float(self.traj_weight_var.get())
            mt = int(self.min_track_var.get())
            disp = float(self.min_displacement_var.get())
            if md < 1 or not (0.0 <= tw <= 1.0) or mt < 1 or disp < 0:
                self.config_status_label.config(text="Invalid tracking parameters", foreground='red'); return False
            self.config.max_distance, self.config.trajectory_weight = md, tw
            self.config.min_track_length, self.config.min_displacement_distance = mt, disp
            self.update_config_summary(); return True
        except ValueError:
            self.config_status_label.config(text="Invalid tracking values", foreground='red'); return False

    def validate_nose_params(self, event=None):
        try:
            sf = int(self.nose_smooth_var.get())
            mm = float(self.nose_movement_var.get())
            if not (2 <= sf <= 10) or mm < 0.1:
                self.config_status_label.config(text="Invalid nose parameters", foreground='red'); return False
            self.config.nose_smoothing_frames, self.config.min_movement_threshold = sf, mm
            self.update_config_summary(); return True
        except ValueError:
            self.config_status_label.config(text="Invalid nose values", foreground='red'); return False

    def update_algorithm(self, event=None):
        self.config.use_hungarian = (self.algorithm_var.get() == "Hungarian")
        self.update_config_summary()

    def update_nose_detection(self):
        self.config.nose_detection_enabled = self.nose_enabled_var.get()
        self.update_config_summary()

    def update_movement_filter(self):
        self.config.filter_stationary_tracks = self.filter_stationary_var.get()
        self.update_config_summary()

    def validate_all_parameters(self):
        valid = (self.validate_threshold_params() and self.validate_blob_params() and
                self.validate_tracking_params() and self.validate_nose_params())
        if valid:
            self.config_status_label.config(text="All parameters valid", foreground='green')
            self.tracker = BatchWormTracker(self.config)
        return valid

    def reset_to_defaults(self):
        self.config = BatchConfig()
        self.min_thresh_var.set(str(self.config.threshold_min))
        self.max_thresh_var.set(str(self.config.threshold_max))
        self.min_blob_var.set(str(self.config.min_blob_size))
        self.max_blob_var.set(str(self.config.max_blob_size))
        self.max_dist_var.set(str(self.config.max_distance))
        self.traj_weight_var.set(str(self.config.trajectory_weight))
        self.min_track_var.set(str(self.config.min_track_length))
        self.algorithm_var.set("Greedy")
        self.nose_enabled_var.set(self.config.nose_detection_enabled)
        self.nose_smooth_var.set(str(self.config.nose_smoothing_frames))
        self.nose_movement_var.set(str(self.config.min_movement_threshold))
        self.filter_stationary_var.set(self.config.filter_stationary_tracks)
        self.min_displacement_var.set(str(self.config.min_displacement_distance))
        self.config_status_label.config(text="Reset to defaults", foreground='green')
        self.tracker = BatchWormTracker(self.config)
        self.update_config_summary()

    def update_config_summary(self):
        backend = f"{GPU_BACKEND.upper()}: {GPU_DEVICE_NAME}" if GPU_ACCELERATED else "CPU (NumPy fallback)"
        s = f"Compute: {backend}\n"
        s += f"Thresholds: {self.config.threshold_min}-{self.config.threshold_max} | "
        s += f"Blob: {self.config.min_blob_size}-{self.config.max_blob_size} px | "
        s += f"Max Dist: {self.config.max_distance} px | "
        s += f"Traj Weight: {self.config.trajectory_weight} | "
        s += f"Min Track: {self.config.min_track_length} frames | "
        s += f"Algo: {'Hungarian' if self.config.use_hungarian else 'Greedy'} | "
        s += f"Movement Filter: {'ON' if self.config.filter_stationary_tracks else 'OFF'}"
        if self.config.filter_stationary_tracks:
            s += f" (>={self.config.min_displacement_distance}px) | "
        else:
            s += " | "
        s += f"Nose: {'ON' if self.config.nose_detection_enabled else 'OFF'}"
        if hasattr(self, 'config_summary_text'):
            self.config_summary_text.delete(1.0, tk.END)
            self.config_summary_text.insert(1.0, s)

    def save_configuration(self):
        if not self.validate_all_parameters():
            messagebox.showerror("Invalid", "Fix parameter errors first"); return
        fp = filedialog.asksaveasfilename(title="Save Configuration", defaultextension=".txt",
                                          filetypes=[("Text files", "*.txt")])
        if not fp: return
        try:
            with open(fp, 'w') as f:
                f.write("# Batch Worm Tracker GPU Accel Configuration\n")
                for k in ['threshold_min','threshold_max','min_blob_size','max_blob_size',
                          'max_distance','trajectory_weight','min_track_length','use_hungarian',
                          'nose_detection_enabled','nose_smoothing_frames','min_movement_threshold',
                          'filter_stationary_tracks','min_displacement_distance']:
                    f.write(f"{k}={getattr(self.config, k)}\n")
            messagebox.showinfo("Saved", f"Saved to:\n{fp}")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {e}")

    def load_configuration(self):
        fp = filedialog.askopenfilename(title="Load Configuration", filetypes=[("Text files", "*.txt")])
        if not fp: return
        field_map = {
            'threshold_min': (int, 'min_thresh_var'), 'threshold_max': (int, 'max_thresh_var'),
            'min_blob_size': (int, 'min_blob_var'), 'max_blob_size': (int, 'max_blob_var'),
            'max_distance': (int, 'max_dist_var'), 'trajectory_weight': (float, 'traj_weight_var'),
            'min_track_length': (int, 'min_track_var'), 'nose_smoothing_frames': (int, 'nose_smooth_var'),
            'min_movement_threshold': (float, 'nose_movement_var'),
            'min_displacement_distance': (float, 'min_displacement_var'),
        }
        bool_map = {
            'use_hungarian': None, 'nose_detection_enabled': 'nose_enabled_var',
            'filter_stationary_tracks': 'filter_stationary_var',
        }
        try:
            with open(fp, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line: continue
                    key, value = line.split('=', 1)
                    key, value = key.strip(), value.strip()
                    try:
                        if key in field_map:
                            cast, var_name = field_map[key]
                            setattr(self.config, key, cast(value))
                            getattr(self, var_name).set(value)
                        elif key in bool_map:
                            bv = value.lower() == 'true'
                            setattr(self.config, key, bv)
                            if bool_map[key]: getattr(self, bool_map[key]).set(bv)
                            if key == 'use_hungarian': self.algorithm_var.set("Hungarian" if bv else "Greedy")
                    except ValueError: continue
            if self.validate_all_parameters():
                self.tracker = BatchWormTracker(self.config)
                self.update_config_summary()
                messagebox.showinfo("Loaded", "Configuration loaded successfully")
            else:
                messagebox.showwarning("Invalid", "Some parameters are invalid")
        except Exception as e:
            messagebox.showerror("Error", f"Load failed: {e}")

    # --- directory management ---

    def add_directories(self):
        dialog = SmartDirectoryDialog(self.root)
        dirs = dialog.show()
        if dirs:
            added = sum(1 for d in dirs if d not in self.directories and (self.directories.append(d) or self.dir_listbox.insert(tk.END, d) or True))
            if added: self.update_status(f"Added {added} director{'y' if added == 1 else 'ies'}")

    def remove_directory(self):
        sel = self.dir_listbox.curselection()
        if not sel: messagebox.showinfo("No Selection", "Select directories to remove"); return
        for i in reversed(sel): self.directories.pop(i); self.dir_listbox.delete(i)
        self.update_status(f"Removed {len(sel)} director{'y' if len(sel) == 1 else 'ies'}")

    def clear_directories(self):
        if self.directories and messagebox.askyesno("Clear", "Remove all?"):
            self.directories.clear(); self.dir_listbox.delete(0, tk.END)
            self.update_status("Cleared all directories")

    def update_status(self, msg):
        self.status_label.config(text=msg); self.root.update_idletasks()

    def update_current_dir_status(self, msg):
        self.current_dir_label.config(text=msg); self.root.update_idletasks()

    # --- batch processing ---

    def start_batch_processing(self):
        if not self.validate_all_parameters():
            messagebox.showerror("Invalid", "Fix errors first"); return
        if not self.directories:
            messagebox.showwarning("Empty", "Add at least one directory"); return
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("Busy", "Already processing"); return
        self.tracker = BatchWormTracker(self.config)
        backend_str = f"{GPU_BACKEND.upper()}: {GPU_DEVICE_NAME}" if GPU_ACCELERATED else "CPU (NumPy)"
        msg = (f"Process {len(self.directories)} directories?\n\n"
               f"Compute: {backend_str}\n"
               f"Background: GPU tiled median, image loading: {N_WORKERS} threads\n"
               f"Tracking: sequential with I/O prefetch (12 frames ahead)\n\n"
               "tracks_debug.csv will be saved (overwriting existing).")
        if not messagebox.askyesno("Confirm", msg): return
        self.results.clear(); self.results_text.delete(1.0, tk.END)
        self.processing_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self.processing_thread.start()

    def _batch_worker(self):
        try:
            total = len(self.directories)
            self.root.after(0, lambda: self.overall_progress.config(maximum=total, value=0))

            for i, directory in enumerate(self.directories):
                self.root.after(0, lambda d=directory:
                    self.update_current_dir_status(f"Processing: {os.path.basename(d)}"))
                self.root.after(0, lambda: self.update_status(
                    f"Directory {i+1}/{total}"))
                self.root.after(0, lambda: self.current_progress.config(maximum=100, value=0))

                def progress_callback(cur, tot):
                    pv = (cur / tot) * 100
                    self.root.after(0, lambda pv=pv: self.current_progress.config(value=pv))

                result = self.tracker.process_directory(directory, progress_callback)
                self.results.append(result)
                self.root.after(0, lambda p=i+1: self.overall_progress.config(value=p))
                self.root.after(0, lambda r=result: self._update_results_display(r))

            self.root.after(0, self._processing_complete)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Batch error: {e}"))
            self.root.after(0, lambda: self.update_status("Batch processing failed"))

    def _update_results_display(self, result):
        self.results_text.insert(tk.END, self._format_result(result))
        self.results_text.see(tk.END)
        self._update_results_summary()

    def _update_results_summary(self):
        if not self.results: return
        ok = len([r for r in self.results if r.success])
        self.results_summary_label.config(text=f"Processed: {len(self.results)} | OK: {ok} | Failed: {len(self.results)-ok}")

    def _format_result(self, r):
        status = "SUCCESS" if r.success else "FAILED"
        quality = {"normal": "NORMAL", "noisy": "NOISY", "empty": "EMPTY"}[r.quality_flag]
        t = f"\n[{status}] {os.path.basename(r.directory)} [{quality}]\n"
        t += f"   Images: {r.num_images}, Tracks: {r.num_accepted_tracks}, Time: {r.processing_time:.1f}s\n"
        if r.quality_flag == "noisy": t += "   WARNING: HIGH TRACK COUNT\n"
        elif r.quality_flag == "empty": t += "   WARNING: NO TRACKS FOUND\n"
        if not r.success: t += f"   Error: {r.error_message}\n"
        return t

    def _processing_complete(self):
        self.update_status("Batch processing complete!")
        self.update_current_dir_status("")
        self.current_progress.config(value=0)
        summary = self._generate_summary()
        self.results_text.insert(tk.END, f"\n{'='*50}\nBATCH PROCESSING SUMMARY\n{'='*50}\n{summary}")
        self.results_text.see(tk.END)
        self.notebook.select(self.results_tab)
        messagebox.showinfo("Complete", "Batch processing finished!\nCheck the Results tab.")

    def _generate_summary(self):
        if not self.results: return "No results.\n"
        ok = [r for r in self.results if r.success]
        fail = [r for r in self.results if not r.success]
        noisy = [r for r in ok if r.quality_flag == "noisy"]
        empty = [r for r in ok if r.quality_flag == "empty"]
        normal = [r for r in ok if r.quality_flag == "normal"]

        s = f"Total: {len(self.results)} | OK: {len(ok)} | Failed: {len(fail)}\n\n"
        if ok:
            tt = sum(r.num_accepted_tracks for r in ok)
            tp = sum(r.processing_time for r in ok)
            s += f"Quality: {len(normal)} normal, {len(noisy)} noisy, {len(empty)} empty\n"
            s += f"Tracks: {tt} total | Time: {tp:.1f}s ({tp/len(ok):.1f}s avg)\n\n"

        s += "Compute:\n"
        if GPU_ACCELERATED:
            s += f"  {GPU_BACKEND.upper()}: {GPU_DEVICE_NAME}\n"
        else:
            s += "  CPU (NumPy) — no GPU backend\n"
        s += f"  I/O threads: {N_WORKERS} (of {os.cpu_count()} cores)\n"
        s += "  Background: parallel image loading + GPU tiled median\n"
        s += "  Tracking: sequential with I/O prefetch\n\n"

        s += "Config:\n"
        s += f"  Thresh: {self.config.threshold_min}-{self.config.threshold_max} | "
        s += f"Blob: {self.config.min_blob_size}-{self.config.max_blob_size} | "
        s += f"MaxDist: {self.config.max_distance} | TrajW: {self.config.trajectory_weight}\n"
        s += f"  MinTrack: {self.config.min_track_length} | "
        s += f"Algo: {'Hungarian' if self.config.use_hungarian else 'Greedy'} | "
        s += f"MovFilter: {'ON' if self.config.filter_stationary_tracks else 'OFF'}"
        if self.config.filter_stationary_tracks:
            s += f" (>={self.config.min_displacement_distance}px)"
        s += f" | Nose: {'ON' if self.config.nose_detection_enabled else 'OFF'}\n\n"

        if noisy:
            s += "NEEDS QC:\n" + "".join(f"  - {os.path.basename(r.directory)} ({r.num_accepted_tracks} tracks)\n" for r in noisy) + "\n"
        if empty:
            s += "NO TRACKS:\n" + "".join(f"  - {os.path.basename(r.directory)}\n" for r in empty) + "\n"
        if fail:
            s += "FAILED:\n" + "".join(f"  - {os.path.basename(r.directory)}: {r.error_message}\n" for r in fail) + "\n"
        s += "All tracks_debug.csv files saved.\n"
        return s

    def export_summary(self):
        if not self.results: messagebox.showwarning("Empty", "No results"); return
        fp = filedialog.asksaveasfilename(title="Save summary", defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if not fp: return
        try:
            with open(fp, 'w') as f:
                f.write(f"BATCH WORM TRACKER GPU ACCEL REPORT\n{'='*60}\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Backend: {GPU_BACKEND.upper()} — {GPU_DEVICE_NAME}\n\n")
                for r in self.results:
                    f.write(f"{'='*40}\n{r.directory}\n")
                    f.write(f"  OK={r.success} Images={r.num_images} Tracks={r.num_accepted_tracks} "
                           f"Quality={r.quality_flag} Time={r.processing_time:.1f}s\n")
                    if r.error_message: f.write(f"  Error: {r.error_message}\n")
                f.write(f"\n{self._generate_summary()}")
            messagebox.showinfo("Exported", f"Saved to:\n{fp}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")

    def clear_results(self):
        self.results.clear(); self.results_text.delete(1.0, tk.END)
        self.overall_progress.config(value=0); self.current_progress.config(value=0)
        self.update_status(f"Ready — {N_WORKERS} threads for I/O ({os.cpu_count()} cores detected)")
        self.update_current_dir_status("")
        self.results_summary_label.config(text="No processing results yet")

    def run(self):
        self.root.mainloop()


def main():
    print("=" * 70)
    print("BATCH WORM TRACKER — GPU-Accelerated (platform-aware cascade)")
    print("=" * 70)
    print(f"\nPlatform: {platform.system()} | GPU vendor (OpenCL): {_gpu_vendor}")
    if GPU_ACCELERATED:
        print(f"Backend:  {GPU_BACKEND.upper()} — {GPU_DEVICE_NAME}")
        print("GPU ops:  median (background), absdiff (subtraction), inRange (threshold)")
    else:
        print("Backend:  CPU (NumPy) — no GPU library detected")
        print("\nTo enable GPU acceleration, install one of:")
        if IS_WINDOWS:
            print("  NVIDIA: pip install cupy-cuda12x   (or cupy-cuda11x)")
            print("  AMD:    pip install torch torch-directml")
        elif IS_LINUX:
            print("  NVIDIA: pip install cupy-cuda12x")
            print("  AMD:    pip install torch (ROCm wheel from pytorch.org)")
    print(f"\nCascade order: {'Win' if IS_WINDOWS else 'Linux'} + "
          f"{_gpu_vendor.upper()} → {GPU_BACKEND.upper()}")
    print(f"\nI/O threads: {N_WORKERS} (of {os.cpu_count()} cores)")
    print("Background: parallel image loading + GPU tiled median")
    print("Tracking: sequential per directory with I/O prefetch")
    print("\nSTARTING GUI...")

    try:
        app = BatchWormTrackerGUI()
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
