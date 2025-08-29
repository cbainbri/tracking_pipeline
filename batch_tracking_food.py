#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch Worm Tracker - Automated processing of multiple image directories
Extracts core functionality from the main WormTracker for automated processing
"""

import os
import sys
import gc
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import time
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from pathlib import Path
import logging
from dataclasses import dataclass


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
    quality_flag: str = "normal"  # "normal", "noisy", "empty"


class BatchWormTracker:
    """Core tracking functionality extracted for batch processing"""

    def __init__(self, config: BatchConfig):
        self.config = config
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for batch processing"""
        logger = logging.getLogger('BatchWormTracker')
        logger.setLevel(logging.INFO)

        # Clear existing handlers
        logger.handlers.clear()

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        return logger

    def find_image_files(self, directory: str) -> List[str]:
        """Find all image files in directory"""
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
        """Generate background image using memory-efficient sampling"""
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

        batch_size = 10
        all_backgrounds = []
        reference_shape = None

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
                except Exception as e:
                    self.logger.warning(f"Error loading image {img_path}: {e}")
                    continue

            gc.collect()

        if all_backgrounds:
            if len(all_backgrounds) > 30:
                idx = np.random.choice(len(all_backgrounds), 30, replace=False)
                sampled_backgrounds = [all_backgrounds[i] for i in idx]
                background = np.median(sampled_backgrounds, axis=0).astype(np.uint8)
            else:
                background = np.median(all_backgrounds, axis=0).astype(np.uint8)

            self.logger.info(f"Background generated from {len(all_backgrounds)} images")
            return background
        else:
            self.logger.error("No valid images found for background generation")
            return None

    def apply_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply threshold + blob-size filtering"""
        if image is None or image.size == 0:
            return np.zeros((100, 100), dtype=np.uint8)

        try:
            min_thresh = max(0, min(255, self.config.threshold_min))
            max_thresh = max(min_thresh, min(255, self.config.threshold_max))
            min_blob = max(1, self.config.min_blob_size)
            max_blob = max(min_blob, self.config.max_blob_size)

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
            self.logger.error(f"Error in apply_threshold: {e}")
            return np.zeros_like(image)

    def load_and_process_frame(self, image_path: str, background: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load and background-subtract a frame"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None

        if background is not None:
            if img.shape != background.shape:
                img = cv2.resize(img, (background.shape[1], background.shape[0]))
            if img.dtype != background.dtype:
                img = img.astype(background.dtype, copy=False)
            subtracted = cv2.absdiff(background, img)
        else:
            subtracted = img

        return img, subtracted

    def predict_next_position(self, track_positions: List[Tuple[float, float, int]], min_frames: int = 3) -> Optional[Tuple[float, float]]:
        """Predict next position based on recent trajectory"""
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
        predicted_x = last_x + avg_vx
        predicted_y = last_y + avg_vy

        return (predicted_x, predicted_y)

    def assign_tracks_with_trajectory(self, active_tracks: Dict, centroids: List[Tuple[float, float]]) -> Dict[int, int]:
        """Enhanced track assignment with trajectory prediction"""
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

                predicted_pos = self.predict_next_position(track_data['positions'])
                track_predictions.append(predicted_pos)

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
        """Greedy assignment fallback"""
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
        """Calculate locomotion direction from recent positions"""
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

        normalized_direction = (avg_vx / direction_magnitude, avg_vy / direction_magnitude)
        return normalized_direction

    def find_nose_position(self, contour: np.ndarray, locomotion_direction: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Find nose position with sub-pixel precision"""
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
        """Detect nose position for a specific track"""
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

        contour = contours[centroid_idx]
        nose_position = self.find_nose_position(contour, locomotion_direction)
        return nose_position

    def process_directory(self, image_directory: str, progress_callback=None) -> ProcessingResult:
        """Process a single image directory"""
        start_time = time.time()
        result = ProcessingResult(
            directory=image_directory,
            success=False,
            num_images=0,
            num_tracks=0,
            num_accepted_tracks=0,
            processing_time=0.0
        )

        try:
            self.logger.info(f"Processing directory: {image_directory}")

            # Find image files
            image_files = self.find_image_files(image_directory)
            if not image_files:
                result.error_message = "No image files found"
                return result

            result.num_images = len(image_files)
            self.logger.info(f"Found {len(image_files)} images")

            # Generate background
            background = self.generate_background(image_files)
            if background is None:
                result.error_message = "Failed to generate background"
                return result

            # Run tracking
            tracks, nose_tracks, track_statistics = self._run_tracking(image_files, background, progress_callback)

            if not tracks:
                result.error_message = "No tracks generated"
                return result

            result.num_tracks = len(tracks)
            result.num_accepted_tracks = len([s for s in track_statistics if s['final_status'] == 'accepted'])

            # Assess quality
            result.quality_flag = self._assess_quality(result.num_accepted_tracks, result.num_images)

            # Export CSV
            csv_path = os.path.join(image_directory, "tracks.csv")
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
        """Assess quality of tracking results"""
        if num_tracks == 0:
            return "empty"

        # Flag as noisy if we have way too many tracks
        tracks_per_image = num_tracks / max(num_images, 1)
        if tracks_per_image > 2.0:  # More than 2 tracks per image on average
            return "noisy"
        elif num_tracks > 100:  # Absolute threshold
            return "noisy"

        return "normal"

    def _run_tracking(self, image_files: List[str], background: np.ndarray, progress_callback=None) -> Tuple[Dict, Dict, List]:
        """Run the tracking algorithm"""
        next_track_id = 1
        active_tracks = {}
        inactive_tracks = {}
        MAX_MISSING_FRAMES = 5

        total_frames = len(image_files)

        for frame_idx, img_path in enumerate(image_files):
            try:
                if progress_callback:
                    progress_callback(frame_idx + 1, total_frames)

                img, subtracted = self.load_and_process_frame(img_path, background)
                if img is None or subtracted is None:
                    continue

                thresholded = self.apply_threshold(subtracted)
                contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                centroids = []
                valid_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if self.config.min_blob_size <= area <= self.config.max_blob_size:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = M["m10"] / M["m00"]
                            cy = M["m01"] / M["m00"]
                            centroids.append((cx, cy))
                            valid_contours.append(contour)

                # Deactivate old tracks
                tracks_to_deactivate = []
                for track_id, track_data in active_tracks.items():
                    frames_missing = frame_idx - track_data['last_frame']
                    if frames_missing > MAX_MISSING_FRAMES:
                        tracks_to_deactivate.append(track_id)

                for track_id in tracks_to_deactivate:
                    inactive_tracks[track_id] = active_tracks[track_id]
                    del active_tracks[track_id]

                # Assign tracks
                assignments = self.assign_tracks_with_trajectory(active_tracks, centroids)

                # Update assigned tracks
                assigned_centroids = set()
                for track_id, centroid_idx in assignments.items():
                    cx, cy = centroids[centroid_idx]

                    active_tracks[track_id]['current_centroid_idx'] = centroid_idx
                    active_tracks[track_id]['positions'].append((cx, cy, frame_idx))
                    active_tracks[track_id]['last_frame'] = frame_idx

                    # Nose detection
                    if self.config.nose_detection_enabled:
                        nose_position = self.detect_nose_for_track(track_id, active_tracks, valid_contours, centroids)
                        if nose_position is not None:
                            if 'nose_positions' not in active_tracks[track_id]:
                                active_tracks[track_id]['nose_positions'] = []
                            active_tracks[track_id]['nose_positions'].append((nose_position[0], nose_position[1], frame_idx))

                    assigned_centroids.add(centroid_idx)

                # Start new tracks
                for i, (cx, cy) in enumerate(centroids):
                    if i not in assigned_centroids:
                        active_tracks[next_track_id] = {
                            'positions': [(cx, cy, frame_idx)],
                            'nose_positions': [],
                            'last_frame': frame_idx
                        }
                        next_track_id += 1

            except Exception as e:
                self.logger.warning(f"Error processing frame {frame_idx}: {e}")
                continue

        # Finalize tracks
        all_final_tracks = {**active_tracks, **inactive_tracks}
        tracks = {}
        nose_tracks = {}
        track_statistics = []

        for track_id, track_data in all_final_tracks.items():
            positions = track_data['positions']
            nose_positions = track_data.get('nose_positions', [])
            track_length = len(positions)

            track_stats = {
                'track_id': track_id,
                'track_length': track_length,
                'nose_detections': len(nose_positions),
                'nose_success_rate': len(nose_positions) / track_length if track_length > 0 else 0,
                'passed_length_filter': track_length >= self.config.min_track_length,
                'final_status': 'pending'
            }

            if track_length < self.config.min_track_length:
                track_stats['final_status'] = 'rejected_short'
            else:
                track_stats['final_status'] = 'accepted'
                tracks[track_id] = positions
                if nose_positions:
                    nose_tracks[track_id] = nose_positions

            track_statistics.append(track_stats)

        return tracks, nose_tracks, track_statistics

    def _export_tracks_csv(self, tracks: Dict, nose_tracks: Dict, csv_path: str):
        """Export tracks to CSV file"""
        all_frames = sorted({pos[2] for positions in tracks.values() for pos in positions})
        track_ids = sorted(tracks.keys())

        columns = ['frame']
        for tid in track_ids:
            columns.extend([f"worm_{tid}_x", f"worm_{tid}_y", f"worm_{tid}_nose_x", f"worm_{tid}_nose_y"])

        data = []
        for frame in all_frames:
            row = [frame]
            for track_id in track_ids:
                # Centroid position
                centroid_pos = next(((p[0], p[1]) for p in tracks[track_id] if p[2] == frame), None)
                if centroid_pos:
                    row.extend([round(centroid_pos[0], 4), round(centroid_pos[1], 4)])
                else:
                    row.extend([None, None])

                # Nose position
                nose_pos = None
                if track_id in nose_tracks:
                    nose_pos = next(((p[0], p[1]) for p in nose_tracks[track_id] if p[2] == frame), None)

                if nose_pos:
                    row.extend([round(nose_pos[0], 4), round(nose_pos[1], 4)])
                else:
                    row.extend([None, None])

            data.append(row)

        df = pd.DataFrame(data, columns=columns)

        # Overwrite existing file if it exists
        if os.path.exists(csv_path):
            self.logger.info(f"Overwriting existing tracks.csv at {csv_path}")

        df.to_csv(csv_path, index=False)
        self.logger.info(f"Exported tracks to {csv_path}")


class BatchWormTrackerGUI:
    """GUI for batch processing multiple directories"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Batch Worm Tracker - Multiple Directory Processing")
        self.root.geometry("1000x700")

        self.config = BatchConfig()
        self.tracker = BatchWormTracker(self.config)
        self.directories = []
        self.results = []
        self.processing_thread = None

        self.setup_gui()

    def setup_gui(self):
        """Setup the GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # Configuration section
        config_frame = ttk.LabelFrame(main_frame, text="Batch Configuration (Using Default Settings)", padding="10")
        config_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(config_frame, text=f"Threshold Range: {self.config.threshold_min}-{self.config.threshold_max}").pack(anchor='w')
        ttk.Label(config_frame, text=f"Blob Size: {self.config.min_blob_size}-{self.config.max_blob_size} pixels").pack(anchor='w')
        ttk.Label(config_frame, text=f"Max Distance: {self.config.max_distance} px, Trajectory Weight: {self.config.trajectory_weight}").pack(anchor='w')
        ttk.Label(config_frame, text=f"Min Track Length: {self.config.min_track_length} frames").pack(anchor='w')
        ttk.Label(config_frame, text=f"Nose Detection: {'Enabled' if self.config.nose_detection_enabled else 'Disabled'}").pack(anchor='w')
        ttk.Label(config_frame, text="Output: tracks.csv saved to each directory (overwrites existing)").pack(anchor='w')

        # Directory selection section
        dir_frame = ttk.LabelFrame(main_frame, text="Directory Selection", padding="10")
        dir_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        buttons_frame = ttk.Frame(dir_frame)
        buttons_frame.pack(fill='x', pady=(0, 10))

        ttk.Button(buttons_frame, text="Add Directory", command=self.add_directory).pack(side='left', padx=(0, 5))
        ttk.Button(buttons_frame, text="Remove Selected", command=self.remove_directory).pack(side='left', padx=(0, 5))
        ttk.Button(buttons_frame, text="Clear All", command=self.clear_directories).pack(side='left', padx=(0, 5))
        ttk.Button(buttons_frame, text="Start Batch Processing", command=self.start_batch_processing).pack(side='right')

        # Directory list
        list_frame = ttk.Frame(dir_frame)
        list_frame.pack(fill='both', expand=True)

        # Listbox with scrollbar
        list_scroll_frame = ttk.Frame(list_frame)
        list_scroll_frame.pack(fill='both', expand=True)

        self.dir_listbox = tk.Listbox(list_scroll_frame, height=8)
        dir_scrollbar = ttk.Scrollbar(list_scroll_frame, orient='vertical', command=self.dir_listbox.yview)
        self.dir_listbox.configure(yscrollcommand=dir_scrollbar.set)

        self.dir_listbox.pack(side='left', fill='both', expand=True)
        dir_scrollbar.pack(side='right', fill='y')

        # Progress section
        progress_frame = ttk.LabelFrame(main_frame, text="Processing Progress", padding="10")
        progress_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        # Progress bars
        ttk.Label(progress_frame, text="Overall Progress:").pack(anchor='w')
        self.overall_progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.overall_progress.pack(fill='x', pady=(2, 10))

        ttk.Label(progress_frame, text="Current Directory:").pack(anchor='w')
        self.current_progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.current_progress.pack(fill='x', pady=(2, 10))

        # Status labels
        self.status_label = ttk.Label(progress_frame, text="Ready to process directories")
        self.status_label.pack(anchor='w')

        self.current_dir_label = ttk.Label(progress_frame, text="")
        self.current_dir_label.pack(anchor='w')

        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Processing Results", padding="10")
        results_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        main_frame.rowconfigure(3, weight=2)  # Give more space to results

        # Results text area
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, wrap='word')
        self.results_text.pack(fill='both', expand=True)

        # Bottom buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

        ttk.Button(button_frame, text="Export Summary Report", command=self.export_summary).pack(side='left')
        ttk.Button(button_frame, text="Clear Results", command=self.clear_results).pack(side='left', padx=(10, 0))
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side='right')

    def add_directory(self):
        """Add a directory to the processing list"""
        directory = filedialog.askdirectory(title="Select directory containing images")
        if directory and directory not in self.directories:
            self.directories.append(directory)
            self.dir_listbox.insert(tk.END, directory)
            self.update_status(f"Added directory: {os.path.basename(directory)}")

    def remove_directory(self):
        """Remove selected directory from the list"""
        selection = self.dir_listbox.curselection()
        if selection:
            index = selection[0]
            removed_dir = self.directories.pop(index)
            self.dir_listbox.delete(index)
            self.update_status(f"Removed directory: {os.path.basename(removed_dir)}")

    def clear_directories(self):
        """Clear all directories from the list"""
        self.directories.clear()
        self.dir_listbox.delete(0, tk.END)
        self.update_status("Cleared all directories")

    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def update_current_dir_status(self, message):
        """Update current directory label"""
        self.current_dir_label.config(text=message)
        self.root.update_idletasks()

    def start_batch_processing(self):
        """Start batch processing in a separate thread"""
        if not self.directories:
            messagebox.showwarning("No Directories", "Please add at least one directory to process")
            return

        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("Processing Active", "Batch processing is already running")
            return

        # Confirm processing
        num_dirs = len(self.directories)
        confirm_msg = f"Start processing {num_dirs} directories?\n\n"
        confirm_msg += "This will:\n"
        confirm_msg += "• Generate background images for each directory\n"
        confirm_msg += "• Run trajectory-aware tracking with nose detection\n"
        confirm_msg += "• Save tracks.csv to each directory (overwriting existing files)\n"
        confirm_msg += "• Flag directories with quality issues\n\n"
        confirm_msg += "This process may take several minutes per directory."

        if not messagebox.askyesno("Confirm Batch Processing", confirm_msg):
            return

        # Clear previous results
        self.results.clear()
        self.results_text.delete(1.0, tk.END)

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self.processing_thread.start()

    def _batch_worker(self):
        """Worker thread for batch processing"""
        try:
            total_dirs = len(self.directories)

            self.root.after(0, lambda: self.overall_progress.config(maximum=total_dirs))
            self.root.after(0, lambda: self.overall_progress.config(value=0))

            for i, directory in enumerate(self.directories):
                self.root.after(0, lambda d=directory: self.update_current_dir_status(f"Processing: {os.path.basename(d)}"))

                # Progress callback for individual directory
                def progress_callback(current_frame, total_frames):
                    progress_value = (current_frame / total_frames) * 100
                    self.root.after(0, lambda pv=progress_value: self.current_progress.config(value=pv))

                self.root.after(0, lambda: self.current_progress.config(maximum=100, value=0))

                # Process the directory
                result = self.tracker.process_directory(directory, progress_callback)
                self.results.append(result)

                # Update overall progress
                self.root.after(0, lambda prog=i+1: self.overall_progress.config(value=prog))

                # Update results display
                self.root.after(0, lambda r=result: self._update_results_display(r))

            # Processing complete
            self.root.after(0, self._processing_complete)

        except Exception as e:
            error_msg = f"Batch processing error: {e}"
            self.root.after(0, lambda: messagebox.showerror("Processing Error", error_msg))
            self.root.after(0, lambda: self.update_status("Batch processing failed"))

    def _update_results_display(self, result: ProcessingResult):
        """Update results display with new result"""
        self.results_text.insert(tk.END, self._format_result(result))
        self.results_text.see(tk.END)

    def _format_result(self, result: ProcessingResult) -> str:
        """Format a single result for display"""
        status_symbol = "SUCCESS" if result.success else "FAILED"
        quality_symbols = {"normal": "NORMAL", "noisy": "NOISY", "empty": "EMPTY"}
        quality_symbol = quality_symbols[result.quality_flag]

        text = f"\n[{status_symbol}] {os.path.basename(result.directory)} [{quality_symbol}]\n"
        text += f"   Images: {result.num_images}, Tracks: {result.num_accepted_tracks}, Time: {result.processing_time:.1f}s\n"

        if result.quality_flag == "noisy":
            text += f"   WARNING: HIGH TRACK COUNT - Manual QC recommended\n"
        elif result.quality_flag == "empty":
            text += f"   WARNING: NO TRACKS FOUND - Check parameters\n"

        if not result.success:
            text += f"   Error: {result.error_message}\n"

        return text

    def _processing_complete(self):
        """Handle completion of batch processing"""
        self.update_status("Batch processing complete!")
        self.update_current_dir_status("")
        self.current_progress.config(value=0)

        # Generate summary
        summary = self._generate_summary()
        self.results_text.insert(tk.END, f"\n{'='*50}\n")
        self.results_text.insert(tk.END, "BATCH PROCESSING SUMMARY\n")
        self.results_text.insert(tk.END, f"{'='*50}\n")
        self.results_text.insert(tk.END, summary)
        self.results_text.see(tk.END)

        # Show completion dialog
        messagebox.showinfo("Batch Complete", "Batch processing finished!\n\nCheck the results below for quality flags and any errors.")

    def _generate_summary(self) -> str:
        """Generate summary of batch processing results"""
        if not self.results:
            return "No results to summarize.\n"

        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        normal_quality = [r for r in successful if r.quality_flag == "normal"]
        noisy_quality = [r for r in successful if r.quality_flag == "noisy"]
        empty_quality = [r for r in successful if r.quality_flag == "empty"]

        summary = f"Total directories processed: {len(self.results)}\n"
        summary += f"Successful: {len(successful)}\n"
        summary += f"Failed: {len(failed)}\n\n"

        if successful:
            total_tracks = sum(r.num_accepted_tracks for r in successful)
            total_time = sum(r.processing_time for r in successful)
            avg_time = total_time / len(successful)

            summary += f"Normal quality: {len(normal_quality)}\n"
            summary += f"Noisy (needs QC): {len(noisy_quality)}\n"
            summary += f"Empty (no tracks): {len(empty_quality)}\n\n"

            summary += f"Total tracks generated: {total_tracks}\n"
            summary += f"Average processing time: {avg_time:.1f}s per directory\n"
            summary += f"Total processing time: {total_time:.1f}s\n\n"

        if noisy_quality:
            summary += "DIRECTORIES NEEDING MANUAL QC (High track count):\n"
            for result in noisy_quality:
                summary += f"   • {os.path.basename(result.directory)} ({result.num_accepted_tracks} tracks)\n"
            summary += "\n"

        if empty_quality:
            summary += "DIRECTORIES WITH NO TRACKS FOUND:\n"
            for result in empty_quality:
                summary += f"   • {os.path.basename(result.directory)}\n"
            summary += "\n"

        if failed:
            summary += "FAILED DIRECTORIES:\n"
            for result in failed:
                summary += f"   • {os.path.basename(result.directory)}: {result.error_message}\n"
            summary += "\n"

        summary += "All tracks.csv files have been saved to their respective directories.\n"

        return summary

    def export_summary(self):
        """Export summary report to file"""
        if not self.results:
            messagebox.showwarning("No Results", "No results to export")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save batch processing summary",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            with open(file_path, 'w') as f:
                f.write("BATCH WORM TRACKER PROCESSING REPORT\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Configuration
                f.write("CONFIGURATION USED:\n")
                f.write(f"Threshold Range: {self.config.threshold_min}-{self.config.threshold_max}\n")
                f.write(f"Blob Size: {self.config.min_blob_size}-{self.config.max_blob_size} pixels\n")
                f.write(f"Max Distance: {self.config.max_distance} px\n")
                f.write(f"Trajectory Weight: {self.config.trajectory_weight}\n")
                f.write(f"Min Track Length: {self.config.min_track_length} frames\n")
                f.write(f"Nose Detection: {'Enabled' if self.config.nose_detection_enabled else 'Disabled'}\n")
                f.write(f"Algorithm: {'Hungarian' if self.config.use_hungarian else 'Greedy'}\n\n")

                # Detailed results
                f.write("DETAILED RESULTS:\n")
                f.write("-" * 30 + "\n")
                for result in self.results:
                    f.write(f"Directory: {result.directory}\n")
                    f.write(f"Success: {result.success}\n")
                    f.write(f"Images: {result.num_images}\n")
                    f.write(f"Total Tracks: {result.num_tracks}\n")
                    f.write(f"Accepted Tracks: {result.num_accepted_tracks}\n")
                    f.write(f"Quality Flag: {result.quality_flag}\n")
                    f.write(f"Processing Time: {result.processing_time:.1f}s\n")
                    if result.error_message:
                        f.write(f"Error: {result.error_message}\n")
                    f.write("-" * 30 + "\n")

                # Summary
                f.write("\n" + self._generate_summary())

            messagebox.showinfo("Export Complete", f"Summary report saved to:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save summary: {e}")

    def clear_results(self):
        """Clear the results display"""
        self.results.clear()
        self.results_text.delete(1.0, tk.END)
        self.overall_progress.config(value=0)
        self.current_progress.config(value=0)
        self.update_status("Results cleared")
        self.update_current_dir_status("")

    def run(self):
        """Run the GUI application"""
        self.root.mainloop()


def main():
    """Main function to run the batch tracker"""
    print("=" * 60)
    print("BATCH WORM TRACKER")
    print("Automated processing of multiple image directories")
    print("=" * 60)
    print("\nFEATURES:")
    print("• Uses same settings as main tracker (trajectory-aware + nose detection)")
    print("• Processes multiple directories automatically")
    print("• Saves tracks.csv to each directory (overwrites existing)")
    print("• Quality assessment (flags noisy/empty results)")
    print("• Comprehensive summary report")
    print("• Network drive compatible")
    print("\nQUALITY FLAGS:")
    print("NORMAL: Standard number of tracks detected")
    print("NOISY: High track count - manual QC recommended")
    print("EMPTY: No tracks found - check parameters")
    print("\nSTARTING BATCH TRACKER GUI...")

    try:
        app = BatchWormTrackerGUI()
        app.run()
    except Exception as e:
        print(f"Error starting batch tracker: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
