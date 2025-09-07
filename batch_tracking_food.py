#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Batch Worm Tracker - Automated processing with adjustable parameters
Extracts core functionality from the main WormTracker for automated processing
Now includes GUI controls for all tracking parameters and stationary track filtering
WITH SMART MULTIPLE DIRECTORY SELECTION SUPPORT
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

    def calculate_track_displacement(self, track_positions: List[Tuple[float, float, int]]) -> float:
        """Calculate total displacement from first to last position in a track"""
        if len(track_positions) < 2:
            return 0.0

        first_pos = track_positions[0]
        last_pos = track_positions[-1]

        displacement = np.sqrt((last_pos[0] - first_pos[0])**2 + (last_pos[1] - first_pos[1])**2)
        return displacement

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
        """Run the tracking algorithm with proper track ID management"""
        # FIXED: Reset everything for each directory (like main tracker does for each session)
        next_track_id = 1
        active_tracks = {}
        inactive_tracks = {}
        used_track_ids = set()  # NEW: Track all IDs that have ever been used
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

                # FIXED: Deactivate old tracks (same as main tracker)
                tracks_to_deactivate = []
                for track_id, track_data in active_tracks.items():
                    frames_missing = frame_idx - track_data['last_frame']
                    if frames_missing > MAX_MISSING_FRAMES:
                        tracks_to_deactivate.append(track_id)

                for track_id in tracks_to_deactivate:
                    inactive_tracks[track_id] = active_tracks[track_id]
                    del active_tracks[track_id]
                    # CRITICAL: Do NOT remove from used_track_ids - keep it reserved forever

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

                # FIXED: Start new tracks with guaranteed unique IDs
                for i, (cx, cy) in enumerate(centroids):
                    if i not in assigned_centroids:
                        # CRITICAL: Ensure this ID has never been used before
                        while next_track_id in used_track_ids:
                            next_track_id += 1

                        # Mark this ID as used
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

        # FIXED: Same finalization as main tracker - no track ID conflicts possible
        all_final_tracks = {**active_tracks, **inactive_tracks}
        tracks = {}
        nose_tracks = {}
        track_statistics = []

        # Sort tracks by their original creation order (lowest ID first)
        sorted_track_ids = sorted(all_final_tracks.keys())

        # Renumber tracks starting from 1 for accepted tracks only
        new_track_id = 1

        for original_track_id in sorted_track_ids:
            track_data = all_final_tracks[original_track_id]
            positions = track_data['positions']
            nose_positions = track_data.get('nose_positions', [])
            track_length = len(positions)

            # Calculate displacement distance
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

            # Apply filters
            if track_length < self.config.min_track_length:
                track_stats['final_status'] = 'rejected_short'
                track_stats['final_track_id'] = None
            elif self.config.filter_stationary_tracks and displacement < self.config.min_displacement_distance:
                track_stats['final_status'] = 'rejected_stationary'
                track_stats['final_track_id'] = None
            else:
                track_stats['final_status'] = 'accepted'
                track_stats['final_track_id'] = new_track_id

                # Use the new sequential track ID
                tracks[new_track_id] = positions
                if nose_positions:
                    nose_tracks[new_track_id] = nose_positions

                new_track_id += 1

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


class SmartDirectoryDialog:
    """Smart dialog for selecting single or multiple directories"""

    def __init__(self, parent):
        self.parent = parent
        self.selected_directories = []
        self.result = None

    def show(self):
        """Show the smart directory selection dialog"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Add Directories")
        self.dialog.geometry("800x600")
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (800 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (600 // 2)
        self.dialog.geometry(f"800x600+{x}+{y}")

        self.create_widgets()

        # Wait for dialog to close
        self.dialog.wait_window()
        return self.result

    def create_widgets(self):
        """Create the dialog widgets"""
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill='both', expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Add Directories for Batch Processing",
                               font=('TkDefaultFont', 12, 'bold'))
        title_label.pack(pady=(0, 10))

        # Instructions
        instructions = ("• Click 'Browse & Add Directory' to add individual directories\n"
                       "• Click 'Add All Subdirectories' to add all subdirs from a parent folder\n"
                       "• Use the list below to manage your selection\n"
                       "• Click OK when finished")

        instruction_label = ttk.Label(main_frame, text=instructions, justify='left')
        instruction_label.pack(pady=(0, 10), anchor='w')

        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(0, 10))

        ttk.Button(button_frame, text="Browse & Add Directory",
                  command=self.add_single_directory).pack(side='left', padx=(0, 5))
        ttk.Button(button_frame, text="Add All Subdirectories",
                  command=self.add_subdirectories).pack(side='left', padx=(0, 5))
        ttk.Button(button_frame, text="Remove Selected",
                  command=self.remove_selected).pack(side='left', padx=(0, 5))
        ttk.Button(button_frame, text="Clear All",
                  command=self.clear_all).pack(side='left', padx=(0, 5))

        # Directory list with scrollbar
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill='both', expand=True, pady=(0, 10))

        self.dir_listbox = tk.Listbox(list_frame, selectmode='extended', height=15)
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.dir_listbox.yview)
        self.dir_listbox.configure(yscrollcommand=scrollbar.set)

        self.dir_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Status label
        self.status_label = ttk.Label(main_frame, text="No directories selected")
        self.status_label.pack(pady=(0, 10))

        # Bottom buttons
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill='x')

        ttk.Button(bottom_frame, text="Cancel", command=self.cancel).pack(side='right', padx=(5, 0))
        ttk.Button(bottom_frame, text="OK", command=self.ok).pack(side='right')

    def add_single_directory(self):
        """Add a single directory"""
        directory = filedialog.askdirectory(
            parent=self.dialog,
            title="Select directory containing images"
        )

        if directory and directory not in self.selected_directories:
            self.selected_directories.append(directory)
            self.dir_listbox.insert(tk.END, directory)
            self.update_status()

    def add_subdirectories(self):
        """Add all subdirectories from a parent directory"""
        parent_dir = filedialog.askdirectory(
            parent=self.dialog,
            title="Select parent directory (all subdirectories will be added)"
        )

        if not parent_dir:
            return

        try:
            subdirs = []
            for item in os.listdir(parent_dir):
                item_path = os.path.join(parent_dir, item)
                if os.path.isdir(item_path):
                    subdirs.append(item_path)

            if not subdirs:
                messagebox.showinfo("No Subdirectories",
                                   "No subdirectories found in the selected parent directory.",
                                   parent=self.dialog)
                return

            # Ask for confirmation
            message = f"Found {len(subdirs)} subdirectories. Add all of them?"
            if messagebox.askyesno("Confirm Add Subdirectories", message, parent=self.dialog):
                added_count = 0
                for subdir in subdirs:
                    if subdir not in self.selected_directories:
                        self.selected_directories.append(subdir)
                        self.dir_listbox.insert(tk.END, subdir)
                        added_count += 1

                messagebox.showinfo("Subdirectories Added",
                                   f"Added {added_count} new directories (skipped {len(subdirs) - added_count} duplicates).",
                                   parent=self.dialog)
                self.update_status()

        except Exception as e:
            messagebox.showerror("Error", f"Error reading subdirectories: {e}", parent=self.dialog)

    def remove_selected(self):
        """Remove selected directories from list"""
        selection = self.dir_listbox.curselection()
        if not selection:
            return

        # Remove in reverse order to maintain indices
        for index in reversed(selection):
            self.selected_directories.pop(index)
            self.dir_listbox.delete(index)

        self.update_status()

    def clear_all(self):
        """Clear all directories"""
        if self.selected_directories:
            if messagebox.askyesno("Clear All", "Remove all directories from the list?", parent=self.dialog):
                self.selected_directories.clear()
                self.dir_listbox.delete(0, tk.END)
                self.update_status()

    def update_status(self):
        """Update status label"""
        count = len(self.selected_directories)
        if count == 0:
            self.status_label.config(text="No directories selected")
        elif count == 1:
            self.status_label.config(text="1 directory selected")
        else:
            self.status_label.config(text=f"{count} directories selected")

    def ok(self):
        """OK button pressed"""
        self.result = self.selected_directories.copy()
        self.dialog.destroy()

    def cancel(self):
        """Cancel button pressed"""
        self.result = None
        self.dialog.destroy()


class BatchWormTrackerGUI:
    """Enhanced GUI for batch processing with adjustable parameters"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Enhanced Batch Worm Tracker - Multiple Directory Processing with Adjustable Parameters")
        self.root.geometry("1200x800")

        self.config = BatchConfig()
        self.tracker = BatchWormTracker(self.config)
        self.directories = []
        self.results = []
        self.processing_thread = None

        self.setup_gui()

    def setup_gui(self):
        """Setup the enhanced GUI interface"""
        # Main frame with notebook for tabbed interface
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Tab 1: Parameter Configuration
        self.config_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.config_tab, text="1. Configuration")
        self.create_config_tab()

        # Tab 2: Batch Processing
        self.batch_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.batch_tab, text="2. Batch Processing")
        self.create_batch_tab()

        # Tab 3: Results
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="3. Results")
        self.create_results_tab()

    def create_config_tab(self):
        """Create the parameter configuration tab"""
        # Main scrollable frame
        canvas = tk.Canvas(self.config_tab)
        scrollbar = ttk.Scrollbar(self.config_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mousewheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)

        # Configuration sections
        self.create_threshold_config(scrollable_frame)
        self.create_blob_config(scrollable_frame)
        self.create_tracking_config(scrollable_frame)
        self.create_nose_config(scrollable_frame)
        self.create_action_buttons(scrollable_frame)

    def create_threshold_config(self, parent):
        """Create threshold configuration section"""
        threshold_frame = ttk.LabelFrame(parent, text="Threshold Parameters", padding="10")
        threshold_frame.pack(fill='x', padx=5, pady=5)

        # Min threshold
        min_frame = ttk.Frame(threshold_frame)
        min_frame.pack(fill='x', pady=2)
        ttk.Label(min_frame, text="Minimum Threshold:").pack(side='left', anchor='w')
        self.min_thresh_var = tk.StringVar(value=str(self.config.threshold_min))
        min_thresh_entry = ttk.Entry(min_frame, textvariable=self.min_thresh_var, width=10)
        min_thresh_entry.pack(side='right')
        min_thresh_entry.bind('<KeyRelease>', self.validate_threshold_params)

        # Max threshold
        max_frame = ttk.Frame(threshold_frame)
        max_frame.pack(fill='x', pady=2)
        ttk.Label(max_frame, text="Maximum Threshold:").pack(side='left', anchor='w')
        self.max_thresh_var = tk.StringVar(value=str(self.config.threshold_max))
        max_thresh_entry = ttk.Entry(max_frame, textvariable=self.max_thresh_var, width=10)
        max_thresh_entry.pack(side='right')
        max_thresh_entry.bind('<KeyRelease>', self.validate_threshold_params)

        # Info label
        ttk.Label(threshold_frame, text="Range: 0-255. Higher values detect brighter objects.",
                 font=('TkDefaultFont', 8), foreground='gray').pack(anchor='w', pady=(5,0))

    def create_blob_config(self, parent):
        """Create blob size configuration section"""
        blob_frame = ttk.LabelFrame(parent, text="Blob Size Filters", padding="10")
        blob_frame.pack(fill='x', padx=5, pady=5)

        # Min blob size
        min_blob_frame = ttk.Frame(blob_frame)
        min_blob_frame.pack(fill='x', pady=2)
        ttk.Label(min_blob_frame, text="Minimum Blob Size (pixels):").pack(side='left', anchor='w')
        self.min_blob_var = tk.StringVar(value=str(self.config.min_blob_size))
        min_blob_entry = ttk.Entry(min_blob_frame, textvariable=self.min_blob_var, width=10)
        min_blob_entry.pack(side='right')
        min_blob_entry.bind('<KeyRelease>', self.validate_blob_params)

        # Max blob size
        max_blob_frame = ttk.Frame(blob_frame)
        max_blob_frame.pack(fill='x', pady=2)
        ttk.Label(max_blob_frame, text="Maximum Blob Size (pixels):").pack(side='left', anchor='w')
        self.max_blob_var = tk.StringVar(value=str(self.config.max_blob_size))
        max_blob_entry = ttk.Entry(max_blob_frame, textvariable=self.max_blob_var, width=10)
        max_blob_entry.pack(side='right')
        max_blob_entry.bind('<KeyRelease>', self.validate_blob_params)

        # Info label
        ttk.Label(blob_frame, text="Filters detected objects by area. Adjust for worm size.",
                 font=('TkDefaultFont', 8), foreground='gray').pack(anchor='w', pady=(5,0))

    def create_tracking_config(self, parent):
        """Create tracking configuration section"""
        tracking_frame = ttk.LabelFrame(parent, text="Trajectory-Aware Tracking Parameters", padding="10")
        tracking_frame.pack(fill='x', padx=5, pady=5)

        # Max distance
        max_dist_frame = ttk.Frame(tracking_frame)
        max_dist_frame.pack(fill='x', pady=2)
        ttk.Label(max_dist_frame, text="Max Distance Between Frames (pixels):").pack(side='left', anchor='w')
        self.max_dist_var = tk.StringVar(value=str(self.config.max_distance))
        max_dist_entry = ttk.Entry(max_dist_frame, textvariable=self.max_dist_var, width=10)
        max_dist_entry.pack(side='right')
        max_dist_entry.bind('<KeyRelease>', self.validate_tracking_params)

        # Trajectory weight
        traj_weight_frame = ttk.Frame(tracking_frame)
        traj_weight_frame.pack(fill='x', pady=2)
        ttk.Label(traj_weight_frame, text="Trajectory Weight (0.0-1.0):").pack(side='left', anchor='w')
        self.traj_weight_var = tk.StringVar(value=str(self.config.trajectory_weight))
        traj_weight_entry = ttk.Entry(traj_weight_frame, textvariable=self.traj_weight_var, width=10)
        traj_weight_entry.pack(side='right')
        traj_weight_entry.bind('<KeyRelease>', self.validate_tracking_params)

        # Min track length
        min_track_frame = ttk.Frame(tracking_frame)
        min_track_frame.pack(fill='x', pady=2)
        ttk.Label(min_track_frame, text="Minimum Track Length (frames):").pack(side='left', anchor='w')
        self.min_track_var = tk.StringVar(value=str(self.config.min_track_length))
        min_track_entry = ttk.Entry(min_track_frame, textvariable=self.min_track_var, width=10)
        min_track_entry.pack(side='right')
        min_track_entry.bind('<KeyRelease>', self.validate_tracking_params)

        # Algorithm selection
        algo_frame = ttk.Frame(tracking_frame)
        algo_frame.pack(fill='x', pady=2)
        ttk.Label(algo_frame, text="Assignment Algorithm:").pack(side='left', anchor='w')
        self.algorithm_var = tk.StringVar(value="Greedy" if not self.config.use_hungarian else "Hungarian")
        algo_combo = ttk.Combobox(algo_frame, textvariable=self.algorithm_var,
                                  values=["Greedy", "Hungarian"], width=12, state="readonly")
        algo_combo.pack(side='right')
        algo_combo.bind('<<ComboboxSelected>>', self.update_algorithm)

        # Info labels
        ttk.Label(tracking_frame, text="Trajectory weight: 0.7 recommended for ID swap prevention",
                 font=('TkDefaultFont', 8), foreground='gray').pack(anchor='w', pady=(5,0))
        ttk.Label(tracking_frame, text="Greedy algorithm recommended for worm tracking",
                 font=('TkDefaultFont', 8), foreground='gray').pack(anchor='w')

        # Movement filter section
        movement_filter_frame = ttk.Frame(tracking_frame)
        movement_filter_frame.pack(fill='x', pady=(10,2))
        self.filter_stationary_var = tk.BooleanVar(value=self.config.filter_stationary_tracks)
        movement_check = ttk.Checkbutton(movement_filter_frame, text="Filter Stationary Tracks",
                                        variable=self.filter_stationary_var,
                                        command=self.update_movement_filter)
        movement_check.pack(side='left', anchor='w')

        # Min displacement distance
        displacement_frame = ttk.Frame(tracking_frame)
        displacement_frame.pack(fill='x', pady=2)
        ttk.Label(displacement_frame, text="Min Displacement Distance (pixels):").pack(side='left', anchor='w')
        self.min_displacement_var = tk.StringVar(value=str(self.config.min_displacement_distance))
        displacement_entry = ttk.Entry(displacement_frame, textvariable=self.min_displacement_var, width=10)
        displacement_entry.pack(side='right')
        displacement_entry.bind('<KeyRelease>', self.validate_tracking_params)

        # Movement filter info
        ttk.Label(tracking_frame, text="Removes tracks that don't move significantly from start to end position",
                 font=('TkDefaultFont', 8), foreground='gray').pack(anchor='w')

    def create_nose_config(self, parent):
        """Create nose detection configuration section"""
        nose_frame = ttk.LabelFrame(parent, text="Nose Detection Parameters", padding="10")
        nose_frame.pack(fill='x', padx=5, pady=5)

        # Enable/disable nose detection
        nose_enable_frame = ttk.Frame(nose_frame)
        nose_enable_frame.pack(fill='x', pady=2)
        self.nose_enabled_var = tk.BooleanVar(value=self.config.nose_detection_enabled)
        nose_check = ttk.Checkbutton(nose_enable_frame, text="Enable Nose Detection",
                                     variable=self.nose_enabled_var,
                                     command=self.update_nose_detection)
        nose_check.pack(side='left', anchor='w')

        # Smoothing frames
        smooth_frame = ttk.Frame(nose_frame)
        smooth_frame.pack(fill='x', pady=2)
        ttk.Label(smooth_frame, text="Smoothing Frames (2-10):").pack(side='left', anchor='w')
        self.nose_smooth_var = tk.StringVar(value=str(self.config.nose_smoothing_frames))
        smooth_entry = ttk.Entry(smooth_frame, textvariable=self.nose_smooth_var, width=10)
        smooth_entry.pack(side='right')
        smooth_entry.bind('<KeyRelease>', self.validate_nose_params)

        # Min movement threshold
        movement_frame = ttk.Frame(nose_frame)
        movement_frame.pack(fill='x', pady=2)
        ttk.Label(movement_frame, text="Min Movement Threshold (pixels/frame):").pack(side='left', anchor='w')
        self.nose_movement_var = tk.StringVar(value=str(self.config.min_movement_threshold))
        movement_entry = ttk.Entry(movement_frame, textvariable=self.nose_movement_var, width=10)
        movement_entry.pack(side='right')
        movement_entry.bind('<KeyRelease>', self.validate_nose_params)

        # Info label
        ttk.Label(nose_frame, text="Detects worm front based on locomotion direction",
                 font=('TkDefaultFont', 8), foreground='gray').pack(anchor='w', pady=(5,0))

    def create_action_buttons(self, parent):
        """Create action buttons for configuration"""
        action_frame = ttk.Frame(parent, padding="10")
        action_frame.pack(fill='x', padx=5, pady=10)

        ttk.Button(action_frame, text="Reset to Defaults", command=self.reset_to_defaults).pack(side='left', padx=5)
        ttk.Button(action_frame, text="Validate All Parameters", command=self.validate_all_parameters).pack(side='left', padx=5)
        ttk.Button(action_frame, text="Save Configuration", command=self.save_configuration).pack(side='left', padx=5)
        ttk.Button(action_frame, text="Load Configuration", command=self.load_configuration).pack(side='left', padx=5)

        # Status label
        self.config_status_label = ttk.Label(action_frame, text="Parameters valid", foreground='green')
        self.config_status_label.pack(side='right', padx=10)

    def create_batch_tab(self):
        """Create the batch processing tab"""
        batch_frame = ttk.Frame(self.batch_tab, padding="10")
        batch_frame.pack(fill='both', expand=True)

        batch_frame.columnconfigure(1, weight=1)
        batch_frame.rowconfigure(2, weight=1)

        # Configuration summary
        summary_frame = ttk.LabelFrame(batch_frame, text="Current Configuration Summary", padding="10")
        summary_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        self.config_summary_text = tk.Text(summary_frame, height=6, wrap='word', font=('TkDefaultFont', 8))
        self.config_summary_text.pack(fill='x')
        self.update_config_summary()

        # Directory selection section
        dir_frame = ttk.LabelFrame(batch_frame, text="Directory Selection", padding="10")
        dir_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        buttons_frame = ttk.Frame(dir_frame)
        buttons_frame.pack(fill='x', pady=(0, 10))

        # SIMPLIFIED: Single smart "Add Directories" button
        ttk.Button(buttons_frame, text="Add Directories", command=self.add_directories).pack(side='left', padx=(0, 5))
        ttk.Button(buttons_frame, text="Remove Selected", command=self.remove_directory).pack(side='left', padx=(0, 5))
        ttk.Button(buttons_frame, text="Clear All", command=self.clear_directories).pack(side='left', padx=(0, 5))
        ttk.Button(buttons_frame, text="Start Batch Processing", command=self.start_batch_processing).pack(side='right')

        # Directory list
        list_frame = ttk.Frame(dir_frame)
        list_frame.pack(fill='both', expand=True)

        # Listbox with scrollbar - Multiple selection mode
        list_scroll_frame = ttk.Frame(list_frame)
        list_scroll_frame.pack(fill='both', expand=True)

        self.dir_listbox = tk.Listbox(list_scroll_frame, height=8, selectmode='extended')
        dir_scrollbar = ttk.Scrollbar(list_scroll_frame, orient='vertical', command=self.dir_listbox.yview)
        self.dir_listbox.configure(yscrollcommand=dir_scrollbar.set)

        self.dir_listbox.pack(side='left', fill='both', expand=True)
        dir_scrollbar.pack(side='right', fill='y')

        # Progress section
        progress_frame = ttk.LabelFrame(batch_frame, text="Processing Progress", padding="10")
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

    def create_results_tab(self):
        """Create the results tab"""
        results_frame = ttk.Frame(self.results_tab, padding="10")
        results_frame.pack(fill='both', expand=True)

        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)

        # Results summary
        summary_frame = ttk.LabelFrame(results_frame, text="Processing Summary", padding="10")
        summary_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.results_summary_label = ttk.Label(summary_frame, text="No processing results yet")
        self.results_summary_label.pack(anchor='w')

        # Results text area
        results_text_frame = ttk.LabelFrame(results_frame, text="Detailed Results", padding="10")
        results_text_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.results_text = scrolledtext.ScrolledText(results_text_frame, height=20, wrap='word')
        self.results_text.pack(fill='both', expand=True)

        # Bottom buttons
        button_frame = ttk.Frame(results_frame)
        button_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        ttk.Button(button_frame, text="Export Summary Report", command=self.export_summary).pack(side='left')
        ttk.Button(button_frame, text="Clear Results", command=self.clear_results).pack(side='left', padx=(10, 0))
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side='right')

    # Parameter validation methods (same as before)
    def validate_threshold_params(self, event=None):
        """Validate threshold parameters"""
        try:
            min_val = int(self.min_thresh_var.get())
            max_val = int(self.max_thresh_var.get())

            if not (0 <= min_val <= 255) or not (0 <= max_val <= 255) or min_val >= max_val:
                self.config_status_label.config(text="Invalid threshold range", foreground='red')
                return False

            self.config.threshold_min = min_val
            self.config.threshold_max = max_val
            self.update_config_summary()
            return True
        except ValueError:
            self.config_status_label.config(text="Invalid threshold values", foreground='red')
            return False

    def validate_blob_params(self, event=None):
        """Validate blob size parameters"""
        try:
            min_val = int(self.min_blob_var.get())
            max_val = int(self.max_blob_var.get())

            if min_val < 1 or max_val < min_val:
                self.config_status_label.config(text="Invalid blob size range", foreground='red')
                return False

            self.config.min_blob_size = min_val
            self.config.max_blob_size = max_val
            self.update_config_summary()
            return True
        except ValueError:
            self.config_status_label.config(text="Invalid blob size values", foreground='red')
            return False

    def validate_tracking_params(self, event=None):
        """Validate tracking parameters"""
        try:
            max_dist = int(self.max_dist_var.get())
            traj_weight = float(self.traj_weight_var.get())
            min_track = int(self.min_track_var.get())
            min_displacement = float(self.min_displacement_var.get())

            if max_dist < 1 or not (0.0 <= traj_weight <= 1.0) or min_track < 1 or min_displacement < 0:
                self.config_status_label.config(text="Invalid tracking parameters", foreground='red')
                return False

            self.config.max_distance = max_dist
            self.config.trajectory_weight = traj_weight
            self.config.min_track_length = min_track
            self.config.min_displacement_distance = min_displacement
            self.update_config_summary()
            return True
        except ValueError:
            self.config_status_label.config(text="Invalid tracking values", foreground='red')
            return False

    def validate_nose_params(self, event=None):
        """Validate nose detection parameters"""
        try:
            smooth_frames = int(self.nose_smooth_var.get())
            min_movement = float(self.nose_movement_var.get())

            if not (2 <= smooth_frames <= 10) or min_movement < 0.1:
                self.config_status_label.config(text="Invalid nose parameters", foreground='red')
                return False

            self.config.nose_smoothing_frames = smooth_frames
            self.config.min_movement_threshold = min_movement
            self.update_config_summary()
            return True
        except ValueError:
            self.config_status_label.config(text="Invalid nose values", foreground='red')
            return False

    def update_algorithm(self, event=None):
        """Update algorithm selection"""
        self.config.use_hungarian = (self.algorithm_var.get() == "Hungarian")
        self.update_config_summary()

    def update_nose_detection(self):
        """Update nose detection enabled state"""
        self.config.nose_detection_enabled = self.nose_enabled_var.get()
        self.update_config_summary()

    def update_movement_filter(self):
        """Update movement filter enabled state"""
        self.config.filter_stationary_tracks = self.filter_stationary_var.get()
        self.update_config_summary()

    def validate_all_parameters(self):
        """Validate all parameters"""
        valid = (self.validate_threshold_params() and
                self.validate_blob_params() and
                self.validate_tracking_params() and
                self.validate_nose_params())

        if valid:
            self.config_status_label.config(text="All parameters valid", foreground='green')
            # Update tracker with new config
            self.tracker = BatchWormTracker(self.config)

        return valid

    def reset_to_defaults(self):
        """Reset all parameters to defaults"""
        self.config = BatchConfig()

        # Update GUI elements
        self.min_thresh_var.set(str(self.config.threshold_min))
        self.max_thresh_var.set(str(self.config.threshold_max))
        self.min_blob_var.set(str(self.config.min_blob_size))
        self.max_blob_var.set(str(self.config.max_blob_size))
        self.max_dist_var.set(str(self.config.max_distance))
        self.traj_weight_var.set(str(self.config.trajectory_weight))
        self.min_track_var.set(str(self.config.min_track_length))
        self.algorithm_var.set("Greedy" if not self.config.use_hungarian else "Hungarian")
        self.nose_enabled_var.set(self.config.nose_detection_enabled)
        self.nose_smooth_var.set(str(self.config.nose_smoothing_frames))
        self.nose_movement_var.set(str(self.config.min_movement_threshold))
        self.filter_stationary_var.set(self.config.filter_stationary_tracks)
        self.min_displacement_var.set(str(self.config.min_displacement_distance))

        self.config_status_label.config(text="Reset to defaults", foreground='green')
        self.tracker = BatchWormTracker(self.config)
        self.update_config_summary()

    def update_config_summary(self):
        """Update the configuration summary display"""
        summary = f"Thresholds: {self.config.threshold_min}-{self.config.threshold_max} | "
        summary += f"Blob Size: {self.config.min_blob_size}-{self.config.max_blob_size} px | "
        summary += f"Max Distance: {self.config.max_distance} px | "
        summary += f"Trajectory Weight: {self.config.trajectory_weight} | "
        summary += f"Min Track: {self.config.min_track_length} frames | "
        summary += f"Algorithm: {'Hungarian' if self.config.use_hungarian else 'Greedy'} | "
        summary += f"Movement Filter: {'ON' if self.config.filter_stationary_tracks else 'OFF'}"
        if self.config.filter_stationary_tracks:
            summary += f" (>={self.config.min_displacement_distance}px) | "
        else:
            summary += " | "
        summary += f"Nose Detection: {'ON' if self.config.nose_detection_enabled else 'OFF'}"

        if hasattr(self, 'config_summary_text'):
            self.config_summary_text.delete(1.0, tk.END)
            self.config_summary_text.insert(1.0, summary)

    def save_configuration(self):
        """Save current configuration to file"""
        if not self.validate_all_parameters():
            messagebox.showerror("Invalid Parameters", "Please fix parameter errors before saving")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            with open(file_path, 'w') as f:
                f.write("# Enhanced Batch Worm Tracker Configuration\n")
                f.write(f"threshold_min={self.config.threshold_min}\n")
                f.write(f"threshold_max={self.config.threshold_max}\n")
                f.write(f"min_blob_size={self.config.min_blob_size}\n")
                f.write(f"max_blob_size={self.config.max_blob_size}\n")
                f.write(f"max_distance={self.config.max_distance}\n")
                f.write(f"trajectory_weight={self.config.trajectory_weight}\n")
                f.write(f"min_track_length={self.config.min_track_length}\n")
                f.write(f"use_hungarian={self.config.use_hungarian}\n")
                f.write(f"nose_detection_enabled={self.config.nose_detection_enabled}\n")
                f.write(f"nose_smoothing_frames={self.config.nose_smoothing_frames}\n")
                f.write(f"min_movement_threshold={self.config.min_movement_threshold}\n")
                f.write(f"filter_stationary_tracks={self.config.filter_stationary_tracks}\n")
                f.write(f"min_displacement_distance={self.config.min_displacement_distance}\n")

            messagebox.showinfo("Configuration Saved", f"Configuration saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save configuration:\n{e}")

    def load_configuration(self):
        """Load configuration from file"""
        file_path = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue

                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()

                        try:
                            if key == 'threshold_min':
                                self.config.threshold_min = int(value)
                                self.min_thresh_var.set(value)
                            elif key == 'threshold_max':
                                self.config.threshold_max = int(value)
                                self.max_thresh_var.set(value)
                            elif key == 'min_blob_size':
                                self.config.min_blob_size = int(value)
                                self.min_blob_var.set(value)
                            elif key == 'max_blob_size':
                                self.config.max_blob_size = int(value)
                                self.max_blob_var.set(value)
                            elif key == 'max_distance':
                                self.config.max_distance = int(value)
                                self.max_dist_var.set(value)
                            elif key == 'trajectory_weight':
                                self.config.trajectory_weight = float(value)
                                self.traj_weight_var.set(value)
                            elif key == 'min_track_length':
                                self.config.min_track_length = int(value)
                                self.min_track_var.set(value)
                            elif key == 'use_hungarian':
                                self.config.use_hungarian = value.lower() == 'true'
                                self.algorithm_var.set("Hungarian" if self.config.use_hungarian else "Greedy")
                            elif key == 'nose_detection_enabled':
                                self.config.nose_detection_enabled = value.lower() == 'true'
                                self.nose_enabled_var.set(self.config.nose_detection_enabled)
                            elif key == 'nose_smoothing_frames':
                                self.config.nose_smoothing_frames = int(value)
                                self.nose_smooth_var.set(value)
                            elif key == 'min_movement_threshold':
                                self.config.min_movement_threshold = float(value)
                                self.nose_movement_var.set(value)
                            elif key == 'filter_stationary_tracks':
                                self.config.filter_stationary_tracks = value.lower() == 'true'
                                self.filter_stationary_var.set(self.config.filter_stationary_tracks)
                            elif key == 'min_displacement_distance':
                                self.config.min_displacement_distance = float(value)
                                self.min_displacement_var.set(value)
                        except ValueError:
                            continue

            if self.validate_all_parameters():
                self.tracker = BatchWormTracker(self.config)
                self.update_config_summary()
                messagebox.showinfo("Configuration Loaded", "Configuration loaded successfully")
            else:
                messagebox.showwarning("Invalid Configuration", "Some parameters in the file are invalid")

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load configuration:\n{e}")

    # SIMPLIFIED: Smart directory selection method
    def add_directories(self):
        """Smart directory selection - handles both single and multiple"""
        dialog = SmartDirectoryDialog(self.root)
        selected_dirs = dialog.show()

        if selected_dirs:
            added_count = 0
            for directory in selected_dirs:
                if directory not in self.directories:
                    self.directories.append(directory)
                    self.dir_listbox.insert(tk.END, directory)
                    added_count += 1

            if added_count > 0:
                if added_count == 1:
                    self.update_status(f"Added 1 directory")
                else:
                    self.update_status(f"Added {added_count} directories")

                if len(selected_dirs) > added_count:
                    messagebox.showinfo("Directories Added",
                                       f"Successfully added {added_count} directories.\n"
                                       f"Skipped {len(selected_dirs) - added_count} duplicates.")
            else:
                self.update_status("No new directories added (all were duplicates)")

    def remove_directory(self):
        """Remove selected directories from the list"""
        selection = self.dir_listbox.curselection()
        if not selection:
            messagebox.showinfo("No Selection", "Please select one or more directories to remove")
            return

        # Remove in reverse order to maintain indices
        removed_count = 0
        for index in reversed(selection):
            removed_dir = self.directories.pop(index)
            self.dir_listbox.delete(index)
            removed_count += 1

        if removed_count == 1:
            self.update_status(f"Removed 1 directory")
        else:
            self.update_status(f"Removed {removed_count} directories")

    def clear_directories(self):
        """Clear all directories from the list"""
        if self.directories:
            if messagebox.askyesno("Clear All", "Remove all directories from the list?"):
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
        """Start batch processing with current configuration"""
        if not self.validate_all_parameters():
            messagebox.showerror("Invalid Parameters", "Please fix parameter errors before processing")
            return

        if not self.directories:
            messagebox.showwarning("No Directories", "Please add at least one directory to process")
            return

        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("Processing Active", "Batch processing is already running")
            return

        # Update tracker with current config
        self.tracker = BatchWormTracker(self.config)

        # Confirm processing
        num_dirs = len(self.directories)
        confirm_msg = f"Start processing {num_dirs} directories with current configuration?\n\n"
        confirm_msg += "This will:\n"
        confirm_msg += "• Generate background images for each directory\n"
        confirm_msg += "• Run trajectory-aware tracking with current parameters\n"
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
        self._update_results_summary()

    def _update_results_summary(self):
        """Update the results summary label"""
        if not self.results:
            return

        successful = len([r for r in self.results if r.success])
        total = len(self.results)
        summary = f"Processed: {total} directories | Successful: {successful} | Failed: {total - successful}"
        self.results_summary_label.config(text=summary)

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

        # Switch to results tab
        self.notebook.select(self.results_tab)

        # Show completion dialog
        messagebox.showinfo("Batch Complete", "Batch processing finished!\n\nCheck the Results tab for detailed information.")

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

            summary += f"Quality Assessment:\n"
            summary += f"  Normal quality: {len(normal_quality)}\n"
            summary += f"  Noisy (needs QC): {len(noisy_quality)}\n"
            summary += f"  Empty (no tracks): {len(empty_quality)}\n\n"

            summary += f"Performance:\n"
            summary += f"  Total tracks generated: {total_tracks}\n"
            summary += f"  Average processing time: {avg_time:.1f}s per directory\n"
            summary += f"  Total processing time: {total_time:.1f}s\n\n"

        # Configuration used
        summary += f"Configuration Used:\n"
        summary += f"  Thresholds: {self.config.threshold_min}-{self.config.threshold_max}\n"
        summary += f"  Blob Size: {self.config.min_blob_size}-{self.config.max_blob_size} pixels\n"
        summary += f"  Max Distance: {self.config.max_distance} px\n"
        summary += f"  Trajectory Weight: {self.config.trajectory_weight}\n"
        summary += f"  Min Track Length: {self.config.min_track_length} frames\n"
        summary += f"  Movement Filter: {'Enabled' if self.config.filter_stationary_tracks else 'Disabled'}\n"
        if self.config.filter_stationary_tracks:
            summary += f"  Min Displacement: {self.config.min_displacement_distance} pixels\n"
        summary += f"  Algorithm: {'Hungarian' if self.config.use_hungarian else 'Greedy'}\n"
        summary += f"  Nose Detection: {'Enabled' if self.config.nose_detection_enabled else 'Disabled'}\n"
        if self.config.nose_detection_enabled:
            summary += f"  Nose Smoothing: {self.config.nose_smoothing_frames} frames\n"
            summary += f"  Movement Threshold: {self.config.min_movement_threshold} px/frame\n"
        summary += "\n"

        if noisy_quality:
            summary += "DIRECTORIES NEEDING MANUAL QC (High track count):\n"
            for result in noisy_quality:
                summary += f"  • {os.path.basename(result.directory)} ({result.num_accepted_tracks} tracks)\n"
            summary += "\n"

        if empty_quality:
            summary += "DIRECTORIES WITH NO TRACKS FOUND:\n"
            for result in empty_quality:
                summary += f"  • {os.path.basename(result.directory)}\n"
            summary += "\n"

        if failed:
            summary += "FAILED DIRECTORIES:\n"
            for result in failed:
                summary += f"  • {os.path.basename(result.directory)}: {result.error_message}\n"
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
                f.write("ENHANCED BATCH WORM TRACKER PROCESSING REPORT\n")
                f.write("=" * 60 + "\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Configuration
                f.write("CONFIGURATION USED:\n")
                f.write(f"Threshold Range: {self.config.threshold_min}-{self.config.threshold_max}\n")
                f.write(f"Blob Size: {self.config.min_blob_size}-{self.config.max_blob_size} pixels\n")
                f.write(f"Max Distance: {self.config.max_distance} px\n")
                f.write(f"Trajectory Weight: {self.config.trajectory_weight}\n")
                f.write(f"Min Track Length: {self.config.min_track_length} frames\n")
                f.write(f"Movement Filter: {'Enabled' if self.config.filter_stationary_tracks else 'Disabled'}\n")
                if self.config.filter_stationary_tracks:
                    f.write(f"Min Displacement Distance: {self.config.min_displacement_distance} pixels\n")
                f.write(f"Algorithm: {'Hungarian' if self.config.use_hungarian else 'Greedy'}\n")
                f.write(f"Nose Detection: {'Enabled' if self.config.nose_detection_enabled else 'Disabled'}\n")
                if self.config.nose_detection_enabled:
                    f.write(f"Nose Smoothing Frames: {self.config.nose_smoothing_frames}\n")
                    f.write(f"Min Movement Threshold: {self.config.min_movement_threshold}\n")
                f.write("\n")

                # Detailed results
                f.write("DETAILED RESULTS:\n")
                f.write("-" * 40 + "\n")
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
                    f.write("-" * 40 + "\n")

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
        self.results_summary_label.config(text="No processing results yet")

    def run(self):
        """Run the GUI application"""
        self.root.mainloop()


def main():
    """Main function to run the enhanced batch tracker"""
    print("=" * 70)
    print("ENHANCED BATCH WORM TRACKER WITH SMART DIRECTORY SELECTION")
    print("Automated processing with adjustable parameters")
    print("=" * 70)
    print("\nNEW FEATURES:")
    print("• Single smart 'Add Directories' button handles both single and multiple selection")
    print("• Add all subdirectories from parent folder")
    print("• Multi-select removal of directories")
    print("• Full parameter customization with validation")
    print("• Tabbed interface for better organization")
    print("• Save/load configuration files")
    print("• Real-time parameter validation")
    print("• Enhanced results summary")
    print("• Stationary track filtering")
    print("• Same core tracking as main tracker")
    print("\nFEATURES:")
    print("• Trajectory-aware tracking with configurable weight")
    print("• Adjustable thresholding and blob filtering")
    print("• Nose detection with configurable parameters")
    print("• Movement distance filtering for stationary objects")
    print("• Quality assessment (flags noisy/empty results)")
    print("• Comprehensive summary reports")
    print("• Network drive compatible")
    print("\nQUALITY FLAGS:")
    print("NORMAL: Standard number of tracks detected")
    print("NOISY: High track count - manual QC recommended")
    print("EMPTY: No tracks found - check parameters")
    print("\nWORKFLOW:")
    print("1. Configure parameters in the Configuration tab")
    print("2. Add directories in the Batch Processing tab")
    print("   - Click 'Add Directories' for flexible selection options")
    print("   - Select single directories or use bulk operations")
    print("3. Start processing and monitor progress")
    print("4. Review results in the Results tab")
    print("\nSTARTING ENHANCED BATCH TRACKER GUI...")

    try:
        app = BatchWormTrackerGUI()
        app.run()
    except Exception as e:
        print(f"Error starting enhanced batch tracker: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
