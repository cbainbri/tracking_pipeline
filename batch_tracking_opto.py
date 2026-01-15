#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Batch Worm Tracker - TWO-STAGE BACKGROUND SUBTRACTION
Uses rolling ball background subtraction + max projection
Handles optogenetics lighting changes and spatial illumination gradients
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
    """Configuration for batch processing with two-stage background subtraction"""
    threshold_min: int = 42  # Higher threshold for cleaner detection after rolling ball
    threshold_max: int = 255
    min_blob_size: int = 120
    max_blob_size: int = 2400
    max_distance: int = 75
    trajectory_weight: float = 0.7
    min_track_length: int = 5  # Changed from 50 to 5 for optogenetics with gaps
    use_hungarian: bool = False
    nose_detection_enabled: bool = True
    nose_smoothing_frames: int = 2
    min_movement_threshold: float = 2.0
    filter_stationary_tracks: bool = True
    min_displacement_distance: float = 75.0
    # Two-stage background parameters
    rolling_ball_radius: int = 50  # Radius for rolling ball filter
    use_max_projection: bool = True  # Use max vs median for stage 2
    save_debug_images: bool = False  # Save intermediate images for diagnosis


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
    """Core tracking functionality with two-stage background subtraction"""

    def __init__(self, config: BatchConfig):
        self.config = config
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for batch processing"""
        logger = logging.getLogger('BatchWormTracker')
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
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

    def create_rolling_ball(self, radius: int, shrinkfactor: int) -> np.ndarray:
        """
        Create precomputed rolling ball patch (top of sphere)
        Based on Fiji's RollingBall class
        
        Args:
            radius: Ball radius in original image coordinates
            shrinkfactor: Image shrink factor
            
        Returns:
            2D array representing height of ball patch above xy-plane
        """
        # Determine arc trim percentage based on radius
        if radius <= 10:
            arc_trim_per = 12  # trim 24% in x and y
        elif radius <= 30:
            arc_trim_per = 12
        elif radius <= 100:
            arc_trim_per = 16  # trim 32%
        else:
            arc_trim_per = 20  # trim 40%
        
        # Scale to shrunk image
        small_ball_radius = max(1, radius // shrinkfactor)
        rsquare = small_ball_radius * small_ball_radius
        diam = small_ball_radius * 2
        
        # Trim edges to create patch (only top of sphere)
        xtrim = (arc_trim_per * diam) // 100
        patchwidth = diam - 2 * xtrim
        halfwidth = small_ball_radius - xtrim
        
        # Create ball patch
        ball = np.zeros((patchwidth + 1, patchwidth + 1), dtype=np.uint8)
        
        for i in range(ball.size):
            x = (i % (patchwidth + 1)) - halfwidth
            y = (i // (patchwidth + 1)) - halfwidth
            temp = rsquare - x*x - y*y
            if temp >= 0:
                ball.flat[i] = int(np.sqrt(temp))
        
        return ball
    
    def shrink_image(self, image: np.ndarray, shrinkfactor: int) -> np.ndarray:
        """
        Shrink image by taking minimum in each shrinkfactor x shrinkfactor block
        VECTORIZED for speed
        
        Args:
            image: Input image
            shrinkfactor: Reduction factor
            
        Returns:
            Shrunk image
        """
        height, width = image.shape
        sheight = height // shrinkfactor
        swidth = width // shrinkfactor
        
        # Smooth first to reduce noise
        smoothed = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Crop to exact multiple of shrinkfactor
        cropped = smoothed[:sheight*shrinkfactor, :swidth*shrinkfactor]
        
        # Reshape to group into blocks and take min
        # Shape: (sheight, shrinkfactor, swidth, shrinkfactor)
        reshaped = cropped.reshape(sheight, shrinkfactor, swidth, shrinkfactor)
        
        # Take minimum across the shrinkfactor dimensions (axes 1 and 3)
        small = reshaped.min(axis=1).min(axis=2)
        
        return small
    
    def roll_ball_on_image(self, image: np.ndarray, ball: np.ndarray, shrinkfactor: int) -> np.ndarray:
        """
        Roll ball across image to find background
        Implements Fiji's rolling ball algorithm (OPTIMIZED)
        
        Args:
            image: Shrunk image (already reduced resolution)
            ball: Precomputed ball patch
            shrinkfactor: Shrink factor for final background size
            
        Returns:
            Background at full resolution
        """
        sheight, swidth = image.shape
        patchwidth = ball.shape[0] - 1
        halfpatch = patchwidth // 2
        
        # Background at full resolution
        height = sheight * shrinkfactor
        width = swidth * shrinkfactor
        background = np.zeros((height, width), dtype=np.uint8)
        
        # Z-center of rolling ball
        zctr = 0
        
        # Progress tracking
        total_positions = (sheight - 2 + patchwidth) * (swidth - 2 + patchwidth)
        positions_done = 0
        last_log = 0
        
        # Roll ball across shrunk image
        for ypt in range(1, sheight - 1 + patchwidth + 1):
            for xpt in range(1, swidth - 1 + patchwidth + 1):
                
                # Find minimum z-difference (ball should be tangent below image)
                zmin = 255
                
                # Calculate bounds to avoid redundant checks
                y_start = max(1, ypt - patchwidth)
                y_end = min(sheight - 1, ypt)
                x_start = max(1, xpt - patchwidth)
                x_end = min(swidth - 1, xpt)
                
                for ypt2 in range(y_start, y_end + 1):
                    by = ypt2 - (ypt - patchwidth)
                    for xpt2 in range(x_start, x_end + 1):
                        bx = xpt2 - (xpt - patchwidth)
                        
                        # Z-difference between image and ball
                        zdif = int(image[ypt2, xpt2]) - (zctr + int(ball[by, bx]))
                        if zdif < zmin:
                            zmin = zdif
                
                # Adjust ball height
                if zmin != 0:
                    zctr += zmin
                
                # Determine which points to skip (optimization)
                ptsbelowlast = halfpatch if zmin < 0 else 0
                
                # Update background with ball heights (only in valid region)
                y_start = max(1, ypt - patchwidth)
                y_end = min(sheight - 1, ypt)
                
                for ypt2 in range(y_start, y_end + 1):
                    yval = ypt2
                    by = ypt2 - (ypt - patchwidth)
                    by_full = (yval - 1 + 1) * shrinkfactor
                    
                    if by_full >= height:
                        continue
                    
                    x_start_inner = max(1, xpt - patchwidth + ptsbelowlast)
                    x_end_inner = min(swidth - 1, xpt)
                    
                    for xpt2 in range(x_start_inner, x_end_inner + 1):
                        xval = xpt2
                        bx = xpt2 - (xpt - patchwidth)
                        
                        # Height at this point
                        zadd = zctr + int(ball[by, bx])
                        
                        # Map to full resolution background
                        bx_full = (xval - 1 + 1) * shrinkfactor
                        
                        if bx_full < width:
                            if zadd > background[by_full, bx_full]:
                                background[by_full, bx_full] = zadd
                
                # Progress logging
                positions_done += 1
                progress_pct = (positions_done / total_positions) * 100
                if progress_pct - last_log >= 10:
                    self.logger.info(f"  Rolling ball progress: {progress_pct:.0f}%")
                    last_log = progress_pct
        
        return background
    
    def interpolate_background(self, background: np.ndarray, shrinkfactor: int) -> np.ndarray:
        """
        Interpolate background to fill in missing pixels
        Uses bilinear interpolation
        
        Args:
            background: Sparse background with values at shrinkfactor intervals
            shrinkfactor: Grid spacing
            
        Returns:
            Fully interpolated background
        """
        height, width = background.shape
        
        # Interpolate horizontally first
        for y in range(shrinkfactor, height - shrinkfactor, shrinkfactor):
            for x in range(shrinkfactor, width - shrinkfactor, shrinkfactor):
                x_next = x + shrinkfactor
                val_left = background[y, x]
                val_right = background[y, x_next]
                
                # Fill in between
                for dx in range(1, shrinkfactor):
                    alpha = dx / shrinkfactor
                    background[y, x + dx] = int(val_left * (1 - alpha) + val_right * alpha)
        
        # Then interpolate vertically
        for y in range(shrinkfactor, height - shrinkfactor, shrinkfactor):
            y_next = y + shrinkfactor
            for x in range(width):
                val_top = background[y, x]
                val_bottom = background[y_next, x]
                
                # Fill in between
                for dy in range(1, shrinkfactor):
                    alpha = dy / shrinkfactor
                    background[y + dy, x] = int(val_top * (1 - alpha) + val_bottom * alpha)
        
        # Extrapolate edges
        self._extrapolate_edges(background, shrinkfactor)
        
        return background
    
    def _extrapolate_edges(self, background: np.ndarray, shrinkfactor: int):
        """Extrapolate edges linearly"""
        height, width = background.shape
        
        # Top and bottom edges
        for x in range(shrinkfactor, width - shrinkfactor):
            # Top
            val1 = background[shrinkfactor, x]
            val2 = background[shrinkfactor + 1, x]
            slope = val2 - val1
            for y in range(shrinkfactor):
                val = val1 - slope * (shrinkfactor - y)
                background[y, x] = np.clip(val, 0, 255)
            
            # Bottom
            val1 = background[height - shrinkfactor - 1, x]
            val2 = background[height - shrinkfactor, x]
            slope = val2 - val1
            for y in range(height - shrinkfactor, height):
                val = val2 + slope * (y - height + shrinkfactor)
                background[y, x] = np.clip(val, 0, 255)
        
        # Left and right edges
        for y in range(height):
            # Left
            val1 = background[y, shrinkfactor]
            val2 = background[y, shrinkfactor + 1]
            slope = val2 - val1
            for x in range(shrinkfactor):
                val = val1 - slope * (shrinkfactor - x)
                background[y, x] = np.clip(val, 0, 255)
            
            # Right
            val1 = background[y, width - shrinkfactor - 1]
            val2 = background[y, width - shrinkfactor]
            slope = val2 - val1
            for x in range(width - shrinkfactor, width):
                val = val2 + slope * (x - width + shrinkfactor)
                background[y, x] = np.clip(val, 0, 255)
    
    def rolling_ball_background(self, image: np.ndarray, radius: int) -> np.ndarray:
        """
        Apply rolling ball background subtraction - Fiji style for DARK backgrounds
        
        For dark worms on light background, Fiji:
        1. Inverts (dark→bright)
        2. Does rolling ball
        3. Inverts back (bright→dark)
        
        Args:
            image: Input grayscale image (dark worms on light background)
            radius: Ball radius in pixels (typical: 50)
            
        Returns:
            Background-subtracted image (dark worms on light background)
        """
        if image is None or image.size == 0:
            return image
        
        # INVERT #1: dark worms on light → bright worms on dark
        inverted = cv2.bitwise_not(image)
        
        # Determine shrink factor based on radius (Fiji's heuristic)
        if radius <= 10:
            shrinkfactor = 1
        elif radius <= 30:
            shrinkfactor = 2
        elif radius <= 100:
            shrinkfactor = 4
        else:
            shrinkfactor = 8
        
        # Shrink image
        if shrinkfactor > 1:
            small_image = self.shrink_image(inverted, shrinkfactor)
        else:
            small_image = inverted.copy()
        
        # Scale radius to shrunk image
        small_radius = max(1, radius // shrinkfactor)
        
        # Create ball (disk) structuring element
        disk_size = 2 * small_radius + 1
        y, x = np.ogrid[-small_radius:small_radius+1, -small_radius:small_radius+1]
        disk = (x*x + y*y <= small_radius*small_radius).astype(np.uint8)
        
        # Morphological opening (erosion then dilation)
        # Removes bright features (worms), leaves smooth background
        background_small = cv2.morphologyEx(small_image, cv2.MORPH_OPEN, disk)
        
        # Resize background back to full resolution
        if shrinkfactor > 1:
            background = cv2.resize(background_small, (image.shape[1], image.shape[0]), 
                                   interpolation=cv2.INTER_LINEAR)
        else:
            background = background_small
        
        # Subtract background from inverted image
        subtracted = cv2.subtract(inverted, background)
        
        # INVERT #2: bright worms on dark → dark worms on light (back to original polarity)
        result = cv2.bitwise_not(subtracted)
        
        return result

    def _process_single_frame_rolling_ball(self, args):
        """
        Helper function for parallel processing of single frame
        
        Args:
            args: Tuple of (image_path, radius, reference_shape)
            
        Returns:
            Tuple of (index, corrected_frame) or (index, None) on error
        """
        img_path, radius, reference_shape, idx = args
        
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return (idx, None)
            
            # Ensure consistent shape
            if reference_shape is not None and img.shape != reference_shape:
                img = cv2.resize(img, (reference_shape[1], reference_shape[0]))
            
            # Apply rolling ball background subtraction
            corrected = self.rolling_ball_background(img, radius)
            return (idx, corrected)
            
        except Exception as e:
            self.logger.warning(f"Error processing {img_path}: {e}")
            return (idx, None)

    def process_all_frames_rolling_ball(self, image_files: List[str]) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """
        Stage 1: Apply rolling ball background subtraction to all frames (PARALLELIZED)
        Stage 2: Generate max projection from corrected frames
        
        Args:
            image_files: List of image paths
            
        Returns:
            Tuple of (list of corrected frames, max projection background)
        """
        radius = self.config.rolling_ball_radius
        
        self.logger.info(f"Stage 1: Applying rolling ball filter (radius={radius}, parallelized)...")
        
        # Get reference shape from first image
        reference_shape = None
        first_img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
        if first_img is not None:
            reference_shape = first_img.shape
        
        self.logger.info(f"  Image size: {reference_shape}")
        
        # Determine shrink factor
        if radius <= 10:
            shrinkfactor = 1
        elif radius <= 30:
            shrinkfactor = 2
        elif radius <= 100:
            shrinkfactor = 4
        else:
            shrinkfactor = 8
        
        self.logger.info(f"  Shrink factor: {shrinkfactor} ({reference_shape[0]}x{reference_shape[1]} → {reference_shape[0]//shrinkfactor}x{reference_shape[1]//shrinkfactor})")
        
        # Prepare arguments for parallel processing
        args_list = [(img_path, radius, reference_shape, idx) 
                     for idx, img_path in enumerate(image_files)]
        
        # Use multiprocessing to parallelize rolling ball filter
        from multiprocessing import Pool, cpu_count
        
        # Use all available cores
        num_workers = cpu_count()
        self.logger.info(f"  Using {num_workers} CPU cores for parallel processing")
        
        corrected_frames_dict = {}
        
        with Pool(num_workers) as pool:
            # Process frames in parallel with progress tracking
            self.logger.info(f"  Processing {len(image_files)} frames in parallel...")
            results = []
            for i, result in enumerate(pool.imap(self._process_single_frame_rolling_ball, args_list)):
                results.append(result)
                # Log every 20%
                if (i + 1) % max(1, len(image_files) // 5) == 0:
                    self.logger.info(f"    Progress: {i+1}/{len(image_files)} frames ({100*(i+1)/len(image_files):.0f}%)")
            
            # Collect results
            for idx, corrected in results:
                if corrected is not None:
                    corrected_frames_dict[idx] = corrected
        
        # Convert to ordered list
        corrected_frames = [corrected_frames_dict[i] for i in sorted(corrected_frames_dict.keys())]
        
        if not corrected_frames:
            self.logger.error("No frames successfully processed")
            return [], None
        
        self.logger.info(f"Stage 1 complete: {len(corrected_frames)} frames processed")
        
        # Stage 2: Generate background from corrected frames
        self.logger.info("Stage 2: Generating background from corrected frames...")
        
        # Sample frames for background (all if <=100, otherwise sample)
        if len(corrected_frames) <= 100:
            sampled_frames = corrected_frames
        else:
            indices = np.linspace(0, len(corrected_frames)-1, 100, dtype=int)
            sampled_frames = [corrected_frames[i] for i in indices]
        
        # Convert to float for projection
        frames_float = [f.astype(np.float32) for f in sampled_frames]
        
        # MAX projection: worms are dark on light background
        # At each pixel, max = the lightest value = when NO worm is there
        # Result: pure background with no worms
        background = np.max(frames_float, axis=0).astype(np.uint8)
        method = "max"
        
        self.logger.info(f"Stage 2 complete: {method} projection from {len(sampled_frames)} frames")
        
        return corrected_frames, background

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

    def detect_blobs(self, binary_image: np.ndarray) -> List[Dict]:
        """Detect blobs (worms) in binary image with shape measurements"""
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blobs = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if self.config.min_blob_size <= area <= self.config.max_blob_size:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    
                    # Calculate shape measurements
                    perimeter = cv2.arcLength(contour, True)
                    
                    # Check if we have enough points for reliable convexity/solidity
                    # Need at least 15 points for stable convex hull
                    if len(contour) >= 15:
                        # Convex hull for convexity and solidity
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        hull_perimeter = cv2.arcLength(hull, True)
                        
                        # Solidity: ratio of contour area to convex hull area
                        # Measures how "solid" vs "concave" the shape is
                        # Straight worm ≈ 0.95-1.0 (very solid)
                        # Bent/omega worm ≈ 0.6-0.8 (concave, tail near head)
                        if hull_area > 0:
                            solidity = area / hull_area
                        else:
                            solidity = np.nan
                        
                        # Convexity: ratio of convex hull perimeter to contour perimeter
                        # Straight worm ≈ 0.95-1.0 (outline nearly matches hull)
                        # Bent/omega worm ≈ 0.7-0.9 (outline is longer due to bending)
                        if perimeter > 0:
                            convexity = hull_perimeter / perimeter
                        else:
                            convexity = np.nan
                    else:
                        # Not enough points for reliable hull calculation
                        convexity = np.nan
                        solidity = np.nan
                    
                    # Fit ellipse to get major/minor axes (need at least 5 points)
                    if len(contour) >= 5:
                        try:
                            ellipse = cv2.fitEllipse(contour)
                            (cx_ell, cy_ell), (minor_axis, major_axis), angle = ellipse
                            
                            # OpenCV returns (width, height) but may swap them
                            # Ensure major >= minor
                            if minor_axis > major_axis:
                                minor_axis, major_axis = major_axis, minor_axis
                            
                            # Aspect ratio (major/minor)
                            if minor_axis > 0:
                                aspect_ratio = major_axis / minor_axis
                            else:
                                aspect_ratio = np.nan
                            
                        except:
                            # Ellipse fitting failed
                            major_axis = np.nan
                            minor_axis = np.nan
                            aspect_ratio = np.nan
                    else:
                        major_axis = np.nan
                        minor_axis = np.nan
                        aspect_ratio = np.nan

                    blob = {
                        'id': i,
                        'centroid': (cx, cy),
                        'area': area,
                        'perimeter': perimeter,
                        'convexity': convexity,
                        'solidity': solidity,
                        'major_axis': major_axis,
                        'minor_axis': minor_axis,
                        'aspect_ratio': aspect_ratio,
                        'contour': contour,
                        'frame_idx': None
                    }
                    blobs.append(blob)

        return blobs

    def calculate_trajectory_vector(self, track: List[Dict], smoothing_window: int = 3) -> np.ndarray:
        """Calculate smoothed trajectory direction vector"""
        if len(track) < 2:
            return np.array([0.0, 0.0])

        recent_positions = [det['centroid'] for det in track[-smoothing_window:]]
        if len(recent_positions) < 2:
            return np.array([0.0, 0.0])

        positions = np.array(recent_positions)
        trajectory = positions[-1] - positions[0]
        norm = np.linalg.norm(trajectory)

        if norm > 1e-6:
            return trajectory / norm
        else:
            return np.array([0.0, 0.0])

    def match_detections_greedy(self, tracks: List[List[Dict]], detections: List[Dict],
                                frame_idx: int) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Greedy matching with trajectory prediction"""
        if not tracks or not detections:
            return [], list(range(len(detections)))

        cost_matrix = np.full((len(tracks), len(detections)), np.inf)

        for t_idx, track in enumerate(tracks):
            last_det = track[-1]
            last_pos = np.array(last_det['centroid'])
            trajectory_vec = self.calculate_trajectory_vector(track)

            for d_idx, det in enumerate(detections):
                det_pos = np.array(det['centroid'])
                euclidean_dist = np.linalg.norm(det_pos - last_pos)

                if euclidean_dist > self.config.max_distance:
                    continue

                predicted_pos = last_pos + trajectory_vec * euclidean_dist
                trajectory_dist = np.linalg.norm(det_pos - predicted_pos)

                w_traj = self.config.trajectory_weight
                combined_cost = w_traj * trajectory_dist + (1 - w_traj) * euclidean_dist
                cost_matrix[t_idx, d_idx] = combined_cost

        assignments = []
        unassigned_detections = set(range(len(detections)))

        track_indices = list(range(len(tracks)))
        track_indices.sort(key=lambda t: cost_matrix[t].min() if cost_matrix[t].min() != np.inf else np.inf)

        for t_idx in track_indices:
            if len(unassigned_detections) == 0:
                break

            valid_costs = [(d_idx, cost_matrix[t_idx, d_idx])
                          for d_idx in unassigned_detections
                          if cost_matrix[t_idx, d_idx] != np.inf]

            if valid_costs:
                best_det, best_cost = min(valid_costs, key=lambda x: x[1])
                assignments.append((t_idx, best_det))
                unassigned_detections.remove(best_det)

        return assignments, list(unassigned_detections)

    def match_detections_hungarian(self, tracks: List[List[Dict]], detections: List[Dict],
                                   frame_idx: int) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Hungarian algorithm matching with trajectory prediction"""
        if not tracks or not detections:
            return [], list(range(len(detections)))

        cost_matrix = np.full((len(tracks), len(detections)), np.inf)

        for t_idx, track in enumerate(tracks):
            last_det = track[-1]
            last_pos = np.array(last_det['centroid'])
            trajectory_vec = self.calculate_trajectory_vector(track)

            for d_idx, det in enumerate(detections):
                det_pos = np.array(det['centroid'])
                euclidean_dist = np.linalg.norm(det_pos - last_pos)

                if euclidean_dist > self.config.max_distance:
                    continue

                predicted_pos = last_pos + trajectory_vec * euclidean_dist
                trajectory_dist = np.linalg.norm(det_pos - predicted_pos)

                w_traj = self.config.trajectory_weight
                combined_cost = w_traj * trajectory_dist + (1 - w_traj) * euclidean_dist
                cost_matrix[t_idx, d_idx] = combined_cost

        finite_mask = np.isfinite(cost_matrix)
        if not finite_mask.any():
            return [], list(range(len(detections)))

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignments = []
        assigned_detections = set()

        for t_idx, d_idx in zip(row_ind, col_ind):
            if np.isfinite(cost_matrix[t_idx, d_idx]):
                assignments.append((t_idx, d_idx))
                assigned_detections.add(d_idx)

        unassigned_detections = [d for d in range(len(detections)) if d not in assigned_detections]
        return assignments, unassigned_detections

    def detect_nose(self, contour: np.ndarray, prev_nose: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Detect worm nose (most elongated point)"""
        if len(contour) < 5:
            return None

        try:
            ellipse = cv2.fitEllipse(contour)
            center, (MA, ma), angle = ellipse

            if MA < 1e-6:
                return None

            angle_rad = np.deg2rad(angle)
            direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])

            contour_points = contour.reshape(-1, 2).astype(np.float32)
            center_np = np.array(center, dtype=np.float32)
            projections = np.dot(contour_points - center_np, direction)

            max_idx = np.argmax(projections)
            min_idx = np.argmin(projections)

            candidate_nose_pos = contour_points[max_idx]
            candidate_tail_pos = contour_points[min_idx]

            if prev_nose is not None:
                dist_to_nose = np.linalg.norm(candidate_nose_pos - prev_nose)
                dist_to_tail = np.linalg.norm(candidate_tail_pos - prev_nose)

                if dist_to_tail < dist_to_nose:
                    candidate_nose_pos, candidate_tail_pos = candidate_tail_pos, candidate_nose_pos

            return candidate_nose_pos

        except Exception as e:
            return None

    def smooth_nose_positions(self, track: List[Dict]) -> List[Dict]:
        """Apply smoothing to nose positions"""
        if not self.config.nose_detection_enabled:
            return track

        window = self.config.nose_smoothing_frames
        if window < 1 or len(track) < 2:
            return track

        for i in range(len(track)):
            if track[i].get('nose') is None:
                continue

            start_idx = max(0, i - window)
            end_idx = min(len(track), i + window + 1)

            nose_positions = [track[j]['nose'] for j in range(start_idx, end_idx)
                            if track[j].get('nose') is not None]

            if nose_positions:
                smoothed_nose = np.mean(nose_positions, axis=0)
                track[i]['nose_smoothed'] = smoothed_nose
            else:
                track[i]['nose_smoothed'] = track[i]['nose']

        return track

    def filter_stationary_tracks(self, tracks: List[List[Dict]]) -> List[List[Dict]]:
        """Filter out tracks that don't move sufficiently"""
        if not self.config.filter_stationary_tracks:
            return tracks

        filtered_tracks = []

        for track in tracks:
            if len(track) < 2:
                continue

            positions = np.array([det['centroid'] for det in track])
            start_pos = positions[0]
            end_pos = positions[-1]
            total_displacement = np.linalg.norm(end_pos - start_pos)

            if total_displacement >= self.config.min_displacement_distance:
                filtered_tracks.append(track)

        return filtered_tracks

    def process_directory(self, directory: str, progress_callback=None) -> ProcessingResult:
        """Process a single directory with two-stage background subtraction"""
        start_time = time.time()

        try:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing: {directory}")
            self.logger.info(f"{'='*60}")

            image_files = self.find_image_files(directory)
            if not image_files:
                return ProcessingResult(
                    directory=directory,
                    success=False,
                    num_images=0,
                    num_tracks=0,
                    num_accepted_tracks=0,
                    processing_time=time.time() - start_time,
                    error_message="No image files found"
                )

            total_frames = len(image_files)
            self.logger.info(f"Found {total_frames} images")

            # Two-stage background subtraction
            corrected_frames, background = self.process_all_frames_rolling_ball(image_files)
            
            if not corrected_frames or background is None:
                return ProcessingResult(
                    directory=directory,
                    success=False,
                    num_images=total_frames,
                    num_tracks=0,
                    num_accepted_tracks=0,
                    processing_time=time.time() - start_time,
                    error_message="Failed background subtraction"
                )
            
            # Save debug images to see what's happening
            if self.config.save_debug_images:
                debug_dir = os.path.join(directory, 'debug_images')
                os.makedirs(debug_dir, exist_ok=True)
                self.logger.info(f"Saving debug images to {debug_dir}")
                
                # Save samples from beginning, middle, end, and during stim transitions
                sample_indices = [0, 29, 30, 49, 50, 69, 70, len(corrected_frames)-1]
                
                for idx in sample_indices:
                    if idx >= len(corrected_frames):
                        continue
                    
                    # 1. Original image
                    orig = cv2.imread(image_files[idx], cv2.IMREAD_GRAYSCALE)
                    cv2.imwrite(os.path.join(debug_dir, f'frame_{idx:03d}_step1_original.png'), orig)
                    
                    # 2. After rolling ball background subtraction
                    corrected = corrected_frames[idx]
                    cv2.imwrite(os.path.join(debug_dir, f'frame_{idx:03d}_step2_after_rolling_ball.png'), corrected)
                    
                    # 3. Max projection background (same for all frames)
                    if idx == 0:
                        cv2.imwrite(os.path.join(debug_dir, f'step3_max_projection_background.png'), background)
                    
                    # 4. Final difference (what gets thresholded)
                    diff = cv2.absdiff(corrected, background)
                    cv2.imwrite(os.path.join(debug_dir, f'frame_{idx:03d}_step4_final_difference.png'), diff)
                    
                    # 5. After morphological opening
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    cleaned = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
                    cv2.imwrite(os.path.join(debug_dir, f'frame_{idx:03d}_step5_after_morpho_open.png'), cleaned)
                    
                    # 6. After thresholding (binary)
                    binary = self.apply_threshold(cleaned)
                    cv2.imwrite(os.path.join(debug_dir, f'frame_{idx:03d}_step6_binary_threshold.png'), binary)
                
                self.logger.info(f"Debug images saved. Check {debug_dir} to diagnose pipeline")

            # Initialize tracking
            active_tracks = []
            finished_tracks = []
            next_track_id = 0

            self.logger.info("Starting frame-by-frame tracking...")

            for frame_idx in range(len(corrected_frames)):
                # Final difference: EXACTLY as Fiji does it
                # corrected_frames: dark worms on very light background (after rolling ball)
                # background: max projection = pure white background
                # ABSDIFF: |background - frame| = |white - (dark worm on light)| = LIGHT worm on BLACK
                corrected_frame = corrected_frames[frame_idx]
                diff = cv2.absdiff(background, corrected_frame)
                
                # Clean up noise with morphological opening
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
                
                # Result: bright worms on dark background (ready for thresholding)
                # Apply thresholding
                binary_frame = self.apply_threshold(diff)

                # Detect blobs
                detections = self.detect_blobs(binary_frame)
                
                # Debug: log detections every 10 frames
                if frame_idx % 10 == 0:
                    self.logger.info(f"  Frame {frame_idx}: {len(detections)} detections, {len(active_tracks)} active tracks")

                for det in detections:
                    det['frame_idx'] = frame_idx

                    if self.config.nose_detection_enabled and 'contour' in det:
                        prev_nose = None
                        if active_tracks:
                            for track in active_tracks:
                                if track and 'nose_smoothed' in track[-1]:
                                    prev_nose = track[-1]['nose_smoothed']
                                    break

                        nose_pos = self.detect_nose(det['contour'], prev_nose)
                        det['nose'] = nose_pos

                # Match detections to tracks
                if self.config.use_hungarian:
                    assignments, unassigned = self.match_detections_hungarian(
                        active_tracks, detections, frame_idx
                    )
                else:
                    assignments, unassigned = self.match_detections_greedy(
                        active_tracks, detections, frame_idx
                    )

                # Update existing tracks
                for track_idx, det_idx in assignments:
                    active_tracks[track_idx].append(detections[det_idx])

                # Create new tracks
                for det_idx in unassigned:
                    new_track = [detections[det_idx]]
                    active_tracks.append(new_track)
                    next_track_id += 1

                # Remove stale tracks
                still_active = []
                for track in active_tracks:
                    if track[-1]['frame_idx'] == frame_idx:
                        still_active.append(track)
                    else:
                        if len(track) >= self.config.min_track_length:
                            finished_tracks.append(track)

                active_tracks = still_active

                # Progress callback
                if progress_callback:
                    progress_callback(frame_idx + 1, len(corrected_frames))

                # Memory management
                if frame_idx % 50 == 0:
                    gc.collect()

            # Finalize remaining active tracks
            for track in active_tracks:
                if len(track) >= self.config.min_track_length:
                    finished_tracks.append(track)

            self.logger.info(f"Initial tracking complete: {len(finished_tracks)} tracks")

            # Apply nose smoothing
            if self.config.nose_detection_enabled:
                self.logger.info("Smoothing nose positions...")
                finished_tracks = [self.smooth_nose_positions(track) for track in finished_tracks]

            # Filter stationary tracks
            if self.config.filter_stationary_tracks:
                before_filter = len(finished_tracks)
                finished_tracks = self.filter_stationary_tracks(finished_tracks)
                after_filter = len(finished_tracks)
                filtered_out = before_filter - after_filter
                self.logger.info(f"Movement filter: {after_filter} tracks kept, {filtered_out} removed")

            # Save results
            self.save_tracks_to_csv(finished_tracks, directory)

            # Determine quality flag
            quality_flag = "normal"
            if len(finished_tracks) == 0:
                quality_flag = "empty"
            elif len(finished_tracks) > 20:
                quality_flag = "noisy"

            processing_time = time.time() - start_time

            self.logger.info(f"Processing complete: {len(finished_tracks)} accepted tracks")
            self.logger.info(f"Processing time: {processing_time:.1f}s")
            self.logger.info(f"Quality assessment: {quality_flag.upper()}")

            # Cleanup
            del corrected_frames
            del background
            gc.collect()

            return ProcessingResult(
                directory=directory,
                success=True,
                num_images=total_frames,
                num_tracks=len(active_tracks) + len(finished_tracks),
                num_accepted_tracks=len(finished_tracks),
                processing_time=processing_time,
                quality_flag=quality_flag
            )

        except Exception as e:
            self.logger.error(f"Error processing {directory}: {e}")
            import traceback
            traceback.print_exc()

            return ProcessingResult(
                directory=directory,
                success=False,
                num_images=len(image_files) if 'image_files' in locals() else 0,
                num_tracks=0,
                num_accepted_tracks=0,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )

    def save_tracks_to_csv(self, tracks: List[List[Dict]], directory: str):
        """Save tracking results to CSV with shape measurements"""
        if not tracks:
            self.logger.warning("No tracks to save")
            return

        # Save long format (original)
        data_rows = []
        for track_id, track in enumerate(tracks):
            for det in track:
                row = {
                    'track_id': track_id,
                    'frame': det['frame_idx'],
                    'x': det['centroid'][0],
                    'y': det['centroid'][1],
                    'area': det['area'],
                    'perimeter': det['perimeter'],
                    'major_axis': det['major_axis'],
                    'minor_axis': det['minor_axis'],
                    'aspect_ratio': det['aspect_ratio'],
                    'convexity': det['convexity'],
                    'solidity': det['solidity']
                }

                if self.config.nose_detection_enabled:
                    if 'nose_smoothed' in det and det['nose_smoothed'] is not None:
                        row['nose_x'] = det['nose_smoothed'][0]
                        row['nose_y'] = det['nose_smoothed'][1]
                    elif 'nose' in det and det['nose'] is not None:
                        row['nose_x'] = det['nose'][0]
                        row['nose_y'] = det['nose'][1]
                    else:
                        row['nose_x'] = np.nan
                        row['nose_y'] = np.nan

                data_rows.append(row)

        df_long = pd.DataFrame(data_rows)
        
        # Reorder columns for readability
        column_order = ['track_id', 'frame', 'x', 'y', 
                       'major_axis', 'minor_axis', 'aspect_ratio',
                       'area', 'perimeter', 'convexity', 'solidity']
        if self.config.nose_detection_enabled:
            column_order.extend(['nose_x', 'nose_y'])
        
        df_long = df_long[column_order]
        
        # Save long format
        output_path_long = os.path.join(directory, 'tracks_long.csv')
        df_long.to_csv(output_path_long, index=False)
        self.logger.info(f"Saved long format tracks to: {output_path_long}")
        
        # Create wide format (one column per animal)
        df_wide = self._convert_to_wide_format(df_long)
        output_path_wide = os.path.join(directory, 'tracks.csv')
        df_wide.to_csv(output_path_wide, index=False)
        self.logger.info(f"Saved wide format tracks to: {output_path_wide}")
        
        # Print summary statistics
        self.logger.info(f"Shape metrics summary:")
        self.logger.info(f"  Mean aspect ratio: {df_long['aspect_ratio'].mean():.3f}")
        self.logger.info(f"  Mean convexity: {df_long['convexity'].mean():.3f}")
        self.logger.info(f"  Mean solidity: {df_long['solidity'].mean():.3f}")
        self.logger.info(f"  Mean major axis: {df_long['major_axis'].mean():.1f} pixels")
        self.logger.info(f"  Mean minor axis: {df_long['minor_axis'].mean():.1f} pixels")
    
    def _convert_to_wide_format(self, df_long: pd.DataFrame) -> pd.DataFrame:
        """
        Convert long format (one row per detection) to wide format (one column per worm)
        Uses the original row-by-row approach to avoid DataFrame NaN issues
        
        Args:
            df_long: Long format dataframe
            
        Returns:
            Wide format dataframe with columns: frame, worm_1_x, worm_1_y, worm_1_major_axis, ...
        """
        # Get all unique track IDs and frames
        track_ids = sorted(df_long['track_id'].unique())
        all_frames = sorted(df_long['frame'].unique())
        
        # Metrics to include for each worm
        metrics = ['x', 'y', 'major_axis', 'minor_axis', 'aspect_ratio', 
                  'area', 'perimeter', 'convexity', 'solidity']
        
        if 'nose_x' in df_long.columns:
            metrics.extend(['nose_x', 'nose_y'])
        
        # Build column names
        columns = ['frame']
        for track_id in track_ids:
            worm_num = track_id + 1
            for metric in metrics:
                columns.append(f'worm_{worm_num}_{metric}')
        
        # Build data row by row (like original script)
        data = []
        for frame in all_frames:
            row = [int(frame)]  # Frame as integer
            
            for track_id in track_ids:
                # Get this track's data for this frame
                track_frame_data = df_long[(df_long['track_id'] == track_id) & 
                                          (df_long['frame'] == frame)]
                
                if len(track_frame_data) > 0:
                    # Track exists at this frame
                    for metric in metrics:
                        value = track_frame_data[metric].iloc[0]
                        row.append(value)
                else:
                    # Track doesn't exist at this frame - fill with None
                    row.extend([None] * len(metrics))
            
            data.append(row)
        
        # Create DataFrame from list of rows (frame column is already int)
        df_wide = pd.DataFrame(data, columns=columns)
        
        return df_wide


class BatchWormTrackerGUI:
    """GUI for batch processing with two-stage background subtraction"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Batch Worm Tracker - Two-Stage Background Edition")
        self.root.geometry("1000x800")

        self.config = BatchConfig()
        self.directories = []
        self.results = []
        self.processing = False

        self.setup_gui()

    def setup_gui(self):
        """Setup the GUI layout"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.config_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.config_frame, text="Configuration")
        self.setup_config_tab()

        self.batch_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.batch_frame, text="Batch Processing")
        self.setup_batch_tab()

        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results")
        self.setup_results_tab()

    def setup_config_tab(self):
        """Setup configuration controls"""
        canvas = tk.Canvas(self.config_frame)
        scrollbar = ttk.Scrollbar(self.config_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Two-stage background parameters
        bg_frame = ttk.LabelFrame(scrollable_frame, text="Two-Stage Background Subtraction", padding=10)
        bg_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(bg_frame, text="Rolling Ball Radius (pixels):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.rb_radius_var = tk.IntVar(value=self.config.rolling_ball_radius)
        ttk.Spinbox(bg_frame, from_=10, to=200, textvariable=self.rb_radius_var, width=10).grid(row=0, column=1, sticky=tk.W, pady=2)
        ttk.Label(bg_frame, text="(50 typical for worms)").grid(row=0, column=2, sticky=tk.W, pady=2, padx=5)

        ttk.Label(bg_frame, text="Stage 2 Projection:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.max_proj_var = tk.BooleanVar(value=self.config.use_max_projection)
        ttk.Checkbutton(bg_frame, text="Use Max Projection", variable=self.max_proj_var).grid(row=1, column=1, columnspan=2, sticky=tk.W, pady=2)
        ttk.Label(bg_frame, text="(uncheck for median)").grid(row=2, column=1, columnspan=2, sticky=tk.W, pady=2, padx=5)

        info_frame = ttk.Frame(bg_frame)
        info_frame.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=10)
        ttk.Label(info_frame, text="How it works:", font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W)
        ttk.Label(info_frame, text="  Stage 1: Rolling ball removes spatial gradients → uniform lighting").pack(anchor=tk.W)
        ttk.Label(info_frame, text="  Stage 2: Max projection → identifies empty plate background").pack(anchor=tk.W)
        ttk.Label(info_frame, text="  Final: Difference reveals only worms").pack(anchor=tk.W)
        ttk.Label(info_frame, text="✓ Handles any lighting changes automatically", font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W, pady=(5,0))

        # Threshold parameters
        thresh_frame = ttk.LabelFrame(scrollable_frame, text="Threshold Parameters", padding=10)
        thresh_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(thresh_frame, text="Min Threshold:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.threshold_min_var = tk.IntVar(value=self.config.threshold_min)
        ttk.Spinbox(thresh_frame, from_=0, to=255, textvariable=self.threshold_min_var, width=10).grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(thresh_frame, text="Max Threshold:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.threshold_max_var = tk.IntVar(value=self.config.threshold_max)
        ttk.Spinbox(thresh_frame, from_=0, to=255, textvariable=self.threshold_max_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=2)

        # Blob size parameters
        blob_frame = ttk.LabelFrame(scrollable_frame, text="Blob Size Parameters", padding=10)
        blob_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(blob_frame, text="Min Blob Size (pixels):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.min_blob_var = tk.IntVar(value=self.config.min_blob_size)
        ttk.Spinbox(blob_frame, from_=1, to=10000, textvariable=self.min_blob_var, width=10).grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(blob_frame, text="Max Blob Size (pixels):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.max_blob_var = tk.IntVar(value=self.config.max_blob_size)
        ttk.Spinbox(blob_frame, from_=1, to=20000, textvariable=self.max_blob_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=2)

        # Tracking parameters
        track_frame = ttk.LabelFrame(scrollable_frame, text="Tracking Parameters", padding=10)
        track_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(track_frame, text="Max Distance (pixels):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.max_distance_var = tk.IntVar(value=self.config.max_distance)
        ttk.Spinbox(track_frame, from_=1, to=500, textvariable=self.max_distance_var, width=10).grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(track_frame, text="Trajectory Weight:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.traj_weight_var = tk.DoubleVar(value=self.config.trajectory_weight)
        ttk.Spinbox(track_frame, from_=0, to=1, increment=0.1, textvariable=self.traj_weight_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=2)

        ttk.Label(track_frame, text="Min Track Length (frames):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.min_track_var = tk.IntVar(value=self.config.min_track_length)
        ttk.Spinbox(track_frame, from_=1, to=1000, textvariable=self.min_track_var, width=10).grid(row=2, column=1, sticky=tk.W, pady=2)

        self.hungarian_var = tk.BooleanVar(value=self.config.use_hungarian)
        ttk.Checkbutton(track_frame, text="Use Hungarian Algorithm", variable=self.hungarian_var).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)

        # Movement filter
        movement_frame = ttk.LabelFrame(scrollable_frame, text="Movement Filter", padding=10)
        movement_frame.pack(fill=tk.X, padx=10, pady=5)

        self.filter_stationary_var = tk.BooleanVar(value=self.config.filter_stationary_tracks)
        ttk.Checkbutton(movement_frame, text="Filter Stationary Tracks", variable=self.filter_stationary_var).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=2)

        ttk.Label(movement_frame, text="Min Displacement (pixels):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.min_displacement_var = tk.DoubleVar(value=self.config.min_displacement_distance)
        ttk.Spinbox(movement_frame, from_=0, to=1000, increment=5, textvariable=self.min_displacement_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=2)

        # Nose detection
        nose_frame = ttk.LabelFrame(scrollable_frame, text="Nose Detection", padding=10)
        nose_frame.pack(fill=tk.X, padx=10, pady=5)

        self.nose_var = tk.BooleanVar(value=self.config.nose_detection_enabled)
        ttk.Checkbutton(nose_frame, text="Enable Nose Detection", variable=self.nose_var).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=2)

        ttk.Label(nose_frame, text="Smoothing Frames:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.nose_smooth_var = tk.IntVar(value=self.config.nose_smoothing_frames)
        ttk.Spinbox(nose_frame, from_=1, to=10, textvariable=self.nose_smooth_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=2)

        # Debug options
        debug_frame = ttk.LabelFrame(scrollable_frame, text="Debug Options", padding=10)
        debug_frame.pack(fill=tk.X, padx=10, pady=5)

        self.debug_images_var = tk.BooleanVar(value=self.config.save_debug_images)
        ttk.Checkbutton(debug_frame, text="Save Debug Images (shows pipeline stages)", variable=self.debug_images_var).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Label(debug_frame, text="Saves to: directory/debug_images/", font=('TkDefaultFont', 8)).grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=20)

        self.nose_enabled_var = tk.BooleanVar(value=self.config.nose_detection_enabled)
        ttk.Checkbutton(nose_frame, text="Enable Nose Detection", variable=self.nose_enabled_var).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=2)

        ttk.Label(nose_frame, text="Smoothing Frames:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.nose_smoothing_var = tk.IntVar(value=self.config.nose_smoothing_frames)
        ttk.Spinbox(nose_frame, from_=0, to=20, textvariable=self.nose_smoothing_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=2)

        ttk.Label(nose_frame, text="Min Movement Threshold:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.min_movement_var = tk.DoubleVar(value=self.config.min_movement_threshold)
        ttk.Spinbox(nose_frame, from_=0, to=50, increment=0.5, textvariable=self.min_movement_var, width=10).grid(row=2, column=1, sticky=tk.W, pady=2)

        # Buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="Apply Configuration", command=self.apply_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_config).pack(side=tk.LEFT, padx=5)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def setup_batch_tab(self):
        """Setup batch processing controls"""
        dir_frame = ttk.LabelFrame(self.batch_frame, text="Directory Management", padding=10)
        dir_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        button_container = ttk.Frame(dir_frame)
        button_container.pack(fill=tk.X, pady=5)

        ttk.Button(button_container, text="Add Directories", command=self.add_directories).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_container, text="Remove Selected", command=self.remove_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_container, text="Clear All", command=self.clear_directories).pack(side=tk.LEFT, padx=5)

        list_frame = ttk.Frame(dir_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.dir_listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED, yscrollcommand=scrollbar.set)
        self.dir_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.dir_listbox.yview)

        process_frame = ttk.LabelFrame(self.batch_frame, text="Processing", padding=10)
        process_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(process_frame, text="Start Processing", command=self.start_processing).pack(side=tk.LEFT, padx=5)
        ttk.Button(process_frame, text="Stop Processing", command=self.stop_processing).pack(side=tk.LEFT, padx=5)

        progress_frame = ttk.Frame(process_frame)
        progress_frame.pack(fill=tk.X, pady=10)

        ttk.Label(progress_frame, text="Overall Progress:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.overall_progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.overall_progress.grid(row=0, column=1, sticky=tk.EW, pady=2, padx=5)

        ttk.Label(progress_frame, text="Current Directory:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.current_progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.current_progress.grid(row=1, column=1, sticky=tk.EW, pady=2, padx=5)

        progress_frame.columnconfigure(1, weight=1)

        self.status_label = ttk.Label(process_frame, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(fill=tk.X, pady=5)

        self.current_dir_label = ttk.Label(process_frame, text="", relief=tk.SUNKEN)
        self.current_dir_label.pack(fill=tk.X, pady=2)

    def setup_results_tab(self):
        """Setup results display"""
        summary_frame = ttk.LabelFrame(self.results_frame, text="Summary", padding=10)
        summary_frame.pack(fill=tk.X, padx=10, pady=5)

        self.results_summary_label = ttk.Label(summary_frame, text="No processing results yet")
        self.results_summary_label.pack()

        details_frame = ttk.LabelFrame(self.results_frame, text="Detailed Results", padding=10)
        details_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        scrollbar = ttk.Scrollbar(details_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.results_text = scrolledtext.ScrolledText(details_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.results_text.yview)

        button_frame = ttk.Frame(self.results_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(button_frame, text="Clear Results", command=self.clear_results).pack(side=tk.LEFT, padx=5)

    def apply_config(self):
        """Apply configuration"""
        try:
            self.config.rolling_ball_radius = self.rb_radius_var.get()
            self.config.use_max_projection = self.max_proj_var.get()
            self.config.threshold_min = self.threshold_min_var.get()
            self.config.threshold_max = self.threshold_max_var.get()
            self.config.min_blob_size = self.min_blob_var.get()
            self.config.max_blob_size = self.max_blob_var.get()
            self.config.max_distance = self.max_distance_var.get()
            self.config.trajectory_weight = self.traj_weight_var.get()
            self.config.min_track_length = self.min_track_var.get()
            self.config.use_hungarian = self.hungarian_var.get()
            self.config.filter_stationary_tracks = self.filter_stationary_var.get()
            self.config.min_displacement_distance = self.min_displacement_var.get()
            self.config.nose_detection_enabled = self.nose_enabled_var.get()
            self.config.nose_smoothing_frames = self.nose_smoothing_var.get()
            self.config.min_movement_threshold = self.min_movement_var.get()
            self.config.save_debug_images = self.debug_images_var.get()

            messagebox.showinfo("Success", "Configuration applied")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply configuration: {e}")

    def reset_config(self):
        """Reset to defaults"""
        self.config = BatchConfig()
        self.update_gui_from_config()
        messagebox.showinfo("Reset", "Configuration reset to defaults")

    def update_gui_from_config(self):
        """Update GUI from config"""
        self.rb_radius_var.set(self.config.rolling_ball_radius)
        self.max_proj_var.set(self.config.use_max_projection)
        self.threshold_min_var.set(self.config.threshold_min)
        self.threshold_max_var.set(self.config.threshold_max)
        self.min_blob_var.set(self.config.min_blob_size)
        self.max_blob_var.set(self.config.max_blob_size)
        self.max_distance_var.set(self.config.max_distance)
        self.traj_weight_var.set(self.config.trajectory_weight)
        self.min_track_var.set(self.config.min_track_length)
        self.hungarian_var.set(self.config.use_hungarian)
        self.filter_stationary_var.set(self.config.filter_stationary_tracks)
        self.min_displacement_var.set(self.config.min_displacement_distance)
        self.nose_enabled_var.set(self.config.nose_detection_enabled)
        self.nose_smoothing_var.set(self.config.nose_smoothing_frames)
        self.min_movement_var.set(self.config.min_movement_threshold)
        self.debug_images_var.set(self.config.save_debug_images)

    def add_directories(self):
        """Add directories"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Directories")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Choose how to add directories:", font=('TkDefaultFont', 10, 'bold')).pack(pady=10)

        def add_single():
            directory = filedialog.askdirectory(title="Select directory")
            if directory and directory not in self.directories:
                self.directories.append(directory)
                self.dir_listbox.insert(tk.END, directory)
            dialog.destroy()

        def add_multiple():
            parent_dir = filedialog.askdirectory(title="Select parent directory")
            if parent_dir:
                subdirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir)
                          if os.path.isdir(os.path.join(parent_dir, d))]
                
                new_dirs = [d for d in subdirs if d not in self.directories]
                
                if new_dirs:
                    self.directories.extend(new_dirs)
                    for d in new_dirs:
                        self.dir_listbox.insert(tk.END, d)
                    messagebox.showinfo("Success", f"Added {len(new_dirs)} directories")
                else:
                    messagebox.showinfo("Info", "No new directories")
            dialog.destroy()

        ttk.Button(dialog, text="Add Single Directory", command=add_single, width=30).pack(pady=5)
        ttk.Button(dialog, text="Add All Subdirectories", command=add_multiple, width=30).pack(pady=5)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy, width=30).pack(pady=5)

    def remove_selected(self):
        """Remove selected"""
        selected_indices = self.dir_listbox.curselection()
        for index in reversed(selected_indices):
            self.dir_listbox.delete(index)
            del self.directories[index]

    def clear_directories(self):
        """Clear all"""
        self.directories.clear()
        self.dir_listbox.delete(0, tk.END)

    def update_status(self, message: str):
        """Update status"""
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def update_current_dir_status(self, message: str):
        """Update current directory status"""
        self.current_dir_label.config(text=message)
        self.root.update_idletasks()

    def start_processing(self):
        """Start processing"""
        if not self.directories:
            messagebox.showwarning("No Directories", "Add directories first")
            return

        if self.processing:
            messagebox.showwarning("Processing", "Already processing")
            return

        self.apply_config()
        self.results.clear()
        self.results_text.delete(1.0, tk.END)

        self.processing = True
        self.process_thread = threading.Thread(target=self.process_directories, daemon=True)
        self.process_thread.start()

    def stop_processing(self):
        """Stop processing"""
        self.processing = False
        self.update_status("Stopping...")

    def process_directories(self):
        """Process all directories"""
        tracker = BatchWormTracker(self.config)
        total_dirs = len(self.directories)

        for idx, directory in enumerate(self.directories):
            if not self.processing:
                break

            self.overall_progress.config(value=(idx / total_dirs) * 100)
            self.update_status(f"Processing {idx+1}/{total_dirs}")
            self.update_current_dir_status(f"Current: {os.path.basename(directory)}")

            def update_current_progress(current, total):
                if self.processing:
                    self.current_progress.config(value=(current / total) * 100)
                    self.root.update_idletasks()

            result = tracker.process_directory(directory, progress_callback=update_current_progress)
            self.results.append(result)
            self.display_result(result)
            self.current_progress.config(value=0)

        self.overall_progress.config(value=100)
        self.update_status("Complete")
        self.update_current_dir_status("")
        self.processing = False
        self.update_results_summary()
        self.notebook.select(self.results_frame)

    def display_result(self, result: ProcessingResult):
        """Display result"""
        self.results_text.insert(tk.END, f"\n{'='*60}\n")
        self.results_text.insert(tk.END, f"Directory: {os.path.basename(result.directory)}\n")
        self.results_text.insert(tk.END, f"Status: {'SUCCESS' if result.success else 'FAILED'}\n")

        if result.success:
            self.results_text.insert(tk.END, f"Images: {result.num_images}\n")
            self.results_text.insert(tk.END, f"Tracks: {result.num_accepted_tracks}\n")
            self.results_text.insert(tk.END, f"Quality: {result.quality_flag.upper()}\n")
            self.results_text.insert(tk.END, f"Time: {result.processing_time:.1f}s\n")
        else:
            self.results_text.insert(tk.END, f"Error: {result.error_message}\n")

        self.results_text.see(tk.END)
        self.root.update_idletasks()

    def update_results_summary(self):
        """Update summary"""
        if not self.results:
            self.results_summary_label.config(text="No results yet")
            return

        successful = len([r for r in self.results if r.success])
        failed = len([r for r in self.results if not r.success])
        total_tracks = sum(r.num_accepted_tracks for r in self.results if r.success)

        summary = f"Processed: {len(self.results)} | Success: {successful} | Failed: {failed} | Tracks: {total_tracks}"
        self.results_summary_label.config(text=summary)

    def clear_results(self):
        """Clear results"""
        self.results.clear()
        self.results_text.delete(1.0, tk.END)
        self.overall_progress.config(value=0)
        self.current_progress.config(value=0)
        self.update_status("Results cleared")
        self.update_current_dir_status("")
        self.results_summary_label.config(text="No results yet")

    def run(self):
        """Run GUI"""
        self.root.mainloop()


def main():
    """Main function"""
    print("=" * 70)
    print("BATCH WORM TRACKER - TWO-STAGE BACKGROUND SUBTRACTION")
    print("Rolling ball + max projection for robust optogenetics tracking")
    print("=" * 70)
    print("\nFEATURES:")
    print("• Stage 1: Rolling ball removes spatial lighting gradients")
    print("• Stage 2: Max projection identifies empty plate background")
    print("• Handles any lighting protocol automatically")
    print("• No epoch boundaries or timing parameters needed")
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
