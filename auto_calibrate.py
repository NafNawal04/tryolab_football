#!/usr/bin/env python3
"""
Automatic calibration module for pixel-to-meters conversion.

This module automatically detects field dimensions in soccer videos and
calculates the pixels-to-meters conversion factor without user intervention.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import hashlib

# Standard soccer field dimensions (in meters)
STANDARD_FIELD_DIMENSIONS = {
    "penalty_area_length": 16.5,  # Length of penalty area
    "penalty_area_width": 40.32,  # Width of penalty area
    "goal_area_length": 5.5,      # Length of goal area (6-yard box)
    "goal_area_width": 18.32,     # Width of goal area
    "center_circle_radius": 9.15, # Center circle radius
    "field_length": 105.0,        # Full field length
    "field_width": 68.0,          # Full field width
}


class AutoCalibrator:
    """Automatically calibrate pixel-to-meters conversion from video frames."""
    
    def __init__(self, cache_dir: str = ".calibration_cache"):
        """
        Initialize auto calibrator.
        
        Parameters
        ----------
        cache_dir : str
            Directory to cache calibration results
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_video_hash(self, video_path: str) -> str:
        """Generate hash for video file for caching."""
        path = Path(video_path)
        if not path.exists():
            return None
        
        # Use file path and size for hash (faster than reading entire file)
        stat = path.stat()
        content = f"{str(path.absolute())}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_cached_calibration(self, video_path: str) -> Optional[float]:
        """Load cached calibration result if available."""
        video_hash = self._get_video_hash(video_path)
        if not video_hash:
            return None
        
        cache_file = self.cache_dir / f"{video_hash}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    return data.get('pixels_to_meters')
            except Exception:
                return None
        return None
    
    def _save_calibration(self, video_path: str, pixels_to_meters: float):
        """Save calibration result to cache."""
        video_hash = self._get_video_hash(video_path)
        if not video_hash:
            return
        
        cache_file = self.cache_dir / f"{video_hash}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'video_path': str(video_path),
                    'pixels_to_meters': pixels_to_meters,
                }, f, indent=2)
        except Exception:
            pass
    
    def _extract_frame(self, video_path: str, frame_number: int = None) -> Optional[np.ndarray]:
        """Extract a frame from video."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Try multiple frames if frame_number not specified
        if frame_number is None:
            # Try frames at 25%, 50%, 75% of video
            frames_to_try = [
                int(total_frames * 0.25),
                int(total_frames * 0.50),
                int(total_frames * 0.75),
            ]
        else:
            frames_to_try = [frame_number]
        
        for frame_num in frames_to_try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.release()
                return frame
        
        cap.release()
        return None
    
    def _detect_field_lines(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect field lines using edge detection and Hough line transform.
        
        Returns
        -------
        Tuple of (horizontal_lines, vertical_lines)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection using Canny
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=50,
            maxLineGap=10
        )
        
        if lines is None or len(lines) == 0:
            return np.array([]), np.array([])
        
        # Separate horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Horizontal lines (angle close to 0 or 180)
            if angle < 15 or angle > 165:
                horizontal_lines.append(line[0])
            # Vertical lines (angle close to 90)
            elif 75 < angle < 105:
                vertical_lines.append(line[0])
        
        return np.array(horizontal_lines), np.array(vertical_lines)
    
    def _measure_distance_between_parallel_lines(
        self, lines: np.ndarray, is_horizontal: bool = True
    ) -> Optional[float]:
        """
        Measure distance between parallel lines (e.g., penalty area boundaries).
        
        Parameters
        ----------
        lines : np.ndarray
            Array of lines
        is_horizontal : bool
            Whether lines are horizontal or vertical
            
        Returns
        -------
        Average distance between parallel lines in pixels
        """
        if len(lines) < 2:
            return None
        
        distances = []
        
        if is_horizontal:
            # For horizontal lines, measure vertical distance
            # Get y-coordinates (average of y1 and y2)
            y_coords = []
            for line in lines:
                y1, y2 = line[1], line[3]
                y_coords.append((y1 + y2) / 2)
            
            y_coords = sorted(y_coords)
            
            # Measure distances between consecutive lines
            for i in range(len(y_coords) - 1):
                dist = abs(y_coords[i+1] - y_coords[i])
                # Filter out very small distances (noise)
                if dist > 20:  # At least 20 pixels
                    distances.append(dist)
        else:
            # For vertical lines, measure horizontal distance
            x_coords = []
            for line in lines:
                x1, x2 = line[0], line[2]
                x_coords.append((x1 + x2) / 2)
            
            x_coords = sorted(x_coords)
            
            for i in range(len(x_coords) - 1):
                dist = abs(x_coords[i+1] - x_coords[i])
                if dist > 20:
                    distances.append(dist)
        
        if not distances:
            return None
        
        # Return median distance (more robust than mean)
        return np.median(distances)
    
    def _estimate_field_dimensions(self, frame: np.ndarray) -> Optional[float]:
        """
        Estimate pixel-to-meters conversion by detecting field markings.
        
        This uses heuristics to identify common field features like:
        - Penalty area boundaries
        - Goal area boundaries
        - Center circle
        
        Returns
        -------
        Estimated pixels_to_meters conversion factor, or None if failed
        """
        height, width = frame.shape[:2]
        
        # Detect field lines
        horizontal_lines, vertical_lines = self._detect_field_lines(frame)
        
        conversion_factors = []
        
        # Method 1: Try to identify penalty area length (horizontal lines)
        # Penalty area is 16.5m, usually visible as a clear horizontal line
        if len(horizontal_lines) >= 2:
            # Get all horizontal line distances
            y_coords = []
            for line in horizontal_lines:
                y1, y2 = line[1], line[3]
                y_coords.append((y1 + y2) / 2)
            
            y_coords = sorted(set(y_coords))  # Remove duplicates
            
            # Find distances that could be penalty area (16.5m)
            # Penalty area should be a significant distance, typically 100-400 pixels
            for i in range(len(y_coords) - 1):
                for j in range(i + 1, len(y_coords)):
                    dist = abs(y_coords[j] - y_coords[i])
                    if 100 < dist < 500:  # Reasonable range for penalty area
                        # Could be penalty area (16.5m) or goal area (5.5m)
                        # Try both and see which is more reasonable
                        penalty_factor = 16.5 / dist
                        goal_factor = 5.5 / dist
                        
                        # Check if this would give reasonable field dimensions
                        field_length_px = width
                        field_width_px = height
                        
                        # Penalty area should be about 15-20% of field length
                        if 0.01 < penalty_factor < 0.1:
                            field_length_m = field_length_px * penalty_factor
                            if 80 < field_length_m < 120:  # Reasonable field length
                                conversion_factors.append(penalty_factor)
                        
                        # Goal area should be about 5-6% of field length
                        if 0.01 < goal_factor < 0.1:
                            field_length_m = field_length_px * goal_factor
                            if 70 < field_length_m < 130:
                                conversion_factors.append(goal_factor * 3)  # Scale up (5.5m -> 16.5m ratio)
        
        # Method 2: Try vertical lines for field width
        if len(vertical_lines) >= 2:
            x_coords = []
            for line in vertical_lines:
                x1, x2 = line[0], line[2]
                x_coords.append((x1 + x2) / 2)
            
            x_coords = sorted(set(x_coords))
            
            for i in range(len(x_coords) - 1):
                for j in range(i + 1, len(x_coords)):
                    dist = abs(x_coords[j] - x_coords[i])
                    if 50 < dist < 300:  # Reasonable range
                        # Could be penalty area width (40.32m) or goal area width (18.32m)
                        penalty_width_factor = 40.32 / dist
                        goal_width_factor = 18.32 / dist
                        
                        field_width_px = width
                        field_length_px = height
                        
                        if 0.01 < penalty_width_factor < 0.1:
                            field_width_m = field_width_px * penalty_width_factor
                            if 50 < field_width_m < 80:
                                conversion_factors.append(penalty_width_factor)
                        
                        if 0.01 < goal_width_factor < 0.1:
                            field_width_m = field_width_px * goal_width_factor
                            if 50 < field_width_m < 85:
                                # Scale to match penalty area ratio
                                conversion_factors.append(goal_width_factor * 2.2)
        
        # Use median of all reasonable conversion factors
        if conversion_factors:
            # Filter outliers (more than 2x median or less than 0.5x median)
            median_factor = np.median(conversion_factors)
            filtered_factors = [
                f for f in conversion_factors
                if median_factor * 0.5 < f < median_factor * 2.0
            ]
            
            if filtered_factors:
                return np.median(filtered_factors)
        
        # Fallback: Use frame dimensions as rough estimate
        # IMPORTANT: Stabilized coordinates might be in a different scale than original frame
        # The motion compensation might crop/scale the image, so field dimensions are uncertain
        # Use a VERY conservative estimate to avoid overestimating distances
        
        # Standard field: 105m x 68m
        # We want to estimate pixels_to_meters such that:
        # - At 30 fps, a player moving 10 px/frame should be ~0.3-0.4 m/frame (sprint speed)
        # - That means: 10 px * pixels_to_meters = 0.3-0.4 m
        # - So pixels_to_meters should be ~0.03-0.04 m/px for reasonable speeds
        
        # More conservative: Assume field is much larger in pixels than we think
        # This will give us a smaller conversion factor (safer - underestimates rather than overestimates)
        if width > height:
            # Landscape: assume field takes up most of the width, but be conservative
            # Use a larger pixel estimate to get a smaller conversion factor
            estimated_field_pixels = width * 1.2  # Assume field is larger than frame (very conservative)
        else:
            estimated_field_pixels = height * 1.2
        
        # Calculate conversion: meters per pixel
        pixels_to_meters = 105.0 / estimated_field_pixels
        
        # Cap the conversion factor to a maximum reasonable value
        # For realistic speeds (10 px/frame at 30 fps = 0.3 m/frame), we need pixels_to_meters <= 0.03
        # But we'll allow up to 0.05 as a safety margin
        max_reasonable_factor = 0.05
        if pixels_to_meters > max_reasonable_factor:
            # If factor is too large, use a fixed conservative estimate
            # Assume field is at least 2500 pixels wide (very conservative)
            estimated_field_pixels = max(width, height) * 1.5
            pixels_to_meters = 105.0 / estimated_field_pixels
            
            # Final cap: never exceed 0.05 m/px
            if pixels_to_meters > max_reasonable_factor:
                pixels_to_meters = max_reasonable_factor
        
        # Ensure minimum reasonable value (field shouldn't be smaller than 500 pixels)
        min_reasonable_factor = 105.0 / 5000  # ~0.021 m/px
        if pixels_to_meters < min_reasonable_factor:
            pixels_to_meters = min_reasonable_factor
        
        return pixels_to_meters
    
    def calibrate(self, video_path: str, frame_number: int = None, verbose: bool = True) -> Optional[float]:
        """
        Automatically calibrate pixel-to-meters conversion for a video.
        
        Parameters
        ----------
        video_path : str
            Path to video file
        frame_number : int, optional
            Specific frame number to use (default: tries multiple frames)
        verbose : bool, optional
            Whether to print calibration progress (default: True)
            
        Returns
        -------
        pixels_to_meters conversion factor, or None if calibration failed
        """
        video_path = Path(video_path)
        
        # Check cache first
        cached = self._load_cached_calibration(str(video_path))
        if cached is not None:
            if verbose:
                print(f"✓ Using cached calibration: {cached:.6f} m/px")
            return cached
        
        if verbose:
            print("Calibrating pixel-to-meters conversion...")
        
        # Try multiple frames for better accuracy
        conversion_factors = []
        
        # Extract frames at different positions in the video
        frame_numbers_to_try = []
        if frame_number is not None:
            frame_numbers_to_try = [frame_number]
        else:
            # Try frames at 20%, 40%, 60%, 80% of video
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if total_frames > 0:
                    frame_numbers_to_try = [
                        int(total_frames * 0.2),
                        int(total_frames * 0.4),
                        int(total_frames * 0.6),
                        int(total_frames * 0.8),
                    ]
        
        for fn in frame_numbers_to_try:
            frame = self._extract_frame(str(video_path), fn)
            if frame is None:
                continue
            
            # Try to estimate conversion factor from this frame
            factor = self._estimate_field_dimensions(frame)
            if factor is not None and factor > 0:
                conversion_factors.append(factor)
        
        # Use median of all conversion factors for robustness
        if conversion_factors:
            # Filter outliers
            median_factor = np.median(conversion_factors)
            filtered_factors = [
                f for f in conversion_factors
                if median_factor * 0.7 < f < median_factor * 1.3
                # Also validate that factor is in reasonable range (much stricter)
                and 0.01 < f < 0.05  # Very strict: 0.01 to 0.05 m/px only
            ]
            
            if filtered_factors:
                pixels_to_meters = np.median(filtered_factors)
            else:
                # If median is outside reasonable range, use fallback
                if 0.01 < median_factor < 0.05:
                    pixels_to_meters = median_factor
                else:
                    # Fallback to frame-based estimation
                    pixels_to_meters = None
        else:
            # Fallback: extract one frame and use fallback estimation
            frame = self._extract_frame(str(video_path), frame_numbers_to_try[0] if frame_numbers_to_try else None)
            if frame is None:
                if verbose:
                    print("✗ Failed to extract frame from video")
                return None
            
            if verbose:
                print("  Using fallback estimation based on frame size...")
            
            height, width = frame.shape[:2]
            
            # Very conservative estimate: assume field is larger in pixels than we think
            # This gives us a smaller conversion factor (safer)
            if width > height:
                estimated_field_pixels = width * 1.2
            else:
                estimated_field_pixels = height * 1.2
            
            pixels_to_meters = 105.0 / estimated_field_pixels
            
            # Cap to maximum reasonable value (0.05 m/px)
            max_reasonable_factor = 0.05
            if pixels_to_meters > max_reasonable_factor:
                if verbose:
                    print(f"  Warning: Calculated conversion factor ({pixels_to_meters:.6f}) seems too large.")
                    print(f"  Adjusting to more conservative estimate...")
                # Use even more conservative estimate
                estimated_field_pixels = max(width, height) * 1.5
                pixels_to_meters = 105.0 / estimated_field_pixels
                
                # Final cap
                if pixels_to_meters > max_reasonable_factor:
                    pixels_to_meters = max_reasonable_factor
            
            # Ensure minimum
            min_reasonable_factor = 105.0 / 5000
            if pixels_to_meters < min_reasonable_factor:
                pixels_to_meters = min_reasonable_factor
        
        # Final validation: Ensure conversion factor is reasonable
        if pixels_to_meters is not None:
            # Stricter validation: factor should be between 0.01 and 0.05 m/px
            if pixels_to_meters > 0.05:
                if verbose:
                    print(f"  Warning: Conversion factor ({pixels_to_meters:.6f}) is too large.")
                    print(f"  Capping to 0.05 m/px for safety.")
                pixels_to_meters = 0.05
            elif pixels_to_meters < 0.01:
                if verbose:
                    print(f"  Warning: Conversion factor ({pixels_to_meters:.6f}) is too small.")
                    print(f"  Setting to 0.01 m/px for safety.")
                pixels_to_meters = 0.01
            
            # Cache the result
            self._save_calibration(str(video_path), pixels_to_meters)
            
            if verbose:
                print(f"✓ Calibration complete: {pixels_to_meters:.6f} m/px")
                print(f"  Estimated: 1 pixel ≈ {pixels_to_meters*100:.2f} cm")
                # Calculate realistic speed example
                example_pixels_per_frame = 10
                example_fps = 30
                example_speed = pixels_to_meters * example_pixels_per_frame * example_fps
                print(f"  (At {example_fps} fps, {example_pixels_per_frame} px/frame ≈ {example_speed:.1f} m/s)")
                if example_speed > 15:
                    print(f"  ⚠ Warning: This might overestimate distances. Consider manual calibration.")
                print(f"  (Cached for future runs)")
        
        return pixels_to_meters


def auto_calibrate(video_path: str, frame_number: int = None, verbose: bool = True) -> Optional[float]:
    """
    Convenience function to auto-calibrate a video.
    
    Parameters
    ----------
    video_path : str
        Path to video file
    frame_number : int, optional
        Specific frame number to use
    verbose : bool, optional
        Whether to print calibration progress (default: True)
        
    Returns
    -------
    pixels_to_meters conversion factor, or None if failed
    """
    calibrator = AutoCalibrator()
    return calibrator.calibrate(video_path, frame_number, verbose=verbose)

