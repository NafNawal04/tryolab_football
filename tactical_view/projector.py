from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from soccer.player import Player
from tactical_view.homography import Homography


@dataclass
class TacticalProjection:
    """Container storing tactical-view coordinates for a player."""

    player_id: Optional[int]
    team_color: Tuple[int, int, int]
    position: np.ndarray


def _order_points(points: np.ndarray) -> np.ndarray:
    """Return the 4 points ordered as top-left, top-right, bottom-left, bottom-right."""
    pts = points.astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[3] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(diff)]

    return rect


def _detect_pitch_corners(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect four pitch corner points using improved green mask + contour search.
    Uses multiple HSV ranges and validation to improve robustness.
    Returns points ordered TL, TR, BL, BR in image coordinates.
    """
    if frame is None or frame.size == 0:
        return None

    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Try multiple HSV ranges for different lighting conditions
    green_ranges = [
        # Standard green range
        (np.array([30, 30, 30]), np.array([90, 255, 255])),
        # Brighter/darker conditions
        (np.array([35, 40, 40]), np.array([85, 255, 255])),
        # More saturated greens
        (np.array([40, 50, 50]), np.array([80, 255, 255])),
        # Wider range for artificial lighting
        (np.array([25, 20, 20]), np.array([95, 255, 255])),
    ]
    
    best_corners = None
    best_score = 0
    
    for lower_green, upper_green in green_ranges:
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Improved morphological operations
        kernel_size = max(5, min(w, h) // 100)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Remove small noise
        min_area = (w * h) * 0.01  # At least 1% of frame
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        
        # Filter contours by area
        large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        if not large_contours:
            continue
        
        # Try the largest contour first
        largest = max(large_contours, key=cv2.contourArea)
        
        # Validate that it's a reasonable size (should be a significant portion of the frame)
        contour_area = cv2.contourArea(largest)
        frame_area = w * h
        if contour_area < frame_area * 0.05:  # At least 5% of frame
            continue
        
        hull = cv2.convexHull(largest)
        
        # Try to get 4 corners with adaptive epsilon
        for epsilon_factor in [0.01, 0.02, 0.03, 0.05]:
            epsilon = epsilon_factor * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            
            if len(approx) == 4:
                corners = approx.reshape(-1, 2).astype(np.float32)
                # Validate corners
                if _validate_corners(corners, w, h):
                    ordered = _order_points(corners)
                    score = _score_corners(ordered, w, h)
                    if score > best_score:
                        best_score = score
                        best_corners = ordered
                break
            elif len(approx) > 4:
                # Too many points, try to get bounding rectangle
                rect = cv2.minAreaRect(approx)
                box_points = cv2.boxPoints(rect)
                corners = box_points.astype(np.float32)
                if _validate_corners(corners, w, h):
                    ordered = _order_points(corners)
                    score = _score_corners(ordered, w, h)
                    if score > best_score:
                        best_score = score
                        best_corners = ordered
                break
        
        # If we got 4 corners, try this result
        if best_corners is not None and best_score > 0.5:
            break
    
    return best_corners


def _validate_corners(corners: np.ndarray, img_width: int, img_height: int) -> bool:
    """Validate that detected corners are reasonable."""
    if corners is None or len(corners) != 4:
        return False
    
    # Check that corners are within image bounds (with some margin)
    margin = 50
    for corner in corners:
        x, y = corner
        if x < -margin or x > img_width + margin or y < -margin or y > img_height + margin:
            return False
    
    # Check that corners form a reasonable quadrilateral
    # Calculate area of the quadrilateral
    area = cv2.contourArea(corners)
    if area < (img_width * img_height) * 0.1:  # At least 10% of frame
        return False
    
    # Check aspect ratio (should be roughly rectangular, not too skewed)
    # Calculate bounding box
    x_coords = corners[:, 0]
    y_coords = corners[:, 1]
    width = np.max(x_coords) - np.min(x_coords)
    height = np.max(y_coords) - np.min(y_coords)
    
    if width < img_width * 0.2 or height < img_height * 0.2:
        return False
    
    # Aspect ratio should be reasonable (not too extreme)
    aspect_ratio = width / height if height > 0 else 0
    if aspect_ratio < 0.3 or aspect_ratio > 3.0:
        return False
    
    return True


def _score_corners(corners: np.ndarray, img_width: int, img_height: int) -> float:
    """Score corners based on how well they represent a pitch."""
    if corners is None or len(corners) != 4:
        return 0.0
    
    # Higher score for corners that:
    # 1. Cover a larger area of the frame
    area = cv2.contourArea(corners)
    frame_area = img_width * img_height
    area_score = min(area / frame_area, 1.0)
    
    # 2. Are well-distributed (not all clustered)
    x_coords = corners[:, 0]
    y_coords = corners[:, 1]
    width = np.max(x_coords) - np.min(x_coords)
    height = np.max(y_coords) - np.min(y_coords)
    
    coverage_score = (width / img_width) * (height / img_height)
    
    # 3. Form a reasonable aspect ratio
    aspect_ratio = width / height if height > 0 else 0
    if 0.5 <= aspect_ratio <= 2.0:
        aspect_score = 1.0
    else:
        aspect_score = max(0.0, 1.0 - abs(aspect_ratio - 1.5) / 1.5)
    
    return (area_score * 0.4 + coverage_score * 0.4 + aspect_score * 0.2)


class TacticalViewProjector:
    """
    Build a tactical top-down projection of the broadcast footage using a homography
    derived from automatically detected pitch corners (no manual annotation required).
    """

    def __init__(self, width: int = 400, height: int = 240, initialization_frames: int = 5) -> None:
        self.width = width
        self.height = height
        self._homography: Optional[Homography] = None
        self._ready = False
        self._initialization_frames = initialization_frames
        self._frame_buffer: List[np.ndarray] = []
        self._corner_buffer: List[np.ndarray] = []
        self._initialization_attempts = 0
        self._max_initialization_attempts = 30  # Try for 30 frames before giving up

    @property
    def ready(self) -> bool:
        return self._ready and self._homography is not None

    def try_initialize(self, frame: np.ndarray) -> bool:
        """
        Detect pitch corners on the provided frame and build the homography once.
        Uses frame averaging for more stable detection.
        """
        if self.ready:
            return True

        if frame is None or frame.size == 0:
            return False

        # Try to detect corners
        corners = _detect_pitch_corners(frame)
        
        if corners is not None:
            # Store frame and corners for averaging
            self._frame_buffer.append(frame.copy())
            self._corner_buffer.append(corners.copy())
            
            # Keep only recent frames
            if len(self._frame_buffer) > self._initialization_frames:
                self._frame_buffer.pop(0)
                self._corner_buffer.pop(0)
        
        self._initialization_attempts += 1
        
        # Need at least a few successful detections before averaging
        if len(self._corner_buffer) < max(2, self._initialization_frames // 2):
            # If we've tried many times and still can't get enough detections, use what we have
            if self._initialization_attempts >= self._max_initialization_attempts and len(self._corner_buffer) > 0:
                corners = np.mean(self._corner_buffer, axis=0)
            else:
                return False
        else:
            # Average the corners for stability
            corners = np.mean(self._corner_buffer, axis=0)

        target = np.array(
            [
                [0, 0],
                [self.width, 0],
                [0, self.height],
                [self.width, self.height],
            ],
            dtype=np.float32,
        )

        try:
            self._homography = Homography(corners, target)
            # Validate homography by testing transformed points
            transformed = self._homography.transform_points(corners)
            # Check if transformed points are reasonable
            if np.any(transformed < -self.width) or np.any(transformed > self.width * 2):
                return False
            if np.any(transformed[:, 1] < -self.height) or np.any(transformed[:, 1] > self.height * 2):
                return False
        except (ValueError, Exception):
            return False

        self._ready = True
        return True

    def project_players(self, players: Sequence[Player]) -> List[TacticalProjection]:
        """Project current player centers onto the tactical view."""
        if not self.ready or not players:
            return []

        projections: List[TacticalProjection] = []

        for player in players:
            # Try center_abs first (absolute coordinates), fallback to center (stabilized)
            center = None
            if hasattr(player, 'center_abs') and player.center_abs is not None:
                try:
                    center = player.center_abs
                    if not isinstance(center, np.ndarray) or center.size < 2:
                        center = None
                except (AttributeError, TypeError):
                    pass
            
            if center is None and hasattr(player, 'center') and player.center is not None:
                try:
                    center = player.center
                    if not isinstance(center, np.ndarray) or center.size < 2:
                        center = None
                except (AttributeError, TypeError):
                    pass
            
            if center is None:
                continue

            try:
                # Ensure center is 2D
                if center.ndim == 1 and len(center) >= 2:
                    center = center[:2]
                elif center.ndim > 1:
                    center = center.reshape(-1)[:2]
                else:
                    continue

                transformed = self._homography.transform_points(
                    np.array([center], dtype=np.float32)
                )[0]

                # Validate transformed position
                if np.any(np.isnan(transformed)) or np.any(np.isinf(transformed)):
                    continue

                team_color = (255, 255, 255)
                if player.team and hasattr(player.team, 'color') and player.team.color:
                    # Convert RGB (drawing helpers) to BGR for OpenCV rendering
                    try:
                        r, g, b = player.team.color[:3]
                        team_color = (int(b), int(g), int(r))
                    except (TypeError, IndexError):
                        pass

                projections.append(
                    TacticalProjection(
                        player_id=getattr(player, 'player_id', None),
                        team_color=team_color,
                        position=transformed,
                    )
                )
            except (ValueError, IndexError, TypeError):
                # Skip this player if projection fails
                continue

        return projections

    def render_view(self, frame: np.ndarray, players: Sequence[Player]) -> Optional[np.ndarray]:
        """Render a tactical overview image with projected player dots."""
        if frame is None or frame.size == 0:
            return None

        # Try to initialize if not ready
        if not self.ready:
            if not self.try_initialize(frame=frame):
                return None
        
        if not self.ready:
            return None

        try:
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            canvas[:, :] = (40, 120, 40)

            # Draw simple pitch markings
            cv2.rectangle(canvas, (5, 5), (self.width - 5, self.height - 5), (255, 255, 255), 1)
            cv2.line(
                canvas,
                (self.width // 2, 5),
                (self.width // 2, self.height - 5),
                (255, 255, 255),
                1,
            )
            cv2.circle(
                canvas,
                (self.width // 2, self.height // 2),
                30,
                (255, 255, 255),
                1,
            )
            cv2.circle(
                canvas,
                (self.width // 2, self.height // 2),
                3,
                (255, 255, 255),
                -1,
            )

            # Draw penalty boxes
            box_w = int(self.width * 0.2)
            box_h = int(self.height * 0.6)
            top = (self.height - box_h) // 2
            cv2.rectangle(canvas, (5, top), (5 + box_w, top + box_h), (255, 255, 255), 1)
            cv2.rectangle(
                canvas,
                (self.width - box_w - 5, top),
                (self.width - 5, top + box_h),
                (255, 255, 255),
                1,
            )

            # Project and draw players
            if players:
                projections = self.project_players(players)
                for proj in projections:
                    try:
                        x, y = int(proj.position[0]), int(proj.position[1])
                        # Allow some margin outside bounds for players near edges
                        margin = 10
                        if x < -margin or y < -margin or x >= self.width + margin or y >= self.height + margin:
                            continue
                        # Clamp to bounds
                        x = max(0, min(self.width - 1, x))
                        y = max(0, min(self.height - 1, y))
                        
                        radius = 4
                        cv2.circle(canvas, (x, y), radius, proj.team_color, -1)
                        if proj.player_id is not None:
                            cv2.putText(
                                canvas,
                                str(proj.player_id),
                                (x + 6, y + 4),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.3,
                                (255, 255, 255),
                                1,
                                lineType=cv2.LINE_AA,
                            )
                    except (ValueError, TypeError, IndexError):
                        # Skip this projection if drawing fails
                        continue

            return canvas
        except Exception:
            # Return None if rendering fails
            return None

