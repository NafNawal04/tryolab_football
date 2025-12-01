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
    """
    Return the 4 points ordered as top-left, top-right, bottom-left, bottom-right.
    Uses a more reliable method based on centroid and angle.
    """
    pts = points.astype(np.float32)
    if len(pts) != 4:
        return pts
    
    # Calculate centroid
    centroid = np.mean(pts, axis=0)
    
    # Sort points by angle from centroid
    # This helps identify which corner is which
    angles = []
    for pt in pts:
        dx = pt[0] - centroid[0]
        dy = pt[1] - centroid[1]
        angle = np.arctan2(dy, dx)
        angles.append(angle)
    
    # Sort by angle and identify corners
    sorted_indices = np.argsort(angles)
    sorted_pts = pts[sorted_indices]
    
    # Now identify which is TL, TR, BL, BR based on position
    # Top points have smaller y, bottom points have larger y
    # Left points have smaller x, right points have larger x
    y_sorted = np.argsort(pts[:, 1])  # Sort by y coordinate
    x_sorted = np.argsort(pts[:, 0])  # Sort by x coordinate
    
    # Top-left: smallest x and smallest y
    # Top-right: largest x and smallest y
    # Bottom-left: smallest x and largest y
    # Bottom-right: largest x and largest y
    
    top_indices = y_sorted[:2]  # Two points with smallest y
    bottom_indices = y_sorted[2:]  # Two points with largest y
    
    # Find left and right among top points
    top_pts = pts[top_indices]
    top_x_sorted = np.argsort(top_pts[:, 0])
    tl_idx = top_indices[top_x_sorted[0]]
    tr_idx = top_indices[top_x_sorted[1]]
    
    # Find left and right among bottom points
    bottom_pts = pts[bottom_indices]
    bottom_x_sorted = np.argsort(bottom_pts[:, 0])
    bl_idx = bottom_indices[bottom_x_sorted[0]]
    br_idx = bottom_indices[bottom_x_sorted[1]]
    
    rect = np.array([
        pts[tl_idx],  # Top-left
        pts[tr_idx],  # Top-right
        pts[bl_idx],  # Bottom-left
        pts[br_idx],  # Bottom-right
    ], dtype=np.float32)
    
    return rect


def _detect_field_lines(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect field lines (white lines on green background) to identify the playing field.
    Returns a mask of the field region.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Detect white lines (field markings)
    # White in HSV: low saturation, high value
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Detect green field
    green_ranges = [
        (np.array([30, 30, 30]), np.array([90, 255, 255])),
        (np.array([35, 40, 40]), np.array([85, 255, 255])),
        (np.array([40, 50, 50]), np.array([80, 255, 255])),
    ]
    
    green_mask = None
    for lower_green, upper_green in green_ranges:
        mask = cv2.inRange(hsv, lower_green, upper_green)
        if green_mask is None:
            green_mask = mask.copy()
        else:
            green_mask = cv2.bitwise_or(green_mask, mask)
    
    if green_mask is None:
        return None
    
    # Combine: field should have green AND potentially white lines
    # But prioritize regions with white lines (actual field markings)
    field_mask = green_mask.copy()
    
    # Dilate white lines to connect them
    kernel = np.ones((5, 5), np.uint8)
    white_dilated = cv2.dilate(white_mask, kernel, iterations=2)
    
    # Field region should be green with white lines nearby
    # Use morphological operations to find regions with both
    field_mask = cv2.bitwise_and(field_mask, cv2.bitwise_not(white_dilated))
    field_mask = cv2.addWeighted(field_mask, 0.7, white_dilated, 0.3, 0)
    field_mask = (field_mask > 100).astype(np.uint8) * 255
    
    # Clean up the mask
    kernel = np.ones((9, 9), np.uint8)
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return field_mask


def _find_field_region_from_lines(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Find the field region by detecting field lines and using their intersections.
    This is more reliable than just green color detection.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Use HoughLinesP to detect field lines
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=100, 
        minLineLength=min(w, h) // 10, 
        maxLineGap=20
    )
    
    if lines is None or len(lines) < 4:
        return None
    
    # Find line intersections to identify field corners
    intersections = []
    for i in range(len(lines)):
        x1, y1, x2, y2 = lines[i][0]
        for j in range(i + 1, len(lines)):
            x3, y3, x4, y4 = lines[j][0]
            
            # Calculate intersection
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                continue
            
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
            
            # Check if intersection is within line segments or close to them
            if -0.5 <= t <= 1.5 and -0.5 <= u <= 1.5:
                px = x1 + t * (x2 - x1)
                py = y1 + t * (y2 - y1)
                
                # Only keep intersections within frame bounds
                if 0 <= px < w and 0 <= py < h:
                    intersections.append([px, py])
    
    if len(intersections) < 4:
        return None
    
    # Find the convex hull of intersections to get field boundary
    intersections = np.array(intersections, dtype=np.float32)
    hull = cv2.convexHull(intersections)
    
    if len(hull) < 4:
        return None
    
    # Approximate to 4 corners
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    
    if len(approx) == 4:
        return approx.reshape(-1, 2).astype(np.float32)
    elif len(approx) > 4:
        # Get minimum area rectangle
        rect = cv2.minAreaRect(approx)
        box_points = cv2.boxPoints(rect)
        return box_points.astype(np.float32)
    
    return None


def _detect_pitch_corners(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect four pitch corner points using multiple strategies:
    1. Field line detection (most reliable)
    2. Green mask with field line validation
    3. Edge-based detection
    
    Returns points ordered TL, TR, BL, BR in image coordinates.
    """
    if frame is None or frame.size == 0:
        return None

    h, w = frame.shape[:2]
    
    # Strategy 1: Use field line intersections (most reliable)
    corners = _find_field_region_from_lines(frame)
    if corners is not None and _validate_corners(corners, w, h):
        ordered = _order_points(corners)
        score = _score_corners(ordered, w, h)
        if score > 0.3:  # Lower threshold for line-based detection
            return ordered
    
    # Strategy 2: Green mask with field line validation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    field_mask = _detect_field_lines(frame)
    
    if field_mask is not None:
        # Find contours in the field mask
        contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Filter by area and position (field should be in center/lower portion)
            min_area = (w * h) * 0.1  # At least 10% of frame
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue
                # Check if contour is in reasonable position (not just top corners)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cy = int(M["m01"] / M["m00"])
                    # Field should be in lower 80% of frame
                    if cy > h * 0.2:
                        valid_contours.append((contour, area))
            
            if valid_contours:
                # Sort by area and position (prefer larger, lower contours)
                valid_contours.sort(key=lambda x: (x[1], -cv2.moments(x[0])["m01"] / cv2.moments(x[0])["m00"] if cv2.moments(x[0])["m00"] != 0 else 0), reverse=True)
                
                for contour, _ in valid_contours[:3]:  # Try top 3 candidates
                    hull = cv2.convexHull(contour)
                    
                    # Try to get 4 corners
                    for epsilon_factor in [0.01, 0.02, 0.03, 0.05]:
                        epsilon = epsilon_factor * cv2.arcLength(hull, True)
                        approx = cv2.approxPolyDP(hull, epsilon, True)
                        
                        if len(approx) == 4:
                            corners = approx.reshape(-1, 2).astype(np.float32)
                            if _validate_corners(corners, w, h):
                                ordered = _order_points(corners)
                                score = _score_corners(ordered, w, h)
                                if score > 0.4:
                                    return ordered
                        elif len(approx) > 4:
                            rect = cv2.minAreaRect(approx)
                            box_points = cv2.boxPoints(rect)
                            corners = box_points.astype(np.float32)
                            if _validate_corners(corners, w, h):
                                ordered = _order_points(corners)
                                score = _score_corners(ordered, w, h)
                                if score > 0.4:
                                    return ordered
    
    # Strategy 3: Fallback to original green mask approach (but with better filtering)
    green_ranges = [
        (np.array([30, 30, 30]), np.array([90, 255, 255])),
        (np.array([35, 40, 40]), np.array([85, 255, 255])),
        (np.array([40, 50, 50]), np.array([80, 255, 255])),
    ]
    
    best_corners = None
    best_score = 0
    
    for lower_green, upper_green in green_ranges:
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        kernel_size = max(5, min(w, h) // 100)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        min_area = (w * h) * 0.1  # Higher threshold
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        
        # Filter by area and position
        large_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            M = cv2.moments(c)
            if M["m00"] != 0:
                cy = int(M["m01"] / M["m00"])
                # Prefer contours in lower portion of frame
                if cy > h * 0.15:
                    large_contours.append((c, area, cy))
        
        if not large_contours:
            continue
        
        # Sort by position (lower is better) and area
        large_contours.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        for contour, area, _ in large_contours[:2]:  # Try top 2
            hull = cv2.convexHull(contour)
            
            for epsilon_factor in [0.01, 0.02, 0.03, 0.05]:
                epsilon = epsilon_factor * cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, epsilon, True)
                
                if len(approx) == 4:
                    corners = approx.reshape(-1, 2).astype(np.float32)
                    if _validate_corners(corners, w, h):
                        ordered = _order_points(corners)
                        score = _score_corners(ordered, w, h)
                        if score > best_score:
                            best_score = score
                            best_corners = ordered
                    break
                elif len(approx) > 4:
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
        
        if best_corners is not None and best_score > 0.5:
            break
    
    return best_corners


def _refine_corners_with_edges(frame: np.ndarray, initial_corners: np.ndarray) -> Optional[np.ndarray]:
    """
    Refine corner detection by finding actual field edge boundaries.
    This helps when the green mask finds a region that's not the actual field boundary.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use Canny edge detection to find field boundaries
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to connect nearby lines
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find lines using HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    
    if lines is None or len(lines) == 0:
        return initial_corners
    
    # Find intersection points of lines near the initial corners
    refined = []
    margin = min(w, h) * 0.1
    
    for corner in initial_corners:
        cx, cy = corner
        # Find the closest line intersection near this corner
        best_point = corner
        min_dist = float('inf')
        
        # Look for line intersections within margin of this corner
        for i, line1 in enumerate(lines):
            x1, y1, x2, y2 = line1[0]
            for j, line2 in enumerate(lines[i+1:], i+1):
                x3, y3, x4, y4 = line2[0]
                
                # Calculate intersection
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) < 1e-6:
                    continue
                
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
                u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
                
                if 0 <= t <= 1 and 0 <= u <= 1:
                    px = x1 + t * (x2 - x1)
                    py = y1 + t * (y2 - y1)
                    
                    # Check if intersection is near the corner
                    dist = np.sqrt((px - cx)**2 + (py - cy)**2)
                    if dist < margin and dist < min_dist:
                        min_dist = dist
                        best_point = np.array([px, py], dtype=np.float32)
        
        refined.append(best_point)
    
    if len(refined) == 4:
        return np.array(refined, dtype=np.float32)
    
    return initial_corners


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
            
            # Check if transformed points are reasonable (should map to target corners)
            # Allow some tolerance for numerical errors
            tolerance = 5
            expected = target
            for i in range(4):
                dist = np.linalg.norm(transformed[i] - expected[i])
                if dist > tolerance:
                    # Homography doesn't map corners correctly
                    return False
            
            # Additional validation: check if homography is well-conditioned
            # by testing a few points in the middle of the field
            test_points = np.array([
                [corners[0][0] + (corners[1][0] - corners[0][0]) * 0.5, 
                 corners[0][1] + (corners[2][1] - corners[0][1]) * 0.5],  # Center
                [corners[0][0] + (corners[1][0] - corners[0][0]) * 0.25, 
                 corners[0][1] + (corners[2][1] - corners[0][1]) * 0.25],  # Quarter point
            ], dtype=np.float32)
            
            test_transformed = self._homography.transform_points(test_points)
            # Check that transformed points are within reasonable bounds
            for pt in test_transformed:
                if pt[0] < -self.width * 0.5 or pt[0] > self.width * 1.5:
                    return False
                if pt[1] < -self.height * 0.5 or pt[1] > self.height * 1.5:
                    return False
            
        except (ValueError, Exception):
            return False

        self._ready = True
        return True

    def _get_foot_position(self, player: Player) -> Optional[np.ndarray]:
        """
        Get the foot position (bottom center of bounding box) for a player.
        This is more accurate for tactical view as players are on the ground.
        """
        if player.detection is None:
            return None
        
        # Try to get absolute points first (accounts for camera movement)
        points = None
        if hasattr(player.detection, 'absolute_points') and player.detection.absolute_points is not None:
            points = player.detection.absolute_points
        elif hasattr(player.detection, 'points') and player.detection.points is not None:
            points = player.detection.points
        
        if points is None or len(points) < 2:
            return None
        
        # Get bounding box coordinates
        x1, y1 = points[0]
        x2, y2 = points[1]
        
        # Foot position is bottom center of bounding box (where player touches ground)
        foot_x = (x1 + x2) / 2
        foot_y = max(y1, y2)  # Bottom of bounding box
        
        return np.array([foot_x, foot_y], dtype=np.float32)

    def project_players(self, players: Sequence[Player]) -> List[TacticalProjection]:
        """
        Project current player foot positions onto the tactical view.
        Uses foot position (bottom center of bbox) instead of center for better accuracy.
        """
        if not self.ready or not players:
            return []

        projections: List[TacticalProjection] = []

        for player in players:
            # Get foot position (bottom center of bbox)
            foot_pos = self._get_foot_position(player)
            
            if foot_pos is None:
                continue

            try:
                # Transform foot position to tactical view
                transformed = self._homography.transform_points(
                    np.array([foot_pos], dtype=np.float32)
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

