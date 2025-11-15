from typing import Dict, List, Optional, Tuple

import numpy as np

from soccer.player import Player


class PlayerDistanceTracker:
    """
    Tracks cumulative distance traveled by players across frames.
    Uses stabilized coordinates (detection.points) for consistent distance calculations.
    """
    
    def __init__(self, pixels_to_meters: Optional[float] = None, min_movement_threshold: float = 3.0):
        """
        Initialize the distance tracker.
        
        Parameters
        ----------
        pixels_to_meters : Optional[float], optional
            Conversion factor from pixels to meters. If None, distances are in pixels.
            For example, if 100 pixels = 1 meter, set to 0.01.
            By default None (distances in pixels)
        min_movement_threshold : float, optional
            Minimum movement in pixels to count as actual movement (default: 3.0).
            Movements smaller than this are ignored as they're likely due to:
            - Camera/stabilization jitter
            - Detection/tracking noise
            This prevents false distance accumulation when players aren't actually moving.
        """
        # Dictionary mapping player_id -> (last_position, cumulative_distance_pixels, cumulative_distance_meters)
        self.player_positions: Dict[int, Tuple[Optional[np.ndarray], float, float]] = {}
        # Track which players were present in the previous frame to detect new appearances
        self.previous_frame_players: set = set()
        self.pixels_to_meters = pixels_to_meters
        self.min_movement_threshold = min_movement_threshold  # Minimum pixels to count as movement
        
    def update_player_distance(self, player: Player) -> Tuple[float, float]:
        """
        Update the cumulative distance for a player based on their current position.
        
        Important:
        - Players introduced in later frames start at 0 meters (last_pos is None)
        - Small movements (< min_movement_threshold) are ignored to filter out camera/stabilization jitter
        - Only significant movements are counted as actual player movement
        
        Parameters
        ----------
        player : Player
            Player object with current detection
            
        Returns
        -------
        Tuple[float, float]
            (distance_pixels, distance_meters) for this frame update.
            If meters conversion is not set, distance_meters will be 0.0.
            First appearance always returns (0.0, 0.0).
        """
        player_id = player.player_id
        current_center = player.center
        
        if player_id is None or current_center is None:
            return (0.0, 0.0)
        
        # Check if this player was present in the previous frame
        # If not, this is their first appearance (or reappearance after disappearing)
        is_new_appearance = (player_id not in self.previous_frame_players)
        
        # If this is a new appearance, reset their distance tracking to 0
        # This handles both:
        # 1. Players appearing for the very first time
        # 2. Players reappearing after disappearing (they start fresh)
        if is_new_appearance:
            # Reset distance tracking for this player (start at 0)
            last_pos = None
            cumul_pixels = 0.0
            cumul_meters = 0.0
        else:
            # Player was present in previous frame - get their last position and distances
            last_pos, cumul_pixels, cumul_meters = self.player_positions.get(
                player_id, (None, 0.0, 0.0)
            )
            # Safety check: if somehow last_pos is None but player was in previous frame,
            # treat as new appearance
            if last_pos is None:
                cumul_pixels = 0.0
                cumul_meters = 0.0
        
        frame_distance_pixels = 0.0
        frame_distance_meters = 0.0
        
        # Only calculate distance if we have a previous position (not a new appearance)
        if last_pos is not None:
            # Calculate Euclidean distance in pixel space
            frame_distance_pixels = np.linalg.norm(current_center - last_pos)
            
            # CRITICAL: Filter out small movements due to camera/stabilization jitter
            # Movements smaller than min_movement_threshold are ignored
            # This prevents false distance accumulation when players aren't actually moving
            if frame_distance_pixels >= self.min_movement_threshold:
                # Validate: Skip unrealistic large jumps (likely tracking errors or ID switches)
                # At 30 fps, max realistic movement is ~5-10 pixels per frame (sprint ~12 m/s)
                # Allow up to 50 pixels per frame as safety margin
                max_reasonable_pixels_per_frame = 50.0
                
                if frame_distance_pixels <= max_reasonable_pixels_per_frame:
                    # Valid movement - proceed with accumulation
                    # Convert to meters if calibration is available
                    if self.pixels_to_meters is not None:
                        frame_distance_meters = frame_distance_pixels * self.pixels_to_meters
                        
                        # Validate: At 30 fps, max realistic speed is ~12 m/s (world record sprint)
                        # That's ~0.4 m per frame. Allow up to 2 m/frame as safety margin
                        max_reasonable_meters_per_frame = 2.0
                        if frame_distance_meters <= max_reasonable_meters_per_frame:
                            # Valid movement in both pixels and meters - accumulate it
                            cumul_pixels += frame_distance_pixels
                            cumul_meters += frame_distance_meters
                        else:
                            # Movement in meters is too large - likely calibration error or tracking error
                            # Skip this frame (don't accumulate)
                            frame_distance_pixels = 0.0
                            frame_distance_meters = 0.0
                    else:
                        # No meters conversion - just accumulate pixels
                        cumul_pixels += frame_distance_pixels
                else:
                    # Movement in pixels is too large - likely tracking error, ID switch, or camera jump
                    # Skip this frame (don't accumulate false movement)
                    frame_distance_pixels = 0.0
            else:
                # Movement too small - likely camera/stabilization jitter or tracking noise
                # Don't count this as movement (frame_distance_pixels already calculated, set to 0)
                frame_distance_pixels = 0.0
        
        # Update stored position and cumulative distances
        # For first appearance (last_pos is None OR is_new_appearance), store position but keep distances at 0.0
        self.player_positions[player_id] = (current_center.copy(), cumul_pixels, cumul_meters)
        
        return (frame_distance_pixels, frame_distance_meters)
    
    def update_frame_players(self, player_ids: set):
        """
        Update which players were present in the current frame.
        This is used to detect new appearances vs. reappearances.
        
        Parameters
        ----------
        player_ids : set
            Set of player IDs present in the current frame
        """
        self.previous_frame_players = player_ids.copy()
    
    def get_player_distance(self, player_id: int, in_meters: bool = False) -> float:
        """
        Get the cumulative distance traveled by a player.
        
        Parameters
        ----------
        player_id : int
            Player ID
        in_meters : bool, optional
            If True, return distance in meters (requires calibration).
            If False, return distance in pixels. By default False
            
        Returns
        -------
        float
            Cumulative distance. Returns 0.0 if player not found or conversion unavailable.
        """
        if player_id not in self.player_positions:
            return 0.0
        
        _, cumul_pixels, cumul_meters = self.player_positions[player_id]
        
        if in_meters:
            if self.pixels_to_meters is None:
                return 0.0
            return cumul_meters
        else:
            return cumul_pixels
    
    def get_all_distances(self, in_meters: bool = False) -> Dict[int, float]:
        """
        Get cumulative distances for all tracked players.
        
        Parameters
        ----------
        in_meters : bool, optional
            If True, return distances in meters. By default False
            
        Returns
        -------
        Dict[int, float]
            Dictionary mapping player_id -> cumulative_distance
        """
        result = {}
        for player_id in self.player_positions:
            result[player_id] = self.get_player_distance(player_id, in_meters=in_meters)
        return result
    
    def reset(self):
        """Reset all tracked distances."""
        self.player_positions.clear()
        self.previous_frame_players.clear()

