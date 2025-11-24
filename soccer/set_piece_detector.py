from typing import List, Optional, Tuple, Dict
import numpy as np
from soccer.ball import Ball
from soccer.player import Player
from soccer.team import Team


class SetPieceDetector:
    """
    Detects defending set pieces (corners, free kicks) by identifying when players
    from both teams form a wall (clustered together) while the ball and one player
    are far from the cluster. Classifies set pieces into:
    - Short: Quick set piece before wall forms
    - Direct: Direct shot at goal after wall forms
    - Tactical: Pass to teammate after wall forms
    """
    
    TYPE_SHORT = "short"
    TYPE_DIRECT = "direct"
    TYPE_TACTICAL = "tactical"
    
    def __init__(self, fps: int):
        self.fps = max(1, int(fps))
        
        # Ball movement thresholds (pixels)
        self.ball_stationary_threshold = 5.0  # Ball is stationary if movement < 5px
        self.ball_movement_threshold = 20.0  # Ball is moving if movement > 20px
        
        # Wall detection parameters - simplified
        self.wall_min_players = 5  # Minimum players in cluster
        self.wall_max_lateral_spacing = 200  # Not used in simplified detection
        self.wall_max_depth_variance = 80  # Not used in simplified detection
        self.wall_min_frames = 1  # No frame stability requirement - detect immediately
        
        # Set piece detection thresholds
        self.set_piece_ball_stationary_frames = int(0.3 * self.fps)  # Ball stationary for 0.3s
        self.short_set_piece_max_wall_frames = int(0.5 * self.fps)  # Short: kick within 0.5s of wall forming
        
        # History buffers
        self._ball_positions: List[Optional[Tuple[float, float]]] = []
        self._ball_movements: List[float] = []  # Movement per frame
        self._history_size = max(30, int(1.0 * self.fps))
        
        # Current set piece state
        self._active_set_piece: Optional[Dict] = None
        self._resolved_set_pieces: List[Dict] = []
        
        # Wall detection state
        self._wall_detected_frames: int = 0
        self._wall_first_detected_frame: Optional[int] = None
        self._wall_missing_frames: int = 0  # Track how many frames wall has been missing
        
    @staticmethod
    def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return np.sqrt(dx * dx + dy * dy)
    
    def _calculate_ball_movement(self, current_pos: Optional[Tuple[float, float]]) -> float:
        """Calculate ball movement from last frame."""
        if current_pos is None:
            return 0.0
        if not self._ball_positions:
            return 0.0
        
        last_pos = self._ball_positions[-1]
        if last_pos is None:
            return 0.0
        
        return self._distance(current_pos, last_pos)
    
    def _is_ball_stationary(self, num_frames: int = None) -> bool:
        """Check if ball has been stationary for given number of frames."""
        if num_frames is None:
            num_frames = self.set_piece_ball_stationary_frames
        
        if len(self._ball_movements) < num_frames:
            return False
        
        # Check if all recent movements are below threshold
        recent_movements = self._ball_movements[-num_frames:]
        return all(m < self.ball_stationary_threshold for m in recent_movements)
    
    def _is_ball_moving(self) -> bool:
        """Check if ball is currently moving significantly."""
        if not self._ball_movements:
            return False
        
        # Check last few frames for significant movement
        recent_frames = min(3, len(self._ball_movements))
        recent_movements = self._ball_movements[-recent_frames:]
        return any(m > self.ball_movement_threshold for m in recent_movements)
    
    def _detect_wall(self, players: List[Player], ball: Ball, closest_player: Optional[Player]) -> Tuple[bool, List[Player]]:
        """
        Simple wall detection: If most players are close together and only 1-2 are far away,
        then it's a defending set piece wall.
        
        Returns:
            (is_wall, wall_players): True if wall detected, list of players in wall
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Filter players with valid positions
        valid_players = [p for p in players if p.center is not None]
        if len(valid_players) < 6:  # Need at least 6 players total (5+ in cluster, 1-2 isolated)
            return (False, [])
        
        # Distance threshold for players to be considered "close" (standing together)
        close_distance = 200  # pixels - players within this distance are considered together
        
        # Distance threshold for players to be considered "far" from cluster
        far_distance = 150  # pixels - players beyond this from cluster are considered isolated
        
        # Find the largest cluster of players that are close to each other
        clusters = []
        unassigned = valid_players.copy()
        
        while unassigned:
            # Start a new cluster with the first unassigned player
            seed = unassigned.pop(0)
            cluster = [seed]
            
            # Find all players close to this cluster
            changed = True
            while changed:
                changed = False
                for player in unassigned[:]:  # Copy list to iterate safely
                    # Check if player is close to any player in the cluster
                    min_dist_to_cluster = min(
                        self._distance(tuple(np.array(p.center)), tuple(np.array(player.center)))
                        for p in cluster
                    )
                    if min_dist_to_cluster <= close_distance:
                        cluster.append(player)
                        unassigned.remove(player)
                        changed = True
            
            # Keep all clusters (we'll find the largest)
            if len(cluster) > 0:
                clusters.append(cluster)
        
        if not clusters:
            return (False, [])
        
        # Find the largest cluster (this is the "most players" group)
        largest_cluster = max(clusters, key=len)
        
        # Check if this cluster has "most players" (at least 5, and more than half of all players)
        total_players = len(valid_players)
        cluster_size = len(largest_cluster)
        
        # Most players should be in the cluster (at least 5 players, and majority)
        if cluster_size < 5 or cluster_size < total_players * 0.6:
            return (False, [])
        
        # Check if there are 1-2 players far from this cluster
        cluster_center = np.mean([np.array(p.center) for p in largest_cluster], axis=0)
        
        players_far_from_cluster = []
        for player in valid_players:
            if player not in largest_cluster:
                player_pos = np.array(player.center)
                dist_to_cluster = self._distance(tuple(cluster_center), tuple(player_pos))
                if dist_to_cluster > far_distance:
                    players_far_from_cluster.append(player)
        
        # Need exactly 1-2 players far from the cluster
        if len(players_far_from_cluster) < 1 or len(players_far_from_cluster) > 2:
            return (False, [])
        
        # All conditions met: most players clustered together, 1-2 players far away
        logger.info(f"_detect_wall: ✓ DEFENDING SET PIECE WALL! "
                   f"Wall: {cluster_size} players, Isolated: {len(players_far_from_cluster)} players")
        return (True, largest_cluster)
    
    def _are_players_in_line(self, players: List[Player]) -> bool:
        """
        Check if players are arranged in a line (side by side).
        Players should have similar Y coordinates and be spaced reasonably.
        """
        if len(players) < 2:
            return False
        
        # Get player centers
        centers = [p.center for p in players]
        
        # Sort by X coordinate (horizontal position)
        centers_sorted = sorted(centers, key=lambda c: c[0])
        
        # Check spacing between consecutive players
        for i in range(len(centers_sorted) - 1):
            dist = self._distance(centers_sorted[i], centers_sorted[i + 1])
            if dist > self.wall_max_lateral_spacing:
                return False
        
        # Check Y coordinate variance (should be small for a wall)
        y_coords = [c[1] for c in centers]
        y_variance = np.std(y_coords) if len(y_coords) > 1 else 0
        if y_variance > self.wall_max_depth_variance:
            return False
        
        return True
    
    def _calculate_wall_bbox(self, wall_players: List[Player]) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Calculate bounding box around all players in the wall.
        
        Parameters:
        -----------
        wall_players : List[Player]
            List of players forming the wall
        
        Returns:
        --------
        Optional[Tuple[Tuple[float, float], Tuple[float, float]]]
            Bounding box as ((xmin, ymin), (xmax, ymax)) or None if no valid players
        """
        if not wall_players:
            return None
        
        # Get all bounding boxes from player detections or centers
        x_coords = []
        y_coords = []
        
        for player in wall_players:
            # Try to get bounding box from detection first
            if player.detection is not None and player.detection.points is not None:
                x1, y1 = player.detection.points[0]
                x2, y2 = player.detection.points[1]
                x_coords.extend([x1, x2])
                y_coords.extend([y1, y2])
            # Fallback to center if detection points not available
            elif player.center is not None:
                center_x, center_y = player.center
                # Use a default player size around center (approximate player bbox size)
                player_width = 40  # Approximate player width in pixels
                player_height = 60  # Approximate player height in pixels
                x_coords.extend([center_x - player_width/2, center_x + player_width/2])
                y_coords.extend([center_y - player_height/2, center_y + player_height/2])
        
        if not x_coords or not y_coords:
            return None
        
        # Calculate bounding box with some padding
        padding = 30  # Add padding around the wall (increased for better visibility)
        xmin = min(x_coords) - padding
        ymin = min(y_coords) - padding
        xmax = max(x_coords) + padding
        ymax = max(y_coords) + padding
        
        return ((xmin, ymin), (xmax, ymax))
    
    def _trim_history(self):
        """Keep only recent history to avoid memory growth."""
        if len(self._ball_positions) > self._history_size:
            excess = len(self._ball_positions) - self._history_size
            self._ball_positions = self._ball_positions[excess:]
            self._ball_movements = self._ball_movements[excess:]
    
    def _classify_set_piece_type(
        self, 
        set_piece: Dict, 
        current_frame: int,
        players: List[Player],
        attacking_team: Optional[Team]
    ) -> str:
        """
        Classify the type of set piece based on sequence of events.
        
        - Short: Ball kicked before wall forms or very quickly after
        - Direct: Ball kicked directly towards goal after wall forms
        - Tactical: Ball passed to teammate after wall forms
        """
        wall_frame = set_piece.get('wall_detected_frame')
        ball_kicked_frame = set_piece.get('ball_kicked_frame')
        
        if wall_frame is None or ball_kicked_frame is None:
            return self.TYPE_TACTICAL  # Default fallback
        
        frames_after_wall = ball_kicked_frame - wall_frame
        
        # Short: kick happens before wall forms or very quickly after
        if frames_after_wall <= self.short_set_piece_max_wall_frames:
            return self.TYPE_SHORT
        
        # For direct vs tactical, check ball movement and possession change
        # Direct: ball moves significantly (suggesting shot) and possession doesn't change quickly
        # Tactical: ball moves moderately and possession changes to teammate
        
        # Check ball movement at kick time (stored when kick was detected)
        kick_movement = set_piece.get('ball_kick_movement', 0.0)
        
        # Large movement (> 40px) suggests direct shot
        if kick_movement > self.ball_movement_threshold * 2:
            return self.TYPE_DIRECT
        
        # Moderate movement with possession change suggests tactical pass
        # Check if a teammate gets the ball shortly after kick
        if attacking_team is not None and players:
            # Look for players of attacking team near ball position after kick
            # This is a simplified check - in practice, you'd track possession changes
            # For now, if movement is moderate, assume tactical
            if self.ball_movement_threshold < kick_movement <= self.ball_movement_threshold * 2:
                return self.TYPE_TACTICAL
        
        # Default to tactical for smaller movements (likely passes)
        return self.TYPE_TACTICAL
    
    def update(
        self,
        frame_number: int,
        players: List[Player],
        ball: Ball,
        team_possession: Optional[Team],
        closest_player: Optional[Player],
    ):
        """
        Update set piece detection state.
        
        Parameters:
        -----------
        frame_number : int
            Current frame number
        players : List[Player]
            List of all players
        ball : Ball
            Ball object
        team_possession : Optional[Team]
            Team currently in possession
        closest_player : Optional[Player]
            Player closest to the ball
        """
        # Track ball position and movement
        ball_center = tuple(ball.center) if ball and ball.center is not None else None
        ball_movement = self._calculate_ball_movement(ball_center)
        
        self._ball_positions.append(ball_center)
        self._ball_movements.append(ball_movement)
        self._trim_history()
        
        # Detect wall formation (players from both teams clustered together)
        is_wall, wall_players = self._detect_wall(players, ball, closest_player)
        
        # Track wall detection (simplified - just for logging)
        if is_wall:
            if self._wall_first_detected_frame is None:
                self._wall_first_detected_frame = frame_number
            self._wall_detected_frames += 1
            self._wall_missing_frames = 0
        else:
            self._wall_missing_frames += 1
            # Reset after a short time
            if self._wall_missing_frames > int(0.5 * self.fps):
                if self._wall_detected_frames > 0:
                    self._wall_detected_frames = 0
                    self._wall_first_detected_frame = None
        
        # Check if we have an active defending set piece
        if self._active_set_piece is not None:
            # Update active set piece
            self._active_set_piece['last_frame'] = frame_number
            
            # Update wall bounding box if wall still exists
            if is_wall:
                wall_bbox = self._calculate_wall_bbox(wall_players)
                self._active_set_piece['wall_bbox'] = wall_bbox
            
            # Resolve set piece if wall breaks (players disperse)
            if not is_wall:
                # Wall broke - defending set piece is over
                self._active_set_piece['resolved_frame'] = frame_number
                self._resolved_set_pieces.append(self._active_set_piece.copy())
                self._active_set_piece = None
        
        # Try to start a new defending set piece detection
        if self._active_set_piece is None:
            # Simple condition: Wall detected (most players close, 1-2 far)
            # No frame stability, no ball detection, no team checking needed
            
            import logging
            logger = logging.getLogger(__name__)
            
            if is_wall and len(wall_players) >= self.wall_min_players:
                logger.info(f"🎯 DEFENDING SET PIECE DETECTED! Frame {frame_number}, Wall players: {len(wall_players)}")
                
                # Calculate bounding box around the wall
                wall_bbox = self._calculate_wall_bbox(wall_players)
                logger.info(f"Wall bbox: {wall_bbox}")
                
                # Start new defending set piece
                self._active_set_piece = {
                    'start_frame': frame_number,
                    'wall_detected_frame': frame_number,
                    'wall_player_count': len(wall_players),
                    'wall_bbox': wall_bbox,
                }
    
    def get_resolved(self) -> List[Dict]:
        """Get list of resolved set pieces."""
        return self._resolved_set_pieces.copy()
    
    def get_active(self) -> Optional[Dict]:
        """Get currently active set piece, if any."""
        if self._active_set_piece is None:
            return None
        return self._active_set_piece.copy()

