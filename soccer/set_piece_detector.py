from typing import List, Optional, Tuple, Dict
import numpy as np
from soccer.ball import Ball
from soccer.player import Player
from soccer.team import Team


class SetPieceDetector:
    """
    Detects defending set pieces (corners, free kicks) and classifies them into:
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
        
        # Wall detection parameters
        self.wall_min_players = 3  # Minimum players to form a wall
        self.wall_max_lateral_spacing = 80  # Max spacing between players in wall (pixels)
        self.wall_max_depth_variance = 30  # Max depth variance for wall alignment (pixels)
        self.wall_min_frames = int(0.5 * self.fps)  # Wall must exist for at least 0.5 seconds
        
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
    
    def _detect_wall(self, players: List[Player], attacking_team: Optional[Team]) -> Tuple[bool, List[Player]]:
        """
        Detect if defending players form a wall (standing side by side in a row).
        
        Returns:
            (is_wall, wall_players): True if wall detected, list of players in wall
        """
        if attacking_team is None or len(players) < self.wall_min_players:
            return (False, [])
        
        # Get defending team players (opposite of attacking team)
        defending_players = [p for p in players if p.team is not None and p.team != attacking_team]
        
        if len(defending_players) < self.wall_min_players:
            return (False, [])
        
        # Filter players with valid positions
        valid_players = [p for p in defending_players if p.center is not None]
        if len(valid_players) < self.wall_min_players:
            return (False, [])
        
        # Try to find a group of players forming a wall
        # A wall is characterized by players standing side by side (similar Y coordinates)
        # and relatively close together (within max_lateral_spacing)
        
        # Sort players by Y coordinate (vertical position)
        sorted_players = sorted(valid_players, key=lambda p: p.center[1])
        
        # Try to find clusters of players with similar Y coordinates
        wall_candidates = []
        current_cluster = []
        
        for i, player in enumerate(sorted_players):
            if not current_cluster:
                current_cluster = [player]
            else:
                # Check if this player is close enough vertically to join cluster
                last_player = current_cluster[-1]
                y_diff = abs(player.center[1] - last_player.center[1])
                
                if y_diff <= self.wall_max_depth_variance:
                    current_cluster.append(player)
                else:
                    # Check if current cluster is large enough to be a wall
                    if len(current_cluster) >= self.wall_min_players:
                        # Check if players in cluster are close enough horizontally
                        if self._are_players_in_line(current_cluster):
                            wall_candidates.append(current_cluster)
                    current_cluster = [player]
        
        # Check last cluster
        if len(current_cluster) >= self.wall_min_players:
            if self._are_players_in_line(current_cluster):
                wall_candidates.append(current_cluster)
        
        # Return the largest wall candidate
        if wall_candidates:
            largest_wall = max(wall_candidates, key=len)
            return (True, largest_wall)
        
        return (False, [])
    
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
        
        # Detect wall formation
        is_wall, wall_players = self._detect_wall(players, team_possession)
        
        if is_wall:
            if self._wall_first_detected_frame is None:
                self._wall_first_detected_frame = frame_number
            self._wall_detected_frames += 1
        else:
            # Reset wall detection if wall breaks
            if self._wall_detected_frames > 0:
                self._wall_detected_frames = 0
                self._wall_first_detected_frame = None
        
        # Check if we have an active set piece
        if self._active_set_piece is not None:
            # Update active set piece
            self._active_set_piece['last_frame'] = frame_number
            
            # Check if ball is kicked (significant movement after being stationary)
            if (self._is_ball_moving() and 
                self._active_set_piece.get('ball_kicked_frame') is None):
                self._active_set_piece['ball_kicked_frame'] = frame_number
                # Store ball movement at kick time for classification
                if len(self._ball_movements) > 0:
                    # Get max movement from last few frames (when kick happened)
                    recent_movements = self._ball_movements[-min(5, len(self._ball_movements)):]
                    self._active_set_piece['ball_kick_movement'] = max(recent_movements) if recent_movements else 0.0
                else:
                    self._active_set_piece['ball_kick_movement'] = 0.0
                
                # Classify the set piece type
                attacking_team_name = self._active_set_piece.get('attacking_team')
                attacking_team = None
                if attacking_team_name and players:
                    # Find the team from players
                    for player in players:
                        if player.team and player.team.name == attacking_team_name:
                            attacking_team = player.team
                            break
                
                set_piece_type = self._classify_set_piece_type(
                    self._active_set_piece, 
                    frame_number,
                    players,
                    attacking_team
                )
                self._active_set_piece['type'] = set_piece_type
            
            # Resolve set piece if ball moves significantly or wall breaks
            if (self._is_ball_moving() and 
                self._active_set_piece.get('ball_kicked_frame') is not None):
                # Ball was kicked, resolve after a short delay
                frames_since_kick = frame_number - self._active_set_piece['ball_kicked_frame']
                if frames_since_kick >= int(0.3 * self.fps):  # 0.3s after kick
                    self._active_set_piece['resolved_frame'] = frame_number
                    self._resolved_set_pieces.append(self._active_set_piece.copy())
                    self._active_set_piece = None
            
            # Also resolve if wall breaks and ball hasn't been kicked
            if not is_wall and self._active_set_piece.get('ball_kicked_frame') is None:
                # Wall broke before kick - might not be a set piece
                self._active_set_piece = None
        
        # Try to start a new set piece detection
        if self._active_set_piece is None:
            # Conditions for set piece:
            # 1. Ball is stationary
            # 2. Ball is in possession of a player
            # 3. Wall is forming or formed
            # 4. Ball has been stationary for required frames
            
            if (closest_player is not None and 
                closest_player.team is not None and
                self._is_ball_stationary() and
                is_wall and
                self._wall_detected_frames >= self.wall_min_frames):
                
                # Start new set piece
                self._active_set_piece = {
                    'start_frame': frame_number,
                    'wall_detected_frame': self._wall_first_detected_frame,
                    'attacking_team': closest_player.team.name,
                    'defending_team': (team_possession.name if team_possession and team_possession != closest_player.team else None),
                    'ball_kicked_frame': None,
                    'resolved_frame': None,
                    'type': None,
                    'wall_player_count': len(wall_players),
                }
    
    def get_resolved(self) -> List[Dict]:
        """Get list of resolved set pieces."""
        return self._resolved_set_pieces.copy()
    
    def get_active(self) -> Optional[Dict]:
        """Get currently active set piece, if any."""
        if self._active_set_piece is None:
            return None
        return self._active_set_piece.copy()

