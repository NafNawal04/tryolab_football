from typing import List, Optional, Tuple

import numpy as np

from soccer.ball import Ball
from soccer.player import Player
from soccer.tackle_attempt import TackleAttempt
from soccer.team import Team


class TackleDetector:
    """
    Tackle detection: when 70% of an opponent player's bbox is inside the ball-holder's bbox.
    - Tackle detected: 70% of tackler's bbox is inside attacker's bbox
    - Success: Tackler acquires the ball within 10 frames
    - Failure: Tackler does not acquire the ball within 10 frames
    """

    def __init__(self, fps: int):
        self.fps = max(1, int(fps))

        # Thresholds (pixels) - for ball possession (closest player)
        self.ball_control_radius_px = 50        # ball control radius for possession

        # Temporal windows
        self.horizon_frames = 10  # frames to wait before resolving outcome (fixed at 10 frames)
        self.min_tackle_duration = 1  # minimum frames for valid tackle (can be 1 since we check immediately)
        
        # Tackle detection threshold
        self.bbox_overlap_threshold = 0.70  # 70% of tackler's bbox must be inside attacker's bbox

        # History buffers (keep last N frames for stability)
        self._history_size = max(10, int(0.3 * self.fps))
        self._ball_centers: List[Tuple[float, float]] = []
        self._possessor_ids: List[Optional[int]] = []
        self._possessor_team_names: List[str] = []

        # Current active attempt and resolved attempts
        self._active_attempt: Optional[TackleAttempt] = None
        self._resolved_attempts: List[TackleAttempt] = []

    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        return (dx * dx + dy * dy) ** 0.5

    @staticmethod
    def _is_point_in_bbox(point: Tuple[float, float], bbox: np.ndarray) -> bool:
        """
        Check if a point is inside a bounding box.
        
        Parameters
        ----------
        point : Tuple[float, float]
            (x, y) coordinates of the point
        bbox : np.ndarray
            Bounding box as [[xmin, ymin], [xmax, ymax]]
            
        Returns
        -------
        bool
            True if point is inside bbox
        """
        if bbox is None or len(bbox) < 2:
            return False
        x, y = point
        xmin, ymin = float(bbox[0][0]), float(bbox[0][1])
        xmax, ymax = float(bbox[1][0]), float(bbox[1][1])
        return xmin <= x <= xmax and ymin <= y <= ymax

    @staticmethod
    def _bbox_intersection_area(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """
        Calculate the intersection area between two bounding boxes.
        
        Parameters
        ----------
        bbox1 : np.ndarray
            First bounding box as [[xmin, ymin], [xmax, ymax]]
        bbox2 : np.ndarray
            Second bounding box as [[xmin, ymin], [xmax, ymax]]
            
        Returns
        -------
        float
            Intersection area in pixels^2
        """
        if bbox1 is None or bbox2 is None or len(bbox1) < 2 or len(bbox2) < 2:
            return 0.0
        
        x1_min, y1_min = float(bbox1[0][0]), float(bbox1[0][1])
        x1_max, y1_max = float(bbox1[1][0]), float(bbox1[1][1])
        x2_min, y2_min = float(bbox2[0][0]), float(bbox2[0][1])
        x2_max, y2_max = float(bbox2[1][0]), float(bbox2[1][1])
        
        # Calculate intersection
        x_intersect_min = max(x1_min, x2_min)
        y_intersect_min = max(y1_min, y2_min)
        x_intersect_max = min(x1_max, x2_max)
        y_intersect_max = min(y1_max, y2_max)
        
        # Check if there's an intersection
        if x_intersect_min >= x_intersect_max or y_intersect_min >= y_intersect_max:
            return 0.0
        
        # Calculate intersection area
        intersection_area = (x_intersect_max - x_intersect_min) * (y_intersect_max - y_intersect_min)
        return intersection_area

    @staticmethod
    def _bbox_area(bbox: np.ndarray) -> float:
        """
        Calculate the area of a bounding box.
        
        Parameters
        ----------
        bbox : np.ndarray
            Bounding box as [[xmin, ymin], [xmax, ymax]]
            
        Returns
        -------
        float
            Area in pixels^2
        """
        if bbox is None or len(bbox) < 2:
            return 0.0
        xmin, ymin = float(bbox[0][0]), float(bbox[0][1])
        xmax, ymax = float(bbox[1][0]), float(bbox[1][1])
        return (xmax - xmin) * (ymax - ymin)

    def _is_tackler_inside_attacker(self, tackler_bbox: np.ndarray, attacker_bbox: np.ndarray) -> bool:
        """
        Check if 70% of the tackler's bounding box is inside the attacker's bounding box.
        
        Parameters
        ----------
        tackler_bbox : np.ndarray
            Tackler's bounding box as [[xmin, ymin], [xmax, ymax]]
        attacker_bbox : np.ndarray
            Attacker's bounding box as [[xmin, ymin], [xmax, ymax]]
            
        Returns
        -------
        bool
            True if 70% or more of tackler's bbox is inside attacker's bbox
        """
        tackler_area = self._bbox_area(tackler_bbox)
        if tackler_area == 0.0:
            return False
        
        intersection_area = self._bbox_intersection_area(tackler_bbox, attacker_bbox)
        overlap_ratio = intersection_area / tackler_area
        
        return overlap_ratio >= self.bbox_overlap_threshold

    def _trim_history(self):
        """Keep only recent history to avoid memory growth."""
        if len(self._ball_centers) > self._history_size:
            excess = len(self._ball_centers) - self._history_size
            self._ball_centers = self._ball_centers[excess:]
            self._possessor_ids = self._possessor_ids[excess:]
            self._possessor_team_names = self._possessor_team_names[excess:]

    def _estimate_possessor(
        self,
        players: List[Player],
        ball_center: Optional[Tuple[float, float]],
        last_team_possession: Optional[Team],
    ) -> Tuple[Optional[int], str]:
        """
        Estimate ball possessor: first check if ball is inside any player's bbox,
        otherwise use closest player within control radius.
        Returns (player_id or None, team_name or "").
        """
        if ball_center is None or not players:
            return (None, "")
        
        # First priority: check if ball is inside any player's bounding box
        for p in players:
            if p.detection is None or p.detection.points is None:
                continue
            if self._is_point_in_bbox(ball_center, p.detection.points):
                team_name = p.team.name if p.team else (last_team_possession.name if last_team_possession else "")
                return (p.player_id, team_name)
        
        # Fallback: closest player within control radius
        best_player = None
        best_dist = 1e9
        for p in players:
            if p.center is None:
                continue
            d = self._dist((float(p.center[0]), float(p.center[1])), (float(ball_center[0]), float(ball_center[1])))
            if d < best_dist:
                best_dist = d
                best_player = p
        if best_player is not None and best_dist <= self.ball_control_radius_px:
            team_name = best_player.team.name if best_player.team else (last_team_possession.name if last_team_possession else "")
            return (best_player.player_id, team_name)
        return (None, "")

    def _find_tackler(
        self,
        players: List[Player],
        attacker: Player,
        ball_center: Optional[Tuple[float, float]],
    ) -> Optional[Player]:
        """
        Find opponent player where 70% of their bounding box is inside the attacker's bounding box.
        This indicates a tackle attempt.
        
        Returns the tackler player if found, None otherwise.
        """
        if attacker is None:
            return None
        
        # Get attacker's bounding box
        if attacker.detection is None or attacker.detection.points is None:
            return None
        
        attacker_bbox = attacker.detection.points

        # Find opponent players where 70% of their bbox is inside attacker's bbox
        for p in players:
            # Skip same player
            if p.player_id == attacker.player_id:
                continue
            # Skip same team
            if p.team and attacker.team and p.team.name == attacker.team.name:
                continue
            if p.detection is None or p.detection.points is None:
                continue
            
            # Check if 70% of tackler's bbox is inside attacker's bbox
            tackler_bbox = p.detection.points
            if self._is_tackler_inside_attacker(tackler_bbox, attacker_bbox):
                # 70% overlap = tackle attempt!
                return p
        
        return None

    def update(
        self,
        frame_number: int,
        players: List[Player],
        ball: Ball,
        team_possession: Optional[Team],
    ):
        # Append current observations
        ball_center = tuple(ball.center) if ball and ball.center is not None else None
        possessor_id, possessor_team_name = self._estimate_possessor(players, ball_center, team_possession)

        self._ball_centers.append(ball_center if ball_center is not None else (0.0, 0.0))
        self._possessor_ids.append(possessor_id)
        self._possessor_team_names.append(possessor_team_name)
        self._trim_history()

        # Resolve in-progress attempt first (if any)
        if self._active_attempt is not None:
            # Check if enough time has passed to resolve (10 frames)
            frames_since_start = frame_number - self._active_attempt.start_frame
            
            if frames_since_start >= self.horizon_frames:
                # After 10 frames, check if tackler acquired the ball
                tackler = next((p for p in players if p.player_id == self._active_attempt.defender_id), None)
                
                if ball_center is not None and tackler is not None:
                    # Check if ball is inside tackler's bbox = successful tackle
                    if (tackler.detection is not None and 
                        tackler.detection.points is not None and
                        self._is_point_in_bbox(ball_center, tackler.detection.points)):
                        # Tackler has the ball = successful tackle
                        self._active_attempt._resolve(frame_number, TackleAttempt.OUTCOME_SUCCESS)
                    elif possessor_id is not None and possessor_team_name != "":
                        # Check if tackler's team won the ball
                        if possessor_team_name == self._active_attempt.defender_team_name:
                            # Tackler's team has the ball = successful tackle
                            self._active_attempt._resolve(frame_number, TackleAttempt.OUTCOME_SUCCESS)
                        else:
                            # Tackler did not acquire the ball = failed tackle
                            self._active_attempt._resolve(frame_number, TackleAttempt.OUTCOME_FAIL)
                    else:
                        # Ball is loose, tackler didn't get it = failed tackle
                        self._active_attempt._resolve(frame_number, TackleAttempt.OUTCOME_FAIL)
                elif possessor_id is not None and possessor_team_name != "":
                    # Check team possession
                    if possessor_team_name == self._active_attempt.defender_team_name:
                        # Tackler's team has the ball = successful tackle
                        self._active_attempt._resolve(frame_number, TackleAttempt.OUTCOME_SUCCESS)
                    else:
                        # Tackler did not acquire the ball = failed tackle
                        self._active_attempt._resolve(frame_number, TackleAttempt.OUTCOME_FAIL)
                else:
                    # Ball is loose, tackler didn't get it = failed tackle
                    self._active_attempt._resolve(frame_number, TackleAttempt.OUTCOME_FAIL)
                
                # Move to resolved list
                if self._active_attempt.is_done:
                    self._resolved_attempts.append(self._active_attempt)
                    self._active_attempt = None
            else:
                # Still within 10 frames, check early resolution if tackler already got the ball
                tackler = next((p for p in players if p.player_id == self._active_attempt.defender_id), None)
                
                if ball_center is not None and tackler is not None:
                    # Check if ball is inside tackler's bbox = early success
                    if (tackler.detection is not None and 
                        tackler.detection.points is not None and
                        self._is_point_in_bbox(ball_center, tackler.detection.points)):
                        # Tackler already has the ball = successful tackle (early resolution)
                        self._active_attempt._resolve(frame_number, TackleAttempt.OUTCOME_SUCCESS)
                        if self._active_attempt.is_done:
                            self._resolved_attempts.append(self._active_attempt)
                            self._active_attempt = None
                    elif possessor_id is not None and possessor_team_name != "":
                        # Check if tackler's team already has the ball
                        if possessor_team_name == self._active_attempt.defender_team_name:
                            # Tackler's team has the ball = early success
                            self._active_attempt._resolve(frame_number, TackleAttempt.OUTCOME_SUCCESS)
                            if self._active_attempt.is_done:
                                self._resolved_attempts.append(self._active_attempt)
                                self._active_attempt = None

        # If there is no active attempt, try to start one
        if self._active_attempt is None:
            # Need a player in possession
            if possessor_id is not None and possessor_team_name != "":
                # Find the attacker (player in possession)
                attacker = None
                for p in players:
                    if p.player_id == possessor_id:
                        attacker = p
                        break
                
                if attacker is not None:
                    # Find if there's an opponent where 70% of their bbox is inside attacker's bbox
                    tackler = self._find_tackler(players, attacker, ball_center)
                    
                    if tackler is not None:
                        # Start a new tackle attempt
                        attempt = TackleAttempt(
                            start_frame=frame_number,
                            attacker_id=possessor_id,
                            defender_id=tackler.player_id,
                            defender_team_name=tackler.team.name if tackler.team else "",
                            attacker_team_name=possessor_team_name,
                            horizon_frames=self.horizon_frames,
                            confirm_frames=self.min_tackle_duration,
                        )
                        attempt.mark_contact(frame_number)
                        self._active_attempt = attempt

    def get_resolved(self) -> List[dict]:
        return [a.as_dict() for a in self._resolved_attempts]

    def get_active(self) -> Optional[dict]:
        return self._active_attempt.as_dict() if self._active_attempt else None

