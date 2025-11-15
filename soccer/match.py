from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import PIL
from PIL import ImageDraw, ImageFont

from soccer.ball import Ball
from soccer.draw import Draw
from soccer.pass_event import Pass, PassEvent
from soccer.player import Player
from soccer.team import Team


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


class TackleAttempt:
    """
    Tracks a single tackle attempt lifecycle and determines outcome.
    All thresholds are in pixel units and time is in frames.
    """

    STATE_CANDIDATE = "candidate"
    STATE_CONTACT = "contact"
    STATE_RESOLVE = "resolve"
    STATE_DONE = "done"

    OUTCOME_SUCCESS = "success"
    OUTCOME_FAIL = "failure"
    OUTCOME_INCONCLUSIVE = "inconclusive"

    def __init__(
        self,
        start_frame: int,
        attacker_id: int,
        defender_id: int,
        defender_team_name: str,
        attacker_team_name: str,
        horizon_frames: int,
        confirm_frames: int,
    ):
        self.start_frame = start_frame
        self.attacker_id = attacker_id
        self.defender_id = defender_id
        self.defender_team_name = defender_team_name
        self.attacker_team_name = attacker_team_name
        self.horizon_frames = horizon_frames
        self.confirm_frames = confirm_frames

        self.state = TackleAttempt.STATE_CANDIDATE
        self.contact_frame = None
        self.resolved_frame = None
        self.outcome = None

        # Internal persistence counters
        self._attacker_control_frames = 0
        self._defender_team_control_frames = 0
        self._frames_since_contact = 0

    def mark_contact(self, frame_number: int):
        if self.state == TackleAttempt.STATE_CANDIDATE:
            self.state = TackleAttempt.STATE_CONTACT
            self.contact_frame = frame_number
            self._frames_since_contact = 0

    def update_resolution(
        self,
        frame_number: int,
        current_possessor_id: Optional[int],
        current_possessor_team_name: str,
        attacker_distance_to_ball_px: float,
        defender_distance_to_ball_px: float,
    ):
        if self.state not in (TackleAttempt.STATE_CONTACT, TackleAttempt.STATE_RESOLVE):
            return

        # Enter resolve state right after contact
        if self.state == TackleAttempt.STATE_CONTACT:
            self.state = TackleAttempt.STATE_RESOLVE
            self._frames_since_contact = 0

        self._frames_since_contact += 1

        # Persistence counts
        if current_possessor_id is not None and current_possessor_id == self.attacker_id:
            self._attacker_control_frames += 1
        if current_possessor_team_name == self.defender_team_name and self.defender_team_name != "":
            self._defender_team_control_frames += 1

        # Resolution rules (pixels-only, persistence based)
        if self._defender_team_control_frames >= self.confirm_frames:
            self._resolve(frame_number, TackleAttempt.OUTCOME_SUCCESS)
            return

        if self._attacker_control_frames >= self.confirm_frames:
            # Attacker retained control long enough after contact → failed tackle
            self._resolve(frame_number, TackleAttempt.OUTCOME_FAIL)
            return

        # Horizon timeout
        if self._frames_since_contact >= self.horizon_frames:
            self._resolve(frame_number, TackleAttempt.OUTCOME_INCONCLUSIVE)

    def _resolve(self, frame_number: int, outcome: str):
        self.state = TackleAttempt.STATE_DONE
        self.outcome = outcome
        self.resolved_frame = frame_number

    @property
    def is_done(self) -> bool:
        return self.state == TackleAttempt.STATE_DONE

    def as_dict(self) -> dict:
        return {
            "start_frame": self.start_frame,
            "contact_frame": self.contact_frame,
            "resolved_frame": self.resolved_frame,
            "attacker_id": self.attacker_id,
            "defender_id": self.defender_id,
            "attacker_team": self.attacker_team_name,
            "defender_team": self.defender_team_name,
            "outcome": self.outcome or "",
        }


class TackleDetector:
    """
    Rule-based tackle detection using pixels and fps (no meters).
    Integrates a simple FSM per attempt and uses possession flips as primary signal.
    """

    def __init__(self, fps: int):
        self.fps = max(1, int(fps))

        # Thresholds (pixels & frames)
        self.control_radius_px = 40            # ball control radius
        self.contact_radius_px = 25            # defender-to-ball proximity for contact
        self.attacker_defender_close_px = 80   # attacker-defender closeness
        self.dir_change_deg_thresh = 30.0      # ball direction change
        self.speed_drop_ratio = 0.5            # ball speed drop (cur/prev)

        # Temporal windows
        self.confirm_frames = max(6, int(0.3 * self.fps))   # persistence to confirm state
        self.horizon_frames = max(30, int(1.5 * self.fps))  # resolve window

        # History buffers
        self._ball_centers: List[Tuple[float, float]] = []
        self._possessor_ids: List[Optional[int]] = []
        self._possessor_team_names: List[str] = []

        # Current active attempt and resolved attempts
        self._active_attempt: Optional[TackleAttempt] = None
        self._resolved_attempts: List[TackleAttempt] = []

        # Internal for possessor stability (not strictly needed for outcome, but useful if extended)
        self._last_stable_possessor_id: Optional[int] = None
        self._stable_possessor_frames = 0

    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        return (dx * dx + dy * dy) ** 0.5

    @staticmethod
    def _angle_deg(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
        import math
        ux, uy = v1
        vx, vy = v2
        num = ux * vx + uy * vy
        den = (ux * ux + uy * uy) ** 0.5 * (vx * vx + vy * vy) ** 0.5
        if den == 0:
            return 0.0
        cos_t = max(-1.0, min(1.0, num / den))
        return abs(math.degrees(math.acos(cos_t)))

    def _ball_velocity(self, k: int) -> Tuple[float, float]:
        # velocity between frames k-1 and k
        if k <= 0 or k >= len(self._ball_centers):
            return (0.0, 0.0)
        x0, y0 = self._ball_centers[k - 1]
        x1, y1 = self._ball_centers[k]
        return (float(x1 - x0), float(y1 - y0))

    def _ball_dir_change_and_speed_drop(self) -> Tuple[float, float]:
        """
        Returns (dir_change_deg, speed_drop_ratio) comparing last two velocity vectors.
        """
        k = len(self._ball_centers) - 1
        if k < 2:
            return (0.0, 1.0)
        v_prev = self._ball_velocity(k - 1)
        v_cur = self._ball_velocity(k)
        dir_change = self._angle_deg(v_prev, v_cur)
        prev_speed = (v_prev[0] * v_prev[0] + v_prev[1] * v_prev[1]) ** 0.5
        cur_speed = (v_cur[0] * v_cur[0] + v_cur[1] * v_cur[1]) ** 0.5
        drop_ratio = 1.0
        if prev_speed > 1e-6:
            drop_ratio = cur_speed / prev_speed
        return (dir_change, drop_ratio)

    def _update_possessor_stability(self, possessor_id: Optional[int]):
        if possessor_id is not None and possessor_id == self._last_stable_possessor_id:
            self._stable_possessor_frames += 1
        elif possessor_id is not None and possessor_id != self._last_stable_possessor_id:
            self._last_stable_possessor_id = possessor_id
            self._stable_possessor_frames = 1
        else:
            # None possessor resets stability
            self._last_stable_possessor_id = None
            self._stable_possessor_frames = 0

    def _estimate_possessor(
        self,
        players: List["Player"],
        ball_center: Optional[Tuple[float, float]],
        last_team_possession: Optional["Team"],
    ) -> Tuple[Optional[int], str]:
        """
        Estimate ball possessor as closest player within control_radius_px.
        Returns (player_id or None, team_name or "").
        """
        if ball_center is None or not players:
            return (None, "")
        best_player = None
        best_dist = 1e9
        for p in players:
            if p.center is None:
                continue
            d = self._dist((float(p.center[0]), float(p.center[1])), (float(ball_center[0]), float(ball_center[1])))
            if d < best_dist:
                best_dist = d
                best_player = p
        if best_player is not None and best_dist <= self.control_radius_px:
            team_name = best_player.team.name if best_player.team else (last_team_possession.name if last_team_possession else "")
            return (best_player.player_id, team_name)
        return (None, "")

    def _select_defender_candidate(
        self,
        players: List["Player"],
        attacker_id: Optional[int],
        attacker_team_name: str,
        ball_center: Optional[Tuple[float, float]],
    ) -> Tuple[Optional["Player"], float, float]:
        """
        Select nearest opponent to ball, also close to attacker.
        Returns (defender_player or None, dist_def_ball, dist_att_def)
        """
        if attacker_id is None or attacker_team_name == "" or ball_center is None:
            return (None, 0.0, 0.0)
        attacker = None
        for p in players:
            if p.player_id == attacker_id:
                attacker = p
                break
        if attacker is None or attacker.center is None:
            return (None, 0.0, 0.0)

        best = None
        best_score = 1e12
        for p in players:
            if p.player_id == attacker_id:
                continue
            if p.team and p.team.name == attacker_team_name:
                continue
            if p.center is None:
                continue
            d_ball = self._dist(
                (float(p.center[0]), float(p.center[1])),
                (float(ball_center[0]), float(ball_center[1])),
            )
            d_pair = self._dist(
                (float(p.center[0]), float(p.center[1])),
                (float(attacker.center[0]), float(attacker.center[1])),
            )
            score = d_ball + 0.5 * d_pair
            if score < best_score:
                best_score = score
                best = (p, d_ball, d_pair)
        return best if best is not None else (None, 0.0, 0.0)

    def update(
        self,
        frame_number: int,
        players: List["Player"],
        ball: "Ball",
        team_possession: Optional["Team"],
    ):
        # Append current observations
        ball_center = tuple(ball.center) if ball and ball.center is not None else None
        possessor_id, possessor_team_name = self._estimate_possessor(players, ball_center, team_possession)

        self._ball_centers.append(ball_center if ball_center is not None else (0.0, 0.0))
        self._possessor_ids.append(possessor_id)
        self._possessor_team_names.append(possessor_team_name)
        self._update_possessor_stability(possessor_id)

        # Resolve in-progress attempt first (if any)
        if self._active_attempt and self._active_attempt.state in (
            TackleAttempt.STATE_CONTACT,
            TackleAttempt.STATE_RESOLVE,
        ):
            # distances for context
            att_d = 0.0
            def_d = 0.0
            if ball_center is not None:
                attacker = next((p for p in players if p.player_id == self._active_attempt.attacker_id), None)
                defender = next((p for p in players if p.player_id == self._active_attempt.defender_id), None)
                if attacker and attacker.center is not None:
                    att_d = self._dist((float(attacker.center[0]), float(attacker.center[1])), (float(ball_center[0]), float(ball_center[1])))
                if defender and defender.center is not None:
                    def_d = self._dist((float(defender.center[0]), float(defender.center[1])), (float(ball_center[0]), float(ball_center[1])))

            self._active_attempt.update_resolution(
                frame_number=frame_number,
                current_possessor_id=possessor_id,
                current_possessor_team_name=possessor_team_name,
                attacker_distance_to_ball_px=att_d,
                defender_distance_to_ball_px=def_d,
            )
            if self._active_attempt.is_done:
                self._resolved_attempts.append(self._active_attempt)
                self._active_attempt = None

        # If there is no active attempt, try to start one
        if self._active_attempt is None:
            attacker_id = possessor_id
            attacker_team_name = possessor_team_name
            if attacker_id is not None and attacker_team_name != "" and ball_center is not None:
                defender, dist_def_ball, dist_att_def = self._select_defender_candidate(
                    players, attacker_id, attacker_team_name, ball_center
                )
                if defender is not None:
                    # Proximity checks
                    close_enough = (
                        dist_att_def <= self.attacker_defender_close_px
                        and dist_def_ball <= self.contact_radius_px
                    )

                    # Kinematic cue: direction change or speed drop
                    dir_change_deg, drop_ratio = self._ball_dir_change_and_speed_drop()
                    kinematic_contact = (
                        dir_change_deg >= self.dir_change_deg_thresh or drop_ratio <= self.speed_drop_ratio
                    )

                    if close_enough and kinematic_contact:
                        attempt = TackleAttempt(
                            start_frame=frame_number,
                            attacker_id=attacker_id,
                            defender_id=defender.player_id,
                            defender_team_name=defender.team.name if defender.team else "",
                            attacker_team_name=attacker_team_name,
                            horizon_frames=self.horizon_frames,
                            confirm_frames=self.confirm_frames,
                        )
                        attempt.mark_contact(frame_number)
                        self._active_attempt = attempt

    def get_resolved(self) -> List[dict]:
        return [a.as_dict() for a in self._resolved_attempts]

    def get_active(self) -> Optional[dict]:
        return self._active_attempt.as_dict() if self._active_attempt else None


class Match:
    def __init__(
        self,
        home: Team,
        away: Team,
        fps: int = 30,
        pixels_to_meters: Optional[float] = None,
    ):
        """

        Initialize Match

        Parameters
        ----------
        home : Team
            Home team
        away : Team
            Away team
        fps : int, optional
            Fps, by default 30
        pixels_to_meters : Optional[float], optional
            Conversion factor from pixels to meters for distance tracking.
            If None, distances are tracked in pixels only.
            For example, if 100 pixels = 1 meter, set to 0.01.
            By default None
        """
        self.duration = 0
        self.home = home
        self.away = away
        self.team_possession = self.home
        self.current_team = self.home
        self.possession_counter = 0
        self.closest_player = None
        self.ball = None
        # Amount of consecutive frames new team has to have the ball in order to change possession
        self.possesion_counter_threshold = 20
        # Distance in pixels from player to ball in order to consider a player has the ball
        self.ball_distance_threshold = 45
        self.fps = fps
        # Pass detection
        self.pass_event = PassEvent()
        self.frame_number = 0
        # Distance tracking
        self.pixels_to_meters = pixels_to_meters
        self.distance_tracker = PlayerDistanceTracker(pixels_to_meters=pixels_to_meters)
        # Tackle detection (pixels + fps only)
        self.tackle_detector = TackleDetector(fps=fps)
        self.tackles: List[dict] = []

    def _load_font(self, size: int) -> ImageFont.ImageFont:
        font_path = Path(__file__).resolve().parent.parent / "fonts" / "Gidole-Regular.ttf"
        try:
            return ImageFont.truetype(str(font_path), size=size)
        except OSError:
            return ImageFont.load_default()

    def update(self, players: List[Player], ball: Ball, frame: Optional[np.ndarray] = None):
        """
        
        Update match possession and closest player

        Parameters
        ----------
        players : List[Player]
            List of players
        ball : Ball
            Ball
        frame : Optional[np.ndarray]
            Current frame image (BGR format).
        """
        self.update_possession()
        
        # Update distance tracking for all players
        # Collect player IDs present in this frame
        current_frame_player_ids = set()
        for player in players:
            if player.detection is not None and player.player_id is not None:
                current_frame_player_ids.add(player.player_id)
                self.distance_tracker.update_player_distance(player)
        
        # Update which players were present in this frame
        # This is used to detect new appearances in the next frame
        self.distance_tracker.update_frame_players(current_frame_player_ids)

        if ball is None or ball.detection is None:
            self.closest_player = None
        else:
            self.ball = ball

            closest_player = min(players, key=lambda player: player.distance_to_ball(ball))

            self.closest_player = closest_player

            ball_distance = closest_player.distance_to_ball(ball)

            if ball_distance > self.ball_distance_threshold:
                self.closest_player = None
            else:
                # Reset counter if team changed
                if closest_player.team != self.current_team:
                    self.possession_counter = 0
                    self.current_team = closest_player.team

                self.possession_counter += 1

                if (
                    self.possession_counter >= self.possesion_counter_threshold
                    and closest_player.team is not None
                ):
                    self.change_team(self.current_team)

                # Pass detection
                self.pass_event.update(closest_player=closest_player, ball=ball)

        self.pass_event.process_pass()

        self.frame_number += 1

        # Tackle detector update (safe-guarded)
        try:
            self.tackle_detector.update(
                frame_number=self.frame_number,
                players=players,
                ball=ball,
                team_possession=self.team_possession,
            )
            # Sync resolved tackles
            resolved_now = self.tackle_detector.get_resolved()
            if len(resolved_now) > len(self.tackles):
                self.tackles = resolved_now.copy()
        except Exception:
            # Never break the main pipeline
            pass

    def change_team(self, team: Team):
        """

        Change team possession

        Parameters
        ----------
        team : Team, optional
            New team in possession
        """
        previous_team = self.team_possession
        if (
            team is not None
            and previous_team is not None
            and previous_team != team
        ):
            previous_team.interceptions += 1
            team.ball_recoveries += 1

        self.team_possession = team

    def update_possession(self):
        """
        Updates match duration and possession counter of team in possession
        """
        if self.team_possession is None:
            return

        self.team_possession.possession += 1
        self.duration += 1

    @property
    def home_possession_str(self) -> str:
        return f"{self.home.abbreviation}: {self.home.get_time_possession(self.fps)}"

    @property
    def away_possession_str(self) -> str:
        return f"{self.away.abbreviation}: {self.away.get_time_possession(self.fps)}"

    def __str__(self) -> str:
        return f"{self.home_possession_str} | {self.away_possession_str}"

    @property
    def time_possessions(self) -> str:
        return f"{self.home.name}: {self.home.get_time_possession(self.fps)} | {self.away.name}: {self.away.get_time_possession(self.fps)}"

    @property
    def passes(self) -> List["Pass"]:
        home_passes = self.home.passes
        away_passes = self.away.passes

        return home_passes + away_passes

    def possession_bar(self, frame: PIL.Image.Image, origin: tuple) -> PIL.Image.Image:
        """
        Draw possession bar

        Parameters
        ----------
        frame : PIL.Image.Image
            Frame
        origin : tuple
            Origin (x, y)

        Returns
        -------
        PIL.Image.Image
            Frame with possession bar
        """

        bar_x = origin[0]
        bar_y = origin[1]
        bar_height = 29
        bar_width = 310

        ratio = self.home.get_percentage_possession(self.duration)

        # Protect against too small rectangles
        if ratio < 0.07:
            ratio = 0.07

        if ratio > 0.93:
            ratio = 0.93

        left_rectangle = (
            origin,
            [int(bar_x + ratio * bar_width), int(bar_y + bar_height)],
        )

        right_rectangle = (
            [int(bar_x + ratio * bar_width), bar_y],
            [int(bar_x + bar_width), int(bar_y + bar_height)],
        )

        left_color = self.home.board_color
        right_color = self.away.board_color

        frame = self.draw_counter_rectangle(
            frame=frame,
            ratio=ratio,
            left_rectangle=left_rectangle,
            left_color=left_color,
            right_rectangle=right_rectangle,
            right_color=right_color,
        )

        # Draw home text
        if ratio > 0.15:
            home_text = (
                f"{int(self.home.get_percentage_possession(self.duration) * 100)}%"
            )

            frame = Draw.text_in_middle_rectangle(
                img=frame,
                origin=left_rectangle[0],
                width=left_rectangle[1][0] - left_rectangle[0][0],
                height=left_rectangle[1][1] - left_rectangle[0][1],
                text=home_text,
                color=self.home.text_color,
            )

        # Draw away text
        if ratio < 0.85:
            away_text = (
                f"{int(self.away.get_percentage_possession(self.duration) * 100)}%"
            )

            frame = Draw.text_in_middle_rectangle(
                img=frame,
                origin=right_rectangle[0],
                width=right_rectangle[1][0] - right_rectangle[0][0],
                height=right_rectangle[1][1] - right_rectangle[0][1],
                text=away_text,
                color=self.away.text_color,
            )

        return frame

    def draw_counter_rectangle(
        self,
        frame: PIL.Image.Image,
        ratio: float,
        left_rectangle: tuple,
        left_color: tuple,
        right_rectangle: tuple,
        right_color: tuple,
    ) -> PIL.Image.Image:
        """Draw counter rectangle for both teams

        Parameters
        ----------
        frame : PIL.Image.Image
            Video frame
        ratio : float
            counter proportion
        left_rectangle : tuple
            rectangle for the left team in counter
        left_color : tuple
            color for the left team in counter
        right_rectangle : tuple
            rectangle for the right team in counter
        right_color : tuple
            color for the right team in counter

        Returns
        -------
        PIL.Image.Image
            Drawed video frame
        """

        # Draw first one rectangle or another in orther to make the
        # rectangle bigger for better rounded corners

        if ratio < 0.15:
            left_rectangle[1][0] += 20

            frame = Draw.half_rounded_rectangle(
                frame,
                rectangle=left_rectangle,
                color=left_color,
                radius=15,
            )

            frame = Draw.half_rounded_rectangle(
                frame,
                rectangle=right_rectangle,
                color=right_color,
                left=True,
                radius=15,
            )
        else:
            right_rectangle[0][0] -= 20

            frame = Draw.half_rounded_rectangle(
                frame,
                rectangle=right_rectangle,
                color=right_color,
                left=True,
                radius=15,
            )

            frame = Draw.half_rounded_rectangle(
                frame,
                rectangle=left_rectangle,
                color=left_color,
                radius=15,
            )

        return frame

    def passes_bar(self, frame: PIL.Image.Image, origin: tuple) -> PIL.Image.Image:
        """
        Draw passes bar

        Parameters
        ----------
        frame : PIL.Image.Image
            Frame
        origin : tuple
            Origin (x, y)

        Returns
        -------
        PIL.Image.Image
            Frame with passes bar
        """

        bar_x = origin[0]
        bar_y = origin[1]
        bar_height = 29
        bar_width = 310

        home_passes = len(self.home.passes)
        away_passes = len(self.away.passes)
        total_passes = home_passes + away_passes

        if total_passes == 0:
            home_ratio = 0
            away_ratio = 0
        else:
            home_ratio = home_passes / total_passes
            away_ratio = away_passes / total_passes

        ratio = home_ratio

        # Protect against too small rectangles
        if ratio < 0.07:
            ratio = 0.07

        if ratio > 0.93:
            ratio = 0.93

        left_rectangle = (
            origin,
            [int(bar_x + ratio * bar_width), int(bar_y + bar_height)],
        )

        right_rectangle = (
            [int(bar_x + ratio * bar_width), bar_y],
            [int(bar_x + bar_width), int(bar_y + bar_height)],
        )

        left_color = self.home.board_color
        right_color = self.away.board_color

        # Draw first one rectangle or another in orther to make the
        # rectangle bigger for better rounded corners
        frame = self.draw_counter_rectangle(
            frame=frame,
            ratio=ratio,
            left_rectangle=left_rectangle,
            left_color=left_color,
            right_rectangle=right_rectangle,
            right_color=right_color,
        )

        # Draw home text
        if ratio > 0.15:
            home_text = f"{int(home_ratio * 100)}%"

            frame = Draw.text_in_middle_rectangle(
                img=frame,
                origin=left_rectangle[0],
                width=left_rectangle[1][0] - left_rectangle[0][0],
                height=left_rectangle[1][1] - left_rectangle[0][1],
                text=home_text,
                color=self.home.text_color,
            )

        # Draw away text
        if ratio < 0.85:
            away_text = f"{int(away_ratio * 100)}%"

            frame = Draw.text_in_middle_rectangle(
                img=frame,
                origin=right_rectangle[0],
                width=right_rectangle[1][0] - right_rectangle[0][0],
                height=right_rectangle[1][1] - right_rectangle[0][1],
                text=away_text,
                color=self.away.text_color,
            )

        return frame

    def get_possession_background(
        self,
    ) -> PIL.Image.Image:
        """
        Get possession counter background

        Returns
        -------
        PIL.Image.Image
            Counter background
        """

        counter = PIL.Image.open("./images/possession_board.png").convert("RGBA")
        counter = Draw.add_alpha(counter, 210)
        counter = np.array(counter)
        red, green, blue, alpha = counter.T
        counter = np.array([blue, green, red, alpha])
        counter = counter.transpose()
        counter = PIL.Image.fromarray(counter)
        counter = counter.resize((int(315 * 1.2), int(210 * 1.2)))
        return counter

    def get_passes_background(self) -> PIL.Image.Image:
        """
        Get passes counter background

        Returns
        -------
        PIL.Image.Image
            Counter background
        """

        counter = PIL.Image.open("./images/passes_board.png").convert("RGBA")
        counter = Draw.add_alpha(counter, 210)
        counter = np.array(counter)
        red, green, blue, alpha = counter.T
        counter = np.array([blue, green, red, alpha])
        counter = counter.transpose()
        counter = PIL.Image.fromarray(counter)
        counter = counter.resize((int(315 * 1.2), int(210 * 1.2)))
        return counter

    def get_interceptions_background(self) -> PIL.Image.Image:
        """
        Get interceptions counter background

        Returns
        -------
        PIL.Image.Image
            Counter background
        """

        width = int(315 * 1.2)
        height = int(210 * 1.2)
        background = PIL.Image.new("RGBA", (width, height), (10, 14, 28, 220))
        draw = ImageDraw.Draw(background)

        title_font = self._load_font(size=34)
        subtitle_font = self._load_font(size=22)

        def _draw_centered(text: str, y: int, font: ImageFont.ImageFont):
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            x = (width - text_width) // 2
            draw.text((x, y), text, font=font, fill=(255, 255, 255, 235))

        _draw_centered("INTERCEPTIONS", 24, title_font)
        _draw_centered("BALL RECOVERIES", 74, subtitle_font)

        return background

    def draw_counter_background(
        self,
        frame: PIL.Image.Image,
        origin: tuple,
        counter_background: PIL.Image.Image,
    ) -> PIL.Image.Image:
        """
        Draw counter background

        Parameters
        ----------
        frame : PIL.Image.Image
            Frame
        origin : tuple
            Origin (x, y)
        counter_background : PIL.Image.Image
            Counter background

        Returns
        -------
        PIL.Image.Image
            Frame with counter background
        """
        frame.paste(counter_background, origin, counter_background)
        return frame

    def interceptions_bar(self, frame: PIL.Image.Image, origin: tuple) -> PIL.Image.Image:
        """
        Draw interceptions bar

        Parameters
        ----------
        frame : PIL.Image.Image
            Frame
        origin : tuple
            Origin (x, y)

        Returns
        -------
        PIL.Image.Image
            Frame with interceptions bar
        """

        bar_x = origin[0]
        bar_y = origin[1]
        bar_height = 29
        bar_width = 310

        home_interceptions = self.home.interceptions
        away_interceptions = self.away.interceptions
        total_interceptions = home_interceptions + away_interceptions

        if total_interceptions == 0:
            home_ratio = 0
            away_ratio = 0
        else:
            home_ratio = home_interceptions / total_interceptions
            away_ratio = away_interceptions / total_interceptions

        ratio = home_ratio

        if ratio < 0.07:
            ratio = 0.07

        if ratio > 0.93:
            ratio = 0.93

        left_rectangle = (
            origin,
            [int(bar_x + ratio * bar_width), int(bar_y + bar_height)],
        )

        right_rectangle = (
            [int(bar_x + ratio * bar_width), bar_y],
            [int(bar_x + bar_width), int(bar_y + bar_height)],
        )

        left_color = self.home.board_color
        right_color = self.away.board_color

        frame = self.draw_counter_rectangle(
            frame=frame,
            ratio=ratio,
            left_rectangle=left_rectangle,
            left_color=left_color,
            right_rectangle=right_rectangle,
            right_color=right_color,
        )

        if ratio > 0.15:
            home_text = f"{int(home_ratio * 100)}%"

            frame = Draw.text_in_middle_rectangle(
                img=frame,
                origin=left_rectangle[0],
                width=left_rectangle[1][0] - left_rectangle[0][0],
                height=left_rectangle[1][1] - left_rectangle[0][1],
                text=home_text,
                color=self.home.text_color,
            )

        if ratio < 0.85:
            away_text = f"{int(away_ratio * 100)}%"

            frame = Draw.text_in_middle_rectangle(
                img=frame,
                origin=right_rectangle[0],
                width=right_rectangle[1][0] - right_rectangle[0][0],
                height=right_rectangle[1][1] - right_rectangle[0][1],
                text=away_text,
                color=self.away.text_color,
            )

        return frame

    def draw_counter(
        self,
        frame: PIL.Image.Image,
        text: str,
        counter_text: str,
        origin: tuple,
        color: tuple,
        text_color: tuple,
        height: int = 27,
        width: int = 120,
    ) -> PIL.Image.Image:
        """
        Draw counter

        Parameters
        ----------
        frame : PIL.Image.Image
            Frame
        text : str
            Text in left-side of counter
        counter_text : str
            Text in right-side of counter
        origin : tuple
            Origin (x, y)
        color : tuple
            Color
        text_color : tuple
            Color of text
        height : int, optional
            Height, by default 27
        width : int, optional
            Width, by default 120

        Returns
        -------
        PIL.Image.Image
            Frame with counter
        """

        team_begin = origin
        team_width_ratio = 0.417
        team_width = width * team_width_ratio

        team_rectangle = (
            team_begin,
            (team_begin[0] + team_width, team_begin[1] + height),
        )

        time_begin = (origin[0] + team_width, origin[1])
        time_width = width * (1 - team_width_ratio)

        time_rectangle = (
            time_begin,
            (time_begin[0] + time_width, time_begin[1] + height),
        )

        frame = Draw.half_rounded_rectangle(
            img=frame,
            rectangle=team_rectangle,
            color=color,
            radius=20,
        )

        frame = Draw.half_rounded_rectangle(
            img=frame,
            rectangle=time_rectangle,
            color=(239, 234, 229),
            radius=20,
            left=True,
        )

        frame = Draw.text_in_middle_rectangle(
            img=frame,
            origin=team_rectangle[0],
            height=height,
            width=team_width,
            text=text,
            color=text_color,
        )

        frame = Draw.text_in_middle_rectangle(
            img=frame,
            origin=time_rectangle[0],
            height=height,
            width=time_width,
            text=counter_text,
            color="black",
        )

        return frame

    def draw_interceptions_counter(
        self,
        frame: PIL.Image.Image,
        counter_background: PIL.Image.Image,
        debug: bool = False,
    ) -> PIL.Image.Image:
        """
        Draw elements of the interceptions in frame

        Parameters
        ----------
        frame : PIL.Image.Image
            Frame
        counter_background : PIL.Image.Image
            Counter background
        debug : bool, optional
            Whether to draw extra debug information, by default False

        Returns
        -------
        PIL.Image.Image
            Frame with interceptions elements
        """

        frame_width = frame.size[0]
        frame_height = frame.size[1]
        margin_bottom = 40
        margin_left = 40
        background_height = counter_background.size[1]

        counter_origin_y = frame_height - background_height - margin_bottom
        counter_origin = (margin_left, counter_origin_y)

        frame = self.draw_counter_background(
            frame,
            origin=counter_origin,
            counter_background=counter_background,
        )

        interceptions_row_y = counter_origin[1] + 115
        recoveries_row_y = counter_origin[1] + 170
        bar_origin_y = counter_origin[1] + 210

        frame = self.draw_counter(
            frame,
            origin=(counter_origin[0] + 35, interceptions_row_y),
            text=self.home.abbreviation,
            counter_text=f"Int: {self.home.interceptions}",
            color=self.home.board_color,
            text_color=self.home.text_color,
            height=31,
            width=150,
        )

        frame = self.draw_counter(
            frame,
            origin=(counter_origin[0] + 35 + 150 + 10, interceptions_row_y),
            text=self.away.abbreviation,
            counter_text=f"Int: {self.away.interceptions}",
            color=self.away.board_color,
            text_color=self.away.text_color,
            height=31,
            width=150,
        )

        frame = self.draw_counter(
            frame,
            origin=(counter_origin[0] + 35, recoveries_row_y),
            text=self.home.abbreviation,
            counter_text=f"Rec: {self.home.ball_recoveries}",
            color=self.home.board_color,
            text_color=self.home.text_color,
            height=31,
            width=150,
        )

        frame = self.draw_counter(
            frame,
            origin=(counter_origin[0] + 35 + 150 + 10, recoveries_row_y),
            text=self.away.abbreviation,
            counter_text=f"Rec: {self.away.ball_recoveries}",
            color=self.away.board_color,
            text_color=self.away.text_color,
            height=31,
            width=150,
        )

        frame = self.interceptions_bar(frame, origin=(counter_origin[0] + 35, bar_origin_y))

        if self.closest_player:
            frame = self.closest_player.draw_pointer(frame)

        if debug:
            frame = self.draw_debug(frame=frame)

        return frame

    def draw_debug(self, frame: PIL.Image.Image) -> PIL.Image.Image:
        """Draw line from closest player feet to ball

        Parameters
        ----------
        frame : PIL.Image.Image
            Video frame

        Returns
        -------
        PIL.Image.Image
            Drawed video frame
        """
        if self.closest_player and self.ball:
            closest_foot = self.closest_player.closest_foot_to_ball(self.ball)

            color = (0, 0, 0)
            # Change line color if its greater than threshold
            distance = self.closest_player.distance_to_ball(self.ball)
            if distance > self.ball_distance_threshold:
                color = (255, 255, 255)

            draw = PIL.ImageDraw.Draw(frame)
            draw.line(
                [
                    tuple(closest_foot),
                    tuple(self.ball.center),
                ],
                fill=color,
                width=2,
            )

    def draw_possession_counter(
        self,
        frame: PIL.Image.Image,
        counter_background: PIL.Image.Image,
        debug: bool = False,
    ) -> PIL.Image.Image:
        """

        Draw elements of the possession in frame

        Parameters
        ----------
        frame : PIL.Image.Image
            Frame
        counter_background : PIL.Image.Image
            Counter background
        debug : bool, optional
            Whether to draw extra debug information, by default False

        Returns
        -------
        PIL.Image.Image
            Frame with elements of the match
        """

        # get width of PIL.Image
        frame_width = frame.size[0]
        counter_origin = (frame_width - 540, 40)

        frame = self.draw_counter_background(
            frame,
            origin=counter_origin,
            counter_background=counter_background,
        )

        frame = self.draw_counter(
            frame,
            origin=(counter_origin[0] + 35, counter_origin[1] + 130),
            text=self.home.abbreviation,
            counter_text=self.home.get_time_possession(self.fps),
            color=self.home.board_color,
            text_color=self.home.text_color,
            height=31,
            width=150,
        )
        frame = self.draw_counter(
            frame,
            origin=(counter_origin[0] + 35 + 150 + 10, counter_origin[1] + 130),
            text=self.away.abbreviation,
            counter_text=self.away.get_time_possession(self.fps),
            color=self.away.board_color,
            text_color=self.away.text_color,
            height=31,
            width=150,
        )
        frame = self.possession_bar(
            frame, origin=(counter_origin[0] + 35, counter_origin[1] + 195)
        )

        if self.closest_player:
            frame = self.closest_player.draw_pointer(frame)

        if debug:
            frame = self.draw_debug(frame=frame)

        return frame

    def draw_passes_counter(
        self,
        frame: PIL.Image.Image,
        counter_background: PIL.Image.Image,
        debug: bool = False,
    ) -> PIL.Image.Image:
        """

        Draw elements of the passes in frame

        Parameters
        ----------
        frame : PIL.Image.Image
            Frame
        counter_background : PIL.Image.Image
            Counter background
        debug : bool, optional
            Whether to draw extra debug information, by default False

        Returns
        -------
        PIL.Image.Image
            Frame with elements of the match
        """

        # get width and height of PIL.Image
        frame_width = frame.size[0]
        frame_height = frame.size[1]
        # Position at bottom right: counter background is ~260px tall, add margin
        counter_background_height = 260  # Approximate height of counter background
        margin_bottom = 40  # Margin from bottom edge
        counter_origin_y = frame_height - counter_background_height - margin_bottom
        counter_origin = (frame_width - 540, counter_origin_y)

        frame = self.draw_counter_background(
            frame,
            origin=counter_origin,
            counter_background=counter_background,
        )

        frame = self.draw_counter(
            frame,
            origin=(counter_origin[0] + 35, counter_origin[1] + 130),
            text=self.home.abbreviation,
            counter_text=str(len(self.home.passes)),
            color=self.home.board_color,
            text_color=self.home.text_color,
            height=31,
            width=150,
        )
        frame = self.draw_counter(
            frame,
            origin=(counter_origin[0] + 35 + 150 + 10, counter_origin[1] + 130),
            text=self.away.abbreviation,
            counter_text=str(len(self.away.passes)),
            color=self.away.board_color,
            text_color=self.away.text_color,
            height=31,
            width=150,
        )
        frame = self.passes_bar(
            frame, origin=(counter_origin[0] + 35, counter_origin[1] + 195)
        )

        if self.closest_player:
            frame = self.closest_player.draw_pointer(frame)

        if debug:
            frame = self.draw_debug(frame=frame)

        return frame
    
    def get_player_distance(self, player: Player, in_meters: bool = False) -> float:
        """
        Get the cumulative distance traveled by a player.
        
        Parameters
        ----------
        player : Player
            Player object
        in_meters : bool, optional
            If True, return distance in meters (requires calibration).
            If False, return distance in pixels. By default False
            
        Returns
        -------
        float
            Cumulative distance traveled by the player.
            Returns 0.0 if player ID not found.
        """
        if player.player_id is None:
            return 0.0
        return self.distance_tracker.get_player_distance(player.player_id, in_meters=in_meters)
    
    def get_team_total_distance(self, players: List[Player], team: Team, in_meters: bool = False) -> float:
        """
        Get the total cumulative distance traveled by all players on a team.
        
        Parameters
        ----------
        players : List[Player]
            Current list of players (to match IDs to teams)
        team : Team
            Team object
        in_meters : bool, optional
            If True, return distance in meters. By default False
            
        Returns
        -------
        float
            Total cumulative distance for the team.
        """
        total = 0.0
        all_distances = self.distance_tracker.get_all_distances(in_meters=in_meters)
        
        # Create a mapping of player_id -> team from current players
        player_id_to_team = {}
        for player in players:
            if player.player_id is not None and player.team is not None:
                player_id_to_team[player.player_id] = player.team
        
        # Sum distances for players on this team
        for player_id, distance in all_distances.items():
            if player_id in player_id_to_team and player_id_to_team[player_id] == team:
                total += distance
        
        return total
    
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
        return self.distance_tracker.get_all_distances(in_meters=in_meters)
    
    def get_distance_statistics(self, in_meters: bool = False) -> Dict[str, float]:
        """
        Get distance statistics across all tracked players.
        
        Parameters
        ----------
        in_meters : bool, optional
            If True, return distances in meters. By default False
            
        Returns
        -------
        Dict[str, float]
            Dictionary with keys: 'total', 'mean', 'min', 'max', 'median', 'count'
        """
        all_distances = self.distance_tracker.get_all_distances(in_meters=in_meters)
        
        if not all_distances:
            return {
                'total': 0.0,
                'mean': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
                'count': 0
            }
        
        distances_list = list(all_distances.values())
        
        return {
            'total': sum(distances_list),
            'mean': np.mean(distances_list),
            'min': np.min(distances_list),
            'max': np.max(distances_list),
            'median': np.median(distances_list),
            'count': len(distances_list)
        }
    
    def reset_distance_tracking(self):
        """
        Reset all distance tracking data.
        """
        self.distance_tracker.reset()

    # ---------------- Tackle accessors ----------------
    def get_tackles(self) -> List[dict]:
        """
        Return list of resolved tackle events:
        [{
          'start_frame', 'contact_frame', 'resolved_frame',
          'attacker_id', 'defender_id', 'attacker_team', 'defender_team', 'outcome'
        }, ...]
        """
        return self.tackles

    def get_active_tackle(self) -> Optional[dict]:
        """
        Return the currently active (unresolved) tackle attempt as dict, or None.
        """
        return self.tackle_detector.get_active()
