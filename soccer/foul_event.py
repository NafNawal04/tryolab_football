from typing import List, Optional

import numpy as np
import PIL
import PIL.ImageDraw
import PIL.ImageFont

from inference.foul_cnn_classifier import FoulCNNClassifier
from soccer.draw import Draw
from soccer.player import Player
from soccer.team import Team


class Foul:
    def __init__(
        self,
        player1: Player,
        player2: Player,
        location: np.ndarray,
        severity: str = "normal",
        frame_number: int = 0,
    ) -> None:
        """
        Initialize a Foul event

        Parameters
        ----------
        player1 : Player
            First player involved in the foul
        player2 : Player
            Second player involved in the foul
        location : np.ndarray
            Location where the foul occurred (center point between players)
        severity : str, optional
            Severity of the foul ("normal", "serious", "violent"), by default "normal"
        frame_number : int, optional
            Frame number when foul occurred, by default 0
        """
        self.player1 = player1
        self.player2 = player2
        self.location = location
        self.severity = severity
        self.frame_number = frame_number
        self.team = player1.team if player1.team else None

    def draw(
        self, img: PIL.Image.Image, duration_frames: int = 30
    ) -> PIL.Image.Image:
        """
        Draw a foul indicator on the frame

        Parameters
        ----------
        img : PIL.Image.Image
            Video frame
        duration_frames : int, optional
            Number of frames to show the foul indicator, by default 30

        Returns
        -------
        PIL.Image.Image
            Frame with foul indicator drawn
        """
        # Draw circle at foul location
        center = tuple(self.location.astype(int))
        
        # Color based on severity
        if self.severity == "violent":
            color = (255, 0, 0)  # Red
            radius = 25
        elif self.severity == "serious":
            color = (255, 165, 0)  # Orange
            radius = 20
        else:
            color = (255, 255, 0)  # Yellow
            radius = 15

        # Draw circle
        draw = PIL.ImageDraw.Draw(img)
        draw.ellipse(
            [
                center[0] - radius,
                center[1] - radius,
                center[0] + radius,
                center[1] + radius,
            ],
            outline=color,
            width=3,
        )
        
        # Draw "FOUL" text
        font = PIL.ImageFont.load_default()
        text = "FOUL"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        text_pos = (center[0] - text_width // 2, center[1] - text_height - radius - 5)
        draw.text(text_pos, text, fill=color, font=font)

        return img

    @staticmethod
    def draw_foul_list(
        img: PIL.Image.Image, fouls: List["Foul"], current_frame: int
    ) -> PIL.Image.Image:
        """
        Draw all recent fouls on the frame

        Parameters
        ----------
        img : PIL.Image.Image
            Video frame
        fouls : List[Foul]
            List of fouls to draw
        current_frame : int
            Current frame number

        Returns
        -------
        PIL.Image.Image
            Frame with fouls drawn
        """
        # Only show fouls from the last 30 frames (1 second at 30fps)
        recent_fouls = [
            foul
            for foul in fouls
            if current_frame - foul.frame_number <= 30
        ]
        
        for foul in recent_fouls:
            img = foul.draw(img, duration_frames=30)

        return img

    def __str__(self):
        team1 = self.player1.team.name if self.player1.team else "Unknown"
        team2 = self.player2.team.name if self.player2.team else "Unknown"
        return f"Foul: {team1} vs {team2} at {self.location}, severity: {self.severity}"


class FoulEvent:
    def __init__(
        self,
        cnn_model_path: Optional[str] = None,
        use_cnn: bool = True,
        cnn_threshold: float = 0.5,
    ) -> None:
        """
        Initialize FoulEvent detector
        Uses CNN-based classification (primary) and heuristic methods (fallback):
        1. CNN Classification: Pre-trained model classifies collision regions as Foul/Not Foul
        2. Collision detection (player-to-player proximity) - fallback/heuristic pre-filter
        3. Behavior analysis (sudden movements, falls) - fallback

        Parameters
        ----------
        cnn_model_path : Optional[str]
            Path to trained CNN model. If None, uses pre-trained ImageNet weights only.
        use_cnn : bool
            Whether to use CNN classifier (primary method)
        cnn_threshold : float
            CNN confidence threshold for foul classification
        """
        self.players_history = []  # Store recent player positions
        self.fouls = []
        self.use_cnn = use_cnn
        
        # CNN-based classification
        if use_cnn:
            try:
                self.cnn_classifier = FoulCNNClassifier(
                    model_path=cnn_model_path,
                    model_type="densenet121",
                    threshold=cnn_threshold,
                )
                print("CNN-based foul classifier initialized")
            except Exception as e:
                print(f"Warning: Could not initialize CNN classifier: {e}")
                print("Falling back to heuristic methods only")
                self.use_cnn = False
                self.cnn_classifier = None
        else:
            self.cnn_classifier = None
        
        # Heuristic methods (fallback or pre-filter)
        self.collision_threshold = 50  # Minimum distance in pixels for collision
        self.velocity_threshold = 10  # Threshold for sudden movement detection
        self.collision_frame_counter = 0
        self.collision_threshold_frames = 5  # Frames to confirm collision
        self.last_collision_frame = -100  # Prevent duplicate detections
        
        # Method 2: Behavior analysis
        self.player_velocities = {}  # Track player velocities
        self.fall_detection_threshold = 5  # Frames with low movement

    def calculate_player_distance(self, player1: Player, player2: Player) -> float:
        """
        Calculate distance between two players (using bounding box centers)

        Parameters
        ----------
        player1 : Player
            First player
        player2 : Player
            Second player

        Returns
        -------
        float
            Distance between players
        """
        if (
            player1.detection is None
            or player2.detection is None
            or player1 == player2
        ):
            return float("inf")

        # Get center of bounding boxes
        p1_points = player1.detection.points
        p2_points = player2.detection.points

        p1_center = np.array(
            [(p1_points[0][0] + p1_points[1][0]) / 2, (p1_points[0][1] + p1_points[1][1]) / 2]
        )
        p2_center = np.array(
            [(p2_points[0][0] + p2_points[1][0]) / 2, (p2_points[0][1] + p2_points[1][1]) / 2]
        )

        return np.linalg.norm(p1_center - p2_center)

    def calculate_player_velocity(self, player: Player, history: List[List[Player]]) -> float:
        """
        Calculate player velocity based on position history

        Parameters
        ----------
        player : Player
            Current player
        history : List[List[Player]]
            Previous player positions (list of frames, each frame is a list of players)

        Returns
        -------
        float
            Player velocity
        """
        if player.detection is None:
            return 0.0

        player_id = player.detection.data.get("id")
        if player_id is None:
            return 0.0

        # Find player in history (go through frames in reverse order)
        for frame_players in reversed(history):
            # frame_players is a list of Player objects from one frame
            if not isinstance(frame_players, list):
                continue
                
            for prev_player in frame_players:
                if (
                    isinstance(prev_player, Player)
                    and prev_player.detection
                    and "id" in prev_player.detection.data
                    and prev_player.detection.data["id"] == player_id
                ):
                    # Calculate displacement
                    p1_points = prev_player.detection.points
                    p2_points = player.detection.points

                    p1_center = np.array(
                        [
                            (p1_points[0][0] + p1_points[1][0]) / 2,
                            (p1_points[0][1] + p1_points[1][1]) / 2,
                        ]
                    )
                    p2_center = np.array(
                        [
                            (p2_points[0][0] + p2_points[1][0]) / 2,
                            (p2_points[0][1] + p2_points[1][1]) / 2,
                        ]
                    )

                    displacement = np.linalg.norm(p2_center - p1_center)
                    return displacement

        return 0.0

    def detect_collision_cnn(
        self,
        players: List[Player],
        frame: np.ndarray,
        frame_number: int,
    ) -> Optional[Foul]:
        """
        CNN-based foul detection: Classify collision regions using CNN

        Parameters
        ----------
        players : List[Player]
            List of all players
        frame : np.ndarray
            Current frame image (BGR format)
        frame_number : int
            Current frame number

        Returns
        -------
        Optional[Foul]
            Detected foul or None
        """
        if not self.use_cnn or self.cnn_classifier is None:
            return None

        # Prevent duplicate detections
        if frame_number - self.last_collision_frame < 30:
            return None

        # Find potential collisions (players from different teams close together)
        for i, player1 in enumerate(players):
            if player1.detection is None:
                continue

            for player2 in players[i + 1 :]:
                if player2.detection is None:
                    continue

                # Skip if same team (normal play)
                if player1.team == player2.team:
                    continue

                distance = self.calculate_player_distance(player1, player2)

                # Only check collisions within reasonable distance
                if distance < self.collision_threshold * 2:
                    # Crop region around both players
                    p1_points = player1.detection.points
                    p2_points = player2.detection.points

                    cropped_region = self.cnn_classifier.crop_foul_region(
                        frame, p1_points, p2_points, margin=50
                    )

                    # Classify using CNN
                    is_foul, confidence = self.cnn_classifier.predict_single(
                        cropped_region
                    )

                    if is_foul and confidence > self.cnn_classifier.threshold:
                        # Determine severity based on distance and confidence
                        if distance < 30 or confidence > 0.8:
                            severity = "violent"
                        elif distance < 40 or confidence > 0.65:
                            severity = "serious"
                        else:
                            severity = "normal"

                        # Calculate foul location
                        p1_center = np.array(
                            [
                                (p1_points[0][0] + p1_points[1][0]) / 2,
                                (p1_points[0][1] + p1_points[1][1]) / 2,
                            ]
                        )
                        p2_center = np.array(
                            [
                                (p2_points[0][0] + p2_points[1][0]) / 2,
                                (p2_points[0][1] + p2_points[1][1]) / 2,
                            ]
                        )

                        location = (p1_center + p2_center) / 2

                        foul = Foul(
                            player1=player1,
                            player2=player2,
                            location=location,
                            severity=severity,
                            frame_number=frame_number,
                        )

                        self.fouls.append(foul)
                        self.last_collision_frame = frame_number

                        return foul

        return None

    def detect_collision(self, players: List[Player], frame_number: int) -> Optional[Foul]:
        """
        Method 1: Detect fouls based on player collisions (proximity)

        Parameters
        ----------
        players : List[Player]
            List of all players
        frame_number : int
            Current frame number

        Returns
        -------
        Optional[Foul]
            Detected foul or None
        """
        # Prevent duplicate detections
        if frame_number - self.last_collision_frame < 30:
            return None

        # Check all player pairs
        for i, player1 in enumerate(players):
            if player1.detection is None:
                continue

            for player2 in players[i + 1 :]:
                if player2.detection is None:
                    continue

                # Skip if same team (normal play)
                if player1.team == player2.team:
                    continue

                distance = self.calculate_player_distance(player1, player2)

                if distance < self.collision_threshold:
                    self.collision_frame_counter += 1

                    if self.collision_frame_counter >= self.collision_threshold_frames:
                        # Determine severity based on distance
                        if distance < 30:
                            severity = "violent"
                        elif distance < 40:
                            severity = "serious"
                        else:
                            severity = "normal"

                        # Calculate foul location (midpoint between players)
                        p1_points = player1.detection.points
                        p2_points = player2.detection.points

                        p1_center = np.array(
                            [
                                (p1_points[0][0] + p1_points[1][0]) / 2,
                                (p1_points[0][1] + p1_points[1][1]) / 2,
                            ]
                        )
                        p2_center = np.array(
                            [
                                (p2_points[0][0] + p2_points[1][0]) / 2,
                                (p2_points[0][1] + p2_points[1][1]) / 2,
                            ]
                        )

                        location = (p1_center + p2_center) / 2

                        foul = Foul(
                            player1=player1,
                            player2=player2,
                            location=location,
                            severity=severity,
                            frame_number=frame_number,
                        )

                        self.fouls.append(foul)
                        self.last_collision_frame = frame_number
                        self.collision_frame_counter = 0

                        return foul
                else:
                    self.collision_frame_counter = 0

        return None

    def detect_behavior_foul(
        self, players: List[Player], frame_number: int
    ) -> Optional[Foul]:
        """
        Method 2: Detect fouls based on player behavior (sudden stops, falls)

        Parameters
        ----------
        players : List[Player]
            List of all players
        frame_number : int
            Current frame number

        Returns
        -------
        Optional[Foul]
            Detected foul or None
        """
        if len(self.players_history) < 5:
            return None

        # Prevent duplicate detections
        if frame_number - self.last_collision_frame < 30:
            return None

        for player in players:
            if player.detection is None:
                continue

            velocity = self.calculate_player_velocity(player, self.players_history)

            # Detect sudden stop (high velocity to zero)
            player_id = player.detection.data.get("id")
            if player_id in self.player_velocities:
                prev_velocity = self.player_velocities[player_id]
                if prev_velocity > self.velocity_threshold and velocity < 2:
                    # Sudden stop detected - possible foul
                    # Check if there's a nearby opponent
                    for other_player in players:
                        if (
                            other_player.detection is None
                            or other_player.team == player.team
                            or other_player == player
                        ):
                            continue

                        distance = self.calculate_player_distance(player, other_player)
                        if distance < self.collision_threshold * 1.5:
                            # Calculate location
                            p1_points = player.detection.points
                            p2_points = other_player.detection.points

                            p1_center = np.array(
                                [
                                    (p1_points[0][0] + p1_points[1][0]) / 2,
                                    (p1_points[0][1] + p1_points[1][1]) / 2,
                                ]
                            )
                            p2_center = np.array(
                                [
                                    (p2_points[0][0] + p2_points[1][0]) / 2,
                                    (p2_points[0][1] + p2_points[1][1]) / 2,
                                ]
                            )

                            location = (p1_center + p2_center) / 2

                            foul = Foul(
                                player1=player,
                                player2=other_player,
                                location=location,
                                severity="normal",
                                frame_number=frame_number,
                            )

                            self.fouls.append(foul)
                            self.last_collision_frame = frame_number

                            return foul

            # Update velocity history
            if player_id:
                self.player_velocities[player_id] = velocity

        return None

    def update(
        self,
        players: List[Player],
        frame_number: int,
        frame: Optional[np.ndarray] = None,
    ) -> None:
        """
        Update foul detection system

        Parameters
        ----------
        players : List[Player]
            List of all players
        frame_number : int
            Current frame number
        frame : Optional[np.ndarray]
            Current frame image (BGR format). Required for CNN-based detection.
        """
        # Method 1: CNN-based classification (primary if available)
        if self.use_cnn and frame is not None:
            self.detect_collision_cnn(players, frame, frame_number)
        else:
            # Fallback: Heuristic collision detection
            self.detect_collision(players, frame_number)

        # Method 2: Behavior analysis (fallback)
        self.detect_behavior_foul(players, frame_number)

        # Update history (keep last 10 frames)
        self.players_history.append(players)
        if len(self.players_history) > 10:
            self.players_history.pop(0)

    def recent_fouls(self, frames: int = 90) -> List[Foul]:
        """
        Get recent fouls within specified frames

        Parameters
        ----------
        frames : int, optional
            Number of frames to look back, by default 90 (3 seconds at 30fps)

        Returns
        -------
        List[Foul]
            List of recent fouls
        """
        if not self.fouls:
            return []

        # Return fouls from last N frames
        latest_frame = max(foul.frame_number for foul in self.fouls)
        return [
            foul
            for foul in self.fouls
            if latest_frame - foul.frame_number <= frames
        ]

    def get_fouls_by_team(self, team: Team) -> List[Foul]:
        """
        Get all fouls for a specific team

        Parameters
        ----------
        team : Team
            Team to get fouls for

        Returns
        -------
        List[Foul]
            List of fouls for the team
        """
        return [
            foul
            for foul in self.fouls
            if foul.team and foul.team.name == team.name
        ]

