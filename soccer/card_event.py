from typing import List, Optional, TYPE_CHECKING

import numpy as np
import PIL
import PIL.ImageDraw
import PIL.ImageFont

from inference.card_ssd_detector import CardSSDDetector
from soccer.player import Player
from soccer.team import Team

if TYPE_CHECKING:
    from soccer.foul_event import Foul


class Card:
    def __init__(
        self,
        player: Player,
        card_type: str,
        foul: Optional[object] = None,  # Foul type, using object to avoid circular import
        frame_number: int = 0,
    ) -> None:
        """
        Initialize a Card event

        Parameters
        ----------
        player : Player
            Player who received the card
        card_type : str
            Type of card ("yellow" or "red")
        foul : Optional[Foul], optional
            Associated foul that led to the card, by default None
        frame_number : int, optional
            Frame number when card was shown, by default 0
        """
        self.player = player
        self.card_type = card_type
        self.foul = foul
        self.frame_number = frame_number
        self.team = player.team if player.team else None

    def draw(
        self, img: PIL.Image.Image, duration_frames: int = 60
    ) -> PIL.Image.Image:
        """
        Draw a card indicator on the frame

        Parameters
        ----------
        img : PIL.Image.Image
            Video frame
        duration_frames : int, optional
            Number of frames to show the card indicator, by default 60

        Returns
        -------
        PIL.Image.Image
            Frame with card indicator drawn
        """
        if self.player.detection is None:
            return img

        # Get player bounding box top center
        points = self.player.detection.points
        top_center_x = (points[0][0] + points[1][0]) / 2
        top_y = points[0][1]

        # Card position (above player)
        card_x = int(top_center_x)
        card_y = int(top_y - 40)

        # Card size and color
        card_width = 40
        card_height = 55

        if self.card_type == "red":
            color = (255, 0, 0)  # Red
            text_color = (255, 255, 255)  # White text
        else:  # yellow
            color = (255, 255, 0)  # Yellow
            text_color = (0, 0, 0)  # Black text

        draw = PIL.ImageDraw.Draw(img)

        # Draw card rectangle
        card_rect = [
            card_x - card_width // 2,
            card_y - card_height,
            card_x + card_width // 2,
            card_y,
        ]
        draw.rectangle(card_rect, fill=color, outline=(0, 0, 0), width=2)

        # Draw card type text
        try:
            font = PIL.ImageFont.truetype("arial.ttf", 16)
        except:
            font = PIL.ImageFont.load_default()

        text = "Y" if self.card_type == "yellow" else "R"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        text_pos = (
            card_x - text_width // 2,
            card_y - card_height // 2 - text_height // 2,
        )
        draw.text(text_pos, text, fill=text_color, font=font)

        # Draw line connecting card to player
        draw.line(
            [(card_x, card_y), (card_x, int(top_y))],
            fill=color,
            width=2,
        )

        return img

    @staticmethod
    def draw_card_list(
        img: PIL.Image.Image, cards: List["Card"], current_frame: int
    ) -> PIL.Image.Image:
        """
        Draw all recent cards on the frame

        Parameters
        ----------
        img : PIL.Image.Image
            Video frame
        cards : List[Card]
            List of cards to draw
        current_frame : int
            Current frame number

        Returns
        -------
        PIL.Image.Image
            Frame with cards drawn
        """
        # Only show cards from the last 60 frames (2 seconds at 30fps)
        recent_cards = [
            card for card in cards if current_frame - card.frame_number <= 60
        ]

        for card in recent_cards:
            img = card.draw(img, duration_frames=60)

        return img

    def __str__(self):
        team = self.player.team.name if self.player.team else "Unknown"
        player_id = (
            self.player.detection.data.get("id", "?")
            if self.player.detection
            else "?"
        )
        return f"{self.card_type.upper()} Card: Player {player_id} ({team})"


class CardEvent:
    def __init__(
        self,
        ssd_model_path: Optional[str] = None,
        use_ssd: bool = True,
        ssd_confidence: float = 0.5,
    ) -> None:
        """
        Initialize CardEvent detector
        Uses SSD-based object detection (primary) and foul-based heuristics (fallback):
        1. SSD Detection: Detect yellow/red card objects directly in frames
        2. Foul-based: Assign cards based on foul severity

        Parameters
        ----------
        ssd_model_path : Optional[str]
            Path to trained SSD/YOLOv5 model for card detection
        use_ssd : bool
            Whether to use SSD detector (primary method)
        ssd_confidence : float
            Minimum confidence for card detection
        """
        self.cards = []
        self.use_ssd = use_ssd
        
        # SSD-based detection
        if use_ssd:
            try:
                self.ssd_detector = CardSSDDetector(
                    model_path=ssd_model_path,
                    confidence_threshold=ssd_confidence,
                    use_yolo=True,
                )
                print("SSD-based card detector initialized")
            except Exception as e:
                print(f"Warning: Could not initialize SSD detector: {e}")
                print("Falling back to foul-based detection only")
                self.use_ssd = False
                self.ssd_detector = None
        else:
            self.ssd_detector = None
        
        # Foul-based card tracking
        self.player_cards = {}  # Track cards per player: {player_id: {"yellow": count, "red": count}}
        self.last_card_frame = {}  # Track last card frame per player

    def detect_card_from_foul(
        self, foul: object, frame_number: int, referee_present: bool = False  # Foul type
    ) -> Optional[Card]:
        """
        Detect if a card should be shown based on foul severity

        Parameters
        ----------
        foul : Foul
            Detected foul
        frame_number : int
            Current frame number
        referee_present : bool, optional
            Whether referee is present near the foul, by default False

        Returns
        -------
        Optional[Card]
            Detected card or None
        """
        # Determine which player committed the foul (simplified: player1)
        offender = foul.player1
        if offender.detection is None:
            return None

        player_id = offender.detection.data.get("id")
        if player_id is None:
            return None

        # Prevent duplicate cards for same player in short time
        if (
            player_id in self.last_card_frame
            and frame_number - self.last_card_frame[player_id] < 90
        ):
            return None

        # Initialize player card count
        if player_id not in self.player_cards:
            self.player_cards[player_id] = {"yellow": 0, "red": 0}

        card_type = None

        # Determine card type based on foul severity
        if foul.severity == "violent":
            # Red card for violent conduct
            card_type = "red"
            self.player_cards[player_id]["red"] += 1
        elif foul.severity == "serious":
            # Yellow card for serious foul
            if self.player_cards[player_id]["yellow"] < 1:
                card_type = "yellow"
                self.player_cards[player_id]["yellow"] += 1
            else:
                # Second yellow = red
                card_type = "red"
                self.player_cards[player_id]["red"] += 1
                self.player_cards[player_id]["yellow"] = 0
        elif referee_present:
            # Referee present might indicate a card-worthy foul
            if self.player_cards[player_id]["yellow"] < 1:
                card_type = "yellow"
                self.player_cards[player_id]["yellow"] += 1

        if card_type:
            card = Card(
                player=offender,
                card_type=card_type,
                foul=foul,
                frame_number=frame_number,
            )
            self.cards.append(card)
            self.last_card_frame[player_id] = frame_number
            return card

        return None

    def detect_referee_card(
        self, players: List[Player], frame_number: int
    ) -> Optional[Card]:
        """
        Detect cards when referee is present (simplified detection)

        Parameters
        ----------
        players : List[Player]
            List of all players (including referee if classified)
        frame_number : int
            Current frame number

        Returns
        -------
        Optional[Card]
            Detected card or None
        """
        # Check if referee is present
        referee = None
        for player in players:
            if player.team and player.team.name == "Referee":
                referee = player
                break

        if referee is None:
            return None

        # If referee is near a player, might indicate card shown
        # This is a simplified heuristic - in reality would need card detection
        for player in players:
            if player == referee or player.detection is None:
                continue

            # Calculate distance to referee
            ref_points = referee.detection.points
            player_points = player.detection.points

            ref_center = np.array(
                [
                    (ref_points[0][0] + ref_points[1][0]) / 2,
                    (ref_points[0][1] + ref_points[1][1]) / 2,
                ]
            )
            player_center = np.array(
                [
                    (player_points[0][0] + player_points[1][0]) / 2,
                    (player_points[0][1] + player_points[1][1]) / 2,
                ]
            )

            distance = np.linalg.norm(ref_center - player_center)

            # If referee is very close to player, might indicate card
            if distance < 100:
                player_id = player.detection.data.get("id")
                if player_id and (
                    player_id not in self.last_card_frame
                    or frame_number - self.last_card_frame[player_id] > 90
                ):
                    # Simplified: assume yellow card
                    if player_id not in self.player_cards:
                        self.player_cards[player_id] = {"yellow": 0, "red": 0}

                    if self.player_cards[player_id]["yellow"] < 1:
                        card = Card(
                            player=player,
                            card_type="yellow",
                            frame_number=frame_number,
                        )
                        self.cards.append(card)
                        self.player_cards[player_id]["yellow"] += 1
                        self.last_card_frame[player_id] = frame_number
                        return card

        return None

    def detect_cards_ssd(
        self,
        frame: np.ndarray,
        players: List[Player],
        frame_number: int,
    ) -> List[Card]:
        """
        SSD-based card detection: Detect card objects directly in frame

        Parameters
        ----------
        frame : np.ndarray
            Current frame image (BGR format)
        players : List[Player]
            List of all players (to associate cards with players)
        frame_number : int
            Current frame number

        Returns
        -------
        List[Card]
            List of detected cards
        """
        if not self.use_ssd or self.ssd_detector is None:
            return []

        detected_cards = []

        # Detect cards in frame
        card_detections = self.ssd_detector.detect_cards_in_frame(frame)

        for card_type, confidence, bbox in card_detections:
            # Find nearest player to associate card with
            xmin, ymin, xmax, ymax = bbox
            card_center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])

            nearest_player = None
            min_distance = float("inf")

            for player in players:
                if player.detection is None:
                    continue

                player_points = player.detection.points
                player_center = np.array(
                    [
                        (player_points[0][0] + player_points[1][0]) / 2,
                        (player_points[0][1] + player_points[1][1]) / 2,
                    ]
                )

                distance = np.linalg.norm(card_center - player_center)

                # Cards are typically shown near players (within 200 pixels)
                if distance < min_distance and distance < 200:
                    min_distance = distance
                    nearest_player = player

            if nearest_player:
                player_id = (
                    nearest_player.detection.data.get("id")
                    if nearest_player.detection
                    else None
                )

                # Prevent duplicate cards for same player in short time
                if (
                    player_id
                    and player_id in self.last_card_frame
                    and frame_number - self.last_card_frame[player_id] < 90
                ):
                    continue

                # Initialize player card count
                if player_id and player_id not in self.player_cards:
                    self.player_cards[player_id] = {"yellow": 0, "red": 0}

                # Create card
                card = Card(
                    player=nearest_player,
                    card_type=card_type,
                    frame_number=frame_number,
                )

                detected_cards.append(card)
                self.cards.append(card)

                if player_id:
                    self.player_cards[player_id][card_type] += 1
                    self.last_card_frame[player_id] = frame_number

        return detected_cards

    def update(
        self,
        fouls: List[object],  # List[Foul] type
        players: List[Player],
        frame_number: int,
        frame: Optional[np.ndarray] = None,
    ) -> None:
        """
        Update card detection system

        Parameters
        ----------
        fouls : List[Foul]
            List of recent fouls
        players : List[Player]
            List of all players
        frame_number : int
            Current frame number
        frame : Optional[np.ndarray]
            Current frame image (BGR format). Required for SSD-based detection.
        """
        # Method 1: SSD-based detection (primary if available)
        if self.use_ssd and frame is not None:
            self.detect_cards_ssd(frame, players, frame_number)

        # Method 2: Foul-based detection (fallback or additional)
        # Check for referee presence
        referee_present = any(
            player.team and player.team.name == "Referee" for player in players
        )

        # Detect cards from fouls
        for foul in fouls:
            # Only check recent fouls
            if frame_number - foul.frame_number <= 30:
                self.detect_card_from_foul(foul, frame_number, referee_present)

        # Also check for referee-based card detection
        self.detect_referee_card(players, frame_number)

    def recent_cards(self, frames: int = 180) -> List[Card]:
        """
        Get recent cards within specified frames

        Parameters
        ----------
        frames : int, optional
            Number of frames to look back, by default 180 (6 seconds at 30fps)

        Returns
        -------
        List[Card]
            List of recent cards
        """
        if not self.cards:
            return []

        latest_frame = max(card.frame_number for card in self.cards)
        return [
            card
            for card in self.cards
            if latest_frame - card.frame_number <= frames
        ]

    def get_cards_by_team(self, team: Team) -> List[Card]:
        """
        Get all cards for a specific team

        Parameters
        ----------
        team : Team
            Team to get cards for

        Returns
        -------
        List[Card]
            List of cards for the team
        """
        return [
            card
            for card in self.cards
            if card.team and card.team.name == team.name
        ]

