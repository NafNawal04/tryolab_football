from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import PIL
from PIL import ImageDraw, ImageFont

from soccer.ball import Ball
from soccer.distance_tracker import PlayerDistanceTracker
from soccer.draw import Draw
from soccer.pass_event import Pass, PassEvent
from soccer.player import Player
from soccer.set_piece_detector import SetPieceDetector
from soccer.tackle_detector import TackleDetector
from soccer.team import Team


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
        # Set piece detection
        self.set_piece_detector = SetPieceDetector(fps=fps)
        self.set_pieces: List[dict] = []

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
        
        # Set piece detector update (safe-guarded)
        try:
            self.set_piece_detector.update(
                frame_number=self.frame_number,
                players=players,
                ball=ball,
                team_possession=self.team_possession,
                closest_player=self.closest_player,
            )
            # Sync resolved set pieces
            resolved_now = self.set_piece_detector.get_resolved()
            if len(resolved_now) > len(self.set_pieces):
                self.set_pieces = resolved_now.copy()
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

    def get_tackles_background(self) -> PIL.Image.Image:
        """
        Get tackles counter background

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

        _draw_centered("TACKLES", 24, title_font)
        _draw_centered("SUCCESS / FAILED", 74, subtitle_font)

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

    def draw_tackles_counter(
        self,
        frame: PIL.Image.Image,
        counter_background: PIL.Image.Image,
        debug: bool = False,
    ) -> PIL.Image.Image:
        """
        Draw elements of the tackles in frame

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
            Frame with tackles elements
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

        # Count tackles by team and outcome
        # Note: defender_team is the team that made the tackle
        home_successful = sum(1 for t in self.tackles if t.get('defender_team') == self.home.name and t.get('outcome') == 'success')
        home_failed = sum(1 for t in self.tackles if t.get('defender_team') == self.home.name and t.get('outcome') == 'failure')
        away_successful = sum(1 for t in self.tackles if t.get('defender_team') == self.away.name and t.get('outcome') == 'success')
        away_failed = sum(1 for t in self.tackles if t.get('defender_team') == self.away.name and t.get('outcome') == 'failure')

        successful_row_y = counter_origin[1] + 115
        failed_row_y = counter_origin[1] + 170
        bar_origin_y = counter_origin[1] + 210

        # Successful tackles row - show both teams
        frame = self.draw_counter(
            frame,
            origin=(counter_origin[0] + 35, successful_row_y),
            text="Successful",
            counter_text=f"{self.home.abbreviation} [{home_successful}] {self.away.abbreviation} [{away_successful}]",
            color=(0, 200, 0),  # Green for successful
            text_color=(255, 255, 255),
            height=31,
            width=310,
        )

        # Failed tackles row - show both teams
        frame = self.draw_counter(
            frame,
            origin=(counter_origin[0] + 35, failed_row_y),
            text="Failed",
            counter_text=f"{self.home.abbreviation} [{home_failed}] {self.away.abbreviation} [{away_failed}]",
            color=(200, 0, 0),  # Red for failed
            text_color=(255, 255, 255),
            height=31,
            width=310,
        )

        # Bar showing ratio
        total_tackles = len(self.tackles)
        if total_tackles > 0:
            home_total = home_successful + home_failed
            home_ratio = home_total / total_tackles
        else:
            home_ratio = 0.5

        if home_ratio < 0.07:
            home_ratio = 0.07
        if home_ratio > 0.93:
            home_ratio = 0.93

        bar_x = counter_origin[0] + 35
        bar_y = bar_origin_y
        bar_height = 29
        bar_width = 310

        left_rectangle = (
            (bar_x, bar_y),
            [int(bar_x + home_ratio * bar_width), int(bar_y + bar_height)],
        )

        right_rectangle = (
            [int(bar_x + home_ratio * bar_width), bar_y],
            [int(bar_x + bar_width), int(bar_y + bar_height)],
        )

        frame = self.draw_counter_rectangle(
            frame=frame,
            ratio=home_ratio,
            left_rectangle=left_rectangle,
            left_color=self.home.board_color,
            right_rectangle=right_rectangle,
            right_color=self.away.board_color,
        )

        if home_ratio > 0.15:
            home_text = f"{int(home_ratio * 100)}%"
            frame = Draw.text_in_middle_rectangle(
                img=frame,
                origin=left_rectangle[0],
                width=left_rectangle[1][0] - left_rectangle[0][0],
                height=left_rectangle[1][1] - left_rectangle[0][1],
                text=home_text,
                color=self.home.text_color,
            )

        if home_ratio < 0.85:
            away_text = f"{int((1 - home_ratio) * 100)}%"
            frame = Draw.text_in_middle_rectangle(
                img=frame,
                origin=right_rectangle[0],
                width=right_rectangle[1][0] - right_rectangle[0][0],
                height=right_rectangle[1][1] - right_rectangle[0][1],
                text=away_text,
                color=self.away.text_color,
            )

        if debug:
            frame = self.draw_debug(frame=frame)

        return frame

    def draw_active_tackles(
        self,
        frame: PIL.Image.Image,
        players: List[Player],
    ) -> PIL.Image.Image:
        """
        Draw active tackle attempts on the field.
        Shows a line between attacker and defender with color indicating status.

        Parameters
        ----------
        frame : PIL.Image.Image
            Frame
        players : List[Player]
            Current list of players

        Returns
        -------
        PIL.Image.Image
            Frame with active tackle indicators
        """
        active = self.get_active_tackle()
        if active is None:
            return frame

        # Create mapping of player_id -> Player
        player_map = {p.player_id: p for p in players if p.player_id is not None}

        attacker_id = active.get('attacker_id')
        defender_id = active.get('defender_id')

        attacker = player_map.get(attacker_id) if attacker_id is not None else None
        defender = player_map.get(defender_id) if defender_id is not None else None

        if attacker is None or defender is None:
            return frame

        # Get player centers
        attacker_center = attacker.center
        defender_center = defender.center

        if attacker_center is None or defender_center is None:
            return frame

        # Convert to tuple for drawing
        attacker_pos = (int(attacker_center[0]), int(attacker_center[1]))
        defender_pos = (int(defender_center[0]), int(defender_center[1]))

        draw = ImageDraw.Draw(frame)

        # Color based on state: orange for contact (active tackle)
        state = active.get('state', 'contact')
        if state == 'contact':
            line_color = (255, 165, 0)  # Orange - active tackle
            line_width = 3
        elif state == 'resolve':
            line_color = (255, 0, 0)  # Red - resolving
            line_width = 4
        else:
            line_color = (200, 200, 200)  # Gray
            line_width = 2

        # Draw line between attacker and defender
        draw.line([attacker_pos, defender_pos], fill=line_color, width=line_width)

        # Draw circles at player positions
        circle_radius = 8
        draw.ellipse(
            [
                attacker_pos[0] - circle_radius,
                attacker_pos[1] - circle_radius,
                attacker_pos[0] + circle_radius,
                attacker_pos[1] + circle_radius,
            ],
            outline=line_color,
            width=2,
        )
        draw.ellipse(
            [
                defender_pos[0] - circle_radius,
                defender_pos[1] - circle_radius,
                defender_pos[0] + circle_radius,
                defender_pos[1] + circle_radius,
            ],
            outline=line_color,
            width=2,
        )

        # Draw label near defender showing state
        label_font = self._load_font(size=16)
        label_text = f"Tackle: {state}"
        bbox = draw.textbbox((0, 0), label_text, font=label_font)
        label_width = bbox[2] - bbox[0]
        label_height = bbox[3] - bbox[1]
        label_x = defender_pos[0] - label_width // 2
        label_y = defender_pos[1] - label_height - 15

        # Draw background for label
        padding = 4
        draw.rectangle(
            [
                label_x - padding,
                label_y - padding,
                label_x + label_width + padding,
                label_y + label_height + padding,
            ],
            fill=(0, 0, 0, 180),
        )
        draw.text((label_x, label_y), label_text, font=label_font, fill=line_color)

        return frame

    def draw_recent_tackles(
        self,
        frame: PIL.Image.Image,
        players: List[Player],
        recent_frames: int = 60,
    ) -> PIL.Image.Image:
        """
        Draw recently resolved tackles on the field.
        Shows outcome with colored indicators.

        Parameters
        ----------
        frame : PIL.Image.Image
            Frame
        players : List[Player]
            Current list of players
        recent_frames : int, optional
            Number of frames to show recent tackles (default: 60, ~2 seconds at 30fps)

        Returns
        -------
        PIL.Image.Image
            Frame with recent tackle indicators
        """
        if self.frame_number == 0:
            return frame

        # Get recent tackles (resolved in last N frames)
        recent_tackles = [
            t for t in self.tackles
            if t.get('resolved_frame') is not None 
            and self.frame_number - recent_frames <= t.get('resolved_frame', 0) <= self.frame_number
        ]

        if not recent_tackles:
            return frame

        # Create mapping of player_id -> Player
        player_map = {p.player_id: p for p in players if p.player_id is not None}

        draw = ImageDraw.Draw(frame)
        label_font = self._load_font(size=14)

        for tackle in recent_tackles[-3:]:  # Show max 3 most recent
            attacker_id = tackle.get('attacker_id')
            defender_id = tackle.get('defender_id')
            outcome = tackle.get('outcome', 'inconclusive')

            attacker = player_map.get(attacker_id) if attacker_id is not None else None
            defender = player_map.get(defender_id) if defender_id is not None else None

            if attacker is None or defender is None:
                continue

            attacker_center = attacker.center
            defender_center = defender.center

            if attacker_center is None or defender_center is None:
                continue

            attacker_pos = (int(attacker_center[0]), int(attacker_center[1]))
            defender_pos = (int(defender_center[0]), int(defender_center[1]))

            # Color based on outcome
            if outcome == 'success':
                line_color = (0, 255, 0)  # Green
                outcome_text = "✓ SUCCESS"
            elif outcome == 'failure':
                line_color = (255, 0, 0)  # Red
                outcome_text = "✗ FAILED"
            else:
                line_color = (128, 128, 128)  # Gray
                outcome_text = "? INCONCLUSIVE"

            # Draw dashed line (simulated with small segments)
            line_width = 2
            segment_length = 10
            gap_length = 5
            dx = defender_pos[0] - attacker_pos[0]
            dy = defender_pos[1] - attacker_pos[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance > 0:
                unit_x = dx / distance
                unit_y = dy / distance
                num_segments = int(distance / (segment_length + gap_length))
                
                for i in range(num_segments):
                    start_ratio = i * (segment_length + gap_length) / distance
                    end_ratio = (i * (segment_length + gap_length) + segment_length) / distance
                    if end_ratio > 1.0:
                        end_ratio = 1.0
                    
                    start_x = int(attacker_pos[0] + start_ratio * dx)
                    start_y = int(attacker_pos[1] + start_ratio * dy)
                    end_x = int(attacker_pos[0] + end_ratio * dx)
                    end_y = int(attacker_pos[1] + end_ratio * dy)
                    
                    draw.line([(start_x, start_y), (end_x, end_y)], fill=line_color, width=line_width)

            # Draw outcome label near midpoint
            mid_x = (attacker_pos[0] + defender_pos[0]) // 2
            mid_y = (attacker_pos[1] + defender_pos[1]) // 2

            bbox = draw.textbbox((0, 0), outcome_text, font=label_font)
            label_width = bbox[2] - bbox[0]
            label_height = bbox[3] - bbox[1]
            label_x = mid_x - label_width // 2
            label_y = mid_y - label_height - 10

            # Draw background for label
            padding = 4
            draw.rectangle(
                [
                    label_x - padding,
                    label_y - padding,
                    label_x + label_width + padding,
                    label_y + label_height + padding,
                ],
                fill=(0, 0, 0, 200),
            )
            draw.text((label_x, label_y), outcome_text, font=label_font, fill=line_color)

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
    
    # ---------------- Set piece accessors ----------------
    def get_set_pieces(self) -> List[dict]:
        """
        Return list of resolved set piece events:
        [{
          'start_frame', 'wall_detected_frame', 'ball_kicked_frame', 'resolved_frame',
          'attacking_team', 'defending_team', 'type', 'wall_player_count'
        }, ...]
        """
        return self.set_pieces
    
    def get_active_set_piece(self) -> Optional[dict]:
        """
        Return the currently active (unresolved) set piece as dict, or None.
        """
        return self.set_piece_detector.get_active()
    
    def get_match_time_minutes(self) -> float:
        """
        Get current match time in minutes.
        
        Returns:
        --------
        float
            Match time in minutes
        """
        if self.fps == 0:
            return 0.0
        return (self.duration / self.fps) / 60.0
    
    def get_tackles_won_per_minute(self, team: Team) -> float:
        """
        Calculate tackles won per minute for a team.
        
        Parameters:
        -----------
        team : Team
            Team to calculate for
        
        Returns:
        --------
        float
            Tackles won per minute (0.0 if no time has passed)
        """
        match_time_minutes = self.get_match_time_minutes()
        if match_time_minutes == 0:
            return 0.0
        
        # Count successful tackles by this team
        tackles_won = sum(
            1 for t in self.tackles 
            if t.get('defender_team') == team.name and t.get('outcome') == 'success'
        )
        
        return tackles_won / match_time_minutes
    
    def get_passes_per_minute(self, team: Team) -> float:
        """
        Calculate passes per minute for a team.
        
        Parameters:
        -----------
        team : Team
            Team to calculate for
        
        Returns:
        --------
        float
            Passes per minute (0.0 if no time has passed)
        """
        match_time_minutes = self.get_match_time_minutes()
        if match_time_minutes == 0:
            return 0.0
        
        total_passes = len(team.passes)
        return total_passes / match_time_minutes
    
    def draw_per_minute_stats(
        self,
        frame: PIL.Image.Image,
    ) -> PIL.Image.Image:
        """
        Draw tackles won per minute and passes per minute statistics on the video.
        
        Parameters:
        -----------
        frame : PIL.Image.Image
            Frame to draw on
        
        Returns:
        --------
        PIL.Image.Image
            Frame with per-minute stats annotations
        """
        draw = ImageDraw.Draw(frame)
        
        # Get frame dimensions
        frame_width, frame_height = frame.size
        
        # Position stats in top-left corner
        start_x = 40
        start_y = 40
        line_height = 35
        font_size = 20
        
        font = self._load_font(size=font_size)
        
        # Calculate stats
        home_tackles_per_min = self.get_tackles_won_per_minute(self.home)
        away_tackles_per_min = self.get_tackles_won_per_minute(self.away)
        home_passes_per_min = self.get_passes_per_minute(self.home)
        away_passes_per_min = self.get_passes_per_minute(self.away)
        
        # Prepare text lines
        lines = [
            f"{self.home.abbreviation} - Tackles Won/Min: {home_tackles_per_min:.2f}",
            f"{self.home.abbreviation} - Passes/Min: {home_passes_per_min:.2f}",
            f"{self.away.abbreviation} - Tackles Won/Min: {away_tackles_per_min:.2f}",
            f"{self.away.abbreviation} - Passes/Min: {away_passes_per_min:.2f}",
        ]
        
        # Draw background rectangle
        padding = 10
        max_width = max(draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0] for line in lines)
        bg_height = len(lines) * line_height + padding * 2
        bg_width = max_width + padding * 2
        
        # Semi-transparent black background
        bg_rectangle = [
            start_x - padding,
            start_y - padding,
            start_x + bg_width,
            start_y + bg_height
        ]
        draw.rectangle(bg_rectangle, fill=(0, 0, 0, 200))
        
        # Draw text lines - always use white for visibility on black background
        y_offset = start_y
        text_color = (255, 255, 255)  # White text for all lines
        
        for i, line in enumerate(lines):
            draw.text((start_x, y_offset), line, font=font, fill=text_color)
            y_offset += line_height
        
        return frame
    
    def draw_active_set_piece(
        self,
        frame: PIL.Image.Image,
    ) -> PIL.Image.Image:
        """
        Draw active defending set piece bounding box on the field.
        Shows a bounding box around the wall of players.

        Parameters
        ----------
        frame : PIL.Image.Image
            Frame

        Returns
        -------
        PIL.Image.Image
            Frame with active set piece bounding box
        """
        active = self.get_active_set_piece()
        if active is None:
            return frame
        
        wall_bbox = active.get('wall_bbox')
        if wall_bbox is None:
            return frame
        
        # Draw bounding box around the wall
        draw = ImageDraw.Draw(frame)
        
        # Extract coordinates
        try:
            (xmin, ymin), (xmax, ymax) = wall_bbox
        except (ValueError, TypeError):
            return frame
        
        # Convert to integers and validate
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        
        # Validate coordinates are reasonable
        if xmin >= xmax or ymin >= ymax:
            return frame
        
        # Get frame dimensions for bounds checking
        frame_width, frame_height = frame.size
        
        # Clamp coordinates to frame bounds
        xmin = max(0, min(xmin, frame_width))
        ymin = max(0, min(ymin, frame_height))
        xmax = max(0, min(xmax, frame_width))
        ymax = max(0, min(ymax, frame_height))
        
        # Check if bbox is too small or invalid
        if xmax - xmin < 10 or ymax - ymin < 10:
            return frame
        
        # Color for defending set piece (yellow/orange)
        bbox_color = (255, 200, 0)  # Orange-yellow
        line_width = 4
        
        # Draw rounded rectangle for the bounding box
        rectangle = [(xmin, ymin), (xmax, ymax)]
        draw.rounded_rectangle(rectangle, radius=10, outline=bbox_color, width=line_width)
        
        # Draw label above the bounding box
        label_font = self._load_font(size=18)
        label_text = "DEFENDING SET PIECE"
        bbox_text = draw.textbbox((0, 0), label_text, font=label_font)
        label_width = bbox_text[2] - bbox_text[0]
        label_height = bbox_text[3] - bbox_text[1]
        label_x = (xmin + xmax) // 2 - label_width // 2
        label_y = ymin - label_height - 10
        
        # Draw background for label
        padding = 6
        draw.rectangle(
            [
                label_x - padding,
                label_y - padding,
                label_x + label_width + padding,
                label_y + label_height + padding,
            ],
            fill=(0, 0, 0, 220),
        )
        draw.text((label_x, label_y), label_text, font=label_font, fill=bbox_color)
        
        return frame