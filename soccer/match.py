from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import PIL
from PIL import ImageDraw, ImageFont

from soccer.ball import Ball
from soccer.draw import Draw
from soccer.pass_event import Pass, PassEvent
from soccer.player import Player
from soccer.team import Team


class Match:
    def __init__(
        self,
        home: Team,
        away: Team,
        fps: int = 30,
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

    def change_team(self, team: Team):
        """

        Change team possession

        Parameters
        ----------
        team : Team, optional
            New team in possession
        """
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
