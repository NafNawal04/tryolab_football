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
    """Return the 4 points ordered as top-left, top-right, bottom-left, bottom-right."""
    pts = points.astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[3] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(diff)]

    return rect


def _detect_pitch_corners(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect four pitch corner points using a lightweight green mask + contour search.
    Returns points ordered TL, TR, BL, BR in image coordinates.
    """
    if frame is None or frame.size == 0:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 30, 30])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 5000:
        return None

    hull = cv2.convexHull(largest)
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    if len(approx) < 4:
        rect = cv2.minAreaRect(hull)
        approx = cv2.boxPoints(rect)
    elif len(approx) > 4:
        approx = cv2.convexHull(approx)
        rect = cv2.minAreaRect(approx)
        approx = cv2.boxPoints(rect)
    else:
        approx = approx.reshape(-1, 2)

    if len(approx) != 4:
        return None

    return _order_points(np.array(approx, dtype=np.float32))


class TacticalViewProjector:
    """
    Build a tactical top-down projection of the broadcast footage using a homography
    derived from automatically detected pitch corners (no manual annotation required).
    """

    def __init__(self, width: int = 400, height: int = 240) -> None:
        self.width = width
        self.height = height
        self._homography: Optional[Homography] = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready and self._homography is not None

    def try_initialize(self, frame: np.ndarray) -> bool:
        """Detect pitch corners on the provided frame and build the homography once."""
        if self.ready:
            return True

        corners = _detect_pitch_corners(frame)
        if corners is None:
            return False

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
        except ValueError:
            return False

        self._ready = True
        return True

    def project_players(self, players: Sequence[Player]) -> List[TacticalProjection]:
        """Project current player centers onto the tactical view."""
        if not self.ready or not players:
            return []

        projections: List[TacticalProjection] = []

        for player in players:
            center = player.center_abs
            if center is None:
                center = player.center
            if center is None:
                continue

            transformed = self._homography.transform_points(
                np.array([center], dtype=np.float32)
            )[0]

            team_color = (255, 255, 255)
            if player.team and player.team.color:
                # Convert RGB (drawing helpers) to BGR for OpenCV rendering
                r, g, b = player.team.color
                team_color = (b, g, r)

            projections.append(
                TacticalProjection(
                    player_id=player.player_id,
                    team_color=team_color,
                    position=transformed,
                )
            )

        return projections

    def render_view(self, frame: np.ndarray, players: Sequence[Player]) -> Optional[np.ndarray]:
        """Render a tactical overview image with projected player dots."""
        if frame is None or frame.size == 0:
            return None

        if not self.try_initialize(frame=frame):
            return None
        if not self.ready:
            return None

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

        projections = self.project_players(players)
        for proj in projections:
            x, y = int(proj.position[0]), int(proj.position[1])
            if x < 0 or y < 0 or x >= self.width or y >= self.height:
                continue
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

        return canvas

