from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class SoccerNetEvent:
    """Single event returned by SoccerNet predictions."""

    label: str
    game_time: Optional[str]
    position_ms: int
    frame: int
    trigger_frame: int
    confidence: Optional[float] = None
    half: Optional[int] = None


class SoccerNetEventsOverlay:
    """Handles rendering SoccerNet predictions on top of video frames."""

    def __init__(
        self,
        events: List[SoccerNetEvent],
        display_frames: int,
    ) -> None:
        self.events = sorted(events, key=lambda event: event.trigger_frame)
        self.display_frames = max(1, display_frames)
        self._next_index = 0
        self._active_event: Optional[SoccerNetEvent] = None
        self._active_since: Optional[int] = None

        self._font_large = self._load_font(size=38)
        self._font_small = self._load_font(size=26)

    @classmethod
    def from_json(
        cls,
        json_path: str,
        fps: float,
        display_seconds: float = 3.0,
        pre_event_seconds: float = 0.5,
    ) -> "SoccerNetEventsOverlay":
        """Create overlay helper from a SoccerNet predictions JSON file."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"SoccerNet predictions file not found: {json_path}")

        with path.open("r", encoding="utf-8") as fp:
            raw_data = json.load(fp)

        predictions = raw_data.get("predictions", [])
        if not isinstance(predictions, list):
            raise ValueError("Invalid SoccerNet predictions file: expected 'predictions' list")

        events: List[SoccerNetEvent] = []
        fps = float(fps) if fps else 25.0
        display_frames = int(round(display_seconds * fps))
        pre_event_frames = max(0, int(round(pre_event_seconds * fps)))

        for entry in predictions:
            try:
                position_ms = int(float(entry.get("position", 0)))
            except (TypeError, ValueError):
                continue

            frame_number = max(0, int(round((position_ms / 1000.0) * fps)))
            trigger_frame = max(0, frame_number - pre_event_frames)

            label = str(entry.get("label", "Event"))
            game_time = entry.get("gameTime")

            confidence_raw = entry.get("confidence")
            try:
                confidence = float(confidence_raw) if confidence_raw is not None else None
            except (TypeError, ValueError):
                confidence = None

            half_raw = entry.get("half")
            try:
                half = int(half_raw) if half_raw is not None else None
            except (TypeError, ValueError):
                half = None

            events.append(
                SoccerNetEvent(
                    label=label,
                    game_time=str(game_time) if game_time is not None else None,
                    position_ms=position_ms,
                    frame=frame_number,
                    trigger_frame=trigger_frame,
                    confidence=confidence,
                    half=half,
                )
            )

        if not events:
            raise ValueError("No valid predictions found in SoccerNet predictions file")

        return cls(events=events, display_frames=max(1, display_frames))

    def draw(self, image: Image.Image, frame_index: int) -> Image.Image:
        """Draw currently active SoccerNet event onto the provided frame."""
        if not self.events:
            return image

        self._advance_events(frame_index)

        if (
            self._active_event is None
            or self._active_since is None
            or frame_index - self._active_since >= self.display_frames
        ):
            return image

        return self._render_event(image, self._active_event)

    def _advance_events(self, frame_index: int) -> None:
        """Update active event pointer for the current frame index."""
        while self._next_index < len(self.events) and self.events[self._next_index].trigger_frame <= frame_index:
            self._active_event = self.events[self._next_index]
            self._active_since = frame_index
            self._next_index += 1

        if (
            self._active_event is not None
            and self._active_since is not None
            and frame_index - self._active_since >= self.display_frames
        ):
            self._active_event = None
            self._active_since = None

    def _render_event(self, image: Image.Image, event: SoccerNetEvent) -> Image.Image:
        base = image.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        width, height = base.size
        margin = int(width * 0.05)
        box_height = 110
        top = margin
        left = margin
        right = width - margin
        bottom = top + box_height

        draw.rounded_rectangle(
            (left, top, right, bottom),
            radius=20,
            fill=(0, 0, 0, 180),
        )

        primary_text = event.label
        metadata_parts: List[str] = []
        if event.game_time:
            metadata_parts.append(f"Game time: {event.game_time}")
        if event.half is not None:
            metadata_parts.append(f"Half: {event.half}")
        if event.confidence is not None:
            metadata_parts.append(f"Confidence: {event.confidence * 100:.1f}%")

        secondary_text = " | ".join(metadata_parts)

        self._draw_centered_text(draw, primary_text, self._font_large, width, top + 20)
        if secondary_text:
            self._draw_centered_text(draw, secondary_text, self._font_small, width, top + 65)

        combined = Image.alpha_composite(base, overlay)
        return combined.convert("RGB")

    def _draw_centered_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.ImageFont,
        image_width: int,
        y: int,
    ) -> None:
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        x = (image_width - text_width) // 2
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

    def _load_font(self, size: int) -> ImageFont.ImageFont:
        font_path = Path(__file__).resolve().parent.parent / "fonts" / "Gidole-Regular.ttf"
        try:
            return ImageFont.truetype(str(font_path), size=size)
        except OSError:
            return ImageFont.load_default()


