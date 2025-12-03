import argparse
import logging
import re
from pathlib import Path

import cv2
import numpy as np
import PIL
from norfair import Tracker, Video
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean

from inference import Converter, HSVClassifier, InertiaClassifier, YoloV5
from inference.filters import DEFAULT_MATCH_KEY, get_filters_for_match
from run_utils import (
    get_ball_detections,
    get_main_ball,
    get_player_detections,
    update_motion_estimator,
)
from soccer import Match, Player, Team
from soccer.draw import AbsolutePath
from soccer.pass_event import Pass
from soccer.movement_analysis import MovementAnalyzer
from auto_calibrate import auto_calibrate
from tactical_view import TacticalViewProjector
from court_keypoint_detector import CourtKeypointDetector


def build_match_setup(
    match_key: str,
    fps: float,
    pixels_to_meters: float = None,
):
    if match_key == "chelsea_man_city":
        home = Team(
            name="Chelsea",
            abbreviation="CHE",
            color=(255, 0, 0),
            board_color=(244, 86, 64),
            text_color=(255, 255, 255),
        )
        away = Team(
            name="Man City",
            abbreviation="MNC",
            color=(240, 230, 188),
            text_color=(0, 0, 0),
        )
        initial_possession = away
    elif match_key == "real_madrid_barcelona":
        home = Team(
            name="Real Madrid",
            abbreviation="RMA",
            color=(255, 255, 255),
            board_color=(235, 214, 120),
            text_color=(0, 0, 0),
        )
        away = Team(
            name="Barcelona",
            abbreviation="BAR",
            color=(128, 0, 128),
            board_color=(28, 43, 92),
            text_color=(255, 215, 0),
        )
        initial_possession = home
    elif match_key == "france_croatia":
        home = Team(
            name="France",
            abbreviation="FRA",
            color=(0, 56, 168),
            board_color=(16, 44, 87),
            text_color=(255, 255, 255),
        )
        away = Team(
            name="Croatia",
            abbreviation="CRO",
            color=(208, 16, 44),
            board_color=(230, 230, 230),
            text_color=(0, 0, 0),
        )
        initial_possession = home
    else:
        raise ValueError(f"Unsupported match key '{match_key}'")

    match = Match(
        home=home,
        away=away,
        fps=fps,
        pixels_to_meters=pixels_to_meters,
    )
    match.team_possession = initial_possession

    return match, [home, away]


parser = argparse.ArgumentParser()
parser.add_argument(
    "--video",
    default="videos/soccer_possession1.mp4",
    type=str,
    help="Path to the input video",
)
parser.add_argument(
    "--model", default="models/ball.pt", type=str, help="Path to the model"
)
parser.add_argument(
    "--match",
    type=str,
    choices=["chelsea_man_city", "real_madrid_barcelona", "france_croatia"],
    help="Preset match configuration to use (auto-detected from video if omitted)",
)
parser.add_argument(
    "--passes",
    action="store_true",
    help="Enable pass detection",
)
parser.add_argument(
    "--interceptions",
    action="store_true",
    help="Enable interception and ball recovery counter",
)
parser.add_argument(
    "--possession",
    action="store_true",
    help="Enable possession counter",
)
parser.add_argument(
    "--tackles",
    action="store_true",
    help="Enable tackle detection and counter",
)
parser.add_argument(
    "--defending",
    action="store_true",
    help="Enable set piece detection (corners, free kicks)",
)
parser.add_argument(
    "--stats",
    action="store_true",
    help="Enable per-minute statistics (tackles won/min, passes/min) annotations",
)
parser.add_argument(
    "--tactical-view",
    action="store_true",
    help="Render a tactical top-down view built from automatic homography estimation",
)

parser.add_argument(
    "--movement-analysis",
    action="store_true",
    help="Analyze player paths, off-ball runs, and zone transitions (requires --tactical-view)",
)
parser.add_argument(
    "--pixels-to-meters",
    type=float,
    default=None,
    help="Conversion factor from pixels to meters (e.g., 0.01 for 100px=1m). If not provided, will be automatically calibrated.",
)
args = parser.parse_args()
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")


MATCH_KEYWORDS = {
    "chelsea_man_city": (
        ("chelsea",),
        ("man city", "manchester city", "man-city", "mancity", "mnc"),
    ),
    "real_madrid_barcelona": (
        ("real madrid", "realmadrid", "real-madrid", "rma"),
        ("barcelona", "barca", "fcb", "fcbarcelona", "bar"),
    ),
    "france_croatia": (
        ("france", "fra", "les bleus", "bleus"),
        ("croatia", "hrvatska", "cro", "hrv"),
    ),
}


def infer_match_key_from_path(video_path: str) -> str:
    """
    Infer the configured match from the video path by looking for team keywords.
    Falls back to DEFAULT_MATCH_KEY when nothing matches.
    """

    def _normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text

    path = Path(video_path)
    candidates = [path.stem]
    candidates.extend(part for part in path.parts if part)
    text_blob = _normalize(" ".join(candidates))

    for match_key, keyword_groups in MATCH_KEYWORDS.items():
        found_all = True
        for variants in keyword_groups:
            if not any(_normalize(keyword) in text_blob for keyword in variants):
                found_all = False
                break
        if found_all:
            return match_key

    return DEFAULT_MATCH_KEY


match_key = args.match or infer_match_key_from_path(args.video)
if args.match:
    logging.info("Using match preset provided via CLI: %s", match_key)
else:
    if match_key == DEFAULT_MATCH_KEY:
        logging.info(
            "Unable to infer match from video path; using default preset '%s'",
            DEFAULT_MATCH_KEY,
        )
    else:
        logging.info(
            "Auto-detected match preset '%s' based on video path '%s'",
            match_key,
            args.video,
        )

video = Video(input_path=args.video)
fps = video.video_capture.get(cv2.CAP_PROP_FPS)

# Auto-calibrate pixels_to_meters if not provided
pixels_to_meters = args.pixels_to_meters
if pixels_to_meters is None:
    logging.info("No pixels-to-meters provided, auto-calibrating...")
    pixels_to_meters = auto_calibrate(args.video, verbose=False)
    if pixels_to_meters is None:
        logging.warning("Auto-calibration failed. Distance tracking will use pixel units only.")
        pixels_to_meters = None
    else:
        # Validate: Conversion factor should be reasonable (0.01 to 0.05 m/px)
        if pixels_to_meters < 0.01 or pixels_to_meters > 0.05:
            logging.warning(f"Auto-calibration produced value ({pixels_to_meters:.6f} m/px) outside recommended range (0.01-0.05).")
            logging.warning("This might cause incorrect distance measurements.")
            logging.warning("Consider providing --pixels-to-meters manually for accurate results.")
        else:
            logging.info(f"Auto-calibration successful: {pixels_to_meters:.6f} m/px")
            logging.info(f"  (1 pixel ≈ {pixels_to_meters*100:.2f} cm)")
            # At 30 fps, estimate realistic speed example
            example_pixels_per_frame = 10  # Typical movement
            example_speed_mps = pixels_to_meters * example_pixels_per_frame * fps
            logging.info(f"  (At {fps} fps, {example_pixels_per_frame} px/frame ≈ {example_speed_mps:.1f} m/s)")
            if example_speed_mps > 15:
                logging.warning(f"  ⚠ Estimated speed seems high. Consider manual calibration.")

# Object Detectors
player_detector = YoloV5()
ball_detector = YoloV5(model_path=args.model)
court_detector = CourtKeypointDetector(model_path="models/keypoint_detector.pt")

match, teams = build_match_setup(
    match_key=match_key,
    fps=fps,
    pixels_to_meters=pixels_to_meters,
)

filters_for_match = get_filters_for_match(match_key)
hsv_classifier = HSVClassifier(filters=filters_for_match)

# Add inertia to classifier
classifier = InertiaClassifier(classifier=hsv_classifier, inertia=20)

# Tracking
player_tracker = Tracker(
    distance_function=mean_euclidean,
    distance_threshold=250,
    initialization_delay=3,
    hit_counter_max=90,
)

ball_tracker = Tracker(
    distance_function=mean_euclidean,
    distance_threshold=150,
    initialization_delay=20,
    hit_counter_max=2000,
)
motion_estimator = MotionEstimator()
coord_transformations = None

# Paths
path = AbsolutePath()

tactical_projector = TacticalViewProjector(pixels_to_meters=pixels_to_meters) if (args.tactical_view or args.movement_analysis) else None
movement_analyzer = MovementAnalyzer() if args.movement_analysis else None

if args.movement_analysis and not args.tactical_view:
    logging.warning("--movement-analysis requires --tactical-view. Enabling tactical view automatically.")
    args.tactical_view = True

# Get Counter img
possession_background = match.get_possession_background()
passes_background = match.get_passes_background()
interceptions_background = match.get_interceptions_background() if args.interceptions else None
tackles_background = match.get_tackles_background() if args.tackles else None

for i, frame in enumerate(video):
    frame_number = i  # Track frame number for distance tracking
    
    # Reset distance tracking on first frame to ensure everyone starts at 0
    if frame_number == 0:
        match.reset_distance_tracking()

    # Get Detections
    players_detections = get_player_detections(player_detector, frame)
    ball_detections = get_ball_detections(ball_detector, frame)
    detections = ball_detections + players_detections

    # Detect court keypoints
    if tactical_projector:
        court_keypoints = court_detector.get_court_keypoints([frame])
        if court_keypoints:
            tactical_projector.update_homography_from_keypoints(court_keypoints[0])

    # Update trackers
    coord_transformations = update_motion_estimator(
        motion_estimator=motion_estimator,
        detections=detections,
        frame=frame,
    )

    player_track_objects = player_tracker.update(
        detections=players_detections, coord_transformations=coord_transformations
    )

    ball_track_objects = ball_tracker.update(
        detections=ball_detections, coord_transformations=coord_transformations
    )

    player_detections = Converter.TrackedObjects_to_Detections(player_track_objects)
    ball_detections = Converter.TrackedObjects_to_Detections(ball_track_objects)

    player_detections = classifier.predict_from_detections(
        detections=player_detections,
        img=frame,
    )

    # Match update (pass frame for CNN/SSD detection)
    ball = get_main_ball(ball_detections)
    players = Player.from_detections(detections=players_detections, teams=teams)
    match.update(players, ball, frame=frame)

    # Update movement analysis if enabled
    tactical_positions = {}
    if movement_analyzer and tactical_projector:
        # Ensure tactical projector is initialized
        if tactical_projector.try_initialize(frame) and tactical_projector.ready:
            # Get tactical positions for all players
            projections = tactical_projector.project_players(players)
            for proj in projections:
                if proj.player_id is not None and proj.position is not None:
                    try:
                        tactical_positions[proj.player_id] = (
                            float(proj.position[0]),
                            float(proj.position[1]),
                        )
                    except (IndexError, TypeError):
                        # Skip if position is invalid
                        continue
            
            # Update movement analyzer
            if tactical_positions:
                movement_analyzer.update(
                    players=players,
                    tactical_positions=tactical_positions,
                    closest_player=match.closest_player,
                )

    tactical_overlay = None
    if tactical_projector:
        tactical_overlay = tactical_projector.render_view(frame=frame, players=players)



    # Draw
    frame = PIL.Image.fromarray(frame)

    # Always render player detections, ball trail, and ball overlay (baseline visualization)
    frame = Player.draw_players(players=players, frame=frame, confidence=False, id=True, match=match)

    if ball and ball.detection is not None:
        frame = path.draw(
            img=frame,
            detection=ball.detection,
            coord_transformations=coord_transformations,
            color=match.team_possession.color if match.team_possession else (255, 255, 255),
        )

    if ball:
        frame = ball.draw(frame)

    if args.possession:
        frame = match.draw_possession_counter(
            frame, counter_background=possession_background, debug=False
        )

    if args.passes:
        pass_list = match.passes

        frame = Pass.draw_pass_list(
            img=frame, passes=pass_list, coord_transformations=coord_transformations
        )

        frame = match.draw_passes_counter(
            frame, counter_background=passes_background, debug=False
        )

    if args.interceptions:
        frame = match.draw_interceptions_counter(
            frame,
            counter_background=interceptions_background,
            debug=False,
        )

    if args.tackles:
        # Draw active tackle attempts on field
        frame = match.draw_active_tackles(frame, players)
        
        # Draw recently resolved tackles on field
        frame = match.draw_recent_tackles(frame, players, recent_frames=60)
        
        # Draw tackles counter
        frame = match.draw_tackles_counter(
            frame,
            counter_background=tackles_background,
            debug=False,
        )

    if args.defending:
        # Draw active defending set piece bounding box on field
        frame = match.draw_active_set_piece(frame)
        
        # Log defending set piece information
        active_set_piece = match.get_active_set_piece()
        if active_set_piece:
            wall_count = active_set_piece.get('wall_player_count', 0)
            wall_bbox = active_set_piece.get('wall_bbox')
            logging.info(
                f"Frame {frame_number}: ✓ DEFENDING SET PIECE ACTIVE - "
                f"Wall players: {wall_count}, "
                f"BBox: {wall_bbox}"
            )
        
        # Log newly resolved set pieces
        set_pieces = match.get_set_pieces()
        if len(set_pieces) > 0:
            # Log the most recent set piece
            latest = set_pieces[-1]
            if latest.get('resolved_frame') == frame_number:
                logging.info(
                    f"Set piece detected - Type: {latest.get('type', 'unknown')}, "
                    f"Attacking: {latest.get('attacking_team')}, "
                    f"Frames: {latest.get('start_frame')} to {latest.get('resolved_frame')}"
                )

    if args.stats:
        # Draw per-minute statistics (tackles won/min, passes/min)
        frame = match.draw_per_minute_stats(frame)

    if tactical_overlay is not None:
        overlay_rgb = cv2.cvtColor(tactical_overlay, cv2.COLOR_BGR2RGB)
        overlay_img = PIL.Image.fromarray(overlay_rgb)
        overlay_position = (
            frame.width - overlay_img.width - 20,
            20,
        )
        frame.paste(overlay_img, overlay_position)

    frame = np.array(frame)

    # Write video
    video.write(frame)

# Print movement analysis summary if enabled
if movement_analyzer:
    movement_analyzer.print_summary(fps=fps)
