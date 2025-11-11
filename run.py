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
from soccer.foul_event import Foul
from soccer.card_event import Card
from soccernet import SoccerNetEventsOverlay


def build_match_setup(
    match_key: str,
    fps: float,
    foul_cnn_model_path: str = None,
    card_ssd_model_path: str = None,
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
        foul_cnn_model_path=foul_cnn_model_path,
        card_ssd_model_path=card_ssd_model_path,
    )
    match.team_possession = initial_possession

    return match, [home, away]


parser = argparse.ArgumentParser()
parser.add_argument(
    "--video",
    default="videos/soccer_possession.mp4",
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
    "--fouls",
    action="store_true",
    help="Enable foul and card detection",
)
parser.add_argument(
    "--foul-cnn-model",
    type=str,
    default=None,
    help="Path to trained CNN model for foul classification (optional)",
)
parser.add_argument(
    "--card-ssd-model",
    type=str,
    default=None,
    help="Path to trained SSD/YOLOv5 model for card detection (optional)",
)
parser.add_argument(
    "--soccernet-predictions",
    type=str,
    default=None,
    help="Path to SoccerNet predictions JSON to overlay on the output video",
)
parser.add_argument(
    "--soccernet",
    action="store_true",
    help="Run SoccerNet CALF inference automatically to generate predictions",
)
parser.add_argument(
    "--soccernet-root",
    type=str,
    default="soccernet",
    help="Base directory containing SoccerNet assets and the CALF repository",
)
parser.add_argument(
    "--soccernet-output-dir",
    type=str,
    default="soccernet/generated",
    help="Directory where auto-generated SoccerNet predictions will be stored",
)
parser.add_argument(
    "--soccernet-model",
    type=str,
    default="CALF_benchmark",
    help="Name of the SoccerNet CALF model to use when running inference",
)
parser.add_argument(
    "--soccernet-display-seconds",
    type=float,
    default=3.0,
    help="Seconds to keep SoccerNet event annotations visible",
)
parser.add_argument(
    "--soccernet-preroll-seconds",
    type=float,
    default=0.5,
    help="Seconds before the event time to show the SoccerNet annotation",
)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


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

# Object Detectors
player_detector = YoloV5()
ball_detector = YoloV5(model_path=args.model)

match, teams = build_match_setup(
    match_key=match_key,
    fps=fps,
    foul_cnn_model_path=args.foul_cnn_model,
    card_ssd_model_path=args.card_ssd_model,
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

# Get Counter img
possession_background = match.get_possession_background()
passes_background = match.get_passes_background()
interceptions_background = match.get_interceptions_background() if args.interceptions else None

soccernet_overlay = None
if args.soccernet_predictions:
    try:
        soccernet_overlay = SoccerNetEventsOverlay.from_json(
            json_path=args.soccernet_predictions,
            fps=fps,
            display_seconds=args.soccernet_display_seconds,
            pre_event_seconds=args.soccernet_preroll_seconds,
        )
        logging.info(
            "Loaded %d SoccerNet events from %s",
            len(soccernet_overlay.events),
            args.soccernet_predictions,
        )
    except (FileNotFoundError, ValueError) as exc:
        logging.warning("Unable to load SoccerNet predictions: %s", exc)
        soccernet_overlay = None
elif args.soccernet:
    try:
        from soccernet.predictor import (
            SoccerNetPredictor,
            SoccerNetPredictorConfig,
            SoccerNetPredictorError,
        )

        predictor = SoccerNetPredictor(
            SoccerNetPredictorConfig(
                root_dir=Path(args.soccernet_root),
                output_dir=Path(args.soccernet_output_dir),
                model_name=args.soccernet_model,
            )
        )
        predictions_path = predictor.generate_predictions(Path(args.video))
        soccernet_overlay = SoccerNetEventsOverlay.from_json(
            json_path=str(predictions_path),
            fps=fps,
            display_seconds=args.soccernet_display_seconds,
            pre_event_seconds=args.soccernet_preroll_seconds,
        )
        logging.info(
            "Loaded %d SoccerNet events from %s",
            len(soccernet_overlay.events),
            predictions_path,
        )
    except SoccerNetPredictorError as exc:
        logging.error("Unable to auto-generate SoccerNet predictions: %s", exc)
        soccernet_overlay = None

for i, frame in enumerate(video):

    # Get Detections
    players_detections = get_player_detections(player_detector, frame)
    ball_detections = get_ball_detections(ball_detector, frame)
    detections = ball_detections + players_detections

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

    # Draw
    frame = PIL.Image.fromarray(frame)

    if soccernet_overlay:
        frame = soccernet_overlay.draw(frame, frame_index=i)

    if args.possession:
        frame = Player.draw_players(
            players=players, frame=frame, confidence=False, id=True
        )

        frame = path.draw(
            img=frame,
            detection=ball.detection,
            coord_transformations=coord_transformations,
            color=match.team_possession.color,
        )

        frame = match.draw_possession_counter(
            frame, counter_background=possession_background, debug=False
        )

        if ball:
            frame = ball.draw(frame)

    if args.passes:
        # Draw player tracking bounding boxes
        frame = Player.draw_players(
            players=players, frame=frame, confidence=False, id=True
        )

        pass_list = match.passes

        frame = Pass.draw_pass_list(
            img=frame, passes=pass_list, coord_transformations=coord_transformations
        )

        frame = match.draw_passes_counter(
            frame, counter_background=passes_background, debug=False
        )

    if args.interceptions:
        frame = Player.draw_players(
            players=players, frame=frame, confidence=False, id=True
        )

        if ball:
            frame = ball.draw(frame)

        frame = match.draw_interceptions_counter(
            frame,
            counter_background=interceptions_background,
            debug=False,
        )

    if args.fouls:
        # Draw player tracking bounding boxes
        frame = Player.draw_players(
            players=players, frame=frame, confidence=False, id=True
        )

        # Draw ball trail path
        frame = path.draw(
            img=frame,
            detection=ball.detection,
            coord_transformations=coord_transformations,
            color=match.team_possession.color if match.team_possession else (255, 255, 255),
        )

        # Draw player with ball (triangle/pointer)
        if match.closest_player:
            frame = match.closest_player.draw_pointer(frame)

        # Draw ball
        if ball:
            frame = ball.draw(frame)

        # Draw fouls and cards
        frame = match.draw_fouls_counter(
            frame, counter_background=possession_background, debug=False
        )

    frame = np.array(frame)

    # Write video
    video.write(frame)
