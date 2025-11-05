"""
SSD-based Card Detector
Implements object detection for yellow and red cards using YOLOv5 (or SSD)
Based on the methodology: Object Recognition/Detection for cards
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from inference.base_detector import BaseDetector


class CardSSDDetector(BaseDetector):
    """
    SSD-based detector for yellow and red cards.
    Uses YOLOv5 (which can be trained for card detection) or can use torchvision SSD.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        use_yolo: bool = True,
    ):
        """
        Initialize card detector

        Parameters
        ----------
        model_path : Optional[str]
            Path to trained YOLOv5 model for card detection.
            If None, uses default YOLOv5 (may need fine-tuning for cards)
        confidence_threshold : float
            Minimum confidence for card detection
        use_yolo : bool
            If True, use YOLOv5. If False, use torchvision SSD (requires separate implementation)
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.use_yolo = use_yolo

        if use_yolo:
            # Use YOLOv5 for card detection
            if model_path:
                self.model = torch.hub.load(
                    "ultralytics/yolov5", "custom", path=model_path
                )
            else:
                # Load default YOLOv5 - note: would need training for card detection
                self.model = torch.hub.load(
                    "ultralytics/yolov5", "yolov5s", pretrained=True
                )
                # Set classes to look for (if model was trained for cards)
                # For now, we'll filter results manually
                print(
                    "Warning: Using default YOLOv5. For card detection, train a custom model."
                )

            self.model = self.model.to(self.device)
            self.model.eval()
        else:
            # Alternative: Use torchvision SSD (would require custom implementation)
            raise NotImplementedError(
                "Pure SSD implementation not yet available. Use YOLOv5 (use_yolo=True)."
            )

    def predict(self, input_image: List[np.ndarray]) -> pd.DataFrame:
        """
        Detect cards in images

        Parameters
        ----------
        input_image : List[np.ndarray]
            List of input images

        Returns
        -------
        pd.DataFrame
            DataFrame with detected cards (xmin, ymin, xmax, ymax, confidence, class)
        """
        if not isinstance(input_image, list):
            input_image = [input_image]

        all_results = []

        for img in input_image:
            # Run YOLOv5 detection
            results = self.model(img, size=640)

            # Get results as DataFrame
            df = results.pandas().xyxy[0]

            # Filter for card detections
            # Note: This assumes the model was trained with card classes
            # If not, we'll need to filter by confidence and size/shape heuristics
            if len(df) > 0:
                # Filter by confidence
                df = df[df["confidence"] >= self.confidence_threshold]

                # If model has card classes, filter by name
                if "name" in df.columns:
                    card_df = df[df["name"].str.contains("card", case=False)]
                else:
                    # Heuristic: look for small rectangular objects (card-like)
                    # Cards are typically small (50-150px width)
                    df["width"] = df["xmax"] - df["xmin"]
                    df["height"] = df["ymax"] - df["ymin"]
                    df["aspect_ratio"] = df["width"] / df["height"]

                    # Cards are roughly rectangular (aspect ratio ~0.6-0.8)
                    # and small (50-200px)
                    card_df = df[
                        (df["aspect_ratio"] >= 0.5)
                        & (df["aspect_ratio"] <= 1.0)
                        & (df["width"] >= 30)
                        & (df["width"] <= 250)
                        & (df["height"] >= 40)
                        & (df["height"] <= 300)
                    ]

                    # Add placeholder class
                    card_df["name"] = "card"

                all_results.append(card_df)
            else:
                # Return empty DataFrame with correct columns
                empty_df = pd.DataFrame(
                    columns=["xmin", "ymin", "xmax", "ymax", "confidence", "class"]
                )
                all_results.append(empty_df)

        # Combine all results
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
        else:
            combined_df = pd.DataFrame(
                columns=["xmin", "ymin", "xmax", "ymax", "confidence", "class"]
            )

        return combined_df

    def detect_cards_in_frame(
        self, frame: np.ndarray
    ) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Detect cards in a single frame and return structured results

        Parameters
        ----------
        frame : np.ndarray
            Input frame

        Returns
        -------
        List[Tuple[str, float, Tuple[int, int, int, int]]]
            List of (card_type, confidence, bbox) tuples
            card_type: "yellow" or "red"
            bbox: (xmin, ymin, xmax, ymax)
        """
        df = self.predict([frame])

        cards = []
        for _, row in df.iterrows():
            bbox = (
                int(row["xmin"]),
                int(row["ymin"]),
                int(row["xmax"]),
                int(row["ymax"]),
            )
            confidence = float(row["confidence"])

            # Determine card type
            # If model was trained with specific classes, use that
            if "name" in row:
                name = str(row["name"]).lower()
                if "yellow" in name or "y" in name:
                    card_type = "yellow"
                elif "red" in name or "r" in name:
                    card_type = "red"
                else:
                    # Heuristic: analyze color in bounding box
                    card_type = self._classify_card_color(frame, bbox)

            else:
                # Use color analysis to determine card type
                card_type = self._classify_card_color(frame, bbox)

            cards.append((card_type, confidence, bbox))

        return cards

    def _classify_card_color(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> str:
        """
        Classify card as yellow or red based on color analysis

        Parameters
        ----------
        frame : np.ndarray
            Full frame
        bbox : Tuple[int, int, int, int]
            Bounding box (xmin, ymin, xmax, ymax)

        Returns
        -------
        str
            "yellow" or "red"
        """
        xmin, ymin, xmax, ymax = bbox
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(frame.shape[1], xmax), min(frame.shape[0], ymax)

        # Extract card region
        card_region = frame[ymin:ymax, xmin:xmax]

        if card_region.size == 0:
            return "yellow"  # Default

        # Convert to HSV for better color analysis
        import cv2

        hsv = cv2.cvtColor(card_region, cv2.COLOR_BGR2HSV)

        # Calculate average hue
        avg_hue = np.mean(hsv[:, :, 0])
        avg_saturation = np.mean(hsv[:, :, 1])
        avg_value = np.mean(hsv[:, :, 2])

        # Yellow cards: Hue around 20-30 (in OpenCV HSV, yellow is ~15-30)
        # Red cards: Hue around 0-10 or 170-180 (red wraps around)
        if avg_saturation > 100:  # Ensure it's not gray
            if 15 <= avg_hue <= 35:
                return "yellow"
            elif avg_hue <= 10 or avg_hue >= 170:
                return "red"

        # Fallback: check RGB values
        avg_bgr = np.mean(card_region.reshape(-1, 3), axis=0)
        if avg_bgr[2] > avg_bgr[0] and avg_bgr[2] > avg_bgr[1]:  # High red
            return "red"
        elif avg_bgr[1] > avg_bgr[0] and avg_bgr[1] > avg_bgr[2]:  # High green/yellow
            return "yellow"

        return "yellow"  # Default fallback

