"""
CNN-based Foul Classifier
Implements binary classification (Foul/Not Foul) using pre-trained CNN models
Based on the methodology: Image Classification with pre-trained InceptionResNetV2 or DenseNet121
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import PIL
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from inference.base_classifier import BaseClassifier


class FoulCNNClassifier(BaseClassifier):
    """
    CNN-based binary classifier for foul detection.
    Uses pre-trained models (InceptionResNetV2 or DenseNet121) with transfer learning.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "densenet121",  # "densenet121" or "inceptionresnetv2"
        threshold: float = 0.5,
    ):
        """
        Initialize CNN-based foul classifier

        Parameters
        ----------
        model_path : Optional[str]
            Path to trained model weights. If None, uses pre-trained ImageNet weights.
        model_type : str
            Model architecture: "densenet121" or "inceptionresnetv2"
        threshold : float
            Classification threshold (0.5 = 50% confidence)
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.model_type = model_type.lower()

        # Load pre-trained model
        if model_type == "densenet121":
            self.model = models.densenet121(pretrained=True)
            # Replace classifier for binary classification
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 1),
                nn.Sigmoid(),  # Binary classification output
            )
        elif model_type == "inceptionresnetv2":
            try:
                self.model = models.inceptionresnetv2(
                    pretrained=True, num_classes=1000
                )
                # Replace classifier for binary classification
                num_features = self.model.last_linear.in_features
                self.model.last_linear = nn.Sequential(
                    nn.Linear(num_features, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(1024, 1),
                    nn.Sigmoid(),
                )
            except Exception:
                # Fallback to DenseNet if InceptionResNetV2 not available
                print(
                    "InceptionResNetV2 not available, falling back to DenseNet121"
                )
                self.model = models.densenet121(pretrained=True)
                num_features = self.model.classifier.in_features
                self.model.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(num_features, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(1024, 1),
                    nn.Sigmoid(),
                )
                self.model_type = "densenet121"
        else:
            raise ValueError(
                f"Unknown model_type: {model_type}. Use 'densenet121' or 'inceptionresnetv2'"
            )

        # Load custom weights if provided
        if model_path:
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"Loaded custom model weights from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load custom weights: {e}")
                print("Using pre-trained ImageNet weights only")

        self.model = self.model.to(self.device)
        self.model.eval()

        # Image preprocessing: resize to 224x224 as per documentation
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ImageNet normalization
            ]
        )

    def crop_foul_region(
        self,
        img: np.ndarray,
        player1_points: np.ndarray,
        player2_points: np.ndarray,
        margin: int = 50,
    ) -> np.ndarray:
        """
        Crop region around two players for foul classification

        Parameters
        ----------
        img : np.ndarray
            Full frame image
        player1_points : np.ndarray
            Bounding box points for player 1: [[x1, y1], [x2, y2]]
        player2_points : np.ndarray
            Bounding box points for player 2: [[x1, y1], [x2, y2]]
        margin : int
            Extra margin around bounding boxes in pixels

        Returns
        -------
        np.ndarray
            Cropped region containing both players
        """
        h, w = img.shape[:2]

        # Get bounding box coordinates
        x1_min = min(player1_points[0][0], player2_points[0][0])
        y1_min = min(player1_points[0][1], player2_points[0][1])
        x2_max = max(player1_points[1][0], player2_points[1][0])
        y2_max = max(player1_points[1][1], player2_points[1][1])

        # Add margin
        x1 = max(0, int(x1_min - margin))
        y1 = max(0, int(y1_min - margin))
        x2 = min(w, int(x2_max + margin))
        y2 = min(h, int(y2_max + margin))

        # Crop region
        cropped = img[y1:y2, x1:x2]

        # Ensure minimum size
        if cropped.size == 0 or cropped.shape[0] < 50 or cropped.shape[1] < 50:
            # Fallback: use center of image
            center_x, center_y = w // 2, h // 2
            size = 200
            x1 = max(0, center_x - size // 2)
            y1 = max(0, center_y - size // 2)
            x2 = min(w, center_x + size // 2)
            y2 = min(h, center_y + size // 2)
            cropped = img[y1:y2, x1:x2]

        return cropped

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for CNN input

        Parameters
        ----------
        image : np.ndarray
            Input image (BGR format from OpenCV)

        Returns
        -------
        torch.Tensor
            Preprocessed tensor ready for model
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image

        # Convert to PIL Image
        pil_image = PIL.Image.fromarray(rgb_image)

        # Apply transforms
        tensor = self.transform(pil_image)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        tensor = tensor.to(self.device)

        return tensor

    def predict_single(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Predict if a single image contains a foul

        Parameters
        ----------
        image : np.ndarray
            Input image (cropped region around players)

        Returns
        -------
        Tuple[bool, float]
            (is_foul, confidence) where confidence is probability of foul
        """
        with torch.no_grad():
            tensor = self.preprocess_image(image)
            output = self.model(tensor)
            confidence = output.item()  # Sigmoid output [0, 1]
            is_foul = confidence >= self.threshold

        return (is_foul, confidence)

    def predict(self, input_image: List[np.ndarray]) -> List[str]:
        """
        Predict foul classification for list of images

        Parameters
        ----------
        input_image : List[np.ndarray]
            List of images to classify

        Returns
        -------
        List[str]
            List of predictions: "Foul" or "Not Foul"
        """
        if not isinstance(input_image, list):
            input_image = [input_image]

        results = []
        for img in input_image:
            is_foul, confidence = self.predict_single(img)
            results.append("Foul" if is_foul else "Not Foul")

        return results

    def predict_with_confidence(
        self, input_image: List[np.ndarray]
    ) -> List[Tuple[str, float]]:
        """
        Predict with confidence scores

        Parameters
        ----------
        input_image : List[np.ndarray]
            List of images to classify

        Returns
        -------
        List[Tuple[str, float]]
            List of (prediction, confidence) tuples
        """
        if not isinstance(input_image, list):
            input_image = [input_image]

        results = []
        for img in input_image:
            is_foul, confidence = self.predict_single(img)
            prediction = "Foul" if is_foul else "Not Foul"
            results.append((prediction, confidence))

        return results

