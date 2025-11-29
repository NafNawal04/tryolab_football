import numpy as np
import cv2


class Homography:
    """Small wrapper around cv2.findHomography/perspectiveTransform."""

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        if source.shape != target.shape:
            raise ValueError("Source and target must have the same shape.")
        if source.shape[1] != 2:
            raise ValueError("Source and target points must be 2D coordinates.")

        source = source.astype(np.float32)
        target = target.astype(np.float32)

        self.matrix, _ = cv2.findHomography(source, target)
        if self.matrix is None:
            raise ValueError("Homography matrix could not be calculated.")

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")

        points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(points, self.matrix)
        return transformed.reshape(-1, 2).astype(np.float32)

