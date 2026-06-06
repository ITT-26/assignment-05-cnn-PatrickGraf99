import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from camera_action import CameraAction


class ImageProcessor:

    def __init__(self):
        print("Initializing Image Processor")

        base_options = python.BaseOptions(
            model_asset_path="03-camera-app/hand_landmarker.task"
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2
        )

        self.hand_detector = vision.HandLandmarker.create_from_options(
            options
        )

    def find_hands(self, image: np.ndarray) -> list[np.ndarray]:
        h, w, _ = image.shape

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.hand_detector.detect(mp_image)

        crops = []

        if not result.hand_landmarks:
            return crops

        for hand in result.hand_landmarks:

            xs = [int(landmark.x * w) for landmark in hand]
            ys = [int(landmark.y * h) for landmark in hand]

            x_min = min(xs)
            x_max = max(xs)

            y_min = min(ys)
            y_max = max(ys)

            padding = 30

            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)

            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            hand_crop = image[y_min:y_max, x_min:x_max]

            if hand_crop.size == 0:
                continue

            crops.append(hand_crop)

        return crops

    # region filters

    def apply_filter(self, target_image: np.ndarray, camera_filter: CameraAction) -> np.ndarray:
        """
        Applies the specified camera filter to the target image and returns the filtered image.
        :param target_image: The image to be manipulated.
        :param camera_filter: The filter to be applied.
        :return: The modified image.
        """
        if camera_filter == CameraAction.APPLY_GRAYSCALE:
            return self._apply_grayscale(target_image)
        elif camera_filter == CameraAction.APPLY_PORTRAIT:
            return self._apply_portrait(target_image)
        elif camera_filter == CameraAction.APPLY_CANNY:
            return self._apply_canny(target_image)
        else:
            return target_image

    def _apply_grayscale(self, target_image: np.ndarray) -> np.ndarray:
        """
        Applies a grayscale filter to the target image.
        :param target_image: The image to be manipulated.
        :return: The modified image.
        """
        gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _apply_portrait(self, target_image: np.ndarray) -> np.ndarray:
        """
        Applies a portrait filter to the target image.
        :param target_image: The image to be manipulated.
        :return: The modified image.
        """
        return cv2.GaussianBlur(target_image, (31, 31), 0)

    def _apply_canny(self, target_image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # endregion