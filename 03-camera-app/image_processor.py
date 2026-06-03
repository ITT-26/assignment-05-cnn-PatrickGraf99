import cv2
import numpy as np
import mediapipe as mp

from camera_filter import CameraAction


class ImageProcessor:

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def find_hands(self, image: np.ndarray) -> list[np.ndarray]:
        """
        Finds hands in the given image.
        :param image: The image to be scanned
        :return: A list of all hands detected.
        """
        h, w, _ = image.shape
        # Use RGB for mediapipe
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)

        crops = []

        if not results.multi_hand_landmarks:
            return []

        for hand_landmarks in results.multi_hand_landmarks:

            xs = []
            ys = []

            # Collect all landmark coordinates
            for lm in hand_landmarks.landmark:
                xs.append(int(lm.x * w))
                ys.append(int(lm.y * h))

            # Bounding box
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            # Add padding (important for fingers)
            padding = 30
            x_min = max(x_min - padding, 0)
            y_min = max(y_min - padding, 0)
            x_max = min(x_max + padding, w)
            y_max = min(y_max + padding, h)

            # Crop hand region
            hand_crop = image[y_min:y_max, x_min:x_max]

            # Avoid empty crops
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

    # endregion