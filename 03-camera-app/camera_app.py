import sys
from datetime import datetime
import os
import cv2
from jedi.third_party.typeshed.stubs.openpyxl.openpyxl.worksheet.filters import Filters

from image_processor import ImageProcessor
from image_classifier import ImageClassifier
from camera_filter import CameraAction

class CameraApp:
    def __init__(self, model_size, save_dir):
        self.save_dir = save_dir
        video_id = 0
        if len(sys.argv) > 1:
            video_id = int(sys.argv[1])

        # Create a video capture object for the webcam
        self.cap = cv2.VideoCapture(video_id)
        ret, self.frame = self.cap.read()

        self.WINDOW_NAME = "camera app"
        cv2.namedWindow(self.WINDOW_NAME)

        self.image_processor = ImageProcessor()
        self.classifier = ImageClassifier(model_size)

        self.active_camera_filters = []
        self.label_filter_map = {
            'like': CameraAction.APPLY_GRAYSCALE,
            'dislike': CameraAction.APPLY_PORTRAIT,
            'peace': ...,
            'fist': ...
        }

    def start(self):
        while True:
            self.update()

            key = cv2.waitKey(1) & 0xFF

            # Q to quit
            if key == ord('q'):
                self.stop()


    def update(self):
        ret, self.frame = self.cap.read()
        if not ret:
            print('Something went wrong with capturing a camera image.')
            return
        hand_crops = self.image_processor.find_hands(self.frame)
        if len(hand_crops) > 0:
            for hand_crop in hand_crops:
                label, confidence = self.classifier.predict(hand_crop)
                if confidence < 0.95:
                    # Be at least 95% confident
                    break
                self.handle_gesture_detected(label)
        display_image = self.frame.copy()
        for camera_filter in self.active_camera_filters:
            display_image = self.image_processor.apply_filter(display_image, camera_filter)
        cv2.imshow(self.WINDOW_NAME, display_image)

    def handle_gesture_detected(self, label):
        if label == 'stop':
            self.start_selfie_timer()
            return
        elif label == 'one':
            self.start_selfie_timer()
        action = self.label_filter_map[label]
        if action in self.active_camera_filters:
            self.active_camera_filters.remove(action)
        else:
            self.active_camera_filters.append(action)

    def start_selfie_timer(self):
        ...

    def save_picture(self):
        filename = datetime.now().strftime("%m_%d_%Y-%H_%M_%S.jpg")
        file_path = os.path.join(self.save_dir, filename)
        cv2.imwrite(file_path, self.frame)

    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()
        sys.exit()


