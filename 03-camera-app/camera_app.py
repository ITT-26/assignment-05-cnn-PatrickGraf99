import math
import sys
import time
from datetime import datetime
import os
import cv2
from google.protobuf import duration

from image_processor import ImageProcessor
from image_classifier import ImageClassifier
from camera_action import CameraAction

class CameraApp:
    def __init__(self, model_size, save_dir, camera_id=0, delay=3, timer_duration = 3):
        print("Initializing Camera App")
        self.cap = cv2.VideoCapture(camera_id)
        self.save_dir = save_dir
        video_id = camera_id

        # Create a video capture object for the webcam
        self.cap = cv2.VideoCapture(video_id)
        ret, self.frame = self.cap.read()

        self.WINDOW_NAME = "camera app"
        cv2.namedWindow(self.WINDOW_NAME)

        # Delay to not trigger gestures repeatedly
        self.detection_delay = delay
        self.detection_ready = True
        self.detection_ready_time = None
        self.timer_running = False
        self.gestures_active = False
        self.timer_end = None
        self.timer_duration = timer_duration


        self.image_processor = ImageProcessor()
        self.classifier = ImageClassifier(model_size)

        self.active_camera_filters = []
        self.label_filter_map = {
            'like': CameraAction.APPLY_GRAYSCALE,
            'dislike': CameraAction.APPLY_PORTRAIT,
            'peace': CameraAction.APPLY_CANNY,
            'fist': CameraAction.ACTIVATE_SELFIE_TIMER,
            'one': ...,
        }

    def start(self):
        while True:
            self.update()

            key = cv2.waitKey(1) & 0xFF

            self.handle_key_press(key)


    def handle_key_press(self, key):
        # Q to quit
        if key == ord('q'):
            self.stop()

        # Disable keyboard if timer is running
        if self.timer_running:
            return

        # G will apply our grayscale filter
        if key == ord('g'):
            self.handle_gesture_detected('like')

        # D for portrait
        if key == ord('b'):
            self.handle_gesture_detected('dislike')

        if key == ord('p'):
            if self.gestures_active:
                self.gestures_active = False
                print('Gesture detection turned off')
            else:
                self.gestures_active = True
                print('Gesture detection turned on')

        # C for canny
        if key == ord('c'):
            self.handle_gesture_detected('peace')

        if key == ord('t'):
            self.start_selfie_timer()

    def update(self):
        ret, self.frame = self.cap.read()
        if not ret:
            print('Something went wrong with capturing a camera image.')
            return
        hand_crops = self.image_processor.find_hands(self.frame)
        # Only ask cnn when detection is turned on and no timer is running and detection is ready
        if not self.timer_running and self.gestures_active and self.detection_ready and len(hand_crops) > 0:
            for hand_crop in hand_crops:
                label, confidence = self.classifier.predict(hand_crop)
                if confidence < 0.95:
                    # Be at least 95% confident
                    break
                self.handle_gesture_detected(label)

        display_image = self.frame.copy()
        for camera_filter in self.active_camera_filters:
            display_image = self.image_processor.apply_filter(display_image, camera_filter)
        if self.timer_running:
            now = time.time()
            remaining = max(0, math.ceil(self.timer_end - now))
            display_image = cv2.putText(display_image, f'Selfie in {remaining}', (20, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            if now >= self.timer_end:
                print('Timer ran down, taking selfie')
                self.timer_running = False
                self.save_picture()
                # Turn entire frame white to imitate a flash effect when taking a picture
                display_image = cv2.rectangle(display_image, (0, 0), (display_image.shape[1], display_image.shape[0]),
                                              (255, 255, 255), -1)
        if not self.detection_ready:
            now = time.time()
            if now >= self.detection_ready_time:
                print('Ready for new gesture')
                self.detection_ready = True
        cv2.imshow(self.WINDOW_NAME, display_image)


    def handle_gesture_detected(self, label):
        print('Handling detected gesture {label}'.format(label=label))
        self.detection_ready = False
        self.detection_ready_time = time.time() + self.detection_delay
        if label == 'fist':
            self.start_selfie_timer()
            return
        action = self.label_filter_map[label]
        if action in self.active_camera_filters:
            self.active_camera_filters.remove(action)
        else:
            self.active_camera_filters.append(action)

    def start_selfie_timer(self):
        print('Starting timer for selfie')
        self.timer_running = True
        self.timer_end = time.time() + self.timer_duration

    def save_picture(self):
        print('Saving picture')
        filename = datetime.now().strftime("%m_%d_%Y-%H_%M_%S.jpg")
        file_path = os.path.join(self.save_dir, filename)
        cv2.imwrite(file_path, self.frame)

    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()
        sys.exit()


