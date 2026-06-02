import json
import uuid

import cv2
import sys
import os


class ImageProcessor:

    def __init__(self):
        video_id = 0
        if len(sys.argv) > 1:
            video_id = int(sys.argv[1])

        # Create a video capture object for the webcam
        self.cap = cv2.VideoCapture(video_id)
        ret, self.frame = self.cap.read()

        self.WINDOW_NAME = "annotater"
        cv2.namedWindow(self.WINDOW_NAME)
        self.frozen = False
        self.annot_points = []
        self.annotating = False
        self.img_frozen = None
        self.img_frozen_original = None
        self.bboxes = []
        self.labels = []

    def update(self):
        if self.frozen:
            return
        if self.annotating:
            return
        # Capture a frame from the webcam
        ret, self.frame = self.cap.read()
        if not ret:
            print('Something went wrong with capturing a camera image.')
            return
        cv2.imshow(self.WINDOW_NAME, self.frame)


    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()
        sys.exit()

    def freeze(self):
        if self.annotating:
            print('Freezing is not allowed during annotation')
            return
        if self.frozen:
            self.frozen = False
            return
        self.frozen = True
        self.img_frozen = self.frame.copy()
        self.img_frozen_original = self.img_frozen.copy()
        cv2.imshow(self.WINDOW_NAME, self.img_frozen)

    def handle_mouse_input(self, event, x, y, flags, param):
        if self.annotating:
            return
        if not self.frozen:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f'Registered click at {x}, {y}')
            cv2.circle(self.img_frozen, (x, y), 3, (0, 0, 255), -1) # Draw a circle on the freeze
            cv2.imshow(self.WINDOW_NAME, self.img_frozen)
            cv2.waitKey(1)
            self.annot_points.append((x, y))
            if len(self.annot_points) == 2:
                self.annotate()

    def annotate(self):
        print('Calculating bbox')
        self.annotating = True
        top_left = self.annot_points[0]
        bottom_right = self.annot_points[1]
        cv2.rectangle(self.img_frozen, top_left, bottom_right, (0, 0, 255), 2)
        cv2.imshow(self.WINDOW_NAME, self.img_frozen)
        self.annot_points = []

        x = top_left[0]
        y = top_left[1]

        width = bottom_right[0] - top_left[0]
        height = bottom_right[1] - top_left[1]
        bbox = [
            x / self.img_frozen.shape[1],
            y / self.img_frozen.shape[0],
            width / self.img_frozen.shape[1],
            height / self.img_frozen.shape[0]
        ]
        label = input('Please enter a label for the captured hand:')
        self.bboxes.append(bbox)
        self.labels.append(label)
        print(f'Added {label}')
        print(f'Total annotations: {len(self.labels)}')
        self.annotating = False

    def save_annot(self):
        if not os.path.exists('annot-name.json'):
            with open('annot-name.json', 'w') as file:
                json.dump({}, file, indent=4)
        annot_id = str(uuid.uuid4())
        cv2.imwrite(f'{annot_id}.jpg', self.img_frozen_original)

        with open('annot-name.json', 'r') as file:
            try:
                json_data = json.load(file)
            except json.decoder.JSONDecodeError:
                json_data = {}
        json_data[f'{annot_id}'] = {
            'bboxes': self.bboxes,
            'labels': self.labels
        }
        with open('annot-name.json', 'w') as file:
            json.dump(json_data, file, indent=4)
        self.reset()

    def reset(self):
        self.annot_points = []
        self.annotating = False
        self.frozen = False
        self.img_frozen = None
        self.img_frozen_original = None
        self.labels = []
        self.bboxes = []

    def start(self):
        cv2.setMouseCallback(self.WINDOW_NAME, self.handle_mouse_input)
        while True:
            self.update()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.stop()

            if key == ord('f'):
                self.freeze()

            if key == ord('s'):
                self.save_annot()

annotater = ImageProcessor()
annotater.start()