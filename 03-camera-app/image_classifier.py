import os.path

import cv2
import numpy as np
import keras

class ImageClassifier:
    def __init__(self, model_size):
        self.model_size = model_size
        model_path = os.path.join('..', '01-hyperparameters', f'gesture_classifier_{self.model_size}.keras')
        label_path = os.path.join('..', '01-hyperparameters', f'gesture_classifier_{self.model_size}_labels.txt')
        with open(label_path, 'r') as f:
            self.labels = f.read().split(',')
        print(self.labels)
        self.model = keras.models.load_model(model_path)

    def predict(self, image: np.ndarray) -> tuple[str, float]:
        #print('Predicting...')
        # test_image = cv2.imread('0a245bef-2894-4b80-93bb-15fc26dc5ffb.jpg')
        # test_image = cv2.resize(test_image,(self.model_size, self.model_size))
        # test_image = test_image.astype("float32") / 255.0
        # test_image = np.expand_dims(test_image, axis=0)

        image = cv2.resize(image, (self.model_size, self.model_size))
        image = image.astype("float") / 255.0
        image = np.expand_dims(image, axis=0)

        prediction = self.model.predict(image)
        label_index = np.argmax(prediction[0])
        confidence = float(prediction[0][label_index])
        label = self.labels[label_index]

        return label,confidence

