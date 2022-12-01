import cv2
import imutils
import numpy as np
from cv2 import HOGDescriptor
from imutils.object_detection import non_max_suppression
from imutils.video import FPS

from src.utils import FONT, FONT_COLOR, BOX_COLOR, get_text_size, FONT_SCALE, FONT_THICKNESS, DETAILS_COLOR, \
    FONT_COLOR_SECONDARY


class Detector:
    def __init__(self):
        self._available_methods = {'HOG': self.hog_detect}
        self._image = None
        self._height = 0
        self._width = 0
        self._max_width = 800
        self._number_of_people = 0
        self._model = None

    def load_img(self, path: str):
        img = cv2.imread(path)
        self._image = img
        self._height = img.shape[0]
        self._width = img.shape[1]

        if self._width > self._max_width:
            self._image = imutils.resize(img, width=self._max_width)

    def show_image(self):
        cv2.imshow("Detection", self._image)

    def hog_detect(self):
        descriptor = HOGDescriptor()
        descriptor.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        persons, weights = descriptor.detectMultiScale(self._image, winStride=(4, 4), padding=(8, 8), scale=1.04)

        persons = np.array([[x, y, x + w, y + h] for (x, y, w, h) in persons])
        persons = non_max_suppression(persons, probs=None, overlapThresh=0.5)

        for index, (xA, yA, xB, yB) in enumerate(persons, start=1):
            cv2.rectangle(self._image, (xA, yA), (xB, yB), BOX_COLOR, 2)
            cv2.rectangle(self._image, (xA, yA - 20), (xB, yA), BOX_COLOR, -1)
            cv2.putText(self._image, f'Person-{index}', (xA, yA), FONT, 0.5, FONT_COLOR, )

        self._number_of_people = len(persons)

    def draw_details(self, time):
        text_width, text_height = get_text_size(f' Total: {self._number_of_people} Time: {time}'
                                                "", FONT_SCALE, FONT_THICKNESS)

        cv2.rectangle(self._image, (0, self._height), (self._width, self._height-text_height-15), DETAILS_COLOR, -1)

        cv2.putText(self._image, f' Total: {self._number_of_people} Time: {time}', (10, self._height - 10), FONT,
                    FONT_SCALE, FONT_COLOR_SECONDARY, FONT_THICKNESS)

    def choose_method(self):
        self._available_methods.get(self._model)()

    def check_image(self, path: str, model: any):
        self.load_img(path)
        self._model = model

        time = FPS().start()
        self.choose_method()
        time.stop()

        self.draw_details(time.elapsed())
        self.show_image()
        cv2.waitKey()

    def check_video(self):
        pass
