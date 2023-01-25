from pathlib import Path

import cv2  # type: ignore
import imutils  # type: ignore
import numpy as np  # type: ignore
from cv2 import HOGDescriptor
from imutils.object_detection import non_max_suppression
from imutils.video import FPS

from rest_image_tracker.detector.utils import (
    BOX_COLOR,
    DETAILS_COLOR,
    FONT,
    FONT_COLOR,
    FONT_COLOR_SECONDARY,
    FONT_SCALE,
    FONT_THICKNESS,
    get_text_size,
)


class Detector:
    """Represent class for perform people detecting on image"""
    def __init__(self) -> None:
        self._available_methods = {'HOG': self.hog_detect}
        self._image: np.ndarray = None
        self._height: int = 0
        self._width: int = 0
        self._max_width: int = 800
        self._max_height: int = 600
        self._number_of_people: int = 0
        self._model_name: str = None

    def resize_on_too_big(self) -> None:
        """Resize loaded image if too large"""
        if self._width > self._max_width:
            self._image = imutils.resize(self._image, width=self._image.shape[1])
        if self._height > self._max_height:
            self._image = imutils.resize(self._image, height=self._image.shape[0])

    def update_sizes(self) -> None:
        """Reassign the height and width of the loaded image"""
        self._height = self._image.shape[0]
        self._width = self._image.shape[1]

    def load_img(self, img: np.ndarray) -> None:
        """
        Load and preprocess the image
        :param img: ndarray that represent an image
        """
        self._image = img
        self.resize_on_too_big()
        self.update_sizes()

    def load_img_from_file(self, path: str) -> None:
        """
        Load and preprocess the image from given path
        :param path: path to an image
        """
        img = cv2.imread(path)
        self.load_img(img)

    def load_img_from_bytes(self, bytes: bytes) -> None:
        """
        Load and preprocess the image from bytes
        :param bytes: bytes that represent an image
        """
        nparr = np.frombuffer(bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        self.load_img(img)

    def save_image(self, path: str | Path) -> None:
        """
        Save the image at the given path
        :param path: path to save the image
        """
        cv2.imwrite(str(path), self._image)

    def show_image(self) -> None:
        """Show the image in a window"""
        cv2.imshow("Detection", self._image)

    def hog_detect(self) -> None:
        """Perform a hog detection on the loaded image"""
        descriptor = HOGDescriptor()
        descriptor.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        people, weights = descriptor.detectMultiScale(
            self._image,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.04,
        )
        people = np.array([
            [x, y, x + w, y + h] for ((x, y, w, h), weight) in zip(people, weights) if weight > .3
        ])
        people = non_max_suppression(people, probs=None, overlapThresh=0.5)

        for index, (xA, yA, xB, yB) in enumerate(people, start=1):
            cv2.rectangle(self._image, (xA, yA), (xB, yB), BOX_COLOR, 2)
            cv2.rectangle(self._image, (xA, yA - 20), (xB, yA), BOX_COLOR, -1)
            cv2.putText(self._image, f'Person-{index}', (xA, yA), FONT, 0.5, FONT_COLOR, )

        self._number_of_people = len(people)

    def draw_details(self, time: float) -> None:
        """
        Add the details about total found people and the time it take on the image
        :param time: the time to print on the image
        """
        text_height = get_text_size(
            f' Total: {self._number_of_people} Time: {time}', FONT_SCALE, FONT_THICKNESS
        )[1]

        cv2.rectangle(
            self._image,
            (0, self._height),
            (self._width, self._height - text_height - 15),
            DETAILS_COLOR,
            -1,
        )
        cv2.putText(
            self._image,
            f' Total: {self._number_of_people} Time: {time}',
            (10, self._height - 10),
            FONT, FONT_SCALE, FONT_COLOR_SECONDARY, FONT_THICKNESS,
        )

    def choose_method(self) -> None:
        """Choose the method for performing a detection"""
        self._available_methods.get(self._model_name)()

    def check_image(self, model_name: str) -> None:
        """
        Measure time how long detection takes for given image, write it on the image
        :param model_name: model name to perform detection
        """
        self._model_name = model_name
        time = FPS().start()
        self.choose_method()
        time.stop()
        self.draw_details(time.elapsed())

    def encode_image(self) -> bytes:
        """
        Encode jpg image to the memory buffer
        :return: encoded image as bytes
        """
        return cv2.imencode('.jpg', self._image)[1]
