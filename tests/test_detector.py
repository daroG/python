from pathlib import Path

import cv2  # type: ignore[import]
import numpy as np
import pytest

from rest_image_tracker.detector.detector import Detector


def test_init():
    detector = Detector()
    assert len(detector._available_methods) == 1
    assert list(detector._available_methods.keys()) == ['HOG']
    assert detector._image is None
    assert detector._height == 0
    assert detector._width == 0
    assert detector._max_width == 800
    assert detector._max_height == 600
    assert detector._number_of_people == 0
    assert detector._model_name is None


def test_resize_on_too_big_raise():
    detector = Detector()
    with pytest.raises(Exception):
        detector.resize_on_too_big()


def test_resize_on_too_big_width():
    detector = Detector()
    h = 30
    w = 2000
    detector._image = np.random.rand(h, w, 3)
    detector.resize_on_too_big()
    assert detector._image.shape[1] == detector._max_width


def test_resize_on_too_big_height():
    detector = Detector()
    h = 2000
    w = 20
    detector._image = np.random.rand(h, w, 3)
    detector.resize_on_too_big()
    assert detector._image.shape[0] == detector._max_height


def test_update_sizes_raise():
    detector = Detector()
    with pytest.raises(Exception):
        detector.update_sizes()


@pytest.mark.parametrize('width, height', [
    (400, 500),
    (300, 200),
    (10, 6000),
    (40, 1500),
])
def test_update_sizes_not_raise(width: int, height: int):
    detector = Detector()
    detector._image = np.random.rand(height, width, 3)
    detector.update_sizes()
    assert detector._height == height
    assert detector._width == width


def test_load_image_fit():
    detector = Detector()
    height = 300
    width = 300
    image = np.random.rand(height, width, 3)
    detector.load_img(image)
    assert np.array_equal(detector._image, image)
    assert detector._width == width
    assert detector._height == height


def test_load_image_too_big():
    detector = Detector()
    height = 3000
    width = 30
    image = np.random.rand(height, width, 3)
    detector.load_img(image)
    assert not np.array_equal(detector._image, image)
    assert detector._width < width
    assert detector._height == detector._max_height


def test_load_image_from_file(image_file_path: str):
    detector = Detector()
    detector.load_img_from_file(image_file_path)
    assert detector._width == 40
    assert detector._height == 30


def test_load_image_from_bytes():
    detector = Detector()
    height = 300
    width = 300
    image = np.random.rand(height, width, 3)
    detector.load_img_from_bytes(cv2.imencode('.jpg', image)[1])
    assert detector._width == width
    assert detector._height == height


def test_save_image(tmp_path: Path, image_file_path: str):
    dir = tmp_path / 'dir'
    dir.mkdir()

    detector = Detector()
    detector.load_img_from_file(image_file_path)

    image_path = dir / 'image.jpg'
    detector.save_image(image_path)

    assert image_path.exists()
    assert image_path.is_file()


def test_choose_method_raise():
    detector = Detector()
    with pytest.raises(Exception):
        detector.choose_method()


def test_encode_image(image_file_path: str):
    detector = Detector()
    detector.load_img_from_file(image_file_path)
    image = detector._image
    assert np.array_equal(cv2.imencode('.jpg', image)[1], detector.encode_image())


def test_detection_is_working(people_file_path: str):
    detector = Detector()
    detector.load_img_from_file(people_file_path)
    detector.perform_detecting('HOG')

    assert detector._number_of_people == 4
