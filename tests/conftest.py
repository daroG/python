from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rest_image_tracker import app


@pytest.fixture
def client() -> TestClient:
    """
    Initialize application client

    :return: test client
    """
    return TestClient(app)


@pytest.fixture(scope='package')
def image_file_path() -> str:
    """
    Return stringified path for test image

    :return: path string
    """
    return str((Path(__file__).parent / 'test_data' / 'image.jpg').absolute())


@pytest.fixture(scope='package')
def people_file_path() -> str:
    """
    Return stringified path for test image of people

    :return: path string
    """
    return str((Path(__file__).parent / 'test_data' / 'people.jpg').absolute())
