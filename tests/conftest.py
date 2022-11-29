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
