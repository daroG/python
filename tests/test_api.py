import cv2  # type: ignore[import]
import numpy as np
from fastapi.testclient import TestClient


def test_hello_world_enpoint(client: TestClient):
    response = client.get('/')
    assert response.status_code == 200
    assert 'msg' in response.json()
    assert 'Hello world' == response.json()['msg']


def test_detect_endpoint(client: TestClient, people_file_path: str):
    with open(people_file_path, 'rb') as file:
        response = client.post(
            '/detect', files={'file': ('filename', file, 'image/jpeg')}
        )
    assert response.status_code == 200
    nparr = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    assert img.shape == (600, 800, 3)
