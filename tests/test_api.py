from fastapi.testclient import TestClient


def test_hello_world_enpoint(client: TestClient):
    response = client.get('/')
    assert response.status_code == 200
    assert 'msg' in response.json()
    assert 'Hello world' == response.json()['msg']
