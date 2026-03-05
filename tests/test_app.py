from fastapi.testclient import TestClient

from app.main import app


def test_index():
    client = TestClient(app)
    res = client.get('/')
    assert res.status_code == 200
    assert '動画コーナー解析' in res.text
