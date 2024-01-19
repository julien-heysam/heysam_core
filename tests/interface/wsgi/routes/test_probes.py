from fastapi.testclient import TestClient

from src.interface.wsgi.app import app

client = TestClient(app)


def test_health_check():
    response = client.get("/probes/")
    assert response.status_code == 200

def test_probe_startup_ready():
    # Assuming app.state is properly set up before the test
    app.state = True
    response = client.get("/probes/startup")
    assert response.status_code == 200

def test_probe_startup_not_ready():
    # Mimicking an unset or improperly set app.state
    app.state = None
    response = client.get("/probes/startup")
    assert response.status_code == 503

def test_probe_readiness_ready():
    # Assuming app.state is properly set up before the test
    app.state = True
    response = client.get("/probes/readiness")
    assert response.status_code == 200

def test_probe_readiness_not_ready():
    # Mimicking an unset or improperly set app.state
    app.state = None
    response = client.get("/probes/readiness")
    assert response.status_code == 425

def test_probe_liveness():
    response = client.get("/probes/liveness")
    assert response.status_code == 200
