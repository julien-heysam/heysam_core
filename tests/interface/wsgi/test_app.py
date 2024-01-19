from fastapi.testclient import TestClient

from src.interface.wsgi.app import app

client = TestClient(app)
