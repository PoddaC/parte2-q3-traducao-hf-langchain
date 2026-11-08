from fastapi.testclient import TestClient
from langchain_core.language_models.fake import FakeListLLM

from app.services.translation_service import build_chain, get_chain
from app.main import app

fake = build_chain(FakeListLLM(responses=["Guten Morgen, wie geht es Ihnen?"]))
app.dependency_overrides[get_chain] = lambda: fake
client = TestClient(app)


def test_translate_ok():
    r = client.post("/translate", json={"text": "Good morning, how are you?"})
    assert r.status_code == 200
    assert r.json()["translation"] == "Guten Morgen, wie geht es Ihnen?"
    assert r.json()["model"] == "Helsinki-NLP/opus-mt-en-de"


def test_empty_rejected():
    assert client.post("/translate", json={"text": ""}).status_code == 422
