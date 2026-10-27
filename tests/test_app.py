import pytest

from app import app


@pytest.fixture(scope="module")
def client():
    return app.test_client()


@pytest.mark.parametrize("url", ["/", "/ask", "/preprocess", "/ngrams", "/pos", "/evaluation"])
def test_pages(client, url):
    assert client.get(url).status_code == 200


def test_apis(client):
    assert client.post("/api/preprocess", json={"text": "kitne students hain"}).status_code == 200
    assert client.get("/api/ngram/overview?corpus=questions-hg&n=3&smoothing=kneser_ney").status_code == 200
    assert client.get("/api/ngram/perplexity?corpus=questions-en").status_code == 200
    assert client.post("/api/ngram/sentence", json={"text": "kitne students hain", "corpus": "questions-hg"}).status_code == 200
    assert client.get("/api/autocomplete?text=kitne").status_code == 200
    assert client.post("/api/langid", json={"text": "kitne students hain"}).get_json()["prediction"] == "hg"
    assert client.post("/api/intent", json={"text": "kitne students hain"}).status_code == 200
    assert client.post("/api/pos/tag", json={"text": "kitne students hain"}).status_code == 200
    assert client.get("/api/pos/model?lang=en").status_code == 200
    assert client.get("/api/pos/eval").status_code == 200
    assert client.get("/api/eval").status_code == 200
    d = client.post("/api/ask", json={"text": "how many students are there"}).get_json()
    assert d["sql"] == "SELECT COUNT(*) FROM students" and d["result"]["rows"] == [[120]]


def test_bad_input(client):
    assert client.post("/api/ask", json={"text": ""}).status_code == 400
    assert client.post("/api/ask", json={"text": "x" * 400}).status_code == 400
    assert client.get("/api/ngram/overview?corpus=nope").status_code == 400
    assert client.get("/api/pos/model?lang=fr").status_code == 400
    assert client.post("/api/ask", json={"text": "hello there"}).get_json()["sql"] is None
