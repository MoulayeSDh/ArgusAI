from argusai.ollama_client import OllamaClient


class DummyResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_generate_json_reads_response_first(monkeypatch):
    def fake_post(*args, **kwargs):
        return DummyResponse(
            {
                "response": '{"primary_route": "reasoning"}',
                "thinking": '{"primary_route": "web_search"}',
            }
        )

    monkeypatch.setattr("argusai.ollama_client.requests.post", fake_post)

    client = OllamaClient("http://ollama.local")
    result = client.generate_json("router", "prompt", {"type": "object"})

    assert result["primary_route"] == "reasoning"


def test_generate_json_reads_thinking_when_response_is_empty(monkeypatch):
    def fake_post(*args, **kwargs):
        return DummyResponse(
            {
                "response": "",
                "thinking": '{"primary_route": "web_search", "use_web": true}',
            }
        )

    monkeypatch.setattr("argusai.ollama_client.requests.post", fake_post)

    client = OllamaClient("http://ollama.local")
    result = client.generate_json("router", "prompt", {"type": "object"})

    assert result["primary_route"] == "web_search"
    assert result["use_web"] is True
