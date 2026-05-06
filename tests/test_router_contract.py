from argusai.config import Config
from argusai.router import Router


class HeuristicOnlyClient:
    def generate_json(self, *args, **kwargs):
        raise AssertionError("heuristic routes must not call the router model")


class EmptyStructuredThenWebSearchClient:
    def generate_json(self, *args, **kwargs):
        return {}

    def generate_text(self, *args, **kwargs):
        return """
        {
          "primary_route": "web_search",
          "reason": "The request asks for current external information.",
          "confidence": 0.95,
          "use_memory": false,
          "use_ocr": false,
          "use_web": true,
          "requires_reasoning": true,
          "secondary_routes": ["reasoning"]
        }
        """


class InvalidRouterClient:
    def generate_json(self, *args, **kwargs):
        return {"primary_route": "unknown"}

    def generate_text(self, *args, **kwargs):
        return "{}"


def make_router(client):
    return Router(Config(), client)


def test_code_request_uses_deterministic_heuristic():
    router = make_router(HeuristicOnlyClient())
    pre = router.preprocess_input(user_text="create file app.py with a hello function")

    plan = router.plan_route(pre)

    assert plan.primary_route == "code"
    assert plan.use_web is False
    assert pre.metadata["requested_filename"] == "app.py"


def test_url_request_routes_to_web_scrape_without_llm():
    router = make_router(HeuristicOnlyClient())
    pre = router.preprocess_input(user_text="Summarize https://example.com/page")

    plan = router.plan_route(pre)

    assert plan.primary_route == "web_scrape"
    assert plan.use_web is True


def test_empty_structured_router_output_uses_text_fallback_for_web_search():
    router = make_router(EmptyStructuredThenWebSearchClient())
    pre = router.preprocess_input(user_text="C est quoi le prix du petrole aujourd hui ?")

    plan = router.plan_route(pre)

    assert plan.primary_route == "web_search"
    assert plan.use_web is True
    assert plan.confidence == 0.95


def test_router_failure_is_explicit_low_confidence_reasoning():
    router = make_router(InvalidRouterClient())
    pre = router.preprocess_input(user_text="question ambiguous")

    plan = router.plan_route(pre)

    assert plan.primary_route == "reasoning"
    assert plan.use_web is False
    assert plan.confidence == 0.1
    assert "failed to return a valid JSON route plan" in plan.reason
