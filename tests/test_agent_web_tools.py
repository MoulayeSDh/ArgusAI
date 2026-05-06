from argusai.agent import ArgusAI
from collections import deque

from argusai.config import ArtifactResult, Config, ExecutionBundle, JudgeVerdict, RoutePlan
from argusai.router import Router


class NoopRouterClient:
    pass


class FakeSearchTool:
    def __init__(self):
        self.queries = []

    def is_available(self):
        return True

    def search_context(self, query, max_results=None):
        self.queries.append((query, max_results))
        return "Web search results:\n[1] Oil price source\nSnippet: Brent price context"


class FakeUnavailableSearchTool:
    def is_available(self):
        return False


class FakeCrawler:
    def __init__(self):
        self.requests = []

    def is_enabled(self):
        return True

    def web_context(self, user_text):
        self.requests.append(user_text)
        return "Scraped page evidence"


class FakeDisabledCrawler:
    def is_enabled(self):
        return False


def make_agent(*, web_enabled=True, confirm=True):
    agent = ArgusAI.__new__(ArgusAI)
    agent.config = Config(web_enabled=web_enabled, crawl4ai_enabled=True)
    agent.search_tool = FakeSearchTool()
    agent.web_crawler = FakeCrawler()
    agent.confirm_web_access = lambda pre, plan: confirm
    return agent


def preprocess(text):
    router = Router(Config(), NoopRouterClient())
    return router.preprocess_input(user_text=text)


def web_search_plan():
    return RoutePlan(
        primary_route="web_search",
        reason="Needs current external data.",
        confidence=0.95,
        use_memory=False,
        use_web=True,
        requires_reasoning=True,
        secondary_routes=["reasoning"],
    )


def web_scrape_plan():
    return RoutePlan(
        primary_route="web_scrape",
        reason="URL detected.",
        confidence=0.95,
        use_memory=True,
        use_web=True,
        requires_reasoning=True,
        secondary_routes=["reasoning"],
    )


def test_execute_web_search_tool_adds_search_evidence():
    agent = make_agent()
    pre = preprocess("C est quoi le prix du petrole aujourd hui ?")
    evidence = {}

    agent.execute_web_tool(pre, web_search_plan(), evidence)

    assert "web_search_context" in evidence
    assert "Oil price source" in evidence["web_search_context"]
    assert agent.search_tool.queries == [(pre.user_text, 5)]


def test_execute_web_tool_records_denial_without_calling_search():
    agent = make_agent(confirm=False)
    pre = preprocess("current oil price")
    evidence = {}

    agent.execute_web_tool(pre, web_search_plan(), evidence)

    assert "web_search_context" in evidence
    assert "denied" in evidence["web_search_context"].lower()
    assert agent.search_tool.queries == []


def test_execute_web_tool_records_disabled_web():
    agent = make_agent(web_enabled=False)
    pre = preprocess("current oil price")
    evidence = {}

    agent.execute_web_tool(pre, web_search_plan(), evidence)

    assert "web_search_context" in evidence
    assert "disabled" in evidence["web_search_context"].lower()


def test_execute_web_scrape_tool_adds_scrape_evidence():
    agent = make_agent()
    pre = preprocess("Read https://example.com/page")
    evidence = {}

    agent.execute_web_tool(pre, web_scrape_plan(), evidence)

    assert evidence["web_scrape_context"] == "Scraped page evidence"
    assert agent.web_crawler.requests == [pre.user_text]


def test_execute_web_tool_reports_unavailable_dependencies():
    agent = make_agent()
    agent.search_tool = FakeUnavailableSearchTool()
    pre = preprocess("current oil price")
    evidence = {}

    agent.execute_web_tool(pre, web_search_plan(), evidence)

    assert "web_search_context" in evidence
    assert "unavailable" in evidence["web_search_context"].lower()

    agent.web_crawler = FakeDisabledCrawler()
    pre = preprocess("Read https://example.com/page")
    evidence = {}

    agent.execute_web_tool(pre, web_scrape_plan(), evidence)

    assert "web_scrape_context" in evidence
    assert "unavailable" in evidence["web_scrape_context"].lower()


class FakePipelineRouter:
    def preprocess_input(self, *, user_text, image_path=None, doc_path=None):
        return preprocess(user_text)

    def plan_route(self, pre):
        return web_search_plan()


class FakeMemory:
    def __init__(self):
        self.writes = []
        self.last_retrieved_count = 0

    def is_ready(self):
        return False

    def write(self, **kwargs):
        self.writes.append(kwargs)


def test_run_once_pipeline_handles_web_search_route_without_trace_data():
    agent = ArgusAI.__new__(ArgusAI)
    agent.config = Config(artifact_note_in_answer=False)
    setattr(agent.config, "enable_" + "tra" + "ces", False)
    agent.router = FakePipelineRouter()
    agent.short_memory = deque(maxlen=4)
    agent.memory = FakeMemory()

    def fake_execute_plan(pre, plan):
        return ExecutionBundle(
            route=plan.primary_route,
            memory_context="",
            evidence={
                "web_search_context": (
                    "Web search results:\n"
                    "[1] Market source\n"
                    "Snippet: Current oil market context"
                )
            },
            metadata={"route_reason": plan.reason, "route_confidence": plan.confidence},
        )

    agent.execute_plan = fake_execute_plan
    agent.generate_draft = lambda pre, plan, bundle: "Draft grounded in web evidence."
    agent.judge_draft = lambda pre, plan, bundle, draft: JudgeVerdict(
        score=9.0,
        accept=True,
        critique="ok",
        improvement_instructions="none",
        route_ok=True,
    )
    agent.generate_final_answer = (
        lambda pre, plan, bundle, draft, verdict, stream_callback=None: "Final oil answer."
    )
    agent.maybe_create_artifact = lambda pre, plan, final_answer: ArtifactResult(
        requested=False
    )

    result = agent.run_once(user_text="C est quoi le prix du petrole aujourd hui ?")

    assert result["route"] == "web_search"
    assert result["accepted"] is True
    assert result["route_ok"] is True
    assert result["answer"] == "Final oil answer."
    assert agent.memory.writes[0]["route"] == "web_search"
