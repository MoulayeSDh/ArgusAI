"""
Routing and input preprocessing for ArgusAI.

This module owns:
- input normalization
- metadata extraction
- heuristic routing
- LLM routing fallback through Ollama
- optional re-planning when the judge says route_ok=False

Design principle:
Use fast heuristics first, then fallback to the vision/router model only when
the route is ambiguous.

Web design:
- web_search: general internet search through SearchWebTool/ddgs.
- web_scrape: URL-specific scraping/crawling through Crawl4AI.
- Crawl4AI must not be used for general search.
"""

from __future__ import annotations

import logging
import textwrap
from typing import Any, Dict, Optional, Set, cast

from .config import Config, JudgeVerdict, PreprocessedInput, RoutePlan, RouteType
from .ollama_client import OllamaClient
from .utils import (
    contains_keywords,
    detect_requested_filename,
    encode_file_base64,
    extract_urls,
    get_doc_info,
    get_image_info,
    has_explicit_web_request,
    normalize_text,
    parse_route_list,
    safe_float,
    safe_json_loads,
)


class Router:
    """Hybrid router: heuristics first, LLM fallback second."""

    ALLOWED_ROUTES: Set[str] = {
        "reasoning",
        "vision",
        "ocr",
        "code",
        "memory",
        "web",
        "web_search",
        "web_scrape",
        "hybrid",
    }

    def __init__(self, config: Config, ollama_client: OllamaClient) -> None:
        self.config = config
        self.ollama = ollama_client

        self.code_keywords: Set[str] = {
            # English
            "code",
            "python",
            "bug",
            "debug",
            "traceback",
            "error",
            "exception",
            "script",
            "refactor",
            "function",
            "class",
            "sql",
            "docker",
            "bash",
            "pytorch",
            "tensorflow",
            "sklearn",
            "regex",
            "api",
            "compile",
            "unit test",
            "pytest",
            "fix this",
            "generate file",
            "create file",
            # French
            "corrige",
            "corriger",
            "corrige ce code",
            "débogue",
            "debogue",
            "debuguer",
            "erreur",
            "exception",
            "traceback",
            "fonction",
            "classe",
            "script",
            "refactorise",
            "génère un fichier",
            "genere un fichier",
            "crée un fichier",
            "cree un fichier",
            "écris un fichier",
            "ecris un fichier",
            "fichier python",
            "test unitaire",
            # Arabic / general simple markers
            "كود",
            "بايثون",
            "خطأ",
            "صحح",
        }

        self.ocr_keywords: Set[str] = {
            # English
            "ocr",
            "extract text",
            "read text",
            "receipt",
            "invoice",
            "scan",
            "screenshot",
            "document",
            "table",
            "pdf",
            "transcribe",
            "read this image",
            "extract from image",
            # French
            "extrais le texte",
            "extraire le texte",
            "lis le texte",
            "lire le texte",
            "lis cette image",
            "lire cette image",
            "facture",
            "reçu",
            "recu",
            "scan",
            "capture",
            "capture d'écran",
            "capture d ecran",
            "document scanné",
            "document scanne",
            "tableau",
            "transcrire",
            # Arabic / general simple markers
            "استخرج النص",
            "اقرأ النص",
            "صورة",
            "فاتورة",
        }

        self.memory_keywords: Set[str] = {
            # English
            "remember",
            "what did we discuss",
            "previous conversation",
            "history",
            "recall",
            "my previous",
            # French
            "souviens",
            "souviens-toi",
            "rappelle",
            "rappelle-moi",
            "historique",
            "conversation précédente",
            "conversation precedente",
            "ce qu'on a dit",
            "ce que nous avons dit",
            # Arabic / general simple markers
            "تذكر",
            "محادثة سابقة",
        }

    # ------------------------------------------------------------------
    # Input preprocessing
    # ------------------------------------------------------------------

    def preprocess_input(
        self,
        *,
        user_text: str,
        image_path: Optional[str] = None,
        doc_path: Optional[str] = None,
    ) -> PreprocessedInput:
        """
        Normalize user input and compute routing metadata.

        This method is intentionally deterministic and fast.
        """

        normalized = normalize_text(user_text)

        if image_path and user_text:
            input_type = "multimodal"
        elif image_path:
            input_type = "image"
        elif doc_path:
            input_type = "document"
        else:
            input_type = "text"

        requested_filename = detect_requested_filename(user_text)
        urls = extract_urls(user_text)
        explicit_web_request = has_explicit_web_request(user_text)

        metadata: Dict[str, Any] = {
            "has_code_intent": contains_keywords(normalized, self.code_keywords),
            "has_ocr_intent": contains_keywords(normalized, self.ocr_keywords),
            "has_memory_intent": contains_keywords(normalized, self.memory_keywords),
            "explicit_web_request": explicit_web_request,
            "requested_filename": requested_filename,
            "wants_file_artifact": bool(requested_filename),
            "contains_url": bool(urls),
            "urls": urls,
            "image_info": get_image_info(image_path) if image_path else {},
            "doc_info": get_doc_info(doc_path) if doc_path else {},
        }

        return PreprocessedInput(
            input_type=cast(Any, input_type),
            user_text=(user_text or "").strip(),
            normalized_text=normalized,
            image_path=image_path,
            doc_path=doc_path,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Main routing
    # ------------------------------------------------------------------

    def plan_route(
        self,
        pre: PreprocessedInput,
        *,
        force_llm: bool = False,
        judge_feedback: Optional[JudgeVerdict] = None,
    ) -> RoutePlan:
        """
        Decide the best execution route.

        Args:
            pre: Preprocessed user input.
            force_llm: If True, skip heuristics and ask the router model.
            judge_feedback: Optional judge feedback for re-planning.

        Returns:
            RoutePlan.
        """

        if not force_llm:
            direct_plan = self.heuristic_route(pre)
            if direct_plan is not None:
                return direct_plan

        return self.llm_route(pre, judge_feedback=judge_feedback)

    def replan_after_judge(
        self,
        pre: PreprocessedInput,
        previous_plan: RoutePlan,
        verdict: JudgeVerdict,
    ) -> RoutePlan:
        """
        Re-plan when the judge says route_ok=False.

        This is intentionally simple for v0.1.0-beta:
        - skip heuristics
        - ask the router model again
        - include previous route and judge critique
        """

        if verdict.route_ok:
            return previous_plan

        return self.plan_route(
            pre,
            force_llm=True,
            judge_feedback=verdict,
        )

    # ------------------------------------------------------------------
    # Heuristic routing
    # ------------------------------------------------------------------

    def heuristic_route(self, pre: PreprocessedInput) -> Optional[RoutePlan]:
        """
        Fast deterministic routing for obvious cases.

        Ambiguous cases return None and are delegated to the LLM router.
        """

        if pre.doc_path:
            return RoutePlan(
                primary_route="hybrid",
                reason="Document input usually needs loading, retrieval and synthesis.",
                confidence=0.95,
                use_memory=True,
                use_ocr=bool(pre.metadata.get("doc_info", {}).get("is_pdf", False)),
                use_web=False,
                requires_reasoning=True,
                secondary_routes=["memory", "reasoning"],
            )

        if pre.input_type in {"image", "multimodal"} and pre.metadata.get("has_ocr_intent"):
            return RoutePlan(
                primary_route="ocr",
                reason="Image input with explicit OCR intent.",
                confidence=0.93,
                use_memory=False,
                use_ocr=True,
                use_web=False,
                requires_reasoning=True,
            )

        if (
            pre.metadata.get("has_code_intent")
            or pre.metadata.get("wants_file_artifact")
        ) and not pre.image_path and not pre.doc_path:
            return RoutePlan(
                primary_route="code",
                reason="Strong coding or file-generation intent detected.",
                confidence=0.94,
                use_memory=True,
                use_ocr=False,
                use_web=False,
                requires_reasoning=False,
            )

        if pre.input_type == "image":
            return RoutePlan(
                primary_route="vision",
                reason="Pure image analysis path.",
                confidence=0.88,
                use_memory=False,
                use_ocr=False,
                use_web=False,
                requires_reasoning=True,
            )

        if pre.metadata.get("has_memory_intent"):
            return RoutePlan(
                primary_route="memory",
                reason="The request appears to ask about previous context or memory.",
                confidence=0.82,
                use_memory=True,
                use_ocr=False,
                use_web=False,
                requires_reasoning=True,
            )

        if pre.metadata.get("contains_url"):
            return RoutePlan(
                primary_route="web_scrape",
                reason="URL detected. Use Crawl4AI/web scraping path after user confirmation.",
                confidence=0.94,
                use_memory=True,
                use_ocr=False,
                use_web=True,
                requires_reasoning=True,
                secondary_routes=["reasoning"],
            )

        return None

    # ------------------------------------------------------------------
    # LLM routing
    # ------------------------------------------------------------------

    def llm_route(
        self,
        pre: PreprocessedInput,
        *,
        judge_feedback: Optional[JudgeVerdict] = None,
    ) -> RoutePlan:
        """Ask the router model to decide the route."""

        schema = self.routing_schema()
        prompt = self.build_router_prompt(pre, judge_feedback=judge_feedback)

        model_json = self.ollama.generate_json(
            model=self.config.router_vision_model,
            prompt=prompt,
            schema=schema,
            images=[encode_file_base64(pre.image_path)] if pre.image_path else None,
            temperature=0.0,
        )

        logging.info("Router raw model_json: %s", model_json)

        if not self.is_valid_route_json(model_json):
            logging.info("Router structured output invalid; using text fallback.")

            fallback_prompt = textwrap.dedent(
                f"""
                {prompt}

                Return only one valid JSON object.
                No markdown.
                No explanation.
                No comments.

                Example:
                {{
                  "primary_route": "web_search",
                  "reason": "The user asks for current weather, which requires internet search.",
                  "confidence": 0.95,
                  "use_memory": false,
                  "use_ocr": false,
                  "use_web": true,
                  "requires_reasoning": true,
                  "secondary_routes": ["reasoning"]
                }}
                """
            ).strip()

            raw_text = self.ollama.generate_text(
                model=self.config.router_vision_model,
                prompt=fallback_prompt,
                stream=False,
                temperature=0.0,
                think=False,
            )

            logging.info("Router fallback raw text: %s", raw_text[:1000])

            model_json = safe_json_loads(raw_text)

            logging.info("Router fallback parsed json: %s", model_json)

        if not self.is_valid_route_json(model_json):
            logging.warning("Router failed to return a valid route plan: %s", model_json)
            model_json = self.failed_route_json()

        plan = self.route_from_model_json(model_json)

        logging.info(
            "Router parsed plan: route=%s | use_web=%s | reason=%s",
            plan.primary_route,
            plan.use_web,
            plan.reason,
        )

        return plan

    def build_router_prompt(
        self,
        pre: PreprocessedInput,
        *,
        judge_feedback: Optional[JudgeVerdict] = None,
    ) -> str:
        """Build the router model prompt."""

        feedback_block = "None"

        if judge_feedback is not None:
            feedback_block = textwrap.dedent(
                f"""
                Previous judge feedback:
                - route_ok: {judge_feedback.route_ok}
                - critique: {judge_feedback.critique}
                - improvement_instructions: {judge_feedback.improvement_instructions}
                """
            ).strip()

        return textwrap.dedent(
            f"""
            You are a routing model for a local agent called ArgusAI.
            Decide the best route for the user request.

            Allowed routes:
            - reasoning
            - vision
            - ocr
            - code
            - memory
            - web_search
            - web_scrape
            - hybrid

            Tool definitions:
            - web_search:
              Use for general internet search when the user asks for current,
              live, recent, external, or time-sensitive information and no specific
              URL is provided. Examples: today's weather, current news, latest
              software versions, exchange rates, current prices, recent public
              events, live scores, current company/product/person status.

            - web_scrape:
              Use only when the user provides a specific URL or explicitly asks
              to scrape, crawl, read, summarize, or extract content from a website
              or webpage. This route uses Crawl4AI and expects URL-specific work.

            - reasoning:
              Use for local reasoning, explanations, stable knowledge, planning,
              and normal questions that do not require current external data.

            - code:
              Use for programming, debugging, refactoring, Docker, scripts,
              notebooks, SQL, tests, and file generation.

            - ocr:
              Use for extracting text from images, screenshots, invoices,
              receipts, scanned documents, or PDFs.

            - vision:
              Use for visual understanding of an image when OCR is not the main
              task.

            - memory:
              Use when the request mainly asks about previous conversation,
              stored context, or remembered project details.

            - hybrid:
              Use when multiple tools are clearly required.

            Critical routing rules:
            - Do not route general search to web_scrape.
            - Do not use Crawl4AI for general search.
            - If the user asks for current, live, today, recent, latest, or
              changing information, route to web_search even if the user does
              not explicitly say "internet" or "web".
            - If the user provides a URL, route to web_scrape.
            - If internet access is disabled but the request requires internet,
              still choose web_search or web_scrape and set use_web=true. The
              agent will ask the user to enable/confirm internet access.
            - Do not answer current-information requests from model memory.
            - Return strict JSON only.

            User text:
            {pre.user_text or '<empty>'}

            Input type:
            {pre.input_type}

            Metadata:
            {pre.metadata}

            Internet access armed:
            {self.web_is_enabled()}

            Crawl4AI available for URL scraping:
            {self.web_scrape_is_enabled()}

            Judge feedback for re-planning:
            {feedback_block}
            """
        ).strip()

    def routing_schema(self) -> Dict[str, Any]:
        """JSON schema expected from the router model."""

        return {
            "type": "object",
            "properties": {
                "primary_route": {
                    "type": "string",
                    "enum": [
                        "reasoning",
                        "vision",
                        "ocr",
                        "code",
                        "memory",
                        "web",
                        "web_search",
                        "web_scrape",
                        "hybrid",
                    ],
                },
                "reason": {"type": "string"},
                "confidence": {"type": "number"},
                "use_memory": {"type": "boolean"},
                "use_ocr": {"type": "boolean"},
                "use_web": {"type": "boolean"},
                "requires_reasoning": {"type": "boolean"},
                "secondary_routes": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "primary_route",
                "reason",
                "confidence",
                "use_memory",
                "use_ocr",
                "use_web",
                "requires_reasoning",
                "secondary_routes",
            ],
        }

    def route_from_model_json(self, model_json: Dict[str, Any]) -> RoutePlan:
        """Convert raw LLM JSON into a validated RoutePlan."""

        primary = str(model_json.get("primary_route", "reasoning")).strip().lower()

        if primary == "web":
            primary = "web_search"

        if primary not in self.ALLOWED_ROUTES:
            primary = "reasoning"

        use_web_requested = self._as_bool(model_json.get("use_web", False))

        if primary in {"web_search", "web_scrape"}:
            use_web_requested = True

        confidence = safe_float(model_json.get("confidence", 0.65), 0.65)
        confidence = max(0.0, min(1.0, confidence))

        return RoutePlan(
            primary_route=cast(RouteType, primary),
            reason=str(model_json.get("reason", "Model-based routing fallback.")),
            confidence=confidence,
            use_memory=self._as_bool(model_json.get("use_memory", True)),
            use_ocr=self._as_bool(model_json.get("use_ocr", False)),
            use_web=use_web_requested,
            requires_reasoning=self._as_bool(model_json.get("requires_reasoning", True)),
            secondary_routes=parse_route_list(model_json.get("secondary_routes", [])),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def web_is_enabled(self) -> bool:
        """
        Return True if internet access is armed.

        This does not mean Crawl4AI is available. ddgs/web_search only needs
        Python network access and the ddgs package.
        """

        return bool(self.config.web_enabled)

    def web_scrape_is_enabled(self) -> bool:
        """Return True if URL scraping through Crawl4AI is enabled."""

        return bool(self.config.web_enabled and self.config.crawl4ai_enabled)

    def is_valid_route_json(self, model_json: Dict[str, Any]) -> bool:
        """Return True only when the router produced a complete route plan."""

        if not isinstance(model_json, dict):
            return False

        primary = str(model_json.get("primary_route", "")).strip().lower()
        if primary == "web":
            primary = "web_search"

        if primary not in self.ALLOWED_ROUTES:
            return False

        required_keys = {
            "primary_route",
            "reason",
            "confidence",
            "use_memory",
            "use_ocr",
            "use_web",
            "requires_reasoning",
            "secondary_routes",
        }

        return required_keys.issubset(model_json.keys())

    @staticmethod
    def failed_route_json() -> Dict[str, Any]:
        """Build an explicit fallback plan when the router model fails."""

        return {
            "primary_route": "reasoning",
            "reason": (
                "Router model failed to return a valid JSON route plan; "
                "falling back to local reasoning without pretending routing succeeded."
            ),
            "confidence": 0.1,
            "use_memory": False,
            "use_ocr": False,
            "use_web": False,
            "requires_reasoning": True,
            "secondary_routes": [],
        }

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        """Parse bool-like values safely."""

        if isinstance(value, bool):
            return value

        if value is None:
            return default

        if isinstance(value, (int, float)):
            return bool(value)

        normalized = str(value).strip().lower()

        if normalized in {"true", "1", "yes", "y", "oui", "o"}:
            return True

        if normalized in {"false", "0", "no", "n", "non"}:
            return False

        return default
