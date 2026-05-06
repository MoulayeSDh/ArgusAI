from __future__ import annotations

import json
import logging
import sys
import textwrap
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

from .artifacts import FileArtifactManager, append_artifact_note
from .config import (
    ArtifactResult,
    Config,
    ExecutionBundle,
    JudgeVerdict,
    Message,
    PreprocessedInput,
    RoutePlan,
)
from .documents import DocumentLoader
from .memory import MemoryManager
from .ocr import OCRManager
from .ollama_client import OllamaClient
from .router import Router
from .search import SearchWebTool
from .utils import (
    encode_file_base64,
    is_affirmative,
    model_installed,
    new_run_id,
    safe_float,
    setup_logging,
    write_trace,
)
from .web import WebCrawler


class ArgusAI:
    """Local-first multimodal agent."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.log_file_path = setup_logging(self.config)

        self.short_memory: Deque[Message] = deque(maxlen=self.config.buffer_k * 2)

        self.ollama = OllamaClient(
            base_url=self.config.ollama_base_url,
            request_timeout=self.config.request_timeout,
            keep_alive=self.config.keep_alive,
        )

        self.router = Router(
            config=self.config,
            ollama_client=self.ollama,
        )

        self.file_manager = FileArtifactManager(
            base_dir=self.config.artifacts_dir,
            max_chars=self.config.artifact_max_chars,
        )

        self.ocr_manager = OCRManager(self.config)
        self.document_loader = DocumentLoader(self.config)

        # General internet search: no API key, no Docker required.
        self.search_tool = SearchWebTool()

        # URL-specific scraping/crawling: optional Crawl4AI backend.
        self.web_crawler = WebCrawler(self.config)

        self._verify_ollama_models()

        self.memory = MemoryManager(
            config=self.config,
            ollama_client=self.ollama,
        )

        self._print_startup_status()

    # ------------------------------------------------------------------
    # Startup / status
    # ------------------------------------------------------------------

    def _verify_ollama_models(self) -> None:
        """Check required Ollama models before starting the agent."""

        try:
            installed = self.ollama.list_models()
        except Exception as exc:
            raise RuntimeError(
                f"Unable to reach Ollama at {self.config.ollama_base_url}/api/tags: {exc}"
            ) from exc

        missing = [
            model_name
            for model_name in sorted(self.config.required_models())
            if not model_installed(model_name, installed)
        ]

        if missing:
            raise RuntimeError(
                "Missing Ollama models: "
                + ", ".join(missing)
                + ". Pull them locally or override model tags via CLI arguments."
            )

    def _print_startup_status(self) -> None:
        """Write startup status to logs."""

        status = [
            f"memory={self.memory.status()}",
            f"web_search={self.search_tool.status()}",
            f"web_scrape={self.web_crawler.status()}",
            f"ocr={self.ocr_manager.available_backend()}",
            f"pdf_ocr={self.document_loader.pdf_ocr_backend_status()}",
            f"logs={self.config.log_file_path.as_posix()}",
            f"traces={self.config.traces_file_path.as_posix() if self.config.enable_traces else 'off'}",
        ]

        logging.info(
            "ArgusAI %s ready | %s",
            self.config.version,
            " | ".join(status),
        )

    def status_dict(self) -> Dict[str, Any]:
        """Return runtime status for CLI display."""

        return {
            "version": self.config.version,
            "memory": self.memory.status(),
            "web_enabled": bool(self.config.web_enabled),
            "web_search": self.search_tool.status(),
            "web_scrape": self.web_crawler.status(),
            "ocr": self.ocr_manager.available_backend(),
            "pdf_ocr": self.document_loader.pdf_ocr_backend_status(),
            "ollama_url": self.config.ollama_base_url,
            "qdrant_url": self.config.qdrant_url,
            "logs": self.config.log_file_path.as_posix(),
            "traces": self.config.traces_file_path.as_posix()
            if self.config.enable_traces
            else "off",
        }

    # ------------------------------------------------------------------
    # Public execution API
    # ------------------------------------------------------------------

    def run_once(
        self,
        *,
        user_text: str,
        image_path: Optional[str] = None,
        doc_path: Optional[str] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Execute one user request.

        Pipeline:
        preprocess -> route -> execute tools -> draft -> judge -> optional re-route
        -> improve -> final answer -> artifact -> memory write -> trace.
        """

        run_id = new_run_id()
        start_time = time.time()

        trace_payload: Dict[str, Any] = {
            "run_id": run_id,
            "input_type": None,
            "route": None,
            "secondary_routes": [],
            "tools_called": [],
            "memory_ready": False,
            "retrieved_chunks": 0,
            "judge_score": None,
            "judge_accept": None,
            "route_ok": None,
            "latency_sec": None,
            "artifact_saved": False,
            "error": None,
        }

        try:
            pre = self.router.preprocess_input(
                user_text=user_text,
                image_path=image_path,
                doc_path=doc_path,
            )

            plan = self.router.plan_route(pre)

            trace_payload["input_type"] = pre.input_type
            trace_payload["route"] = plan.primary_route
            trace_payload["secondary_routes"] = plan.secondary_routes

            bundle = self.execute_plan(pre, plan)

            draft = self.generate_draft(pre, plan, bundle)
            verdict = self.judge_draft(pre, plan, bundle, draft)

            if self.config.enable_route_recheck and not verdict.route_ok:
                logging.info(
                    "Judge reported route_ok=False. Re-planning once. Previous route=%s",
                    plan.primary_route,
                )

                plan = self.router.replan_after_judge(
                    pre=pre,
                    previous_plan=plan,
                    verdict=verdict,
                )

                trace_payload["route"] = plan.primary_route
                trace_payload["secondary_routes"] = plan.secondary_routes

                bundle = self.execute_plan(pre, plan)
                draft = self.generate_draft(pre, plan, bundle)
                verdict = self.judge_draft(pre, plan, bundle, draft)

            current_answer = draft
            final_verdict = verdict

            for _ in range(1, max(1, int(self.config.max_judge_rounds))):
                if final_verdict.accept:
                    break

                current_answer = self.improve_draft(
                    pre=pre,
                    plan=plan,
                    bundle=bundle,
                    draft=current_answer,
                    verdict=final_verdict,
                )

                final_verdict = self.judge_draft(
                    pre=pre,
                    plan=plan,
                    bundle=bundle,
                    draft=current_answer,
                )

            final_answer = self.generate_final_answer(
                pre=pre,
                plan=plan,
                bundle=bundle,
                draft=current_answer,
                verdict=final_verdict,
                stream_callback=stream_callback,
            )

            artifact_result = self.maybe_create_artifact(
                pre=pre,
                plan=plan,
                final_answer=final_answer,
            )

            final_answer = append_artifact_note(
                final_answer,
                artifact_result,
                enabled=self.config.artifact_note_in_answer,
            )

            self.memory.write(
                user_text=pre.user_text,
                answer=final_answer,
                route=plan.primary_route,
                score=final_verdict.score,
                short_memory=self.short_memory,
            )

            latency = round(time.time() - start_time, 3)

            trace_payload.update(
                {
                    "input_type": pre.input_type,
                    "route": plan.primary_route,
                    "secondary_routes": plan.secondary_routes,
                    "tools_called": self._infer_tools_called(bundle),
                    "memory_ready": self.memory.is_ready(),
                    "retrieved_chunks": self.memory.last_retrieved_count,
                    "judge_score": final_verdict.score,
                    "judge_accept": final_verdict.accept,
                    "route_ok": final_verdict.route_ok,
                    "latency_sec": latency,
                    "artifact_saved": bool(artifact_result.saved_path),
                    "error": None,
                }
            )

            return {
                "answer": final_answer,
                "route": plan.primary_route,
                "score": final_verdict.score,
                "accepted": final_verdict.accept,
                "route_ok": final_verdict.route_ok,
                "artifact": artifact_result.saved_path,
                "latency_sec": latency,
            }

        except Exception as exc:
            trace_payload["error"] = str(exc)
            trace_payload["latency_sec"] = round(time.time() - start_time, 3)
            logging.exception("ArgusAI run_once failed.")
            raise

        finally:
            write_trace(self.config, trace_payload)

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def execute_plan(
        self,
        pre: PreprocessedInput,
        plan: RoutePlan,
    ) -> ExecutionBundle:
        """Execute tools required by the route plan."""

        evidence: Dict[str, Any] = {}

        if pre.doc_path:
            doc_text = self.document_loader.load_document_text(
                pre.doc_path,
                self.ocr_manager,
            )

            if doc_text:
                evidence["document_excerpt"] = doc_text[:2500]
                self.memory.ingest_document(pre.doc_path, doc_text)

        memory_context = (
            self.memory.retrieve(pre.user_text, self.short_memory)
            if self.should_retrieve_memory(pre, plan)
            else ""
        )

        if memory_context:
            evidence["memory_context"] = memory_context[:1500]

        if plan.use_ocr and (pre.image_path or pre.doc_path):
            target_path = pre.image_path or pre.doc_path

            if target_path:
                evidence["ocr_text"] = self.ocr_manager.run_ocr_on_path(
                    target_path,
                    self.document_loader,
                )

        if plan.primary_route in {"vision", "hybrid"} and pre.image_path:
            evidence["vision_analysis"] = self.vision_analysis(
                user_text=pre.user_text,
                image_path=pre.image_path,
                memory_context=memory_context,
            )

        if self.request_wants_web(pre, plan):
            self.execute_web_tool(pre, plan, evidence)

        return ExecutionBundle(
            route=plan.primary_route,
            memory_context=memory_context,
            evidence=evidence,
            metadata={
                "route_reason": plan.reason,
                "route_confidence": plan.confidence,
            },
        )

    def execute_web_tool(
        self,
        pre: PreprocessedInput,
        plan: RoutePlan,
        evidence: Dict[str, Any],
    ) -> None:
        """
        Execute the correct internet tool.

        - web_search: general search through ddgs.
        - web_scrape: URL/page scraping through Crawl4AI.
        """

        route = plan.primary_route

        if not self.config.web_enabled:
            key = "web_search_context" if route != "web_scrape" else "web_scrape_context"
            evidence[key] = (
                "Internet access is required for this request, but web access is disabled. "
                "Ask the user to enable it with /web on."
            )
            return

        if not self.confirm_web_access(pre, plan):
            key = "web_search_context" if route != "web_scrape" else "web_scrape_context"
            evidence[key] = (
                "Internet access was denied for this request. "
                "Continue without web results and do not invent current facts."
            )
            return

        if route == "web_scrape" or pre.metadata.get("contains_url"):
            if not self.web_crawler.is_enabled():
                evidence["web_scrape_context"] = (
                    "URL scraping was requested, but Crawl4AI/web scraping is unavailable. "
                    "Do not invent page content."
                )
                return

            evidence["web_scrape_context"] = self.web_crawler.web_context(pre.user_text)
            return

        if route in {"web_search", "web", "hybrid"} or plan.use_web:
            if not self.search_tool.is_available():
                evidence["web_search_context"] = (
                    "General web search was requested, but ddgs is unavailable. "
                    "Install ddgs or continue without current web results."
                )
                return

            evidence["web_search_context"] = self.search_tool.search_context(
                query=pre.user_text,
                max_results=5,
            )

    def should_retrieve_memory(
        self,
        pre: PreprocessedInput,
        plan: RoutePlan,
    ) -> bool:
        """
        Decide whether semantic memory should be retrieved for this request.

        Memory grounding policy:
        - Do not retrieve memory for simple greetings.
        - Do not retrieve memory for generic knowledge questions by default.
        - Retrieve memory only when the request is likely contextual.
        """

        if not plan.use_memory:
            return False

        if not self.memory.is_ready():
            return False

        text = pre.normalized_text.strip()

        if not text:
            return False

        simple_greetings = {
            "hi",
            "hello",
            "hey",
            "bonjour",
            "salut",
            "salam",
            "مرحبا",
            "السلام عليكم",
        }

        if text in simple_greetings:
            return False

        if len(text) <= 12 and plan.primary_route == "reasoning":
            return False

        if pre.metadata.get("has_memory_intent"):
            return True

        if plan.primary_route in {"memory", "hybrid", "code"}:
            return True

        contextual_markers = {
            "mon projet",
            "notre projet",
            "ce projet",
            "argusai",
            "rappelle",
            "rappelle-moi",
            "souviens",
            "souviens-toi",
            "précédemment",
            "precedemment",
            "avant",
            "historique",
            "conversation",
            "ce qu'on a dit",
            "ce que nous avons dit",
            "previous",
            "earlier",
            "before",
            "our project",
            "my project",
            "remember",
            "recall",
            "what did we discuss",
        }

        return any(marker in text for marker in contextual_markers)

    def vision_analysis(
        self,
        *,
        user_text: str,
        image_path: str,
        memory_context: str,
    ) -> str:
        """Analyze an image with the router/vision model."""

        prompt = textwrap.dedent(
            f"""
            You are the vision model inside ArgusAI.
            Analyze the image carefully.
            Be precise and concise.
            If the user asked a question, answer it.

            User request:
            {user_text or 'Describe the image.'}

            Optional memory context:
            {memory_context or 'None'}

            Memory policy:
            Use memory only if it is directly relevant to the current image request.
            Ignore unrelated memory.
            Do not invent a continuation of a previous conversation.
            """
        ).strip()

        return self.ollama.generate_text(
            model=self.config.router_vision_model,
            prompt=prompt,
            images=[encode_file_base64(image_path)],
            stream=False,
            temperature=0.1,
            think=False,
        )

    def request_wants_web(
        self,
        pre: PreprocessedInput,
        plan: RoutePlan,
    ) -> bool:
        """Return True if the request needs any internet tool."""

        return bool(
            plan.primary_route in {"web", "web_search", "web_scrape"}
            or plan.use_web
            or pre.metadata.get("contains_url")
            or pre.metadata.get("explicit_web_request")
        )

    def confirm_web_access(
        self,
        pre: PreprocessedInput,
        plan: RoutePlan,
    ) -> bool:
        """
        Ask user confirmation before using internet.

        Non-interactive sessions deny internet by default.
        """

        if not self.config.web_enabled:
            return False

        if not self.request_wants_web(pre, plan):
            return False

        if not sys.stdin.isatty():
            logging.info("Non-interactive session: internet access denied by default.")
            return False

        answer = input(
            f"This request needs internet access for route '{plan.primary_route}'. "
            "Continue? [y/n]: "
        )

        return is_affirmative(answer)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_draft(
        self,
        pre: PreprocessedInput,
        plan: RoutePlan,
        bundle: ExecutionBundle,
    ) -> str:
        """Generate an internal draft answer."""

        prompt = self.build_draft_prompt(pre, plan, bundle)
        model = self._model_for_route(plan)

        return self.ollama.generate_text(
            model=model,
            prompt=prompt,
            stream=False,
            temperature=0.2,
            think=self._think_for_route(plan),
        )

    def improve_draft(
        self,
        *,
        pre: PreprocessedInput,
        plan: RoutePlan,
        bundle: ExecutionBundle,
        draft: str,
        verdict: JudgeVerdict,
    ) -> str:
        """Improve a draft using judge feedback."""

        evidence = json.dumps(
            bundle.evidence,
            ensure_ascii=False,
            indent=2,
        )[:4000]

        prompt = textwrap.dedent(
            f"""
            Improve the draft answer using the judge feedback.
            Produce only the improved answer.
            Keep it concise, grounded, and technically correct.

            User request:
            {pre.user_text}

            Route:
            {plan.primary_route}

            Evidence:
            {evidence}

            Draft:
            {draft[:5000]}

            Judge critique:
            {verdict.critique}

            Improvement instructions:
            {verdict.improvement_instructions}

            Memory policy:
            Use memory only if it is directly relevant to the current user request.
            Ignore unrelated memory.
            Do not invent a continuation of a previous conversation.
            """
        ).strip()

        return self.ollama.generate_text(
            model=self._model_for_route(plan),
            prompt=prompt,
            stream=False,
            temperature=0.15,
            think=self._think_for_route(plan),
        )

    def judge_draft(
        self,
        pre: PreprocessedInput,
        plan: RoutePlan,
        bundle: ExecutionBundle,
        draft: str,
    ) -> JudgeVerdict:
        """Evaluate a draft answer with the judge model."""

        heuristic_score, heuristic_note = self.heuristic_quality_check(plan, draft)

        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "accept": {"type": "boolean"},
                "route_ok": {"type": "boolean"},
                "critique": {"type": "string"},
                "improvement_instructions": {"type": "string"},
            },
            "required": [
                "score",
                "accept",
                "route_ok",
                "critique",
                "improvement_instructions",
            ],
        }

        prompt = self.build_judge_prompt(
            pre=pre,
            plan=plan,
            bundle=bundle,
            draft=draft,
            heuristic_score=heuristic_score,
            heuristic_note=heuristic_note,
        )

        judge_json = self.ollama.generate_json(
            model=self.config.judge_model,
            prompt=prompt,
            schema=schema,
            temperature=0.0,
        )

        score = safe_float(
            judge_json.get("score", heuristic_score),
            heuristic_score,
        )

        accept = bool(
            judge_json.get(
                "accept",
                score >= self.config.judge_accept_threshold,
            )
        )

        return JudgeVerdict(
            score=score,
            accept=accept,
            critique=str(judge_json.get("critique", heuristic_note)),
            improvement_instructions=str(
                judge_json.get(
                    "improvement_instructions",
                    "Improve clarity and grounding.",
                )
            ),
            route_ok=bool(judge_json.get("route_ok", True)),
        )

    def generate_final_answer(
        self,
        *,
        pre: PreprocessedInput,
        plan: RoutePlan,
        bundle: ExecutionBundle,
        draft: str,
        verdict: JudgeVerdict,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Generate the final user-facing answer."""

        prompt = self.build_final_prompt(
            pre=pre,
            plan=plan,
            bundle=bundle,
            draft=draft,
            verdict=verdict,
        )

        should_stream = bool(stream_callback)

        return self.ollama.generate_text(
            model=self._model_for_route(plan),
            prompt=prompt,
            stream=should_stream,
            callback=stream_callback,
            temperature=0.15,
            think=(not should_stream and self._think_for_route(plan)),
        )

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def build_draft_prompt(
        self,
        pre: PreprocessedInput,
        plan: RoutePlan,
        bundle: ExecutionBundle,
    ) -> str:
        """Build draft-generation prompt."""

        evidence = json.dumps(
            bundle.evidence,
            ensure_ascii=False,
            indent=2,
        )[:5000]

        memory_policy = textwrap.dedent(
            """
            Memory policy:
            Use memory only if it is directly relevant to the current user request.
            Ignore unrelated memory.
            Do not invent a continuation of a previous conversation.
            If the user only greets you, answer normally without using retrieved memory.
            """
        ).strip()

        if plan.primary_route == "code":
            filename = pre.metadata.get("requested_filename")

            artifact_instruction = (
                f"If the user explicitly requested a file named {filename}, "
                "generate the file content directly. Prefer returning a single "
                "clean code block or raw file content with no unnecessary commentary."
                if filename
                else "If the user asks for code, prefer a single clean code block with minimal explanation."
            )

            return textwrap.dedent(
                f"""
                You are the coding specialist inside ArgusAI.
                Solve the user's request with practical and correct code.
                Prefer Python unless another language is explicitly requested.
                Keep the answer focused and useful.
                {artifact_instruction}

                User request:
                {pre.user_text}

                Memory/context:
                {bundle.memory_context or 'None'}

                Evidence:
                {evidence or '{}'}

                {memory_policy}
                """
            ).strip()

        if plan.primary_route == "web_search":
            return textwrap.dedent(
                f"""
                You are the web-search synthesis component inside ArgusAI.
                Use the web search evidence to answer the user's request.
                Do not invent current facts if search results are missing or weak.
                Mention uncertainty when the evidence is insufficient.
                Keep the answer concise.

                User request:
                {pre.user_text}

                Evidence:
                {evidence or '{}'}

                {memory_policy}
                """
            ).strip()

        if plan.primary_route == "web_scrape":
            return textwrap.dedent(
                f"""
                You are the URL-scraping synthesis component inside ArgusAI.
                Use the scraped page evidence to answer the user's request.
                Do not invent page content if scraping failed.
                Keep the answer concise and grounded.

                User request:
                {pre.user_text}

                Evidence:
                {evidence or '{}'}

                {memory_policy}
                """
            ).strip()

        return textwrap.dedent(
            f"""
            You are the main reasoning model inside ArgusAI.
            Produce an internal draft answer.
            Do not reveal hidden reasoning.
            Be grounded in the available evidence.
            If evidence is incomplete, state uncertainty clearly.

            Route:
            {plan.primary_route}

            User request:
            {pre.user_text or 'Analyze the provided input.'}

            Memory context:
            {bundle.memory_context or 'None'}

            Evidence:
            {evidence or '{}'}

            {memory_policy}
            """
        ).strip()

    def build_judge_prompt(
        self,
        *,
        pre: PreprocessedInput,
        plan: RoutePlan,
        bundle: ExecutionBundle,
        draft: str,
        heuristic_score: float,
        heuristic_note: str,
    ) -> str:
        """Build judge prompt."""

        evidence = json.dumps(
            bundle.evidence,
            ensure_ascii=False,
            indent=2,
        )[:4000]

        return textwrap.dedent(
            f"""
            You are the judge model for ArgusAI.
            Evaluate the draft answer.

            Rules:
            - Be strict but practical.
            - Prefer concise, technically correct answers.
            - If the answer is already good, accept it.
            - If the route is wrong, set route_ok=false.
            - If route is web_search, the answer must be grounded in web_search_context.
            - If route is web_scrape, the answer must be grounded in web_scrape_context.
            - If internet evidence is unavailable or denied, the answer must not invent current facts.
            - If memory was used but unrelated to the current request, penalize it.
            - If it is weak, provide short but actionable improvement instructions.
            - Return strict JSON only.

            User request:
            {pre.user_text}

            Route:
            {plan.primary_route}

            Heuristic pre-score:
            {heuristic_score}

            Heuristic note:
            {heuristic_note}

            Evidence:
            {evidence}

            Draft answer:
            {draft[:5000]}
            """
        ).strip()

    def build_final_prompt(
        self,
        *,
        pre: PreprocessedInput,
        plan: RoutePlan,
        bundle: ExecutionBundle,
        draft: str,
        verdict: JudgeVerdict,
    ) -> str:
        """Build final-answer prompt."""

        evidence = json.dumps(
            bundle.evidence,
            ensure_ascii=False,
            indent=2,
        )[:4000]

        memory_policy = textwrap.dedent(
            """
            Memory policy:
            Use memory only if it is directly relevant to the current user request.
            Ignore unrelated memory.
            Do not invent a continuation of a previous conversation.
            """
        ).strip()

        if verdict.accept:
            return textwrap.dedent(
                f"""
                Rewrite the approved draft as the final user-facing answer.
                Keep it concise, clear, and directly useful.
                Do not mention judging or hidden chain-of-thought.
                Stay grounded in the evidence if present.
                For current web information, do not invent facts beyond the evidence.

                User request:
                {pre.user_text}

                Evidence:
                {evidence}

                Approved draft:
                {draft[:6000]}

                {memory_policy}
                """
            ).strip()

        return textwrap.dedent(
            f"""
            Improve the draft answer using the judge feedback.
            Produce the final answer only.
            Keep it grounded in the evidence.
            Be concise and direct.
            For current web information, do not invent facts beyond the evidence.

            User request:
            {pre.user_text}

            Evidence:
            {evidence}

            Previous draft:
            {draft[:5000]}

            Judge critique:
            {verdict.critique}

            Improvement instructions:
            {verdict.improvement_instructions}

            {memory_policy}
            """
        ).strip()

    # ------------------------------------------------------------------
    # Quality / artifacts
    # ------------------------------------------------------------------

    def heuristic_quality_check(
        self,
        plan: RoutePlan,
        draft: str,
    ) -> tuple[float, str]:
        """Fast heuristic pre-check before judge model evaluation."""

        content = draft.strip()

        if not content:
            return 2.0, "Empty answer."

        if len(content) < 30:
            return 4.0, "Answer too short."

        lowered = content.lower()

        uncertainty_markers = [
            "i don't know",
            "cannot answer",
            "not enough information",
            "je ne sais pas",
            "information insuffisante",
        ]

        if any(marker in lowered for marker in uncertainty_markers):
            return 5.5, "Answer contains strong uncertainty."

        if plan.primary_route == "code" and "```" in content:
            return 8.0, "Code answer includes a code block."

        if len(content) > 120:
            return 7.5, "Substantive draft with acceptable length."

        return 6.5, "Potentially acceptable but needs judge confirmation."

    def maybe_create_artifact(
        self,
        *,
        pre: PreprocessedInput,
        plan: RoutePlan,
        final_answer: str,
    ) -> ArtifactResult:
        """Create a file artifact when explicitly requested."""

        filename = pre.metadata.get("requested_filename")

        if not filename or plan.primary_route != "code":
            return ArtifactResult(
                requested=bool(filename),
                filename=filename,
            )

        return self.file_manager.save_from_answer(
            filename=str(filename),
            answer=final_answer,
        )

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _model_for_route(self, plan: RoutePlan) -> str:
        """Select generation model for a route."""

        if plan.primary_route == "code":
            return self.config.coder_model

        return self.config.reasoning_model

    def _think_for_route(self, plan: RoutePlan) -> bool:
        """
        Return whether Ollama thinking mode should be enabled for this route.

        Code models such as qwen2.5-coder generally do not support think=True.
        """

        if plan.primary_route == "code":
            return False

        return True

    def _infer_tools_called(self, bundle: ExecutionBundle) -> List[str]:
        """Infer tool calls from execution evidence."""

        tools: List[str] = []

        if "document_excerpt" in bundle.evidence:
            tools.append("documents")

        if "memory_context" in bundle.evidence:
            tools.append("memory")

        if "ocr_text" in bundle.evidence:
            tools.append("ocr")

        if "vision_analysis" in bundle.evidence:
            tools.append("vision")

        if "web_search_context" in bundle.evidence:
            tools.append("web_search")

        if "web_scrape_context" in bundle.evidence:
            tools.append("web_scrape")

        if "web_context" in bundle.evidence:
            tools.append("web")

        return tools

    def stream_callback(self, chunk: str) -> None:
        """Default stream callback for CLI."""

        print(chunk, end="", flush=True)

    def get_history_preview(self) -> List[Dict[str, str]]:
        """Return short memory preview for CLI."""

        return [
            {
                "role": message.role,
                "content": message.content.replace("\n", " ")[:160],
            }
            for message in self.short_memory
        ]

    def clear_short_memory(self) -> None:
        """Clear short-term conversation memory."""

        self.short_memory.clear()

    def toggle_web(self, mode: str) -> str:
        """Enable or disable internet access from CLI."""

        normalized = mode.strip().lower()

        if normalized in {"on", "enable", "enabled"}:
            search_available = self.search_tool.is_available()
            scrape_available = self.web_crawler.is_available()

            if not search_available and not scrape_available:
                self.config.web_enabled = False
                self.config.crawl4ai_enabled = False
                return (
                    "Web tools are unavailable. Install ddgs for general search "
                    "or crawl4ai for URL scraping."
                )

            self.config.web_enabled = True
            self.config.crawl4ai_enabled = bool(scrape_available)

            return (
                "Web access armed. Each request still needs confirmation. "
                f"Search={self.search_tool.status()} | Scrape={self.web_crawler.status()}"
            )

        if normalized in {"off", "disable", "disabled"}:
            self.config.web_enabled = False
            self.config.crawl4ai_enabled = False
            return "Web access disabled."

        return "Use /web on or /web off."