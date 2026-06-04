"""
Command-line interface for ArgusAI.

This module owns:
- argument parsing
- interactive CLI loop
- command handling
- startup configuration

The heavy agent logic stays in agent.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import textwrap
from pathlib import Path
from typing import Optional

from . import __app_name__, __version__
from .agent import ArgusAI
from .config import Config
from .utils import Colors, format_cli_text


try:
    from art import text2art
except ImportError:
    text2art = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Banner / display
# ---------------------------------------------------------------------------


def render_cli_banner() -> str:
    """Render ArgusAI CLI banner."""

    if text2art is not None:
        try:
            title = text2art("ArgusAI", font="small")
        except Exception:
            title = "ARGUSAI"
    else:
        title = r"""
    ___                     ___    ___
   /   |  _________ ___  __/   |  /  _/
  / /| | / ___/ __ `/ / / / /| |  / /
 / ___ |/ /  / /_/ / /_/ / ___ |_/ /
/_/  |_/_/   \__, /\__,_/_/  |_/___/
            /____/
""".strip("\n")

    subtitle = "Welcome to ArgusAI by Moulaye S. Dahi"
    details = (
        "Local-first multimodal CLI | OCR | Vision | Memory | "
        "Web Search | Optional URL Scraping"
    )

    return f"{title}\n{subtitle}\n{details}"


def print_banner() -> None:
    """Print CLI banner and command summary."""

    print()
    print(f"{Colors.CYAN}{render_cli_banner()}{Colors.RESET}")
    print(
        f"{Colors.YELLOW}Commands:{Colors.RESET} "
        "/image <path> | /doc <path> | /web on|off | "
        "/history | /status | /clear | /help | exit"
    )
    print(f"{Colors.DIM}Type /help for usage details.{Colors.RESET}")


def show_help(config: Config) -> None:
    """Print CLI help."""

    help_text = textwrap.dedent(
        f"""
        {Colors.YELLOW}Interactive commands{Colors.RESET}
          /image <path>   Analyze an image.
          /doc <path>     Analyze a document.
          /web on|off     Arm or disable internet access.
                           General search uses ddgs.
                           URL/page scraping uses Crawl4AI when available.
          /history        Show recent conversation history.
          /status         Show current runtime status.
          /clear          Clear the screen and redraw the banner.
          /help           Show this help message.
          exit            Quit ArgusAI.

        {Colors.YELLOW}Startup options{Colors.RESET}
          --enable-web                 Arm web access after per-request confirmation.
          --disable-memory             Start without Qdrant semantic memory.
          --enable-surya-experimental  Enable experimental Surya OCR support.
          --tesseract-cmd <path>        Explicit Tesseract binary path.
          --collection <name>          Qdrant collection name.
          --ollama-url <url>           Ollama base URL.
          --qdrant-url <url>           Qdrant base URL.
          --embedding-model <tag>      Embedding model tag.
          --router-model <tag>         Vision/router model tag.
          --reasoning-model <tag>      Main reasoning model tag.
          --judge-model <tag>          Judge model tag.
          --coder-model <tag>          Coding model tag.
          --artifacts-dir <path>       Directory for generated files, logs, and traces.
          --max-judge-rounds <n>       Maximum draft improvement rounds.
          --no-traces                  Disable JSONL execution traces.
          --disable-route-recheck      Disable one-shot re-routing when judge route_ok=false.
          --status-only                Run startup diagnostics without launching the agent.
          --log-level <level>          Log verbosity for file logs.

        {Colors.YELLOW}Examples{Colors.RESET}
          argusai
          argusai doctor
          argusai --status-only
          argusai --enable-web
          argusai --disable-memory
          argusai --collection argusai_v2_memory_qwen3
          python -m argusai.main --disable-memory

        {Colors.YELLOW}Web behavior{Colors.RESET}
          web_search:
            General internet search through ddgs. No API key. No Docker required.

          web_scrape:
            URL/page scraping through Crawl4AI. Optional dependency.

          ArgusAI always asks confirmation before using internet.

        Logs:
          {config.log_file_path.as_posix()}

        Traces:
          {config.traces_file_path.as_posix() if config.enable_traces else 'off'}
        """
    ).strip()

    print(help_text)


# ---------------------------------------------------------------------------
# CLI loop
# ---------------------------------------------------------------------------


def run_cli_loop(agent: ArgusAI) -> None:
    """Run the interactive ArgusAI CLI loop."""

    print_banner()

    while True:
        try:
            raw = input(f"\n{Colors.BOLD}{Colors.GREEN}You>{Colors.RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Colors.YELLOW}Bye.{Colors.RESET}")
            break

        if not raw:
            continue

        lowered = raw.lower()

        if lowered == "exit":
            print(f"{Colors.YELLOW}Bye.{Colors.RESET}")
            break

        if lowered == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            print_banner()
            continue

        if lowered == "/help":
            show_help(agent.config)
            continue

        if lowered == "/history":
            show_history(agent)
            continue

        if lowered == "/status":
            show_status(agent)
            continue

        if lowered.startswith("/web "):
            mode = raw.split(maxsplit=1)[1].strip()
            message = agent.toggle_web(mode)
            print(message)
            continue

        image_path: Optional[str] = None
        doc_path: Optional[str] = None
        user_text = raw

        if raw.startswith("/image "):
            image_path = raw.replace("/image ", "", 1).strip().strip('"').strip("'")

            if not Path(image_path).exists():
                print(f"{Colors.RED}Image not found.{Colors.RESET}")
                continue

            user_text = input(f"{Colors.BLUE}Question about the image>{Colors.RESET} ").strip()

        elif raw.startswith("/doc "):
            doc_path = raw.replace("/doc ", "", 1).strip().strip('"').strip("'")

            if not Path(doc_path).exists():
                print(f"{Colors.RED}Document not found.{Colors.RESET}")
                continue

            user_text = input(f"{Colors.BLUE}Question about the document>{Colors.RESET} ").strip()

        try:
            print(f"\n{Colors.BOLD}{Colors.CYAN}ArgusAI>{Colors.RESET} ", end="", flush=True)

            result = agent.run_once(
                user_text=user_text,
                image_path=image_path,
                doc_path=doc_path,
                stream_callback=None,
            )

            print(format_cli_text(result["answer"]))

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Interrupted.{Colors.RESET}")

        except Exception:
            logging.exception("Execution failure")
            print(
                f"{Colors.RED}Something went wrong. "
                f"Check the log file for details.{Colors.RESET}"
            )


def show_history(agent: ArgusAI) -> None:
    """Display short-term memory preview."""

    history = agent.get_history_preview()

    if not history:
        print(f"{Colors.DIM}No history yet.{Colors.RESET}")
        return

    for item in history:
        role = item["role"]
        content = item["content"]

        role_color = Colors.GREEN if role == "user" else Colors.CYAN

        print(f"{role_color}{role.capitalize()}:{Colors.RESET} {format_cli_text(content)}")


def show_status(agent: ArgusAI) -> None:
    """Display runtime status."""

    status = agent.status_dict()

    print(
        "Status\n"
        f"- Version: {status['version']}\n"
        f"- Memory: {status['memory']}\n"
        f"- Web enabled: {status['web_enabled']}\n"
        f"- Web search: {status['web_search']}\n"
        f"- Web scrape: {status['web_scrape']}\n"
        f"- OCR: {status['ocr']}\n"
        f"- PDF OCR: {status['pdf_ocr']}\n"
        f"- Ollama: {status['ollama_url']}\n"
        f"- Qdrant: {status['qdrant_url']}\n"
        f"- Logs: {status['logs']}\n"
        f"- Traces: {status['traces']}"
    )


def dependency_available(module_name: str) -> bool:
    """Return True when an importable module is available."""

    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def check_http_endpoint(url: str, timeout: int = 3) -> tuple[bool, str]:
    """Check a local HTTP endpoint for doctor output."""

    try:
        import requests

        response = requests.get(url, timeout=timeout)
        if response.ok:
            return True, f"reachable ({response.status_code})"
        return False, f"HTTP {response.status_code}"
    except Exception as exc:
        return False, str(exc)


def run_status_check(config: Config) -> int:
    """
    Run lightweight startup diagnostics without constructing ArgusAI.

    This intentionally reports local service availability as warnings because
    Ollama and Qdrant may be started separately by the user.
    """

    checks: list[tuple[str, bool, str]] = []

    try:
        config.ensure_runtime_dirs()
        checks.append(("runtime directories", True, config.artifacts_path.as_posix()))
    except Exception as exc:
        checks.append(("runtime directories", False, str(exc)))

    required_modules = {
        "requests": "requests",
        "art": "art",
        "Pillow": "PIL",
        "pytesseract": "pytesseract",
        "pypdf": "pypdf",
        "pypdfium2": "pypdfium2",
        "PyMuPDF": "fitz",
        "PyYAML": "yaml",
        "qdrant-client": "qdrant_client",
        "llama-index-core": "llama_index.core",
        "llama-index-embeddings-ollama": "llama_index.embeddings.ollama",
        "llama-index-vector-stores-qdrant": "llama_index.vector_stores.qdrant",
        "ddgs": "ddgs",
    }

    for label, module_name in required_modules.items():
        available = dependency_available(module_name)
        checks.append((f"dependency {label}", available, "installed" if available else "missing"))

    if dependency_available("pytesseract"):
        try:
            import pytesseract

            if config.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd

            version = pytesseract.get_tesseract_version()
            checks.append(("tesseract runtime", True, str(version)))
        except Exception as exc:
            checks.append(
                (
                    "tesseract runtime",
                    True,
                    f"warning: unavailable ({exc})",
                )
            )

    optional_modules = {
        "crawl4ai": "crawl4ai",
        "surya-ocr": "surya",
    }

    for label, module_name in optional_modules.items():
        available = dependency_available(module_name)
        checks.append(
            (
                f"optional {label}",
                True,
                "installed" if available else "not installed",
            )
        )

    ollama_ok, ollama_detail = check_http_endpoint(
        f"{config.ollama_base_url.rstrip('/')}/api/tags"
    )
    checks.append(("ollama", True, ollama_detail if ollama_ok else f"warning: {ollama_detail}"))

    qdrant_ok, qdrant_detail = check_http_endpoint(
        f"{config.qdrant_url.rstrip('/')}/collections"
    )
    checks.append(("qdrant", True, qdrant_detail if qdrant_ok else f"warning: {qdrant_detail}"))

    print(f"{config.app_name} doctor")
    print(f"- version: {config.version}")
    print(f"- ollama_url: {config.ollama_base_url}")
    print(f"- qdrant_url: {config.qdrant_url}")
    print(f"- artifacts_dir: {config.artifacts_path.as_posix()}")

    failed_required = False

    for name, ok, detail in checks:
        marker = "ok" if ok else "fail"
        print(f"- {name}: {marker} - {detail}")

        if not ok:
            failed_required = True

    return 1 if failed_required else 0


# ---------------------------------------------------------------------------
# Argument parsing / config
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(
        prog="argusai",
        description=f"{__app_name__} {__version__} - local-first multimodal agent",
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=["doctor"],
        help="Run startup diagnostics without launching the interactive agent.",
    )

    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--collection", default=None)

    parser.add_argument("--router-model", default=None)
    parser.add_argument("--reasoning-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--coder-model", default=None)
    parser.add_argument("--embedding-model", default=None)

    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--disable-memory", action="store_true")
    parser.add_argument("--enable-surya-experimental", action="store_true")
    parser.add_argument("--disable-surya-auto-fallback", action="store_true")
    parser.add_argument("--tesseract-cmd", default=None)
    parser.add_argument("--enable-web", action="store_true")
    parser.add_argument("--max-judge-rounds", type=int, default=None)
    parser.add_argument("--request-timeout", type=int, default=None)
    parser.add_argument("--keep-alive", default=None)
    parser.add_argument("--log-level", default=None)

    parser.add_argument("--no-traces", action="store_true")
    parser.add_argument("--disable-route-recheck", action="store_true")
    parser.add_argument("--no-memory-prompt", action="store_true")
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Run startup diagnostics without launching the interactive agent.",
    )

    return parser


def build_config_from_args(args: argparse.Namespace) -> Config:
    """
    Build Config from environment defaults plus CLI overrides.

    Config.from_env() keeps Docker Compose usage clean.
    CLI arguments override environment/default values.
    """

    config = Config.from_env()

    if args.ollama_url is not None:
        config.ollama_base_url = args.ollama_url

    if args.qdrant_url is not None:
        config.qdrant_url = args.qdrant_url

    if args.collection is not None:
        config.qdrant_collection = args.collection

    if args.router_model is not None:
        config.router_vision_model = args.router_model

    if args.reasoning_model is not None:
        config.reasoning_model = args.reasoning_model

    if args.judge_model is not None:
        config.judge_model = args.judge_model

    if args.coder_model is not None:
        config.coder_model = args.coder_model

    if args.embedding_model is not None:
        config.embedding_model = args.embedding_model

    if args.artifacts_dir is not None:
        config.artifacts_dir = args.artifacts_dir

    if args.max_judge_rounds is not None:
        config.max_judge_rounds = max(1, int(args.max_judge_rounds))

    if args.request_timeout is not None:
        config.request_timeout = max(15, int(args.request_timeout))

    if args.keep_alive is not None:
        config.keep_alive = args.keep_alive

    if args.log_level is not None:
        config.log_level = args.log_level

    if args.disable_memory:
        config.use_memory = False

    if args.no_memory_prompt:
        config.ask_before_disable_memory = False

    if args.enable_surya_experimental:
        config.use_surya = True

    if args.disable_surya_auto_fallback:
        config.auto_surya_fallback = False

    if args.tesseract_cmd is not None:
        config.tesseract_cmd = args.tesseract_cmd

    if args.enable_web:
        config.web_enabled = True
        config.crawl4ai_enabled = dependency_available("crawl4ai")

    if args.no_traces:
        config.enable_traces = False

    if args.disable_route_recheck:
        config.enable_route_recheck = False

    return config


def run_cli(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint used by main.py and pyproject.toml."""

    parser = build_parser()
    args = parser.parse_args(argv)

    config = build_config_from_args(args)

    if args.status_only or args.command == "doctor":
        return run_status_check(config)

    try:
        agent = ArgusAI(config)
        run_cli_loop(agent)
        return 0

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted.{Colors.RESET}")
        return 0

    except Exception:
        print(f"{Colors.RED}Startup failed. Check the log file for details.{Colors.RESET}")
        logging.exception("Fatal startup error")
        return 1
