# ArgusAI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![Status](https://img.shields.io/badge/Status-v0.1.0--beta-orange)
![License](https://img.shields.io/badge/License-PolyForm%20Noncommercial-red)

ArgusAI is a local-first multimodal agentic AI assistant for local models, local semantic memory, OCR, document analysis, artifact generation, and optional explicit URL crawling.

The project is built for developers, researchers, and ML engineers who want more control over their local AI stack.

## Status

Current version: `v0.1.0-beta`

This is an early beta. The current focus is:

- Local LLM inference with Ollama
- Optional long-term semantic memory with Qdrant
- Image and document OCR
- PDF and document text extraction
- Optional URL crawling with Crawl4AI
- CLI-first interaction
- Docker Compose deployment
- JSONL execution traces
- Basic artifact generation and validation

## Repository Structure

```text
ArgusAI/
|-- src/
|   `-- argusai/
|       |-- __init__.py
|       |-- main.py
|       |-- cli.py
|       |-- config.py
|       |-- agent.py
|       |-- router.py
|       |-- ollama_client.py
|       |-- memory.py
|       |-- ocr.py
|       |-- documents.py
|       |-- web.py
|       |-- artifacts.py
|       `-- utils.py
|-- docker/
|   `-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
|-- pyproject.toml
|-- README.md
|-- LICENSE.md
|-- .gitignore
`-- .dockerignore
```

## Architecture

ArgusAI follows this pipeline:

```text
User input
-> preprocessing
-> routing
-> tool execution
-> draft generation
-> judge evaluation
-> optional improvement or re-routing
-> final answer
-> artifact generation
-> memory write
-> trace logging
```

Main components:

| Module | Role |
| --- | --- |
| `agent.py` | Main orchestration pipeline |
| `router.py` | Heuristic and LLM-based routing |
| `ollama_client.py` | Local Ollama HTTP client |
| `memory.py` | Qdrant and LlamaIndex semantic memory |
| `ocr.py` | OCR backend management |
| `documents.py` | PDF, document, and image loading |
| `web.py` | Optional explicit URL crawling |
| `artifacts.py` | File generation and validation |
| `utils.py` | Shared utilities, logging, and traces |

## Default Model Roles

ArgusAI uses Ollama as the local model runtime.

```text
router_vision_model = qwen3-vl:8b
reasoning_model     = deepseek-r1:8b
judge_model         = qwen3.5:9b
coder_model         = qwen2.5-coder:7b
embedding_model     = qwen3-embedding:latest
```

You can override these model tags with CLI arguments or environment variables.

## Installation

### Docker Compose

Prerequisites:

- Docker and Docker Compose
- Ollama running on the host machine

Start Ollama:

```bash
ollama serve
```

Pull the default models:

```bash
ollama pull qwen3-vl:8b
ollama pull deepseek-r1:8b
ollama pull qwen3.5:9b
ollama pull qwen2.5-coder:7b
ollama pull qwen3-embedding:latest
```

Start ArgusAI and Qdrant:

```bash
docker compose up --build
```

Qdrant runs inside Docker Compose. Ollama is expected on the host at:

```text
http://host.docker.internal:11434
```

The compose file includes `host-gateway` support for Linux Docker hosts.

### Local Development

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install the core dependencies and editable package:

```bash
pip install -r requirements.txt
pip install -e .
```

Optional extras:

```bash
pip install -e ".[web]"
pip install -e ".[surya]"
pip install -e ".[dev]"
```

Run ArgusAI:

```bash
argusai
```

Alternative:

```bash
python -m argusai.main
```

## Diagnostics

Run a lightweight smoke check without launching the interactive agent:

```bash
argusai doctor
```

Equivalent:

```bash
argusai --status-only
```

The diagnostic reports dependency availability, runtime output directories, Ollama reachability, and Qdrant reachability. Missing Ollama or Qdrant services are warnings because they may be started separately.

## CLI Commands

```text
/image <path>   Analyze an image
/doc <path>     Analyze a document
/web on|off     Enable or disable web access
/history        Show recent conversation history
/status         Show runtime status
/clear          Clear the terminal screen
/help           Show help
exit            Quit ArgusAI
```

## Usage Examples

Analyze an image:

```text
/image examples/image.png
```

Analyze a document:

```text
/doc examples/document.pdf
```

Enable web crawling:

```text
/web on
```

Ask a normal reasoning question:

```text
Explain the role of Qdrant in ArgusAI.
```

Generate a file:

```text
Generate a file app.py that creates a simple FastAPI endpoint.
```

Generated files are saved under:

```text
outputs/
```

## Runtime Outputs

ArgusAI automatically creates runtime outputs:

```text
outputs/logs/argusai.log
outputs/traces/runs.jsonl
```

Execution traces include metadata such as route, tools called, memory status, judge score, latency, artifact status, and errors.

## Web Access Policy

Web access is disabled by default.

In `v0.1.0-beta`, ArgusAI only supports crawling explicit URLs:

```text
Summarize this page: https://example.com
```

General web search is not implemented yet.

## OCR Notes

Default OCR backend:

```text
pytesseract
```

Experimental backend:

```text
surya-ocr
```

Install Surya support with `pip install -e ".[surya]"`. Surya remains disabled by default because its Python API may change across versions.

OCR on scanned PDFs is limited by default to the first few pages for beta stability.

## Current Limitations

- General web search is not implemented yet.
- Web crawling requires explicit URLs.
- OCR quality depends on the local OCR backend.
- OCR on scanned PDFs is limited by default.
- Surya OCR is experimental.
- Model availability depends on local Ollama tags.
- Docker image is intended for local-first usage, not hardened production deployment.
- Benchmarks and evaluation suites are not yet included.
- The judge model improves quality, but external evals are not yet implemented.

## Roadmap

- Routing benchmark
- Code generation benchmark with pytest
- RAG retrieval evaluation
- Better artifact validators
- Optional local Qdrant launcher scripts
- More robust web search tool
- CI/CD with GitHub Actions
- Optional modular expansion into `tools/`, `llm`, `memory`, and `evaluation`
- More structured agent trajectory evaluation

## License

This project is source-available under the **PolyForm Noncommercial License 1.0.0**.

Commercial use is not permitted without explicit written permission from the author.

See [LICENSE.md](LICENSE.md) for the full license text.

## Author

Built by **Moulaye S. Dahi**.
