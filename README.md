# ArgusAI

[![GitHub Stars](https://img.shields.io/github/stars/MoulayeSDh/ArgusAI?style=social)](https://github.com/MoulayeSDh/ArgusAI/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/MoulayeSDh/ArgusAI?style=social)](https://github.com/MoulayeSDh/ArgusAI/forks)
[![GitHub Issues](https://img.shields.io/github/issues/MoulayeSDh/ArgusAI)](https://github.com/MoulayeSDh/ArgusAI/issues)
[![Last Commit](https://img.shields.io/github/last-commit/MoulayeSDh/ArgusAI)](https://github.com/MoulayeSDh/ArgusAI/commits/main)
[![Release](https://img.shields.io/github/v/release/MoulayeSDh/ArgusAI?include_prereleases)](https://github.com/MoulayeSDh/ArgusAI/releases)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![Status](https://img.shields.io/badge/Status-v0.1.0--beta-orange)
[![License](https://img.shields.io/badge/License-PolyForm%20Noncommercial-red)](LICENSE.md)

ArgusAI is a local-first multimodal agentic AI assistant built around local models, local semantic memory, OCR, document analysis, artifact generation, general web search, and optional explicit URL scraping.

It is designed for developers, researchers, and ML engineers who want a controllable local AI stack instead of a cloud-first assistant.

## Status

Current version: `v0.1.0-beta`

This beta already includes:

- Local LLM inference through Ollama
- Local semantic memory through Qdrant and LlamaIndex
- Short-term conversation memory
- Image OCR and document OCR
- PDF and text document extraction
- General web search through `ddgs`
- Explicit URL scraping through Crawl4AI
- CLI-first interaction
- Docker Compose deployment
- JSONL execution traces
- Artifact generation and validation
- Basic pytest coverage for routing, OCR fallback, Ollama JSON, web tools, and Qdrant compatibility

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
|       |-- search.py
|       |-- web.py
|       |-- artifacts.py
|       `-- utils.py
|-- tests/
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
-> optional re-routing or improvement
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
| `search.py` | General web search with `ddgs` |
| `web.py` | Optional explicit URL scraping with Crawl4AI |
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

## Docker Compose Usage

Prerequisites:

- Docker and Docker Compose
- Ollama installed and running on the host machine
- Required Ollama models pulled locally

Start Ollama on the host:

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

Build and start ArgusAI with Qdrant:

```bash
docker compose up --build
```

For an interactive CLI session, this is often cleaner:

```bash
docker compose run --rm argusai
```

Qdrant runs inside Docker Compose. Ollama is expected on the host at:

```text
http://host.docker.internal:11434
```

The compose file includes `host-gateway` support for Linux Docker hosts.

## Local Development

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

Install dependencies and the editable package:

```bash
pip install -r requirements.txt
pip install -e .
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

Run a lightweight startup check:

```bash
argusai doctor
```

Equivalent:

```bash
argusai --status-only
```

The diagnostic reports dependency availability, output directories, Ollama reachability, Qdrant reachability, OCR runtime status, and optional tool status.

## Tests

Run the developer test suite:

```bash
pip install -e .
pip install pytest
pytest -q
```

The current tests cover:

- Router contracts
- Web search and URL scraping tool behavior
- Ollama JSON response contract
- OCR fallback behavior
- Qdrant/LlamaIndex compatibility detection
- Minimal pipeline handling for web search routes

## CLI Commands

```text
/image <path>   Analyze an image
/doc <path>     Analyze a document
/web on|off     Arm or disable web access
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

Enable web tools:

```text
/web on
```

Ask a current-information question:

```text
What is the latest stable Python version?
```

Scrape a specific URL:

```text
Summarize this page: https://example.com
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

Execution traces include route, tools called, memory status, judge score, latency, artifact status, and errors.

## Web Access Policy

Web access is disabled by default.

When enabled with `/web on`, ArgusAI still asks for confirmation before internet access is used.

Two web paths exist:

```text
web_search  -> general search through ddgs
web_scrape  -> explicit URL/page scraping through Crawl4AI
```

ArgusAI should not use Crawl4AI for general search. Crawl4AI is reserved for explicit URLs.

## OCR Notes

Default OCR backend:

```text
pytesseract + Tesseract runtime
```

Docker includes Tesseract runtime. On Windows, you may need to install Tesseract manually and set the binary path:

```powershell
argusai --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Experimental backend:

```text
surya-ocr
```

Surya remains disabled by default because its Python API changes frequently. ArgusAI uses it defensively as a best-effort fallback when enabled or when Tesseract is unavailable and auto-fallback is active.

## Troubleshooting

If startup fails with missing Ollama models, pull the model tags shown in the error message:

```bash
ollama pull <model-tag>
```

If memory is off, verify Qdrant:

```bash
curl http://localhost:6333/collections
```

If Docker cannot reach Ollama, verify that Ollama is running on the host and reachable from the container through:

```text
http://host.docker.internal:11434
```

If OCR is unavailable locally on Windows, install the Tesseract runtime and pass `--tesseract-cmd`.

## Current Limitations

- Model availability depends on local Ollama tags.
- OCR quality depends on the OCR backend and installed language data.
- OCR on scanned PDFs is limited by default for beta stability.
- Surya OCR is experimental.
- Docker image is local-first and developer-oriented, not hardened production infrastructure.
- The judge model improves answer quality, but external benchmark suites are not yet included.

## Roadmap

- Routing benchmark
- Code generation benchmark with pytest
- RAG retrieval evaluation
- Better artifact validators
- Optional local Qdrant launcher scripts
- CI/CD with GitHub Actions
- More structured agent trajectory evaluation
- Optional modular expansion into `tools/`, `llm`, `memory`, and `evaluation`

## License

This project is source-available under the **PolyForm Noncommercial License 1.0.0**.

Commercial use is not permitted without explicit written permission from the author.

See [LICENSE.md](LICENSE.md) for the full license text.

## Docker Images

The default Docker image is the recommended user-friendly install path. It uses `requirements.txt`, keeps Crawl4AI scraping support, installs Tesseract in Linux, and does not install Surya OCR. This avoids the heavy Torch/CUDA dependency chain during normal Docker builds.

```bash
docker compose build argusai
```

The full Docker image is available for users who explicitly want Surya OCR. It uses `requirements_full.txt`, which includes the standard dependencies plus `surya-ocr`. This build can take much longer because Surya pulls Torch and related ML packages.

```bash
docker build -f docker/Dockerfile_full -t argusai-full .
```

Both dependency files pin `qdrant-client==1.13.3` to stay compatible with the default Compose server image `qdrant/qdrant:v1.13.4`.

## Author

Built by **Moulaye S. Dahi**.
