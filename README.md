<h1 align="center">ArgusAI</h1>

<p align="center">
  <strong>Local-first multimodal agentic AI assistant</strong><br>
  <em>Your models. Your data. Your machine.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/Docker-Ready-blue" alt="Docker">
  <img src="https://img.shields.io/badge/Status-v0.1.0--beta-orange" alt="Status">
  <a href="LICENSE.md">
    <img src="https://img.shields.io/badge/License-PolyForm%20Noncommercial-red" alt="License">
  </a>
</p>
**ArgusAI** is a local-first multimodal agentic AI assistant designed for developers, researchers, and ML engineers who want to run powerful AI workflows on their own machine.

It combines local LLMs, semantic memory, OCR, document analysis, web search, URL scraping, code generation, artifact creation, and self-evaluation inside a CLI-first workflow.

ArgusAI is built around a simple principle:

> Your models, your data, your machine.

## Why ArgusAI?

Most AI assistants are cloud-first. They depend on remote APIs, external infrastructure, and opaque model routing.

ArgusAI takes the opposite direction.

It is designed to run locally with open-source components, while keeping internet access optional and explicit. The goal is to give developers more control over inference, memory, tools, logs, and execution traces.

ArgusAI is not trying to be another chatbot.  
It is an experimental local agentic stack for building, testing, and extending AI assistants that stay closer to the user’s environment.

## Key Features

- Local model inference with Ollama
- Local semantic memory with Qdrant and LlamaIndex
- Multimodal routing for text, images, and documents
- OCR with Tesseract, with optional Surya support
- PDF and document text extraction
- General web search through `ddgs`
- Explicit URL scraping through Crawl4AI
- Code-oriented generation with artifact saving
- Judge model for answer validation and improvement
- CLI-first workflow
- Docker Compose support
- JSONL execution traces
- Developer test suite with pytest

## Local-First Design

ArgusAI is designed to work without depending on proprietary APIs.

By default, the main stack is local:

```text
Ollama     -> local LLM inference
Qdrant     -> local vector memory
Tesseract  -> local OCR
CLI        -> local interaction
outputs/   -> local logs, traces, and generated artifacts
```

Internet access is disabled by default.  
When enabled, ArgusAI still asks for confirmation before using web tools.

This makes ArgusAI suitable for:

- private experimentation
- local AI agent prototyping
- offline-first workflows
- controlled RAG experiments
- document and OCR analysis
- developer-oriented assistant workflows

## Default Model Roles

ArgusAI uses Ollama model tags by default:

```text
router_vision_model = qwen3-vl:8b
reasoning_model     = deepseek-r1:8b
judge_model         = qwen3.5:9b
coder_model         = qwen2.5-coder:7b
embedding_model     = qwen3-embedding:latest
```

You can override these tags through CLI arguments or environment variables.

## Installation

ArgusAI can be installed in two ways:

1. Docker Compose
2. Direct local installation

## Option 1: Docker Compose

Docker is the recommended path if you want a reproducible environment.

### Prerequisites

- Docker Desktop or Docker Engine
- Ollama installed on the host machine
- Required Ollama models pulled locally

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

Build the Docker image:

```bash
docker compose build
```

Run diagnostics:

```bash
docker compose run --rm argusai doctor
```

Start ArgusAI:

```bash
docker compose run --rm argusai
```

Qdrant runs inside Docker Compose.

Ollama is expected on the host at:

```text
http://host.docker.internal:11434
```

The Docker image includes Tesseract for stable OCR support.

## Option 2: Direct Local Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Run diagnostics:

```bash
argusai doctor
```

Start ArgusAI:

```bash
argusai
```

Alternative:

```bash
python -m argusai.main
```

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

## Example Usage

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

ArgusAI creates local runtime outputs:

```text
outputs/logs/argusai.log
outputs/traces/runs.jsonl
```

These files help developers inspect routing decisions, tool calls, memory status, judge scores, latency, artifact generation, and runtime errors.

## Developer Tests

Run the test suite:

```bash
pip install -e .
pip install pytest
pytest -q
```

The current tests cover:

- routing behavior
- web search and URL scraping tools
- Ollama JSON response handling
- OCR fallback logic
- Qdrant and LlamaIndex compatibility
- minimal pipeline execution

## Web Policy

Web access is disabled by default.

When enabled with:

```text
/web on
```

ArgusAI still asks for confirmation before using internet tools.

Two web paths are supported:

```text
web_search  -> general search through ddgs
web_scrape  -> explicit URL/page scraping through Crawl4AI
```

Crawl4AI is used for explicit URLs only.  
General search is handled separately through `ddgs`.

## OCR Notes

Docker includes Tesseract by default.

For local Windows installations, you may need to install Tesseract manually and provide the binary path:

```powershell
argusai --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Surya OCR is treated as experimental and optional.

## Security Model ⚡️

ArgusAI is local-first by design.

- Web access is disabled by default.
- URL scraping is disabled by default.
- Internet access requires explicit user confirmation.
- Qdrant is intended to run locally or inside a private Docker Compose network.
- The default Docker Compose setup is for local development, not public server exposure.
- Do not expose Qdrant ports publicly without authentication.

  
## Project Status

Current version:

```text
v0.1.0-beta
```

ArgusAI is an early beta.  
The current focus is stability, local execution, developer usability, and modular extensibility.

## Roadmap

- Stronger routing evaluation
- Code generation benchmark
- Better RAG retrieval evaluation
- More artifact validators
- CI/CD with GitHub Actions
- Improved local memory management
- Optional MCP-style local actions
- More structured agent trajectory evaluation

## License

This project is source-available under the **PolyForm Noncommercial License 1.0.0**.

Commercial use is not permitted without explicit written permission from the author.

See [LICENSE.md](LICENSE.md) for the full license text.

## Author

Built by **Moulaye S. Dahi**.
