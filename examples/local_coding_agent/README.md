# Local coding agent

**Input**: a precise development request.

**Prompt**:

```text
Generate a Python FastAPI endpoint that accepts a health check and saves it as app.py.
```

**What happens**: ArgusAI routes the request to its coding model, validates the generated content, stores the artifact under `outputs/` and records a local execution trace.

**Expected result**: a saved artifact and a short summary of the validation outcome. Review generated code before using it in production.

![Workflow](../../assets/local-coding.svg)
