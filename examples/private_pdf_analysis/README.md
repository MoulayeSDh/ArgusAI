# Private PDF analysis

**Input**: a PDF you are allowed to analyze, for example `C:\work\contract.pdf`.

**Command**:

```text
/doc C:\work\contract.pdf
```

**What happens**: ArgusAI extracts native text, applies OCR if the pages are scanned, stores bounded local context in Qdrant, then answers questions about the document.

**Try this question**:

```text
List the obligations, deadlines and clauses that need review. Cite the relevant passages.
```

**Expected result**: a structured answer grounded in the extracted document content. The document, local memory and runtime trace remain on your machine.

![Workflow](../../assets/document-analysis.svg)
