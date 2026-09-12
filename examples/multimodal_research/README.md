# Multimodal research

**Input**: an image you are authorized to inspect and a question about it.

**Command**:

```text
/image C:\work\diagram.png
```

Then ask:

```text
Explain the diagram. What information is missing before a decision can be made?
```

**Optional web step**:

```text
/web on
```

ArgusAI requests confirmation before searching or scraping. Keep web access disabled when the task is private or offline.

**Expected result**: image understanding combined with local context. Any web source is used only after explicit approval.

![Workflow](../../assets/multimodal-research.svg)
