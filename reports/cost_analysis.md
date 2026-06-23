# Cost And Time Analysis

## Build Time

| Step | Seconds |
|---|---:|
| Load 70 documents | 0.253 |
| Split into 1796 chunks | 0.312 |
| Build Flat RAG vector index | 14.793 |
| Build NetworkX knowledge graph | 12.305 |
| Total startup/build time | 27.664 |

## Graph Size

| Metric | Value |
|---|---:|
| Document nodes | 70 |
| Entity nodes | 46 |
| Edges | 1753 |
| Corpus characters | 3539541 |
| Corpus tokens estimated | 884903 |
| Chunk characters including overlap | 5966771 |
| Chunk tokens estimated including overlap | 1492142 |

## Token Usage Notes

- Graph construction in this implementation uses local alias-based entity extraction, so API token usage for building the graph is 0.
- Token counts are approximate, using characters / 4.
- LLM answer context is capped at about 1250 prompt tokens per Flat/Graph answer.
- For 20 benchmark questions, answer generation would use about 50000 prompt tokens before model output.
- If `--llm-judge` is enabled, judge prompts add about 31000 more prompt tokens.
- Wall-clock time depends strongly on the configured model endpoint. The local graph build is fast; LLM calls dominate runtime.

Configured model: `deepseek-3.2`
Configured base URL: `https://api.xah.io/v1`
