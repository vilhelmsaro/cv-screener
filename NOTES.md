# NOTES

## Eval results
_Paste the output of `cvs eval` (also saved to `evals/last_run.txt`) here, unedited._

```
TODO: run evals and paste results
```

## Not done / known issues
- TODO: fill in after the run.
- Photo generation depends on the chosen OpenRouter image model; if it fails the run stops for that candidate.
- Skill matching via filters is exact after normalization ("PyTorch" vs "Pytorch" ok, "ML" vs "machine learning" not);
  the agent is instructed to fall back to semantic search.
- No hybrid keyword (BM25) ranking; semantic + metadata filters only.

## Next steps
- BM25 + vector fusion, reranker for "best fit" questions.
- LLM-as-judge eval for answer quality on top of the deterministic checks.
- Skill synonym map / taxonomy for filters.
- Web UI or public deploy.
