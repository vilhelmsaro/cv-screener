# NOTES

## Eval results
_Paste the output of `cvs eval` (also saved to `evals/last_run.txt`) here, unedited._

```
TODO: run evals and paste results
```

## Not done / known issues
- TODO: fill in after the run.
- Photo generation depends on the chosen OpenRouter image model. A failed candidate is skipped and listed at the end;
  rerun with `cvs generate --only <id>`.
- Skill matching via filters is exact after normalization ("PyTorch" vs "Pytorch" ok, "Java 17" and
  "Docker (basic)" also match "Java" / "Docker", "ML" vs "machine learning" not); the agent is instructed to
  fall back to semantic search.
- Field rules assume English headings and common CV conventions (a "Skills" section, "N years of experience"
  or dated jobs, "City, Country" next to the email). All 12 generated CVs parse fully without the LLM; other
  CVs may leave a field empty rather than get a wrong value.
- Seniority is what the CV shows, not the seed's intent: c11 was generated with "Senior QA Automation
  Engineer" as her current title, so she is indexed as senior although the seed says mid.
- No keyword (BM25) ranking; search is metadata filters + semantic similarity over chunks. The profile-level
  embedding is stored but not queried yet.
- Chunking relies on common English section headings and date ranges ("Feb 2022 – Present"). A CV with
  unusual headings falls back to fewer, larger chunks rather than failing.

## Next steps
- BM25 + vector fusion, reranker for "best fit" questions.
- LLM-as-judge eval for answer quality on top of the deterministic checks.
- Skill synonym map / taxonomy for filters.
- Web UI or public deploy.
