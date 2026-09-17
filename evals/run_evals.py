"""Run eval cases against the live agent and print pass/fail per check plus a total.

Usage: python -m evals.run_evals   (or `cvs eval`)
Requires a populated index and OPENROUTER_API_KEY. Output is also saved to evals/last_run.txt.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

from cv_screener.store import fold

HERE = Path(__file__).parent
NO_MATCH_HINTS = ["no candidate", "no one", "nobody", "none of the candidates", "no matching",
                  "not find any", "couldn't find", "could not find", "no candidates", "doesn't appear", "not in the dataset"]


def mentioned(text: str, names: list[str]) -> set[str]:
    t = fold(text)

    def has_word(word: str) -> bool:
        return re.search(rf"\b{re.escape(fold(word))}\b", t) is not None

    # A name counts as mentioned if its first and last token both appear as whole words.
    return {n for n in names if has_word(n.split()[0]) and has_word(n.split()[-1])}


def check_case(case: dict, answer: str, messages, all_names: list[str]) -> list[tuple[str, bool, str]]:
    calls = [p.tool_name for m in messages for p in m.parts if isinstance(p, ToolCallPart)]
    returned = " ".join(str(p.content) for m in messages for p in m.parts if isinstance(p, ToolReturnPart))
    in_answer = mentioned(answer, all_names)
    retrieved = mentioned(returned, all_names)
    checks = [("used a tool", bool(calls), f"calls={calls}")]
    if tool := case.get("expect_tool"):
        checks.append((f"called {tool}", tool in calls, f"calls={calls}"))
    checks.append(("grounded (names came from tools)", in_answer <= retrieved,
                   f"ungrounded={sorted(in_answer - retrieved)}"))
    for n in case.get("expect_all", []):
        checks.append((f"names {n}", n in in_answer, ""))
    for n in case.get("forbid", []):
        checks.append((f"does not name {n}", n not in in_answer, ""))
    if any_of := case.get("expect_contains_any"):
        checks.append(("mentions key facts", any(s in answer for s in any_of), f"one of {any_of}"))
    if case.get("expect_no_match"):
        checks.append(("names nobody", not in_answer, f"named={sorted(in_answer)}"))
        checks.append(("says no match", any(h in answer.lower() for h in NO_MATCH_HINTS), ""))
    return checks


def main() -> int:
    sys.path.insert(0, str(HERE.parent))
    from cv_screener.agent import Deps, build_agent
    from cv_screener.llm import OpenRouterEmbedder
    from cv_screener.store import CandidateStore

    cases = yaml.safe_load((HERE / "cases.yaml").read_text(encoding="utf-8"))
    store = CandidateStore(OpenRouterEmbedder())
    all_names = [m["name"] for m in store.all_profiles()]
    if not all_names:
        print("Index is empty. Run `cvs generate` and `cvs index` first.")
        return 2
    agent, deps = build_agent(), Deps(store=store)

    lines, passed = [], 0
    for case in cases:
        try:
            result = agent.run_sync(case["question"], deps=deps)
            checks = check_case(case, result.output, result.all_messages(), all_names)
            answer = result.output
        except Exception as e:  # a crash is a failed case, not a crashed suite
            checks, answer = [("agent ran", False, repr(e))], ""
        ok = all(c[1] for c in checks)
        passed += ok
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {case['id']}: {case['question']}")
        lines += [f"    {'ok ' if c[1] else 'XX '} {c[0]} {c[2] if not c[1] else ''}".rstrip() for c in checks]
        if not ok:
            lines.append("    answer: " + answer.replace("\n", " ")[:400])
    lines.append(f"\nTOTAL: {passed}/{len(cases)} cases passed")
    report = "\n".join(lines)
    print(report)
    (HERE / "last_run.txt").write_text(report + "\n", encoding="utf-8")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
