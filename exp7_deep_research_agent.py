"""
Experiment 7: Deep Research Agent Workflow
===========================================
An agentic content-generation workflow built around two extra steps beyond
plain "search then write": PLANNING and REFLECTION.

Pipeline:
  1. PLAN       -> break the topic into a small set of research sub-questions
  2. RESEARCH   -> gather information for each sub-question (web search if
                   available, else the user-supplied local corpus)
  3. DRAFT      -> synthesize a first draft from all gathered notes
  4. REFLECT    -> the agent critiques its own draft (coverage, accuracy,
                   structure, missing angles) and produces concrete feedback
  5. REVISE     -> the draft is rewritten to address the feedback
  Steps 4-5 repeat for `--reflection_rounds` iterations (default 2).
  6. FINALIZE   -> return the last draft plus the full trace (plan, notes,
                   every reflection + revision) for transparency/grading.

LLM backend: OpenAI (OPENAI_API_KEY) or Anthropic (ANTHROPIC_API_KEY) if set.
If neither is set, a lightweight rule-based fallback is used for every step
so the *workflow/control-flow* can still be run, inspected and graded without
any API key. Set a key for real-quality output.

Web search: uses the `duckduckgo_search` package if installed (no API key
required). Falls back to a local corpus folder (--corpus) or to fabricated
placeholder notes if neither is available.
"""

import os
import json
import argparse
import textwrap
from dataclasses import dataclass, field
from typing import List, Dict


# ----------------------------------------------------------------------
# LLM client abstraction
# ----------------------------------------------------------------------

class LLMClient:
    def __init__(self):
        self.backend = "fallback"
        if os.environ.get("OPENAI_API_KEY"):
            self.backend = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            self.backend = "anthropic"
        print(f"[LLMClient] using backend: {self.backend}")

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        if self.backend == "openai":
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.4,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()

        if self.backend == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text.strip()

        # ---- rule-based fallback (no API key needed) ----
        return self._fallback(system, user)

    @staticmethod
    def _fallback(system: str, user: str) -> str:
        """Deterministic stand-in so the pipeline is runnable with zero setup.
        Not meant to produce high-quality prose -- it exists so the control
        flow (plan -> research -> draft -> reflect -> revise) can be tested."""
        if "sub-questions" in system.lower() or "plan" in system.lower():
            topic = user.split("Topic:")[-1].strip()
            return json.dumps([
                f"What is {topic}?",
                f"Why does {topic} matter / what problem does it solve?",
                f"What are the main approaches or components of {topic}?",
                f"What are current limitations or open challenges in {topic}?",
                f"What are notable real-world examples or applications of {topic}?",
            ])
        if "critique" in system.lower() or "reflect" in system.lower():
            return json.dumps({
                "coverage_gaps": ["Add concrete examples.", "Add a discussion of limitations."],
                "structure_issues": ["Add clearer section headings."],
                "accuracy_flags": [],
                "overall_verdict": "Draft is a reasonable first pass but needs more depth and examples."
            })
        # draft / revise fallback: just stitch the notes together
        return "[fallback draft]\n" + user[-1500:]


# ----------------------------------------------------------------------
# Research / search tool
# ----------------------------------------------------------------------

class SearchTool:
    def __init__(self, corpus_folder: str = None):
        self.corpus_folder = corpus_folder
        self.mode = "duckduckgo"
        try:
            import duckduckgo_search  # noqa: F401
        except Exception:
            self.mode = "local" if corpus_folder else "stub"

    def search(self, query: str, max_results: int = 4) -> List[Dict]:
        if self.mode == "duckduckgo":
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            return [{"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")}
                    for r in results]

        if self.mode == "local":
            import glob
            hits = []
            for path in glob.glob(os.path.join(self.corpus_folder, "**", "*.*"), recursive=True):
                if path.lower().endswith((".txt", ".md")):
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    if any(w.lower() in text.lower() for w in query.split()):
                        hits.append({"title": os.path.basename(path), "snippet": text[:500], "url": path})
            return hits[:max_results]

        # stub: no search backend and no corpus available
        return [{"title": "stub", "snippet": f"(No search backend available. Placeholder note for: {query})", "url": ""}]


# ----------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------

@dataclass
class AgentTrace:
    topic: str
    plan: List[str] = field(default_factory=list)
    notes: Dict[str, List[Dict]] = field(default_factory=dict)
    drafts: List[str] = field(default_factory=list)
    reflections: List[Dict] = field(default_factory=list)


class DeepResearchAgent:
    PLAN_SYSTEM = (
        "You are a research planning assistant. Given a topic, break it down "
        "into 4-6 focused sub-questions that together give thorough coverage "
        "of the topic for a written report. Respond ONLY as a JSON list of "
        "strings, nothing else."
    )
    DRAFT_SYSTEM = (
        "You are a research writer. Using ONLY the provided research notes, "
        "write a well-structured, informative report on the topic. Use clear "
        "section headings that map to the sub-questions. Be concrete and "
        "avoid filler sentences."
    )
    REFLECT_SYSTEM = (
        "You are a critical editor performing self-reflection on a draft. "
        "Identify: (1) coverage_gaps - topics/sub-questions inadequately "
        "covered, (2) structure_issues - organization/clarity problems, "
        "(3) accuracy_flags - claims that look unsupported by the notes, "
        "(4) overall_verdict - one sentence. Respond ONLY as JSON with those "
        "four keys (three of them lists of short strings, overall_verdict a string)."
    )
    REVISE_SYSTEM = (
        "You are a research writer revising a draft based on editorial "
        "feedback. Produce a complete, improved version of the report that "
        "addresses every point in the feedback. Keep the good parts, expand "
        "or fix the weak parts. Output the full revised report only."
    )

    def __init__(self, llm: LLMClient, search: SearchTool, reflection_rounds: int = 2):
        self.llm = llm
        self.search = search
        self.reflection_rounds = reflection_rounds

    # ---- 1. Planning ----
    def plan(self, topic: str) -> List[str]:
        raw = self.llm.complete(self.PLAN_SYSTEM, f"Topic: {topic}")
        try:
            sub_qs = json.loads(raw)
            assert isinstance(sub_qs, list)
        except Exception:
            # be lenient if the LLM didn't return clean JSON
            sub_qs = [line.strip("-* ").strip() for line in raw.splitlines() if line.strip()]
        return sub_qs

    # ---- 2. Research ----
    def research(self, sub_questions: List[str]) -> Dict[str, List[Dict]]:
        notes = {}
        for q in sub_questions:
            notes[q] = self.search.search(q)
        return notes

    # ---- 3. Draft ----
    def draft(self, topic: str, notes: Dict[str, List[Dict]]) -> str:
        notes_block = ""
        for q, results in notes.items():
            notes_block += f"\nSub-question: {q}\n"
            for r in results:
                notes_block += f"  - ({r.get('title','')}) {r.get('snippet','')}\n"
        user = f"Topic: {topic}\n\nResearch notes:\n{notes_block}\n\nWrite the report now."
        return self.llm.complete(self.DRAFT_SYSTEM, user, max_tokens=1200)

    # ---- 4. Reflect ----
    def reflect(self, topic: str, draft_text: str) -> Dict:
        user = f"Topic: {topic}\n\nDraft:\n{draft_text}\n\nCritique this draft."
        raw = self.llm.complete(self.REFLECT_SYSTEM, user, max_tokens=500)
        try:
            return json.loads(raw)
        except Exception:
            return {"coverage_gaps": [], "structure_issues": [], "accuracy_flags": [],
                    "overall_verdict": raw[:300]}

    # ---- 5. Revise ----
    def revise(self, topic: str, draft_text: str, feedback: Dict) -> str:
        user = (
            f"Topic: {topic}\n\nCurrent draft:\n{draft_text}\n\n"
            f"Editorial feedback (JSON):\n{json.dumps(feedback, indent=2)}\n\n"
            "Produce the revised report."
        )
        return self.llm.complete(self.REVISE_SYSTEM, user, max_tokens=1400)

    # ---- Full workflow ----
    def run(self, topic: str) -> AgentTrace:
        trace = AgentTrace(topic=topic)

        print("[1/5] Planning sub-questions...")
        trace.plan = self.plan(topic)
        for q in trace.plan:
            print(f"    - {q}")

        print("[2/5] Researching each sub-question...")
        trace.notes = self.research(trace.plan)

        print("[3/5] Writing first draft...")
        current_draft = self.draft(topic, trace.notes)
        trace.drafts.append(current_draft)

        for i in range(self.reflection_rounds):
            print(f"[4/5] Reflection round {i+1}/{self.reflection_rounds}...")
            feedback = self.reflect(topic, current_draft)
            trace.reflections.append(feedback)

            print(f"[5/5] Revising based on feedback (round {i+1})...")
            current_draft = self.revise(topic, current_draft, feedback)
            trace.drafts.append(current_draft)

        return trace


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp 7: Deep research agent (planning + reflection)")
    parser.add_argument("--topic", type=str, required=True, help="Research topic / content brief")
    parser.add_argument("--corpus", type=str, default=None, help="Optional local folder of .txt/.md docs to search instead of the web")
    parser.add_argument("--reflection_rounds", type=int, default=2)
    parser.add_argument("--out", type=str, default="report.md", help="Where to save the final report")
    parser.add_argument("--trace_out", type=str, default="trace.json", help="Where to save the full run trace")
    args = parser.parse_args()

    llm = LLMClient()
    search = SearchTool(corpus_folder=args.corpus)
    agent = DeepResearchAgent(llm, search, reflection_rounds=args.reflection_rounds)

    trace = agent.run(args.topic)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(trace.drafts[-1])

    with open(args.trace_out, "w", encoding="utf-8") as f:
        json.dump({
            "topic": trace.topic,
            "plan": trace.plan,
            "notes": trace.notes,
            "num_drafts": len(trace.drafts),
            "reflections": trace.reflections,
            "drafts": trace.drafts,
        }, f, indent=2)

    print("\n=== FINAL REPORT (also saved to {}) ===\n".format(args.out))
    print(textwrap.shorten(trace.drafts[-1], width=1500, placeholder=" ...[truncated, see file]"))
    print(f"\nFull trace saved to {args.trace_out}")


if __name__ == "__main__":
    main()
