# Exp 2 + Exp 7 — Setup & Execution

## 0. Setup (do this once)

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You don't need every package — each script auto-detects what's installed
and falls back gracefully (see notes below). At minimum install `numpy`
and `scikit-learn`.

Optional but recommended, for real LLM-generated answers instead of the
fallback mode:
```bash
export OPENAI_API_KEY="sk-..."          # or
export ANTHROPIC_API_KEY="sk-ant-..."
```
(Windows CMD: `set OPENAI_API_KEY=sk-...`)

---

## Experiment 2 — RAG-based Question Answering

**File:** `exp2_rag_qa.py`
**Stages implemented:** Indexing → Retrieval → Response Generation (see the
three classes `DocumentIndexer`, `Retriever`, `Generator` in the file).

A `docs/` folder with two sample `.txt` files is included so you can test
immediately. Add your own `.txt`/`.md` files to that folder (or point
`--docs` at any folder).

### Run it
```bash
python exp2_rag_qa.py --docs ./docs --query "What are the three stages of a RAG pipeline?"
```

Useful flags:
```bash
python exp2_rag_qa.py \
  --docs ./docs \
  --query "How does FAISS support similarity search?" \
  --top_k 3 \
  --chunk_size 400 \
  --chunk_overlap 80
```

### What you'll see
1. An `[Indexing]` log line showing how many chunks were created and which
   embedder/vector-store backend was auto-selected.
2. The top-k retrieved chunks with their similarity scores.
3. The generated answer (LLM-based if you set an API key, otherwise a
   clearly-labeled extractive fallback so you can still verify retrieval
   quality without any key).

### Backends it auto-selects
| Component   | Preferred            | Fallback (no install needed) |
|-------------|-----------------------|-------------------------------|
| Embeddings  | sentence-transformers | TF-IDF (scikit-learn)         |
| Vector store| FAISS                 | NumPy brute-force cosine sim  |
| Generation  | OpenAI / Anthropic    | Extractive best-chunk answer  |

---

## Experiment 7 — Deep Research Agent (Planning + Reflection)

**File:** `exp7_deep_research_agent.py`
**Workflow implemented:** Plan → Research → Draft → **Reflect → Revise**
(repeated `--reflection_rounds` times) → Final report. See the
`DeepResearchAgent` class — each stage is its own method
(`plan`, `research`, `draft`, `reflect`, `revise`, `run`).

### Run it (web search mode — needs no API key for search itself)
```bash
pip install duckduckgo-search   # if not already installed
python exp7_deep_research_agent.py --topic "Retrieval-Augmented Generation" --reflection_rounds 2
```

### Run it against your own local documents instead of the web
```bash
python exp7_deep_research_agent.py --topic "RAG pipelines" --corpus ./docs --reflection_rounds 2
```

### Outputs
- `report.md` — the final revised report.
- `trace.json` — the full transparent trace: the generated plan
  (sub-questions), every research note gathered, every intermediate draft,
  and every reflection/critique — useful for showing your workflow in a lab
  writeup.

### What "planning + reflection" means here concretely
- **Planning**: the topic is decomposed into 4–6 sub-questions before any
  writing happens (`agent.plan()`).
- **Research**: each sub-question is searched independently, so notes are
  organized per sub-question (`agent.research()`).
- **Reflection**: after each draft, the agent runs a self-critique pass
  that returns structured JSON feedback (`coverage_gaps`,
  `structure_issues`, `accuracy_flags`, `overall_verdict`).
- **Revision**: the draft is rewritten specifically to resolve that
  feedback, and the cycle repeats for `--reflection_rounds` iterations
  (default 2), which is the "improvement loop" the experiment asks for.

### Backends it auto-selects
| Component | Preferred                 | Fallback (no install/key needed)        |
|-----------|----------------------------|------------------------------------------|
| LLM       | OpenAI / Anthropic         | Deterministic rule-based stand-in (lets you test the control flow with zero setup) |
| Search    | duckduckgo-search (web)    | Local `--corpus` folder, else placeholder notes |

---

## Experiment — Text-to-SQL Workflow

**File:** `exp_text_to_sql.py`
**Stages implemented:** Schema Introspection → SQL Generation → Execution →
**Self-Correction Loop** (on SQL error) → Answer Synthesis. See
`SchemaExtractor`, `SQLGenerator`, `QueryExecutor`, `AnswerSynthesizer`, tied
together in `TextToSQLPipeline`.

A sample SQLite database (`departments` + `employees`, with foreign keys) is
generated automatically so you can run it immediately with zero setup.

### Run it
```bash
# one-time: create the sample DB explicitly (optional — it's auto-created
# the first time you ask a question if the DB file doesn't exist yet)
python exp_text_to_sql.py --db ./company.db --init_sample_db

python exp_text_to_sql.py --db ./company.db --question "How many employees are there?"
python exp_text_to_sql.py --db ./company.db --question "What is the average salary of employees in Engineering?"
python exp_text_to_sql.py --db ./company.db --question "List all departments"
```

Point `--db` at your own `.db`/`.sqlite` file to use your own schema —
`SchemaExtractor` reads it automatically via `PRAGMA table_info`/`PRAGMA
foreign_key_list`, no manual schema description needed.

### What you'll see
1. The introspected schema (tables, columns, foreign keys) printed to the
   terminal — this is exactly what gets fed to the LLM as context.
2. The generated SQL query.
3. If execution fails (e.g. the LLM hallucinated a column), the error
   message is fed back to the LLM to regenerate a corrected query — up to
   `--max_retries` times (default 2). Each attempt is logged so you can show
   the self-correction loop in your report.
4. The raw result rows, plus a natural-language answer synthesized from
   them.

### Backends it auto-selects
| Component | Preferred            | Fallback (no key needed)                          |
|-----------|------------------------|----------------------------------------------------|
| SQL gen / fix / answer | OpenAI / Anthropic | Small rule-based NL→SQL for counts, averages, "list all X" — lets you test schema introspection, execution and the retry loop with zero setup |

---

## Notes for your lab report
- All three scripts print progress at each stage to the terminal, and Exp 7
  also writes `trace.json` — screenshot or paste these to document
  indexing/retrieval (Exp 2), plan/reflect/revise cycles (Exp 7), and the
  schema/SQL/self-correction steps (Text-to-SQL).
- To get genuinely good generated text/SQL (not the offline fallback), set
  `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` before running any script.
- Swap in your own document set for Exp 2 by dropping `.txt`/`.md` files
  into `docs/` (or any folder passed via `--docs`).
- Swap in your own database for the Text-to-SQL experiment by pointing
  `--db` at any SQLite file.
