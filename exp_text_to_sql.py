"""
Experiment: Text-to-SQL Workflow
==================================
Given a natural-language question and a database, produce a correct SQL
query, run it, and return the results in plain English.

Pipeline stages (mirrors the other experiments' style — each stage is its
own class/method so it's easy to inspect and grade):

  1. SCHEMA INTROSPECTION -> SchemaExtractor  (read table/column info from the DB)
  2. SQL GENERATION       -> SQLGenerator     (NL question + schema -> SQL, via LLM)
  3. EXECUTION             -> QueryExecutor    (run SQL against SQLite, catch errors)
  4. SELF-CORRECTION LOOP  -> if execution fails, the error is fed back to the
                              LLM to regenerate a fixed query (up to N retries)
  5. ANSWER SYNTHESIS      -> turn the raw SQL result rows into a natural-
                              language answer

LLM backend: OpenAI (OPENAI_API_KEY) or Anthropic (ANTHROPIC_API_KEY) if set.
Without either, a small rule-based fallback handles a handful of common
question patterns (counts, simple filters, "list all X") so the workflow is
still runnable end-to-end with zero setup — useful for testing the pipeline
control flow before you have an API key.

Comes with a script to build a sample SQLite database (`--init_sample_db`)
so you can try it immediately with no setup beyond `pip install`.
"""

import os
import re
import json
import sqlite3
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# ----------------------------------------------------------------------
# 0. Sample database (so this is runnable with zero setup)
# ----------------------------------------------------------------------

def init_sample_db(db_path: str):
    """Creates a small company DB: departments + employees, with sample rows."""
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE departments (
        dept_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        location TEXT
    );

    CREATE TABLE employees (
        emp_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        dept_id INTEGER,
        salary REAL,
        hire_date TEXT,
        FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
    );
    """)
    cur.executemany(
        "INSERT INTO departments (dept_id, name, location) VALUES (?, ?, ?)",
        [
            (1, "Engineering", "Bangalore"),
            (2, "Sales", "Mumbai"),
            (3, "HR", "Hyderabad"),
        ],
    )
    cur.executemany(
        "INSERT INTO employees (emp_id, name, dept_id, salary, hire_date) VALUES (?, ?, ?, ?, ?)",
        [
            (1, "Asha Rao", 1, 95000, "2021-03-01"),
            (2, "Vikram Singh", 1, 88000, "2022-07-15"),
            (3, "Meera Iyer", 2, 72000, "2020-11-20"),
            (4, "Rahul Nair", 3, 65000, "2023-01-10"),
            (5, "Priya Sharma", 1, 102000, "2019-06-05"),
            (6, "Karan Mehta", 2, 78000, "2022-02-28"),
        ],
    )
    conn.commit()
    conn.close()
    print(f"[init_sample_db] created sample DB at {db_path}")


# ----------------------------------------------------------------------
# 1. Schema introspection
# ----------------------------------------------------------------------

@dataclass
class TableSchema:
    name: str
    columns: List[Dict[str, str]] = field(default_factory=list)  # [{"name":..., "type":...}]
    foreign_keys: List[Dict[str, str]] = field(default_factory=list)


class SchemaExtractor:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def extract(self) -> List[TableSchema]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [r[0] for r in cur.fetchall()]

        schemas = []
        for t in tables:
            cur.execute(f"PRAGMA table_info({t})")
            cols = [{"name": row[1], "type": row[2]} for row in cur.fetchall()]

            cur.execute(f"PRAGMA foreign_key_list({t})")
            fks = [{"column": row[3], "references_table": row[2], "references_column": row[4]}
                   for row in cur.fetchall()]

            schemas.append(TableSchema(name=t, columns=cols, foreign_keys=fks))

        conn.close()
        return schemas

    @staticmethod
    def to_prompt_string(schemas: List[TableSchema]) -> str:
        lines = []
        for s in schemas:
            col_str = ", ".join(f"{c['name']} {c['type']}" for c in s.columns)
            lines.append(f"TABLE {s.name}({col_str})")
            for fk in s.foreign_keys:
                lines.append(f"  -- FK: {s.name}.{fk['column']} -> {fk['references_table']}.{fk['references_column']}")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# LLM client (same pattern as the other experiments)
# ----------------------------------------------------------------------

class LLMClient:
    def __init__(self):
        self.backend = "fallback"
        if os.environ.get("OPENAI_API_KEY"):
            self.backend = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            self.backend = "anthropic"
        print(f"[LLMClient] using backend: {self.backend}")

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        if self.backend == "openai":
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0,
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

        return self._fallback_sql(user)

    @staticmethod
    def _fallback_sql(user: str) -> str:
        """Very small rule-based NL->SQL for common patterns, so the pipeline
        runs without any API key. Looks for the question inside `user`."""
        q_match = re.search(r"Question:\s*(.+)", user)
        question = (q_match.group(1) if q_match else user).lower()

        # table guess: prefer a table name (singular or plural) that actually
        # appears in the question; otherwise fall back to the first table
        all_tables = re.findall(r"TABLE (\w+)", user)
        table = all_tables[0] if all_tables else "employees"
        for t in all_tables:
            if t.lower() in question or t.lower().rstrip("s") in question:
                table = t
                break

        # column guess for aggregate questions: prefer a column that belongs
        # to the chosen table and is mentioned in the question
        table_cols_match = re.search(rf"TABLE {re.escape(table)}\(([^)]*)\)", user)
        table_cols = []
        if table_cols_match:
            table_cols = [c.split()[0] for c in table_cols_match.group(1).split(",")]
        agg_col = next((c for c in table_cols if c.lower() in question), "*")

        if "how many" in question or "count" in question:
            return f"SELECT COUNT(*) FROM {table};"
        if "average" in question or "avg" in question:
            return f"SELECT AVG({agg_col}) FROM {table};"
        if "list all" in question or "show all" in question or re.search(r"\ball\b", question):
            return f"SELECT * FROM {table};"
        return f"SELECT * FROM {table} LIMIT 10;"


# ----------------------------------------------------------------------
# 2. SQL generation
# ----------------------------------------------------------------------

class SQLGenerator:
    SYSTEM_PROMPT = (
        "You are an expert SQLite query writer. Given a database schema and a "
        "natural-language question, output ONLY a single valid SQLite SQL "
        "query that answers the question. No explanation, no markdown fences, "
        "no comments — just the raw SQL statement ending in a semicolon."
    )

    FIX_SYSTEM_PROMPT = (
        "You are an expert SQLite query writer fixing a broken query. Given "
        "the schema, the original question, the failed SQL, and the database "
        "error message, output ONLY a corrected single valid SQLite query "
        "(no explanation, no markdown fences)."
    )

    def __init__(self, llm: LLMClient):
        self.llm = llm

    @staticmethod
    def _clean(sql: str) -> str:
        sql = sql.strip()
        sql = re.sub(r"^```(sql)?", "", sql).strip()
        sql = re.sub(r"```$", "", sql).strip()
        return sql

    def generate(self, question: str, schema_str: str) -> str:
        user = f"Schema:\n{schema_str}\n\nQuestion: {question}\n\nSQL:"
        sql = self.llm.complete(self.SYSTEM_PROMPT, user)
        return self._clean(sql)

    def fix(self, question: str, schema_str: str, failed_sql: str, error: str) -> str:
        user = (
            f"Schema:\n{schema_str}\n\nQuestion: {question}\n\n"
            f"Failed SQL:\n{failed_sql}\n\nError:\n{error}\n\nCorrected SQL:"
        )
        sql = self.llm.complete(self.FIX_SYSTEM_PROMPT, user)
        return self._clean(sql)


# ----------------------------------------------------------------------
# 3. Execution
# ----------------------------------------------------------------------

class QueryExecutor:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def run(self, sql: str) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        try:
            cur.execute(sql)
            if sql.strip().lower().startswith("select"):
                cols = [d[0] for d in cur.description] if cur.description else []
                rows = cur.fetchall()
                return {"ok": True, "columns": cols, "rows": rows}
            else:
                conn.commit()
                return {"ok": True, "columns": [], "rows": [], "rowcount": cur.rowcount}
        except Exception as e:
            return {"ok": False, "error": str(e)}
        finally:
            conn.close()


# ----------------------------------------------------------------------
# 5. Answer synthesis
# ----------------------------------------------------------------------

class AnswerSynthesizer:
    SYSTEM_PROMPT = (
        "You turn SQL query results into a short, clear natural-language "
        "answer to the original question. Be concise and factual."
    )

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def synthesize(self, question: str, columns: List[str], rows: List[tuple]) -> str:
        if self.llm.backend == "fallback":
            if not rows:
                return "No matching rows were found."
            if len(rows) == 1 and len(rows[0]) == 1:
                return f"Answer: {rows[0][0]}"
            preview = "\n".join(str(r) for r in rows[:10])
            return f"Columns: {columns}\nRows ({len(rows)} total, showing up to 10):\n{preview}"

        result_str = f"Columns: {columns}\nRows: {rows[:20]}"
        user = f"Question: {question}\n\nSQL result:\n{result_str}\n\nAnswer in plain English:"
        return self.llm.complete(self.SYSTEM_PROMPT, user, max_tokens=300)


# ----------------------------------------------------------------------
# Full pipeline
# ----------------------------------------------------------------------

class TextToSQLPipeline:
    def __init__(self, db_path: str, max_retries: int = 2):
        self.db_path = db_path
        self.schema_extractor = SchemaExtractor(db_path)
        self.llm = LLMClient()
        self.generator = SQLGenerator(self.llm)
        self.executor = QueryExecutor(db_path)
        self.synthesizer = AnswerSynthesizer(self.llm)
        self.max_retries = max_retries

    def ask(self, question: str) -> Dict[str, Any]:
        schemas = self.schema_extractor.extract()
        schema_str = SchemaExtractor.to_prompt_string(schemas)
        print("[Schema]\n" + schema_str)

        sql = self.generator.generate(question, schema_str)
        attempts = [{"sql": sql}]
        result = self.executor.run(sql)

        retries = 0
        while not result["ok"] and retries < self.max_retries:
            print(f"[Self-correction] attempt {retries+1}: query failed with: {result['error']}")
            sql = self.generator.fix(question, schema_str, sql, result["error"])
            attempts.append({"sql": sql})
            result = self.executor.run(sql)
            retries += 1

        if not result["ok"]:
            return {
                "question": question,
                "final_sql": sql,
                "success": False,
                "error": result["error"],
                "attempts": attempts,
                "answer": f"I couldn't produce a working query after {retries} retries. Last error: {result['error']}",
            }

        answer = self.synthesizer.synthesize(question, result.get("columns", []), result.get("rows", []))
        return {
            "question": question,
            "final_sql": sql,
            "success": True,
            "columns": result.get("columns", []),
            "rows": result.get("rows", []),
            "attempts": attempts,
            "answer": answer,
        }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Text-to-SQL workflow experiment")
    parser.add_argument("--db", type=str, default="./company.db", help="Path to SQLite DB file")
    parser.add_argument("--question", type=str, help="Natural-language question to answer")
    parser.add_argument("--init_sample_db", action="store_true", help="Create/overwrite a sample DB at --db and exit")
    parser.add_argument("--max_retries", type=int, default=2, help="Self-correction retries on SQL error")
    args = parser.parse_args()

    if args.init_sample_db:
        init_sample_db(args.db)
        return

    if not args.question:
        parser.error("--question is required unless using --init_sample_db")

    if not os.path.exists(args.db):
        print(f"[warning] {args.db} not found — creating a sample DB there. "
              f"Run with --init_sample_db explicitly to control this.")
        init_sample_db(args.db)

    pipeline = TextToSQLPipeline(args.db, max_retries=args.max_retries)
    result = pipeline.ask(args.question)

    print("\n=== Generated SQL ===")
    print(result["final_sql"])

    if result["success"]:
        print("\n=== Raw result ===")
        print("columns:", result["columns"])
        for row in result["rows"][:10]:
            print(" ", row)
    else:
        print("\n=== Error ===")
        print(result["error"])

    print("\n=== Answer ===")
    print(result["answer"])


if __name__ == "__main__":
    main()
