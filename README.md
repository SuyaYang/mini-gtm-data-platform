# GTM Outreach Agent

Gathers warehouse context about an account or prospect and drafts a personalized outreach email. Schema is discovered dynamically at runtime — no table or column names are hardcoded.

## Quickstart

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- A built warehouse (`warehouse/data.duckdb`) — run `./setup.sh` from the repo root if you haven't already
- An LLM provider: either an OpenAI API key **or** a local [Ollama](https://ollama.com) server

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure your LLM provider

Create `agent/.env` from the example and set your provider:

```bash
cp agent/.env.example agent/.env
```

**Option A — OpenAI (hosted):**

```bash
# agent/.env
OPENAI_API_KEY=sk-your-real-key
```

**Option B — Ollama (local):**

```bash
# Start the server and pull a model first:
ollama serve
ollama pull qwen2.5:7b
```

```bash
# agent/.env
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
GATHERER_MODEL=qwen2.5:7b
DRAFTER_MODEL=qwen2.5:7b
```

### 3. Run it

```bash
# Look up an account and draft an outreach email
uv run python -m agent "Synergy Solutions"

# Look up a prospect by email
uv run python -m agent "sarah.martinez@acme.com"

# Look up a prospect by name
uv run python -m agent "Sarah Martinez"
```

### 4. Verify with a dry run (recommended first time)

`--dry-run` prints the gathered intelligence brief without calling the drafter, so you can inspect what the agent found before generating an email:

```bash
uv run python -m agent "Synergy Solutions" --dry-run --verbose
```

## CLI Reference

```
usage: agent [-h] [--db DB] [--verbose] [--dry-run] [--max-turns N] target
```

| Flag | Description |
|---|---|
| `target` | Account name, prospect name, or prospect email (required) |
| `--db PATH` | Path to DuckDB warehouse (default: `../warehouse/data.duckdb`) |
| `--verbose`, `-v` | Log every SQL query and reasoning to stderr |
| `--dry-run` | Print the intelligence brief only, skip email generation |
| `--max-turns N` | Max gatherer tool-calling turns (default: 10) |

### Environment variables

All configurable via `agent/.env` or shell environment. CLI flags override these where applicable.

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | Yes | — |
| `OPENAI_BASE_URL` | No | OpenAI hosted API |
| `WAREHOUSE_PATH` | No | `../warehouse/data.duckdb` |
| `GATHERER_MODEL` | No | `gpt-4.1` |
| `DRAFTER_MODEL` | No | `gpt-4.1` |
| `AGENT_MAX_TURNS` | No | `10` |

## Example Output

With `--dry-run`, the agent prints the structured brief that feeds the email drafter:

```text
[1/3] Discovering warehouse schema …
      Found 3 schemas, 30 tables

[2/3] Gathering intelligence on "Synergy Solutions" …
      Brief: 5 sections, 7 queries

### DEAL HISTORY
  opportunity_name | stage | amount
  Synergy Solutions - Platform Deal | Proposal | 13788

### PRODUCT USAGE
  feature_name | monthly_active_users | engagement_tier
  Workflow Automation | 42 | High

### AGENT SUMMARY
Recent deal activity and strong product adoption suggest a deal acceleration angle.
```

Without `--dry-run`, the final output is a ready-to-send email:

```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Subject: Quick idea for Synergy Solutions

<personalized email body>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## How It Works

### Approach

The agent runs a 3-stage pipeline:

1. **Schema Discovery** — queries `information_schema` to build a live catalog of every table and column in the warehouse. No schema is hardcoded.
2. **Context Gathering** — a ReAct-style tool loop where the LLM writes SQL against the discovered catalog. Each query is labeled with a purpose (deal history, call intelligence, product usage, marketing engagement, contacts). The loop continues until all five areas are covered.
3. **Email Drafting** — a single LLM call that receives the gathered brief and produces a personalized email. The outreach strategy (prospecting, deal acceleration, re-engagement, expansion, nurture) is inferred from the data, not hardcoded.

### Architecture

```
"Input (Prospect or Account)"
       |
       v
+-------------------------------------+
|  1. Schema Discovery  (catalog.py)  |
|  Queries information_schema.        |
|  Builds live catalog of all tables. |
|  Output -> SchemaCatalog            |
+------------------+------------------+
                   v
+-------------------------------------+
|  2. Context Gathering (gatherer.py) |
|  OpenAI tool-calling loop (max 10). |
|  Tool: query_warehouse(sql,purpose) |
|  Gathers: deals, calls, product    |
|  usage, marketing, contacts.        |
|  Output -> IntelligenceBrief        |
+------------------+------------------+
                   v
+-------------------------------------+
|  3. Email Drafting  (drafter.py)    |
|  Single OpenAI call. Infers         |
|  strategy from brief data.          |
|  Output -> Email(subject, body)     |
+-------------------------------------+
```

**2 LLM calls** per run. ~19 SQL queries (14 catalog + 5-7 context).

### Input handling

- **Account input**: company names such as `"Acme Dynamics"` or `"Synergy Solutions"`.
- **Prospect input**: person identifiers such as `"sarah.martinez@acme.com"` or `"Sarah Martinez"`.
- **Ambiguous names**: the gatherer checks both account-like and person-like matches before deciding.
- **Non-unique account names**: synthetic data can contain duplicates — the agent may need to disambiguate between matching records.

## Files

| File | Purpose |
|---|---|
| `__main__.py` | CLI + orchestration. Loads `agent/.env` automatically. |
| `_config.py` | Shared configuration — model resolution for hosted OpenAI and local Ollama. |
| `catalog.py` | Schema discovery via `information_schema`. Deep introspection on marts, light on staging/raw. |
| `db.py` | Read-only DuckDB wrapper. 50-row limit enforced at driver level. Errors captured, never raised. |
| `gatherer.py` | ReAct agent loop. LLM writes SQL from the catalog, labels each query with a purpose. |
| `drafter.py` | Single-shot email generation. Strategy inferred from data. |

## Design Decisions

- **`read_only=True`** on DuckDB — mutation impossible at driver level, not a prompt instruction.
- **Row limit in `db.py`** — the LLM writes free-form SQL, so every query is wrapped in a `LIMIT` subquery at the driver level. This prevents a single unbounded `SELECT *` from blowing up the conversation context and API cost. 50 rows is more than enough for the representative samples the gatherer needs per domain.
- **Errors as data** — bad SQL returns `QueryResult.error`, never crashes the loop.
- **Free-form purpose labels** — sections in `IntelligenceBrief` are labeled by the LLM, not a hardcoded enum.
- **Required research coverage** — the gatherer must attempt all five GTM context areas before finishing.
- **Strategy inferred** — no if/else rules; the drafter LLM picks the approach from the data.

## Known Issues

- **Schema hallucination in SQL generation** — the gatherer LLM frequently generates SQL with incorrect schema/table references despite receiving the correct catalog. Common failure patterns:
  - **Invented schemas**: e.g. `sales.opportunities` when `sales` does not exist (should be `raw.opportunities` or a marts table).
  - **Missing schema qualifiers**: e.g. `FROM leads` instead of `FROM raw.leads`.
  - **Wrong schema prefix on staging tables**: e.g. `marts.stg_calls` when `stg_*` tables live in `staging`, not `marts`.
- **Ineffective error recovery** — when a query fails, the `catalog_hint` with correct table names is sent back, but the model often repeats the same mistakes across retries until the turn budget is exhausted.
- **Model sensitivity** — smaller or less instruction-following models (e.g. `qwen2.5:7b`) are more prone to these errors. 

## Limitations

- **Ambiguous entity resolution** — if multiple accounts or people match the same name, the agent may choose the wrong record without an explicit disambiguation step.
- **Prompt-led reasoning** — the gatherer relies on model judgment to decide which SQL to write and how to interpret partial results.
- **No automated eval harness** — output quality is easy to inspect manually, but not scored systematically.
- **Turn-budget sensitivity** — if the gatherer hits its turn limit, it returns a partial brief rather than failing.

## Future Work

### Business

- **Explicit disambiguation flow** — when an input matches multiple accounts, show candidate records with details (region, industry, ARR) and let the user choose before continuing.
- **Batch mode for rep workflows** — run the agent across all accounts for a rep, rank by signal strength, and generate a prioritized outreach queue.
- **Role-specific email variants** — generate different drafts for different stakeholders (champion, evaluator, economic buyer) from the same brief.

### Technical

- **Lazy schema exploration** — start with a lightweight table list and let the gatherer request detail on only the tables it needs, reducing prompt size and cost.
- **Brief caching with freshness rules** — reuse a gathered `IntelligenceBrief` for a short window (e.g. 24 hours) unless new warehouse activity appears.
- **Evaluation harness** — score generated emails on factual accuracy, personalization depth, CTA clarity, and tone consistency across a test set.
- **SQL schema validation** — pre-check generated SQL table/schema references against the catalog before executing, and reject invalid queries with a targeted correction instead of relying on database error messages.
- **Stronger model or few-shot examples** — switch to a more capable model (e.g. Claude, GPT-4o) or add few-shot SQL examples to the gatherer prompt showing correct schema-qualified table references.
