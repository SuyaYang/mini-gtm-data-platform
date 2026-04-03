"""ReAct context-gathering agent using OpenAI tool calling."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

from openai import OpenAI

from ._config import resolve_model
from .catalog import SchemaCatalog
from .db import QueryResult, WarehouseConnection

log = logging.getLogger(__name__)

MAX_TURNS = int(os.getenv("AGENT_MAX_TURNS", "10"))

REQUIRED_PURPOSES = (
    "deal_history",
    "call_intelligence",
    "product_usage",
    "marketing_engagement",
    "contacts",
)

# The gatherer exposes exactly one tool: run a labeled SQL query. The purpose
# label becomes the section name in the final brief.
TOOLS = [{
    "type": "function",
    "function": {
        "name": "query_warehouse",
        "description": "Execute a read-only SQL query (DuckDB syntax). Returns up to 50 rows as JSON.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SQL query to execute"},
                "purpose": {"type": "string", "description": "Label: account_lookup, deal_history, call_intelligence, product_usage, marketing_engagement, contacts, etc."},
            },
            "required": ["sql", "purpose"],
        },
    },
}]

# The system prompt teaches the model how to resolve the target, which data
# domains to inspect, and when to stop querying.
SYSTEM = """\
You are a GTM intelligence analyst. Gather context about a target account or
prospect so a sales rep can write a personalized outreach email.

You have one tool: query_warehouse — read-only SQL against DuckDB.
**Only use table/column names from the catalog below.**
The only valid business schemas are: marts, staging, raw.

WAREHOUSE SCHEMA:
{catalog}

INSTRUCTIONS:
1. Resolve the target carefully:
   - If the input is an email, search lead/contact tables first.
   - If the input is a person's name, search lead/contact tables first, then resolve their related account if available.
   - If the input is a company name, search account tables first.
   - If the input is ambiguous and not an email, check both person-like and account-like matches before deciding.
2. Gather across these areas. Use these exact purpose labels:
   - deal_history: open/closed opportunities, stages, amounts
   - call_intelligence: recent calls, tracker topics (competitors, objections, buying signals)
   - product_usage: feature adoption, engagement tier, trends
   - marketing_engagement: lead activities, campaigns
   - contacts: stakeholders, champions, economic buyers
3. Prefer the marts schema (pre-joined). Fall back to staging if needed.
4. For prospects with no account, focus on funnel/lead tables.
5. Do not finish until every required purpose label above has been attempted at least once, even if the result is no data.
6. SQL RULES:
   - Return exactly one SELECT query per tool call.
   - Do NOT include a trailing semicolon.
   - Do NOT invent schemas, tables, or columns that are not in the catalog.
   - If a query fails because a schema/table/column does not exist, inspect the catalog and retry with exact names from the catalog.
7. Do not ask the user for more information. Use best-effort fuzzy matching with the warehouse data you already have.
8. End with a 2-4 sentence summary.
"""


# ── Intelligence brief ───────────────────────────────────────────


@dataclass
class IntelligenceBrief:
    """All warehouse evidence collected for the target during the gather step."""
    target_name: str
    target_type: str = "unknown"  # "account" or "prospect"
    sections: dict[str, list[QueryResult]] = field(default_factory=dict)
    summary: str = ""

    def add(self, qr: QueryResult) -> None:
        """Bucket each query result under the model-provided purpose label."""
        self.sections.setdefault(qr.purpose, []).append(qr)

    @property
    def total_queries(self) -> int:
        return sum(len(v) for v in self.sections.values())

    def missing_sections(self, required: tuple[str, ...] = REQUIRED_PURPOSES) -> list[str]:
        return [name for name in required if name not in self.sections]

    def to_drafter_context(self) -> str:
        """Flatten the brief into markdown-like text for the email drafter."""
        parts: list[str] = []
        for purpose, results in self.sections.items():
            parts.append(f"### {purpose.upper().replace('_', ' ')}")
            for qr in results:
                if qr.error:
                    parts.append(f"  [query failed: {qr.error}]")
                elif not qr.rows:
                    parts.append("  (no data)")
                else:
                    parts.append("  " + " | ".join(qr.columns))
                    for row in qr.rows:
                        parts.append("  " + " | ".join(str(row.get(c, "")) for c in qr.columns))
            parts.append("")
        if self.summary:
            parts.append(f"### AGENT SUMMARY\n{self.summary}")
        return "\n".join(parts)


# ── Public entry point ───────────────────────────────────────────


def gather_context(
    client: OpenAI,
    conn: WarehouseConnection,
    catalog: SchemaCatalog,
    target_name: str,
    *,
    max_turns: int = MAX_TURNS,
) -> IntelligenceBrief:
    """Run the tool loop until the model stops or the turn budget is exhausted."""
    brief = IntelligenceBrief(target_name=target_name)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM.format(catalog=catalog.to_prompt_context())},
        {"role": "user", "content": f"Gather all relevant GTM intelligence on: {target_name}"},
    ]

    for turn in range(1, max_turns + 1):
        log.info("Turn %d/%d", turn, max_turns)
        resp = client.chat.completions.create(
            model=resolve_model("GATHERER_MODEL"),
            max_tokens=4096,
            tools=TOOLS,
            tool_choice="auto",
            messages=messages,
        )
        message = resp.choices[0].message

        # No tool calls → model is either done or needs nudging.
        if not message.tool_calls:
            missing = brief.missing_sections()
            if missing:
                _nudge_for_missing(messages, message.content or "", missing)
                continue
            return _finalize_brief(brief, message.content or "", turn)

        # Append the assistant message, then execute each tool call.
        messages.append(_serialize_assistant(message))
        for tc in message.tool_calls:
            payload = _execute_tool_call(tc, conn, catalog, brief)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": payload})

    # Budget exhausted — return whatever was gathered.
    log.warning("Budget exhausted (%d turns)", max_turns)
    missing = brief.missing_sections()
    brief.summary = (
        "(budget exhausted — partial context"
        + (f"; missing sections: {', '.join(missing)}" if missing else "")
        + ")"
    )
    return brief


# ── Private helpers ──────────────────────────────────────────────


def _serialize_assistant(message) -> dict:
    """Convert an API response message to the dict format for message history."""
    msg: dict[str, object] = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in message.tool_calls
        ],
    }
    if message.content:
        msg["content"] = message.content
    return msg


def _execute_tool_call(
    tool_call,
    conn: WarehouseConnection,
    catalog: SchemaCatalog,
    brief: IntelligenceBrief,
) -> str:
    """Execute a single tool call and return the JSON response payload."""
    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"invalid tool arguments: {exc}"})

    sql = args.get("sql", "")
    purpose = args.get("purpose", "general")
    log.info("  [%s] %s", purpose, sql[:120])

    qr = conn.execute(sql, purpose=purpose)
    brief.add(qr)
    return _format_query_result(qr, catalog)


def _format_query_result(qr: QueryResult, catalog: SchemaCatalog) -> str:
    """Serialize a QueryResult to JSON for the tool response message."""
    if qr.error:
        payload: dict[str, object] = {"error": qr.error}
        if "does not exist" in qr.error or "Parser Error" in qr.error:
            payload["catalog_hint"] = {
                "valid_schemas": ["marts", "staging", "raw"],
                "tables_by_schema": catalog.table_name_hint(),
                "reminder": "Retry using exact table names from the catalog. "
                            "Do not invent new schemas or include a trailing semicolon.",
            }
    else:
        payload = {
            "columns": qr.columns,
            "rows": qr.rows,
            "row_count": qr.row_count,
            "truncated": qr.truncated,
        }
    return json.dumps(payload, default=str)


def _nudge_for_missing(messages: list[dict], text: str, missing: list[str]) -> None:
    """Push the model to cover required sections it hasn't attempted yet."""
    messages.append({"role": "assistant", "content": text})
    messages.append({
        "role": "user",
        "content": (
            f"You are missing these required sections: {', '.join(missing)}. "
            "Continue querying until each section has been attempted at least once, "
            "even if the result is no data. Do not ask the user for more information. "
            "Then return the final summary."
        ),
    })


def _finalize_brief(brief: IntelligenceBrief, summary: str, turn: int) -> IntelligenceBrief:
    """Attach the summary and infer target type from gathered sections."""
    brief.summary = summary
    if "account_lookup" in brief.sections:
        brief.target_type = "account"
    elif any(k in brief.sections for k in ("marketing_engagement", "lead_lookup")):
        brief.target_type = "prospect"
    log.info("Done: %d turns, %d queries", turn, brief.total_queries)
    return brief
