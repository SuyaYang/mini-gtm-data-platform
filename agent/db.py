"""Read-only DuckDB wrapper. Row-limited, exception-safe."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

log = logging.getLogger(__name__)

# Hard cap on rows returned per query — enforced at the driver level,
# so the LLM cannot bypass it regardless of what SQL it writes.
MAX_ROWS = 50


@dataclass
class QueryResult:
    """Structured result from a single warehouse query."""
    sql: str          # The original SQL that was executed
    purpose: str      # LLM-provided label (e.g. "deal_history", "contacts")
    columns: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False       # True if results were capped at MAX_ROWS
    error: str | None = None      # Populated on failure; query never crashes the agent


class WarehouseConnection:
    """Thin wrapper around DuckDB. Opened read_only=True so mutations are
    impossible at the driver level — this is a hard constraint, not a prompt."""

    def __init__(self, db_path: Path) -> None:
        # The connection itself enforces read-only access, which is safer than
        # merely telling the model not to write data.
        self._conn = duckdb.connect(str(db_path), read_only=True)
        log.info("Connected (read-only): %s", db_path)

    def execute(self, sql: str, *, purpose: str = "general", max_rows: int = MAX_ROWS) -> QueryResult:
        """Run SQL, return at most *max_rows*. Errors captured, never raised."""
        try:
            normalized_sql = sql.strip().rstrip(";").strip()

            # Wrap caller's SQL in a subquery with LIMIT to enforce row cap.
            # Fetch one extra row to detect whether results were truncated.
            result = self._conn.execute(f"SELECT * FROM ({normalized_sql}) AS _q LIMIT {max_rows + 1}")
            columns = [d[0] for d in result.description]
            raw = result.fetchall()

            truncated = len(raw) > max_rows
            rows = [dict(zip(columns, r)) for r in raw[:max_rows]]

            log.info("[%s] %d rows%s", purpose, len(rows), " (truncated)" if truncated else "")
            return QueryResult(sql=normalized_sql, purpose=purpose, columns=columns, rows=rows,
                               row_count=len(rows), truncated=truncated)
        except Exception as exc:
            # Return the error as data so the agent loop can continue.
            log.warning("[%s] FAILED: %s", purpose, exc)
            return QueryResult(sql=sql, purpose=purpose, error=str(exc))

    def close(self) -> None:
        """Release the underlying DuckDB connection."""
        self._conn.close()
