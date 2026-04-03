"""
Dynamic schema discovery via information_schema.
Marts get deep introspection (row counts + samples); staging/raw get column listings only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .db import WarehouseConnection

log = logging.getLogger(__name__)

# Skip DuckDB/system schemas because they add noise and should never be used
# by the agent when constructing business-facing queries.
_SKIP = {"information_schema", "pg_catalog", "main"}

# The marts layer is the most analyst-friendly surface, so we invest extra
# catalog budget there by fetching row counts and a small sample.
_DEEP = {"marts"}

_SCHEMA_LABELS = {
    "marts": "Analytics-ready, prefer these",
    "staging": "Cleaned views",
}


@dataclass
class TableInfo:
    """Metadata for one discovered table as seen by the agent."""
    schema_name: str
    table_name: str
    columns: list[tuple[str, str]] = field(default_factory=list)  # (name, type)
    row_count: int | None = None
    sample_rows: list[dict] = field(default_factory=list)

    @property
    def qualified(self) -> str:
        """Return the schema-qualified table name used in generated SQL."""
        return f"{self.schema_name}.{self.table_name}"


@dataclass
class SchemaCatalog:
    """Runtime view of the warehouse schema that will be injected into prompts."""
    tables: list[TableInfo] = field(default_factory=list)

    # ── Discovery ────────────────────────────────────────────────

    @classmethod
    def discover(cls, conn: WarehouseConnection) -> SchemaCatalog:
        """Discover schemas/tables/columns directly from information_schema."""
        catalog = cls()
        schemas = _list_schemas(conn)
        for schema in schemas:
            catalog.tables.extend(_discover_schema(conn, schema))
        log.info("Catalog: %d tables across %s", len(catalog.tables), schemas)
        return catalog

    # ── Prompt rendering ─────────────────────────────────────────

    def to_prompt_context(self) -> str:
        """Render the catalog into compact text for the gatherer's system prompt."""
        by_schema: dict[str, list[TableInfo]] = {}
        for t in self.tables:
            by_schema.setdefault(t.schema_name, []).append(t)

        parts: list[str] = []
        # Show marts first so the model sees the preferred query surface early.
        for schema in sorted(by_schema, key=lambda s: (s != "marts", s)):
            label = _SCHEMA_LABELS.get(schema, "Raw")
            parts.append(f"=== SCHEMA: {schema} ({len(by_schema[schema])} tables) — {label} ===\n")
            for t in sorted(by_schema[schema], key=lambda t: t.table_name):
                parts.append(_render_table(t))
        return "\n".join(parts)

    def table_name_hint(self) -> dict[str, list[str]]:
        """Compact per-schema table list for recovery after bad SQL."""
        by_schema: dict[str, list[str]] = {}
        for table in self.tables:
            by_schema.setdefault(table.schema_name, []).append(table.table_name)
        return {schema: sorted(names) for schema, names in by_schema.items()}

    @property
    def stats(self) -> str:
        """Human-readable summary used by the CLI progress output."""
        return f"{len({t.schema_name for t in self.tables})} schemas, {len(self.tables)} tables"


# ── Private helpers ──────────────────────────────────────────────


def _list_schemas(conn: WarehouseConnection) -> list[str]:
    """Return business schemas, filtering out DuckDB internals."""
    result = conn.execute(
        "SELECT schema_name FROM information_schema.schemata",
        purpose="catalog",
    )
    if result.error:
        return []
    # Preserve discovery order while deduplicating.
    return list(dict.fromkeys(
        r["schema_name"] for r in result.rows if r["schema_name"] not in _SKIP
    ))


def _discover_schema(conn: WarehouseConnection, schema: str) -> list[TableInfo]:
    """Discover all tables and columns in a single schema."""
    cols_r = conn.execute(
        f"SELECT table_name, column_name, data_type "
        f"FROM information_schema.columns "
        f"WHERE table_schema = '{schema}' "
        f"ORDER BY table_name, ordinal_position",
        purpose="catalog",
        max_rows=2000,
    )
    if cols_r.error:
        return []

    # Rebuild table objects from the flat information_schema row set.
    by_name: dict[str, TableInfo] = {}
    for row in cols_r.rows:
        t = by_name.setdefault(row["table_name"], TableInfo(schema, row["table_name"]))
        t.columns.append((row["column_name"], row["data_type"]))

    if schema in _DEEP:
        for table in by_name.values():
            _deep_inspect(conn, table)

    return list(by_name.values())


def _deep_inspect(conn: WarehouseConnection, table: TableInfo) -> None:
    """Fetch row count and a sample in a single query using a window function."""
    result = conn.execute(
        f"SELECT *, COUNT(*) OVER () AS _total_rows FROM {table.qualified}",
        purpose="catalog",
        max_rows=3,
    )
    if not result.rows:
        return
    table.row_count = result.rows[0]["_total_rows"]
    # Strip the synthetic column so sample_rows mirrors the real schema.
    table.sample_rows = [
        {k: v for k, v in row.items() if k != "_total_rows"}
        for row in result.rows
    ]


def _render_table(t: TableInfo) -> str:
    """Format a single table's metadata for the prompt."""
    header = f"TABLE: {t.qualified}"
    if t.row_count is not None:
        header += f" ({t.row_count:,} rows)"
    cols = ", ".join(f"{n} ({dt})" for n, dt in t.columns)
    lines = [header, f"  Columns: {cols}"]
    if t.sample_rows:
        lines.append(f"  Sample: {t.sample_rows[0]}")
    lines.append("")
    return "\n".join(lines)
