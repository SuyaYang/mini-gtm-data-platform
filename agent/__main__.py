"""CLI entrypoint for the GTM outreach agent.

This module wires together the three main stages of a run:
1. discover the live warehouse schema,
2. gather account/prospect context with tool-driven SQL,
3. draft an outreach email from the gathered brief.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from urllib.error import URLError
from urllib.request import urlopen
from pathlib import Path

# Auto-load `agent/.env` so the package can be run directly without a
# separate dotenv dependency. For this agent we let the file override stale
# shell exports, which makes local provider/model switching more predictable.
_ENV = Path(__file__).resolve().parent / ".env"
if _ENV.exists():
    for line in _ENV.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

from openai import OpenAI

from .catalog import SchemaCatalog
from .db import WarehouseConnection
from .drafter import draft_email
from .gatherer import gather_context

_AGENT_DIR = Path(__file__).resolve().parent


def _default_db_path() -> Path:
    """Resolve WAREHOUSE_PATH relative to the agent package when needed."""
    raw = os.getenv("WAREHOUSE_PATH", "")
    if raw:
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else (_AGENT_DIR / candidate).resolve()

    return (_AGENT_DIR.parent / "warehouse" / "data.duckdb").resolve()


_DEFAULT_DB = _default_db_path()
_PLACEHOLDER_KEYS = {"", "sk-your-key-here"}


def _has_local_ollama(base_url: str) -> bool:
    """Return True if an Ollama-compatible server responds at *base_url*."""
    try:
        with urlopen(base_url.rstrip("/") + "/models", timeout=1.5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            return "data" in payload
    except (OSError, URLError, TimeoutError, json.JSONDecodeError):
        return False


def _is_placeholder_key(api_key: str | None) -> bool:
    return (api_key or "").strip() in _PLACEHOLDER_KEYS


def _build_client() -> OpenAI:
    """Create an OpenAI-compatible client for hosted OpenAI or local Ollama."""
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    local_ollama = "http://localhost:11434/v1"

    if base_url:
        if "localhost:11434" in base_url and not _has_local_ollama(base_url):
            raise SystemExit(
                "Error: OPENAI_BASE_URL points to a local Ollama server, but no server responded.\n"
                "Start Ollama first, for example:\n"
                "  ollama serve\n"
                "  ollama pull qwen2.5:7b"
            )

        # Ollama exposes an OpenAI-compatible API and accepts any non-empty
        # placeholder key, commonly "ollama".
        return OpenAI(base_url=base_url, api_key=api_key or "ollama")

    # Make local testing easier: if the env file still contains the example
    # placeholder key, prefer a local Ollama server when one is running.
    if _is_placeholder_key(api_key) and _has_local_ollama(local_ollama):
        return OpenAI(base_url=local_ollama, api_key="ollama")

    if _is_placeholder_key(api_key):
        raise SystemExit(
            "Error: OPENAI_API_KEY is still set to the example placeholder.\n"
            "Choose one of these setups:\n"
            "  1. Hosted OpenAI: put a real OPENAI_API_KEY in agent/.env\n"
            "  2. Local Ollama: start Ollama and set OPENAI_BASE_URL=http://localhost:11434/v1\n"
            "     plus OPENAI_API_KEY=ollama in agent/.env"
        )

    return OpenAI(api_key=api_key)


def main() -> None:
    """Parse CLI args, run the pipeline, and print either a brief or email."""
    ap = argparse.ArgumentParser(prog="agent", description="GTM outreach agent")
    ap.add_argument("target", help="Account name, prospect name, or prospect email")
    ap.add_argument("--db", type=Path, default=_DEFAULT_DB, help="DuckDB warehouse path")
    ap.add_argument("--verbose", "-v", action="store_true", help="Log queries and reasoning")
    ap.add_argument("--dry-run", action="store_true", help="Print brief only, skip email")
    ap.add_argument("--max-turns", type=int, default=10, help="Max gatherer turns (default: 10)")
    args = ap.parse_args()

    # Verbose mode is aimed at agent debugging, so logs go to stderr while the
    # final brief/email stays clean on stdout.
    logging.basicConfig(format="%(message)s", level=logging.INFO if args.verbose else logging.WARNING, stream=sys.stderr)

    if not args.db.exists():
        print(f"Error: warehouse not found at {args.db}\nRun ./setup.sh first.", file=sys.stderr)
        sys.exit(1)

    # One warehouse connection is shared across all phases of the run.
    conn = WarehouseConnection(args.db)
    try:
        # 1. Schema discovery
        print("\n[1/3] Discovering warehouse schema …", file=sys.stderr)
        catalog = SchemaCatalog.discover(conn)
        print(f"      Found {catalog.stats}", file=sys.stderr)

        # 2. Context gathering
        client = _build_client()
        print(f'\n[2/3] Gathering intelligence on "{args.target}" …', file=sys.stderr)
        brief = gather_context(client, conn, catalog, args.target, max_turns=args.max_turns)
        print(f"      Brief: {len(brief.sections)} sections, {brief.total_queries} queries", file=sys.stderr)

        if args.dry_run:
            # Dry-run mode is useful for inspecting what the gatherer learned
            # without spending the extra model call on email generation.
            print("\n" + "=" * 60 + "\nDRY RUN — Intelligence Brief\n" + "=" * 60, file=sys.stderr)
            print(brief.to_drafter_context())
            return

        # 3. Email drafting
        print("\n[3/3] Drafting email …", file=sys.stderr)
        email = draft_email(client, brief)
        print(f"      Strategy: {email.strategy}", file=sys.stderr)

        # Keep stdout human-readable so the generated email can be copied
        # directly into a mail client if needed.
        sep = "\u2501" * 60
        print(f"\n{sep}\nSubject: {email.subject}\n\n{email.body}\n{sep}")
        if args.verbose and email.reasoning:
            print(f"\nReasoning: {email.reasoning}", file=sys.stderr)
    finally:
        # Close the database even if schema discovery or an API call fails.
        conn.close()


if __name__ == "__main__":
    main()
