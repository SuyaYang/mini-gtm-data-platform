"""Email drafter — turns an IntelligenceBrief into a personalized outreach email."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from openai import OpenAI

from ._config import resolve_model
from .gatherer import IntelligenceBrief

log = logging.getLogger(__name__)

# This prompt asks the model to both choose an outreach strategy and emit a
# parseable response format that the CLI can print cleanly.
SYSTEM = """\
You are an expert B2B sales copywriter. Draft a personalized outreach email
from the intelligence brief provided.

STRATEGY (choose best fit):
- prospecting: no open deal. Value-led, reference industry peers.
- deal_acceleration: open mid/late-stage deal. Create urgency, reference usage/call topics.
- re_engagement: past closed-lost but continued interest. Lead with what changed.
- expansion: strong product adoption, no expansion deal. Introduce next tier.
- nurture: early prospect, light engagement. Reference specific content they touched.

RULES:
- Address recipient by first name (champion or most senior contact).
- Reference specific data points from the brief. No generic filler.
- Under 150 words. Subject under 60 characters.
- End with a low-friction CTA.
- Professional, conversational. No buzzwords, no exclamation marks.
- Do NOT fabricate data.

FORMAT:
STRATEGY: <name>
REASONING: <1-2 sentences>
SUBJECT: <subject line>
BODY:
<email>
"""

# Recognized section headers in the model's formatted response.
_PARSE_FIELDS = frozenset({"STRATEGY", "REASONING", "SUBJECT", "BODY"})


@dataclass
class Email:
    """Structured email draft returned by the drafter stage."""
    subject: str
    body: str
    strategy: str
    reasoning: str


def draft_email(client: OpenAI, brief: IntelligenceBrief) -> Email:
    """Turn the gathered brief into a final email plus strategy metadata."""
    resp = client.chat.completions.create(
        model=resolve_model("DRAFTER_MODEL"),
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
             f"TARGET: {brief.target_name} ({brief.target_type})\n\n"
             f"INTELLIGENCE BRIEF:\n{brief.to_drafter_context()}"},
        ],
    )
    return _parse(resp.choices[0].message.content or "")


def _parse(raw: str) -> Email:
    """Extract the structured fields from the model's formatted response."""
    values: dict[str, list[str]] = {}
    current: str | None = None

    for line in raw.splitlines():
        header = line.strip().split(":", 1)[0].upper()
        if header in _PARSE_FIELDS:
            current = header.lower()
            rest = line.split(":", 1)[1].strip() if ":" in line else ""
            # Only BODY is multiline; other fields capture the inline value.
            if current == "body":
                values.setdefault(current, [])
            elif rest:
                values[current] = [rest]
        elif current:
            values.setdefault(current, []).append(line)

    body = "\n".join(values.get("body", [])).strip() or raw.strip()
    return Email(
        subject=_join(values, "subject") or "(no subject)",
        body=body,
        strategy=_join(values, "strategy") or "unknown",
        reasoning=_join(values, "reasoning"),
    )


def _join(values: dict[str, list[str]], key: str) -> str:
    return " ".join(values.get(key, [])).strip()
