"""Prompt construction for the three summarization roles.

The library owns prompt assembly; the SummarizeFn owns model contact.
Three prompt roles, each tuned to its job:
  - summarize: faithfulness (preserve decisions, constraints, open threads)
  - refine: incremental fold (integrate new material, drop superseded)
  - tighten: pure compression (cut to length, keep decision-relevant content)

Default prompts are tuned for general chat.  Domain-specific few-shot tuning
is expected once real outputs are observed — the spec-first-then-iterate loop.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Summarize prompt — the workhorse
# ---------------------------------------------------------------------------

SUMMARIZE_SYSTEM = (
    "You are a conversation compressor.  Your output will be injected as "
    "context for a language model that must continue the conversation.  "
    "Produce terse, third-person notes — not prose.  Never invent detail."
)

SUMMARIZE_TEMPLATE = """\
Summarize the following conversation excerpt into compact notes.

Preserve:
- Facts and constraints the user established
- Decisions made and their rationale
- Open / unresolved threads
- The user's goals and stated preferences
- Names, identifiers, numbers that later turns may reference

Drop: greetings, filler, repetition.

--- CONVERSATION ---
{text}
--- END ---

Summary (terse third-person notes):"""


# ---------------------------------------------------------------------------
# Refine prompt — incremental fold
# ---------------------------------------------------------------------------

REFINE_TEMPLATE = """\
You have an existing summary of an earlier portion of a conversation, and
new exchanges that followed it.  Produce ONE updated summary that integrates
the new material and drops anything now superseded.  Output terse third-person
notes — not prose.  Never invent detail.

--- EXISTING SUMMARY ---
{existing_summary}
--- NEW EXCHANGES ---
{new_text}
--- END ---

Updated summary (terse third-person notes):"""


# ---------------------------------------------------------------------------
# Tighten prompt — pure compression
# ---------------------------------------------------------------------------

TIGHTEN_TEMPLATE = """\
The following notes are too long.  Cut them to roughly {target_tokens} tokens.
Keep the most decision-relevant content.  Add nothing.  Output only the
shortened notes.

--- NOTES ---
{text}
--- END ---

Shortened notes:"""


def build_summarize_prompt(text: str) -> str:
    """Assemble a full summarize prompt from the template."""
    return SUMMARIZE_TEMPLATE.format(text=text)


def build_refine_prompt(existing_summary: str, new_text: str) -> str:
    """Assemble a refine prompt for incremental fold."""
    return REFINE_TEMPLATE.format(
        existing_summary=existing_summary,
        new_text=new_text,
    )


def build_tighten_prompt(text: str, target_tokens: int) -> str:
    """Assemble a tighten prompt with an explicit compression target."""
    return TIGHTEN_TEMPLATE.format(text=text, target_tokens=target_tokens)
