import json
import logging
from datetime import datetime
from pathlib import Path

from src.config import (
    ESCALATE_PREFIX,
    FINAL_PREFIX,
    LOGS_DIR,
    STATE_ESCALATE,
    STATE_QUERY,
    STATE_RESPONSE,
)

# Module logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Prompt Templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful customer support assistant. Answer the user's question
using ONLY the context provided below. Do not use any outside knowledge.

Respond using EXACTLY one of these two formats — no other format is acceptable:

  FINAL ANSWER: <your concise answer here>
  ESCALATE

Use ESCALATE if and only if:
  - The context does not contain enough information to answer confidently, OR
  - The query involves a sensitive issue (billing dispute, legal, account security), OR
  - The query is completely unrelated to the provided context.

Do NOT add any explanation, preamble, or text outside the two formats above."""


def build_user_prompt(query: str, context: str) -> str:
    """Combine the retrieved context and user query into a single user message."""
    return (
        f"Context:\n{context}\n\n"
        f"User Question:\n{query}"
    )


# Output Parser 
def parse_llm_output(raw: str) -> tuple[bool, str]:
    stripped = raw.strip()
    if stripped.upper().startswith(ESCALATE_PREFIX):
        logger.info("LLM signalled ESCALATE")
        return True, ""

    if stripped.upper().startswith(FINAL_PREFIX.upper()):
        # Extract everything after "FINAL ANSWER:"
        answer = stripped[len(FINAL_PREFIX):].strip()
        logger.info("LLM returned FINAL ANSWER (%d chars)", len(answer))
        return False, answer

    # Unexpected format — escalate for safety
    logger.warning("Unexpected LLM output format; defaulting to ESCALATE. Raw: %s", stripped[:120])
    return True, ""


# Context Formatter 
def format_context(chunks: list[dict]) -> str:
    if not chunks:
        return ""

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("metadata", {}).get("source", "unknown")
        page   = chunk.get("metadata", {}).get("page", "?")
        text   = chunk.get("page_content", "").strip()
        parts.append(f"[{i}] (source: {source}, page: {page})\n{text}")

    return "\n\n".join(parts)


# HITL Escalation Logger 
ESCALATION_LOG = LOGS_DIR / "escalations.log"

def log_escalation(state: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "query":     state.get(STATE_QUERY, ""),
        "context_snippet": state.get("context", "")[:200],
        "escalate":  state.get(STATE_ESCALATE, True),
        "response":  state.get(STATE_RESPONSE, ""),
    }

    with ESCALATION_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    logger.info("Escalation logged to %s", ESCALATION_LOG)


# HITL Simulated Response 
HITL_RESPONSE = (
    "Your query has been escalated to our human support team. "
    "A representative will reach out to you within 24 hours. "
    "We apologise for any inconvenience."
)
