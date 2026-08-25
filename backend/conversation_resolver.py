"""
IIT Jodhpur V1 — Conversation Resolver

Purpose
-------
Resolve incomplete follow-up questions only when recent conversation
context is required.

Design principles
-----------------
- Standalone questions pass through unchanged.
- Explicit topic switches pass through unchanged.
- Genuine follow-ups may use recent conversation.
- Obviously ambiguous references do not get guessed.
- The resolver does not rewrite every question.
- If resolution fails, the original question is preserved.
- No factual content is invented by this layer.

Architecture
------------
    User Question
        ↓
    Cheap local classification
        ↓
    ┌──────────────────────────────────────┐
    │ Standalone / topic switch            │
    │ → pass through                       │
    │                                      │
    │ Obviously ambiguous reference        │
    │ → preserve original                  │
    │                                      │
    │ Genuine follow-up candidate          │
    │ → Qwen2.5:3B resolver                │
    └──────────────────────────────────────┘
"""

import json
import re
from typing import Any, Dict, List

from backend.llm import query_llm


# =========================================================
# Configuration
# =========================================================

MAX_HISTORY_MESSAGES = 8

FOLLOW_UP_PATTERNS = [
    r"^\s*what about\b",
    r"^\s*how about\b",
    r"^\s*and what about\b",
    r"^\s*how much\b",
    r"^\s*how many\b",
    r"^\s*when is it\b",
    r"^\s*where is it\b",
    r"^\s*where are they\b",
    r"^\s*is it\b",
    r"^\s*is that\b",
    r"^\s*does it\b",
    r"^\s*do they\b",
    r"^\s*can i\b",
    r"^\s*can they\b",
    r"^\s*what are its\b",
    r"^\s*what is its\b",
    r"^\s*tell me more\b",
    r"^\s*more about\b",
]


# These are intentionally limited to clearly vague references.
# We do NOT create a large hard-coded language rule system.
AMBIGUOUS_REFERENCE_PATTERNS = [
    r"^\s*what about (?:that|this|it)\s*[?!.]*\s*$",
    r"^\s*how about (?:that|this|it)\s*[?!.]*\s*$",
    r"^\s*what is (?:that|this|it)\s*[?!.]*\s*$",
    r"^\s*how is (?:that|this|it)\s*[?!.]*\s*$",
    r"^\s*what are (?:those|these|they)\s*[?!.]*\s*$",
]


# =========================================================
# Query-model wrapper
# =========================================================

def invoke_query_llm(prompt: str):
    """
    Invoke the query-understanding model.

    Kept separate so this layer is easy to unit test without
    mutating the ChatOllama/Pydantic instance.
    """

    return query_llm.invoke(prompt)


# =========================================================
# Helpers
# =========================================================

def _normalize_question(
    question: str,
) -> str:
    """
    Normalize whitespace without changing meaning.
    """

    return re.sub(
        r"\s+",
        " ",
        str(question or ""),
    ).strip()


def _is_follow_up_like(
    question: str,
) -> bool:
    """
    Detect questions that are likely incomplete without
    recent conversation context.
    """

    normalized = question.lower().strip()

    if not normalized:
        return False

    return any(
        re.search(
            pattern,
            normalized,
        )
        for pattern in FOLLOW_UP_PATTERNS
    )


def _is_obviously_ambiguous(
    question: str,
) -> bool:
    """
    Detect very vague deictic references where the user has not
    provided enough explicit information to justify a rewrite.

    Example:
        "What about that?"

    These should remain unresolved rather than being guessed.
    """

    normalized = question.lower().strip()

    if not normalized:
        return False

    return any(
        re.search(
            pattern,
            normalized,
        )
        for pattern in AMBIGUOUS_REFERENCE_PATTERNS
    )


def _build_history(
    messages: List[Dict[str, Any]],
) -> str:
    """
    Convert recent conversation messages into compact context.
    """

    recent = messages[
        -MAX_HISTORY_MESSAGES:
    ]

    lines = []

    for message in recent:
        role = str(
            message.get(
                "role",
                "",
            )
        ).strip().lower()

        content = str(
            message.get(
                "content",
                "",
            )
        ).strip()

        if not content:
            continue

        if role not in {
            "user",
            "assistant",
        }:
            continue

        lines.append(
            f"{role.upper()}: {content}"
        )

    return "\n".join(lines)


def _extract_json(
    text: str,
) -> Dict[str, Any]:
    """
    Extract the first valid JSON object from the resolver response.
    """

    text = str(
        text or ""
    ).strip()

    try:
        parsed = json.loads(
            text
        )

        if isinstance(
            parsed,
            dict,
        ):
            return parsed

    except json.JSONDecodeError:
        pass

    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL,
    )

    if not match:
        raise ValueError(
            "Resolver returned no JSON object."
        )

    parsed = json.loads(
        match.group(0)
    )

    if not isinstance(
        parsed,
        dict,
    ):
        raise ValueError(
            "Resolver JSON was not an object."
        )

    return parsed


# =========================================================
# Resolver Prompt
# =========================================================

RESOLVER_PROMPT = """
You are the conversation resolver for a college AI assistant.

Your job is ONLY to determine whether the latest user question
needs recent conversation context and, when necessary, rewrite it
into a self-contained retrieval question.

Rules:
1. Do not rewrite standalone questions.
2. Do not change an explicit new topic.
3. Preserve the active topic/entity for genuine follow-ups.
4. Preserve program boundaries such as B.Tech, M.Tech, M.Sc.,
   and Ph.D.
5. Preserve department boundaries.
6. Never invent missing facts.
7. If context is insufficient to resolve the follow-up,
   return the original question unchanged.
8. Never invent what vague words such as "that", "this", or "it"
   refer to unless the intended reference is genuinely clear.
9. Return JSON only.

Allowed mode values:
- standalone
- topic_switch
- follow_up
- ambiguous

Return exactly this JSON structure:

<JSON_SCHEMA>

Conversation:
<HISTORY>

Latest user question:
<QUESTION>
"""


# =========================================================
# Main Resolver
# =========================================================

def resolve_conversation(
    question: str,
    chat_history: List[Dict[str, Any]] | None = None,
) -> Dict[str, str]:
    """
    Resolve a user question using recent conversation only when
    the question appears context-dependent.
    """

    normalized_question = _normalize_question(
        question
    )

    history = chat_history or []

    # ---------------------------------------------------------
    # No history
    # ---------------------------------------------------------

    if not history:
        return {
            "mode": "standalone",
            "resolved_question": normalized_question,
            "active_topic": "",
            "active_entity": "",
        }

    # ---------------------------------------------------------
    # Clearly ambiguous reference
    # ---------------------------------------------------------

    if _is_obviously_ambiguous(
        normalized_question
    ):
        return {
            "mode": "ambiguous",
            "resolved_question": normalized_question,
            "active_topic": "",
            "active_entity": "",
        }

    # ---------------------------------------------------------
    # Clearly standalone / explicit topic switch
    # ---------------------------------------------------------

    if not _is_follow_up_like(
        normalized_question
    ):
        return {
            "mode": "topic_switch",
            "resolved_question": normalized_question,
            "active_topic": "",
            "active_entity": "",
        }

    history_text = _build_history(
        history
    )

    if not history_text:
        return {
            "mode": "ambiguous",
            "resolved_question": normalized_question,
            "active_topic": "",
            "active_entity": "",
        }

    # ---------------------------------------------------------
    # Prepare resolver prompt safely
    # ---------------------------------------------------------

    json_schema = """
{
  "mode": "standalone",
  "resolved_question": "original or resolved question",
  "active_topic": "",
  "active_entity": ""
}
""".strip()

    prompt = (
        RESOLVER_PROMPT
        .replace(
            "<JSON_SCHEMA>",
            json_schema,
        )
        .replace(
            "<HISTORY>",
            history_text,
        )
        .replace(
            "<QUESTION>",
            normalized_question,
        )
    )

    # ---------------------------------------------------------
    # Follow-up resolution
    # ---------------------------------------------------------

    try:
        response = invoke_query_llm(
            prompt
        )

        data = _extract_json(
            response.content
        )

        mode = str(
            data.get(
                "mode",
                "ambiguous",
            )
        ).strip().lower()

        if mode not in {
            "standalone",
            "topic_switch",
            "follow_up",
            "ambiguous",
        }:
            mode = "ambiguous"

        resolved_question = _normalize_question(
            data.get(
                "resolved_question",
                normalized_question,
            )
        )

        if not resolved_question:
            resolved_question = normalized_question

        # -----------------------------------------------------
        # Safety check after model resolution
        # -----------------------------------------------------

        # If the model itself still tries to interpret an
        # obviously vague reference, reject the rewrite.
        if _is_obviously_ambiguous(
            normalized_question
        ):
            return {
                "mode": "ambiguous",
                "resolved_question": normalized_question,
                "active_topic": "",
                "active_entity": "",
            }

        return {
            "mode": mode,
            "resolved_question": resolved_question,
            "active_topic": _normalize_question(
                data.get(
                    "active_topic",
                    "",
                )
            ),
            "active_entity": _normalize_question(
                data.get(
                    "active_entity",
                    "",
                )
            ),
        }

    except Exception:
        # Conservative fallback:
        # never invent context when resolution fails.
        return {
            "mode": "ambiguous",
            "resolved_question": normalized_question,
            "active_topic": "",
            "active_entity": "",
        }