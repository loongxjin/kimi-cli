from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

from kimi_cli.session_state import load_session_state, save_session_state
from kimi_cli.utils.logging import logger
from kimi_cli.utils.string import shorten

if TYPE_CHECKING:
    from kimi_cli.session import Session


def extract_first_turn_from_wire(session_dir: Path) -> tuple[str, str] | None:
    """Extract the first turn's user message and assistant response from wire.jsonl.

    Returns:
        tuple[str, str] | None: (user_message, assistant_response) or None if not found
    """
    wire_file = session_dir / "wire.jsonl"
    if not wire_file.exists():
        return None

    user_message: str | None = None
    assistant_response_parts: list[str] = []
    in_first_turn = False

    try:
        with open(wire_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    message = record.get("message", {})
                    msg_type = message.get("type")

                    if msg_type == "TurnBegin":
                        if in_first_turn:
                            # Second turn started, stop
                            break
                        in_first_turn = True
                        user_input = message.get("payload", {}).get("user_input")
                        if user_input:
                            from kosong.message import Message

                            msg = Message(role="user", content=user_input)
                            user_message = msg.extract_text(" ")

                    elif msg_type == "ContentPart" and in_first_turn:
                        payload = message.get("payload", {})
                        if payload.get("type") == "text" and payload.get("text"):
                            assistant_response_parts.append(payload["text"])

                    elif msg_type == "TurnEnd" and in_first_turn:
                        break

                except json.JSONDecodeError:
                    continue
    except OSError:
        return None

    if user_message and assistant_response_parts:
        return (user_message, "".join(assistant_response_parts))
    return None


async def _poll_first_turn(session_dir: Path) -> tuple[str, str] | None:
    """Wait briefly for the wire recorder to flush, then extract the first turn."""
    first_turn = None
    for _ in range(5):
        await asyncio.sleep(0.05)
        first_turn = extract_first_turn_from_wire(session_dir)
        if first_turn:
            break
    return first_turn


def _persist_title_result(
    session_dir: Path,
    session: Session,
    ai_title: str | None,
) -> None:
    """Read-modify-write session state after attempting title generation."""
    fresh = load_session_state(session_dir)
    if ai_title:
        fresh.custom_title = ai_title
        fresh.title_generated = True
    else:
        fresh.title_generate_attempts = fresh.title_generate_attempts + 1
    save_session_state(fresh, session_dir)

    session.state.custom_title = fresh.custom_title
    if ai_title:
        session.state.title_generated = True
    else:
        session.state.title_generate_attempts = fresh.title_generate_attempts


async def generate_title_with_llm(
    user_message: str,
    assistant_response: str | None,
) -> str | None:
    """Generate a session title using the configured default LLM.

    Returns the generated title or None if generation failed.
    """
    try:
        from kosong import generate
        from kosong.message import Message

        from kimi_cli.auth.oauth import OAuthManager
        from kimi_cli.config import load_config
        from kimi_cli.llm import create_llm

        config = load_config()
        model_name = config.default_model

        if not model_name or model_name not in config.models:
            return None

        model_config = config.models[model_name]
        provider_config = config.providers.get(model_config.provider)

        if not provider_config:
            return None

        oauth = OAuthManager(config)
        await oauth.ensure_fresh()
        llm = create_llm(provider_config, model_config, oauth=oauth)

        if not llm:
            return None

        system_prompt = (
            "Generate a concise session title (max 50 characters) "
            "based on the conversation. "
            "Only respond with the title text, nothing else. "
            "No quotes, no explanation."
        )

        prompt = f"""User: {user_message[:300]}
Assistant: {(assistant_response or "")[:300]}

Title:"""

        result = await generate(
            chat_provider=llm.chat_provider,
            system_prompt=system_prompt,
            tools=[],
            history=[Message(role="user", content=prompt)],
        )

        generated_title = result.message.extract_text().strip()
        generated_title = generated_title.strip("\"'")

        if generated_title:
            if len(generated_title) <= 50:
                return generated_title
            return shorten(generated_title, width=50)

    except Exception as e:
        logger.warning("Failed to generate title using AI: {error}", error=e)

    return None


async def improve_session_title(
    session: Session,
    user_input: str | None = None,
) -> str | None:
    """Try to improve an existing fallback session title using the LLM.

    This is designed to be run as a background task after the fallback title
    has already been saved synchronously.

    Returns the improved title if LLM succeeds, otherwise None.
    """
    session_dir = session.dir

    first_turn = await _poll_first_turn(session_dir)
    if first_turn:
        user_message, assistant_response = first_turn
    elif user_input:
        user_message = user_input
        assistant_response = None
    else:
        return None

    user_text = " ".join(user_message.strip().split())
    if not user_text:
        return None

    state = load_session_state(session_dir)
    if state.title_generate_attempts >= 3 or state.title_generated:
        return None

    ai_title = await generate_title_with_llm(user_message, assistant_response)
    _persist_title_result(session_dir, session, ai_title)
    return ai_title
