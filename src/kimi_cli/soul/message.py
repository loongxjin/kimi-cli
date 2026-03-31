from __future__ import annotations

from collections.abc import Sequence

from kosong.message import Message
from kosong.tooling.error import ToolRuntimeError

from kimi_cli.llm import ModelCapability
from kimi_cli.wire.types import (
    ContentPart,
    ImageURLPart,
    TextPart,
    ThinkPart,
    ToolResult,
    VideoURLPart,
)


def system(message: str) -> ContentPart:
    return TextPart(text=f"<system>{message}</system>")


def system_reminder(message: str) -> TextPart:
    return TextPart(text=f"<system-reminder>\n{message}\n</system-reminder>")


def is_system_reminder_message(message: Message) -> bool:
    """Check whether a message is an internal system-reminder user message."""
    if message.role != "user" or len(message.content) != 1:
        return False
    part = message.content[0]
    return isinstance(part, TextPart) and part.text.strip().startswith("<system-reminder>")


def tool_result_to_message(tool_result: ToolResult) -> Message:
    """Convert a tool result to a message.

    When the output is plain text (str), the system message and output are
    merged into a single TextPart so that the serialized content is a plain
    string.  Many OpenAI-compatible models expect tool-role message content to
    be a string and fail to parse the list-of-dicts format that kosong produces
    when multiple TextParts are present.
    """
    if tool_result.return_value.is_error:
        assert tool_result.return_value.message, "Error return value should have a message"
        message = tool_result.return_value.message
        if isinstance(tool_result.return_value, ToolRuntimeError):
            message += "\nThis is an unexpected error and the tool is probably not working."
        content: list[ContentPart] = [system(f"ERROR: {message}")]
        if tool_result.return_value.output:
            content.extend(_output_to_content_parts(tool_result.return_value.output))
    else:
        content: list[ContentPart] = []
        msg = tool_result.return_value.message
        output = tool_result.return_value.output
        if output and isinstance(output, str) and output:
            # Merge system message and text output into one TextPart so
            # _serialize_content produces a plain str (not a list[dict]).
            parts: list[str] = []
            if msg:
                parts.append(f"<system>{msg}</system>")
            parts.append(output)
            content.append(TextPart(text="\n".join(parts)))
        else:
            if msg:
                content.append(system(msg))
            if output:
                content.extend(_output_to_content_parts(output))
        if not content:
            content.append(system("Tool output is empty."))
        elif not any(isinstance(part, TextPart) for part in content):
            # Ensure at least one TextPart exists so the LLM API won't reject
            # the message with "text content is empty" (see #1663).
            content.insert(0, system("Tool returned non-text content."))

    return Message(
        role="tool",
        content=content,
        tool_call_id=tool_result.tool_call_id,
    )


def _output_to_content_parts(
    output: str | ContentPart | Sequence[ContentPart],
) -> list[ContentPart]:
    content: list[ContentPart] = []
    match output:
        case str(text):
            if text:
                content.append(TextPart(text=text))
        case ContentPart():
            content.append(output)
        case _:
            content.extend(output)
    return content


def check_message(
    message: Message, model_capabilities: set[ModelCapability]
) -> set[ModelCapability]:
    """Check the message content, return the missing model capabilities."""
    capabilities_needed = set[ModelCapability]()
    for part in message.content:
        if isinstance(part, ImageURLPart):
            capabilities_needed.add("image_in")
        elif isinstance(part, VideoURLPart):
            capabilities_needed.add("video_in")
        elif isinstance(part, ThinkPart):
            capabilities_needed.add("thinking")
    return capabilities_needed - model_capabilities
