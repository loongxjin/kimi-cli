from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from kosong.message import Message

from kimi_cli.soul.dynamic_injection import DynamicInjection, DynamicInjectionProvider

if TYPE_CHECKING:
    from kimi_cli.soul.kimisoul import KimiSoul

_YOLO_INJECTION_TYPE = "yolo_mode"

_YOLO_PROMPT = (
    "You are running in yolo mode. Tool approvals are auto-approved, but you can "
    "still ask the user questions via AskUserQuestion when you genuinely need their "
    "input.\n"
    "- You MAY call AskUserQuestion when the user's preference genuinely matters "
    "and cannot be reasonably inferred.\n"
    "- For EnterPlanMode / ExitPlanMode, they will be auto-approved. You can use "
    "them normally but expect no user feedback."
)


class YoloModeInjectionProvider(DynamicInjectionProvider):
    """Injects a one-time reminder when yolo mode is active."""

    def __init__(self) -> None:
        self._injected: bool = False

    async def get_injections(
        self,
        history: Sequence[Message],
        soul: KimiSoul,
    ) -> list[DynamicInjection]:
        if not soul.is_yolo:
            return []
        if self._injected:
            return []
        self._injected = True
        return [DynamicInjection(type=_YOLO_INJECTION_TYPE, content=_YOLO_PROMPT)]
