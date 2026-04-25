import re
from collections.abc import Callable
from pathlib import Path
from typing import override

from kaos.path import KaosPath
from kosong.tooling import CallableTool2, ToolError, ToolReturnValue
from pydantic import BaseModel, Field

from kimi_cli.soul.agent import Runtime
from kimi_cli.soul.approval import Approval
from kimi_cli.tools.display import DisplayBlock
from kimi_cli.tools.file import FileActions
from kimi_cli.tools.file.plan_mode import inspect_plan_edit_target
from kimi_cli.tools.utils import load_desc
from kimi_cli.utils.diff import build_diff_blocks
from kimi_cli.utils.logging import logger
from kimi_cli.utils.path import is_within_workspace

_BASE_DESCRIPTION = load_desc(Path(__file__).parent / "replace.md")


class Edit(BaseModel):
    old: str = Field(description="The old string to replace. Can be multi-line.")
    new: str = Field(description="The new string to replace with. Can be multi-line.")
    replace_all: bool = Field(description="Whether to replace all occurrences.", default=False)


class Params(BaseModel):
    path: str = Field(
        description=(
            "The path to the file to edit. Absolute paths are required when editing files "
            "outside the working directory."
        )
    )
    edit: Edit | list[Edit] = Field(
        description=(
            "The edit(s) to apply to the file. "
            "You can provide a single edit or a list of edits here."
        )
    )


class StrReplaceFile(CallableTool2[Params]):
    name: str = "StrReplaceFile"
    description: str = _BASE_DESCRIPTION
    params: type[Params] = Params

    def __init__(self, runtime: Runtime, approval: Approval):
        super().__init__()
        self._work_dir = runtime.builtin_args.KIMI_WORK_DIR
        self._additional_dirs = runtime.additional_dirs
        self._approval = approval
        self._plan_mode_checker: Callable[[], bool] | None = None
        self._plan_file_path_getter: Callable[[], Path | None] | None = None

    def bind_plan_mode(
        self, checker: Callable[[], bool], path_getter: Callable[[], Path | None]
    ) -> None:
        """Bind plan mode state checker and plan file path getter."""
        self._plan_mode_checker = checker
        self._plan_file_path_getter = path_getter

    async def _validate_path(self, path: KaosPath) -> ToolError | None:
        """Validate that the path is safe to edit."""
        resolved_path = path.canonical()

        if (
            not is_within_workspace(resolved_path, self._work_dir, self._additional_dirs)
            and not path.is_absolute()
        ):
            return ToolError(
                message=(
                    f"`{path}` is not an absolute path. "
                    "You must provide an absolute path to edit a file "
                    "outside the working directory."
                ),
                brief="Invalid path",
            )
        return None

    def _apply_edit(self, content: str, edit: Edit) -> tuple[str, bool]:
        """Apply a single edit to the content.

        Returns (new_content, was_fuzzy) where was_fuzzy indicates whether
        whitespace-normalized fuzzy matching was used.
        """
        # Try exact match first
        if edit.old in content:
            if edit.replace_all:
                return content.replace(edit.old, edit.new), False
            else:
                return content.replace(edit.old, edit.new, 1), False

        # Fallback: whitespace-normalized fuzzy match
        fuzzy_old = _fuzzy_match_whitespace(content, edit.old)
        if fuzzy_old is not None:
            logger.info(
                "StrReplaceFile: using whitespace-normalized match for edit (exact match failed)"
            )
            if edit.replace_all:
                return content.replace(fuzzy_old, edit.new), True
            else:
                return content.replace(fuzzy_old, edit.new, 1), True

        return content, False

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        if not params.path:
            return ToolError(
                message="File path cannot be empty.",
                brief="Empty file path",
            )

        try:
            p = KaosPath(params.path).expanduser()
            if err := await self._validate_path(p):
                return err
            p = p.canonical()

            plan_target = inspect_plan_edit_target(
                p,
                plan_mode_checker=self._plan_mode_checker,
                plan_file_path_getter=self._plan_file_path_getter,
            )
            if isinstance(plan_target, ToolError):
                return plan_target

            is_plan_file_edit = plan_target.is_plan_target

            if not await p.exists():
                if is_plan_file_edit:
                    return ToolError(
                        message=(
                            "The current plan file does not exist yet. "
                            "Use WriteFile to create it before calling StrReplaceFile."
                        ),
                        brief="Plan file not created",
                    )
                return ToolError(
                    message=f"`{params.path}` does not exist.",
                    brief="File not found",
                )
            if not await p.is_file():
                return ToolError(
                    message=f"`{params.path}` is not a file.",
                    brief="Invalid path",
                )

            # Read the file content
            content = await p.read_text(errors="replace")

            original_content = content
            edits = [params.edit] if isinstance(params.edit, Edit) else params.edit

            # Apply all edits
            any_fuzzy = False
            for edit in edits:
                content, was_fuzzy = self._apply_edit(content, edit)
                any_fuzzy = any_fuzzy or was_fuzzy

            # Check if any changes were made
            if content == original_content:
                # Build a helpful error message with context
                hint = _build_no_match_hint(edits, original_content)
                return ToolError(
                    message=(
                        "No replacements were made. "
                        "The old string was not found in the file."
                        + (f"\n\n{hint}" if hint else "")
                    ),
                    brief="No replacements made",
                )

            diff_blocks: list[DisplayBlock] = await build_diff_blocks(
                str(p), original_content, content
            )

            action = (
                FileActions.EDIT
                if is_within_workspace(p, self._work_dir, self._additional_dirs)
                else FileActions.EDIT_OUTSIDE
            )

            # Plan file edits are auto-approved; all other edits need approval.
            if not is_plan_file_edit:
                result = await self._approval.request(
                    self.name,
                    action,
                    f"Edit file `{p}`",
                    display=diff_blocks,
                )
                if not result:
                    return result.rejection_error()

            # Write the modified content back to the file
            await p.write_text(content, errors="replace")

            # Count changes for success message
            total_replacements = 0
            for edit in edits:
                if edit.replace_all:
                    total_replacements += original_content.count(edit.old)
                else:
                    total_replacements += 1 if edit.old in original_content else 0

            fuzzy_note = " (used whitespace-normalized match)" if any_fuzzy else ""
            return ToolReturnValue(
                is_error=False,
                output="",
                message=(
                    f"File successfully edited{fuzzy_note}. "
                    f"Applied {len(edits)} edit(s) with {total_replacements} total replacement(s)."
                ),
                display=diff_blocks,
            )

        except Exception as e:
            logger.warning("StrReplaceFile failed: {path}: {error}", path=params.path, error=e)
            return ToolError(
                message=f"Failed to edit. Error: {e}",
                brief="Failed to edit file",
            )


def _normalize_whitespace(s: str) -> str:
    """Replace each run of horizontal whitespace (spaces/tabs) with a single space."""
    return re.sub(r"[ \t]+", " ", s)


def _fuzzy_match_whitespace(content: str, old: str) -> str | None:
    """Try to find old in content using whitespace-normalized matching.

    If found, return the actual substring from content (with original whitespace)
    so that the replacement preserves the file's indentation.
    Returns None if no match is found.
    """
    norm_old = _normalize_whitespace(old)
    if not norm_old.strip():
        return None

    norm_content = _normalize_whitespace(content)
    idx = norm_content.find(norm_old)
    if idx < 0:
        return None

    # Build a mapping from normalized positions to original positions.
    # Each normalized character maps to a range of original characters.
    segments: list[tuple[int, int]] = []  # segments[norm_idx] = (orig_start, orig_end)
    i = 0
    while i < len(content):
        if content[i] in (" ", "\t"):
            start = i
            while i < len(content) and content[i] in (" ", "\t"):
                i += 1
            segments.append((start, i))  # entire whitespace run -> 1 normalized char
        else:
            segments.append((i, i + 1))
            i += 1

    if idx >= len(segments):
        return None

    # Map normalized range [idx, idx+len(norm_old)) back to original range
    start_orig = segments[idx][0]
    end_norm = idx + len(norm_old)
    if end_norm > len(segments):
        return None
    end_orig = segments[end_norm - 1][1]

    actual = content[start_orig:end_orig]
    # Sanity check
    if _normalize_whitespace(actual) != norm_old:
        return None

    return actual


def _build_no_match_hint(edits: list[Edit], content: str) -> str:
    """Build a hint message when no match is found, showing nearby content."""
    lines = content.splitlines()
    hints: list[str] = []

    for edit in edits:
        old_lines = edit.old.splitlines()
        if not old_lines:
            continue

        # Try to find the first line of old in the file (with whitespace normalization)
        first_norm = _normalize_whitespace(old_lines[0]).strip()
        if not first_norm:
            continue

        for i, line in enumerate(lines):
            if first_norm in _normalize_whitespace(line).strip():
                # Found a potential match — show surrounding context
                ctx_start = max(0, i - 2)
                ctx_end = min(len(lines), i + len(old_lines) + 2)
                context_lines = lines[ctx_start:ctx_end]
                # Add line numbers
                numbered = [
                    f"  {ctx_start + j + 1}: {line!r}" for j, line in enumerate(context_lines)
                ]
                hints.append(
                    f"The old string might be near line {i + 1}. "
                    f"File content around that area:\n" + "\n".join(numbered)
                )
                break

        if not hints:
            # Show first few lines of old for debugging
            preview = old_lines[:3]
            old_preview = "\n".join(f"  {line!r}" for line in preview)
            hints.append(f"The old string to find (first {len(preview)} lines):\n{old_preview}")

    return "\n\n".join(hints)
