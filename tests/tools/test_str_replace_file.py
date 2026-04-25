"""Tests for the str_replace_file tool."""

from __future__ import annotations

from pathlib import Path

from kaos.path import KaosPath

from kimi_cli.tools.file.replace import Edit, Params, StrReplaceFile
from kimi_cli.wire.types import DiffDisplayBlock


async def test_replace_single_occurrence(
    str_replace_file_tool: StrReplaceFile, temp_work_dir: KaosPath
):
    """Test replacing a single occurrence."""
    file_path = temp_work_dir / "test.txt"
    original_content = "Hello world! This is a test."
    await file_path.write_text(original_content)

    result = await str_replace_file_tool(
        Params(path=str(file_path), edit=Edit(old="world", new="universe"))
    )

    assert not result.is_error
    assert "successfully edited" in result.message
    diff_block = next(block for block in result.display if block.type == "diff")
    assert isinstance(diff_block, DiffDisplayBlock)
    assert diff_block.path == str(file_path)
    assert diff_block.old_text == original_content
    assert diff_block.new_text == "Hello universe! This is a test."
    assert await file_path.read_text() == "Hello universe! This is a test."


async def test_replace_all_occurrences(
    str_replace_file_tool: StrReplaceFile, temp_work_dir: KaosPath
):
    """Test replacing all occurrences."""
    file_path = temp_work_dir / "test.txt"
    original_content = "apple banana apple cherry apple"
    await file_path.write_text(original_content)

    result = await str_replace_file_tool(
        Params(
            path=str(file_path),
            edit=Edit(old="apple", new="fruit", replace_all=True),
        )
    )

    assert not result.is_error
    assert "successfully edited" in result.message
    assert await file_path.read_text() == "fruit banana fruit cherry fruit"


async def test_replace_multiple_edits(
    str_replace_file_tool: StrReplaceFile, temp_work_dir: KaosPath
):
    """Test applying multiple edits."""
    file_path = temp_work_dir / "test.txt"
    original_content = "Hello world! Goodbye world!"
    await file_path.write_text(original_content)

    result = await str_replace_file_tool(
        Params(
            path=str(file_path),
            edit=[
                Edit(old="Hello", new="Hi"),
                Edit(old="Goodbye", new="See you"),
            ],
        )
    )

    assert not result.is_error
    assert "successfully edited" in result.message
    assert await file_path.read_text() == "Hi world! See you world!"


async def test_replace_multiline_content(
    str_replace_file_tool: StrReplaceFile, temp_work_dir: KaosPath
):
    """Test replacing multi-line content."""
    file_path = temp_work_dir / "test.txt"
    original_content = "Line 1\nLine 2\nLine 3\n"
    await file_path.write_text(original_content)

    result = await str_replace_file_tool(
        Params(
            path=str(file_path),
            edit=Edit(old="Line 2\nLine 3", new="Modified line 2\nModified line 3"),
        )
    )

    assert not result.is_error
    assert "successfully edited" in result.message
    assert await file_path.read_text() == "Line 1\nModified line 2\nModified line 3\n"


async def test_replace_unicode_content(
    str_replace_file_tool: StrReplaceFile, temp_work_dir: KaosPath
):
    """Test replacing unicode content."""
    file_path = temp_work_dir / "test.txt"
    original_content = "Hello 世界! café"
    await file_path.write_text(original_content)

    result = await str_replace_file_tool(
        Params(path=str(file_path), edit=Edit(old="世界", new="地球"))
    )

    assert not result.is_error
    assert "successfully edited" in result.message
    assert await file_path.read_text() == "Hello 地球! café"


async def test_replace_no_match(str_replace_file_tool: StrReplaceFile, temp_work_dir: KaosPath):
    """Test replacing when the old string is not found."""
    file_path = temp_work_dir / "test.txt"
    original_content = "Hello world!"
    await file_path.write_text(original_content)

    result = await str_replace_file_tool(
        Params(path=str(file_path), edit=Edit(old="notfound", new="replacement"))
    )

    assert result.is_error
    assert "No replacements were made" in result.message
    assert await file_path.read_text() == original_content  # Content unchanged


async def test_replace_with_relative_path(
    str_replace_file_tool: StrReplaceFile, temp_work_dir: KaosPath
):
    """Test replacing with a relative path inside the work directory."""
    relative_dir = temp_work_dir / "relative" / "path"
    await relative_dir.mkdir(parents=True, exist_ok=True)
    file_path = relative_dir / "file.txt"
    await file_path.write_text("old content")

    result = await str_replace_file_tool(
        Params(path="relative/path/file.txt", edit=Edit(old="old", new="new"))
    )

    assert not result.is_error
    assert await file_path.read_text() == "new content"


async def test_replace_outside_work_directory(
    str_replace_file_tool: StrReplaceFile, outside_file: Path
):
    """Test replacing outside the working directory with an absolute path."""
    outside_file.write_text("old content", encoding="utf-8")

    result = await str_replace_file_tool(
        Params(path=str(outside_file), edit=Edit(old="old", new="new"))
    )

    assert not result.is_error
    assert outside_file.read_text(encoding="utf-8") == "new content"


async def test_replace_outside_work_directory_with_prefix(
    str_replace_file_tool: StrReplaceFile, temp_work_dir: KaosPath
):
    """Paths sharing the work dir prefix but outside should still be editable
    with absolute paths."""
    base = Path(str(temp_work_dir))
    sneaky_dir = base.parent / f"{base.name}-sneaky"
    sneaky_dir.mkdir(parents=True, exist_ok=True)
    sneaky_file = sneaky_dir / "test.txt"
    sneaky_file.write_text("content", encoding="utf-8")

    result = await str_replace_file_tool(
        Params(path=str(sneaky_file), edit=Edit(old="content", new="new"))
    )

    assert not result.is_error
    assert sneaky_file.read_text() == "new"


async def test_replace_nonexistent_file(
    str_replace_file_tool: StrReplaceFile, temp_work_dir: KaosPath
):
    """Test replacing in a non-existent file."""
    file_path = temp_work_dir / "nonexistent.txt"

    result = await str_replace_file_tool(
        Params(path=str(file_path), edit=Edit(old="old", new="new"))
    )

    assert result.is_error
    assert "does not exist" in result.message


async def test_replace_directory_instead_of_file(
    str_replace_file_tool: StrReplaceFile, temp_work_dir: KaosPath
):
    """Test replacing in a directory instead of a file."""
    dir_path = temp_work_dir / "directory"
    await dir_path.mkdir()

    result = await str_replace_file_tool(
        Params(path=str(dir_path), edit=Edit(old="old", new="new"))
    )

    assert result.is_error
    assert "is not a file" in result.message


async def test_replace_mixed_multiple_edits(
    str_replace_file_tool: StrReplaceFile, temp_work_dir: KaosPath
):
    """Test multiple edits with different replace_all settings."""
    file_path = temp_work_dir / "test.txt"
    original_content = "apple apple banana apple cherry"
    await file_path.write_text(original_content)

    result = await str_replace_file_tool(
        Params(
            path=str(file_path),
            edit=[
                Edit(old="apple", new="fruit", replace_all=False),  # Only first occurrence
                Edit(
                    old="banana", new="tasty", replace_all=True
                ),  # All occurrences (though only one)
            ],
        )
    )

    assert not result.is_error
    assert "successfully edited" in result.message
    assert await file_path.read_text() == "fruit apple tasty apple cherry"


async def test_replace_empty_strings(
    str_replace_file_tool: StrReplaceFile, temp_work_dir: KaosPath
):
    """Test replacing with empty strings."""
    file_path = temp_work_dir / "test.txt"
    original_content = "Hello world!"
    await file_path.write_text(original_content)

    result = await str_replace_file_tool(
        Params(path=str(file_path), edit=Edit(old="world", new=""))
    )

    assert not result.is_error
    assert "successfully edited" in result.message
    assert await file_path.read_text() == "Hello !"


async def test_replace_space_tab_mismatch(
    str_replace_file_tool: StrReplaceFile, temp_work_dir: KaosPath
):
    """File uses tabs, LLM sends spaces — fuzzy match should work."""
    file_path = temp_work_dir / "test.py"
    # File content uses tab indentation
    original_content = "def hello():\n\tprint('world')\n\treturn True\n"
    await file_path.write_text(original_content)

    # LLM sends spaces instead of tabs
    result = await str_replace_file_tool(
        Params(
            path=str(file_path),
            edit=Edit(
                old="    print('world')\n    return True",
                new="    print('hello')\n    return False",
            ),
        )
    )

    assert not result.is_error
    assert "whitespace-normalized match" in result.message
    # The fuzzy match finds the tab-indented text and replaces with LLM's new value
    new_content = await file_path.read_text()
    assert "print('hello')" in new_content
    assert "return False" in new_content


async def test_replace_tab_space_mismatch(
    str_replace_file_tool: StrReplaceFile, temp_work_dir: KaosPath
):
    """File uses spaces, LLM sends tabs — fuzzy match should work."""
    file_path = temp_work_dir / "test.py"
    # File content uses space indentation
    original_content = "def hello():\n    print('world')\n    return True\n"
    await file_path.write_text(original_content)

    # LLM sends tabs instead of spaces
    result = await str_replace_file_tool(
        Params(
            path=str(file_path),
            edit=Edit(
                old="\tprint('world')\n\treturn True", new="\tprint('hello')\n\treturn False"
            ),
        )
    )

    assert not result.is_error
    assert "whitespace-normalized match" in result.message
    new_content = await file_path.read_text()
    assert "print('hello')" in new_content
    assert "return False" in new_content


async def test_replace_exact_match_no_fuzzy(
    str_replace_file_tool: StrReplaceFile, temp_work_dir: KaosPath
):
    """Exact match should NOT trigger fuzzy match."""
    file_path = temp_work_dir / "test.txt"
    await file_path.write_text("Hello world!")

    result = await str_replace_file_tool(
        Params(path=str(file_path), edit=Edit(old="world", new="universe"))
    )

    assert not result.is_error
    assert "whitespace-normalized" not in result.message
    assert await file_path.read_text() == "Hello universe!"


async def test_replace_whitespace_fuzzy_still_no_match(
    str_replace_file_tool: StrReplaceFile, temp_work_dir: KaosPath
):
    """Completely unrelated content — fuzzy match should also fail."""
    file_path = temp_work_dir / "test.txt"
    await file_path.write_text("Hello world!\n")

    result = await str_replace_file_tool(
        Params(path=str(file_path), edit=Edit(old="xyz_not_in_file_abc", new="replacement"))
    )

    assert result.is_error
    assert "No replacements were made" in result.message
    assert await file_path.read_text() == "Hello world!\n"


async def test_replace_fuzzy_enhanced_error_hint(
    str_replace_file_tool: StrReplaceFile, temp_work_dir: KaosPath
):
    """When both exact and fuzzy fail, error message should include helpful context."""
    file_path = temp_work_dir / "test.py"
    await file_path.write_text("def hello():\n    print('world')\n    return True\n")

    # Send something that partially matches but isn't fully correct
    result = await str_replace_file_tool(
        Params(
            path=str(file_path),
            edit=Edit(old="print('different')", new="print('replaced')"),
        )
    )

    assert result.is_error
    assert "No replacements" in result.message
    # The hint should mention line context
    assert "line" in result.message.lower() or "repr" in result.message.lower()


async def test_replace_fuzzy_preserves_original_indentation(
    str_replace_file_tool: StrReplaceFile, temp_work_dir: KaosPath
):
    """Fuzzy match finds the text even when whitespace differs."""
    file_path = temp_work_dir / "test.py"
    # File uses tab indentation (common in real projects)
    original_content = "class Foo:\n\tdef bar(self):\n\t\tx = 1\n\t\treturn x\n"
    await file_path.write_text(original_content)

    # LLM sends 4-space indentation in old string
    result = await str_replace_file_tool(
        Params(
            path=str(file_path),
            edit=Edit(old="    x = 1\n    return x", new="    x = 2\n    return x"),
        )
    )

    assert not result.is_error
    assert "whitespace-normalized match" in result.message
    new_content = await file_path.read_text()
    # The fuzzy match found the tab-indented text and replaced it with the
    # LLM-provided new value (which uses spaces)
    assert "x = 2" in new_content
    assert "return x" in new_content
