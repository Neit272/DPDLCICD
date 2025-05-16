import re
from typing import List

def slice_semantic_blocks(code: str, context: int = 2) -> List[str]:
    """
    Extract semantic slices from C/C++ code based on control structures and vulnerable APIs.
    """
    lines = code.splitlines()
    match_lines = set()

    # Patterns for control flow & vulnerable API
    triggers = [
        r"\bif\b", r"\belse\b", r"\bfor\b", r"\bwhile\b", r"\bswitch\b", r"\bcase\b",
        r"\bstrcpy\b", r"\bmemcpy\b", r"\bstrncpy\b", r"\bsprintf\b", r"\bgets\b",
        r"\bmalloc\b", r"\bcalloc\b", r"\brealloc\b", r"\bfree\b",
        r"\bsystem\b", r"\bpopen\b", r"\bfopen\b", r"\bfread\b", r"\bfwrite\b"
    ]
    pattern = re.compile("|".join(triggers))

    # Find trigger lines
    for i, line in enumerate(lines):
        if pattern.search(line):
            for j in range(max(0, i - context), min(len(lines), i + context + 1)):
                match_lines.add(j)

    # Merge into blocks
    match_lines = sorted(match_lines)
    blocks = []
    current_block = []
    prev_line = -2

    for idx in match_lines:
        if idx - prev_line > 1 and current_block:
            blocks.append("\n".join(current_block))
            current_block = []
        current_block.append(lines[idx])
        prev_line = idx

    if current_block:
        blocks.append("\n".join(current_block))

    return blocks
