#!/usr/bin/env python3
"""
Scan Swift source files for public declarations missing documentation comments.
Use Claude API to generate doc comments, then patch the files in-place.
"""

import os
import re
import sys
import json
import urllib.request
import urllib.error

SOURCES_DIR = os.environ.get("SOURCES_DIR", "Sources")
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = os.environ.get("MODEL", "claude-sonnet-4-20250514")

# Matches public declarations (func, var, let, class, struct, enum, protocol, init, subscript, typealias)
DECL_PATTERN = re.compile(
    r'^(\s*)(public\s+(?:static\s+|final\s+|override\s+|class\s+)*'
    r'(?:func|var|let|class|struct|enum|protocol|init|subscript|typealias)\b.+)$',
    re.MULTILINE
)

# A doc comment is any line ending with /// (possibly with text) immediately before the declaration
DOC_COMMENT = re.compile(r'^\s*///')


def find_undocumented(filepath: str) -> list[dict]:
    """Return list of undocumented public declarations with line info."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    results = []
    for i, line in enumerate(lines):
        m = DECL_PATTERN.match(line)
        if not m:
            continue
        # Check if preceding non-blank line is a doc comment
        j = i - 1
        while j >= 0 and lines[j].strip() == "":
            j -= 1
        if j >= 0 and DOC_COMMENT.match(lines[j]):
            continue
        # Also skip if it's just a closing brace or similar
        indent = m.group(1)
        decl = m.group(2).strip()

        # Gather context: up to 15 lines from the declaration to understand the body
        context_end = min(i + 15, len(lines))
        context = "".join(lines[i:context_end])

        results.append({
            "line": i,
            "indent": indent,
            "decl": decl,
            "context": context,
        })
    return results


def generate_doc_comments(filepath: str, undocumented: list[dict]) -> dict[int, str]:
    """Call Claude to generate doc comments for undocumented declarations."""
    if not undocumented or not API_KEY:
        return {}

    # Build a prompt with all undocumented declarations from this file
    decl_list = ""
    for idx, item in enumerate(undocumented):
        decl_list += f"\n--- Declaration {idx} (line {item['line']}) ---\n"
        decl_list += f"```swift\n{item['context']}```\n"

    prompt = f"""You are a Swift documentation expert. Generate concise Swift doc comments (///) for each undocumented public declaration below from the file `{os.path.basename(filepath)}`.

Rules:
- Use /// style comments only
- Include a brief summary line
- Add /// - Parameter name: description for each parameter
- Add /// - Returns: description if the function returns a non-Void type
- Add /// - Throws: description if the function throws
- Keep descriptions concise but informative, inferring purpose from the name, types, and body context
- Do NOT include the declaration itself, only the doc comment lines
- Preserve the original indentation level provided

{decl_list}

Respond with a JSON array where each element has "index" (matching the declaration index) and "comment" (the full doc comment string with newlines, properly indented). Only JSON, no markdown fences."""

    body = json.dumps({
        "model": MODEL,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}]
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"  API error {e.code}: {e.read().decode()}", file=sys.stderr)
        return {}

    # Extract text content
    text = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            text += block["text"]

    # Parse JSON from response
    try:
        # Try to extract JSON if wrapped in markdown fences
        json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL)
        if json_match:
            items = json.loads(json_match.group(1))
        else:
            items = json.loads(text.strip())
    except json.JSONDecodeError:
        print(f"  Failed to parse API response for {filepath}", file=sys.stderr)
        print(f"  Response: {text[:500]}", file=sys.stderr)
        return {}

    result = {}
    for item in items:
        idx = item["index"]
        comment = item["comment"]
        if not comment.endswith("\n"):
            comment += "\n"
        line_no = undocumented[idx]["line"]
        result[line_no] = comment

    return result


def patch_file(filepath: str, doc_comments: dict[int, str]) -> int:
    """Insert doc comments into the file. Returns number of insertions."""
    if not doc_comments:
        return 0

    with open(filepath, "r") as f:
        lines = f.readlines()

    # Insert in reverse order so line numbers stay valid
    for line_no in sorted(doc_comments.keys(), reverse=True):
        comment = doc_comments[line_no]
        lines.insert(line_no, comment)

    with open(filepath, "w") as f:
        f.writelines(lines)

    return len(doc_comments)


def main():
    if not API_KEY:
        print("Error: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    total_patched = 0
    patched_files = []

    for root, dirs, files in os.walk(SOURCES_DIR):
        # Skip test directories
        if "Tests" in root:
            continue
        for fname in sorted(files):
            if not fname.endswith(".swift"):
                continue
            filepath = os.path.join(root, fname)
            undocumented = find_undocumented(filepath)
            if not undocumented:
                continue

            print(f"  {filepath}: {len(undocumented)} undocumented declarations")
            doc_comments = generate_doc_comments(filepath, undocumented)
            count = patch_file(filepath, doc_comments)
            if count > 0:
                total_patched += count
                patched_files.append(filepath)

    print(f"\nTotal: {total_patched} doc comments added across {len(patched_files)} files")

    # Write summary for the PR body
    if patched_files:
        summary = f"Added documentation for {total_patched} public declarations across {len(patched_files)} files:\n\n"
        for f in patched_files:
            summary += f"- `{f}`\n"
        with open("/tmp/doc_summary.md", "w") as f:
            f.write(summary)

    return 0 if total_patched > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
