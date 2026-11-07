"""Stitch every question's answer.md into report/answers.md, with each figure copied next to it.

    python -m experiments.yuqi_questions.report.build        # from tianq (no Hub access)

The questions come in the README's table order. A figure `![...](path.png)` in a question's answer.md (a name in
its folder, or a relative path such as a figure from another experiment) is copied to report/<folder>__<name>.png
and the link rewritten, so the report renders from this folder alone.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]+\.png)\)")


def folders() -> list[str]:
    """The question folders, in the README's table order."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    return re.findall(r"\]\(([a-z_]+)/\)", readme)


def main() -> None:
    parts = [(HERE / "intro.md").read_text(encoding="utf-8").strip()]
    for folder in folders():
        answer = ROOT / folder / "answer.md"
        if not answer.exists():
            continue
        text = answer.read_text(encoding="utf-8").strip()

        def copied(match: re.Match) -> str:
            source = ROOT / folder / match.group(2)
            name = f"{folder}__{source.name}"
            shutil.copyfile(source, HERE / name)
            return f"![{match.group(1)}]({name})"

        parts.append(IMAGE.sub(copied, text))
    (HERE / "answers.md").write_text("\n\n---\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote answers.md from {len(parts) - 1} questions")


if __name__ == "__main__":
    main()
