from pathlib import Path
import re


def test_readme_cli_examples_are_syntactically_valid():
    readme = Path(__file__).resolve().parents[1] / "README.md"
    content = readme.read_text(encoding="utf-8")

    expected_commands = [
        "roof-area infer --threshold 0.6 --tile-size 512",
        "roof-area eval --min-area-m2 10",
        "roof-area train --seed 123",
    ]

    for command in expected_commands:
        assert re.search(re.escape(command), content), f"Missing CLI example: {command}"

    for command in expected_commands:
        assert re.fullmatch(r"roof-area \w+(?: [\w.-]+)+", command), (
            f"CLI example does not match expected syntax pattern: {command}"
        )
