#!/usr/bin/env python3
"""Read-only source/configuration checks; does not import ROS, models, or cameras."""
import ast
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", "local", "__pycache__", "build", "install", "log", ".venv"}


def main():
    counts = {"Python": 0, "YAML": 0, "XML": 0}
    errors = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or SKIP.intersection(path.relative_to(ROOT).parts):
            continue
        try:
            if path.suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path), feature_version=(3, 10))
                counts["Python"] += 1
            elif path.suffix in {".yaml", ".yml"}:
                yaml.safe_load(path.read_text(encoding="utf-8"))
                counts["YAML"] += 1
            elif path.name == "package.xml":
                ET.parse(path)
                counts["XML"] += 1
        except (SyntaxError, ValueError, yaml.YAMLError, ET.ParseError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    for error in errors:
        print(error)
    print(", ".join(f"{name}: {count}" for name, count in counts.items()))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
