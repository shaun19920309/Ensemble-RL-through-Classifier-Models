from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a checksum manifest for the packaged portfolio datasets."
    )
    parser.add_argument("--data-root", default="external_data")
    parser.add_argument(
        "--output", default="results/data_quality/data_file_manifest.csv"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.data_root).resolve()
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == ".DS_Store":
            continue
        rows.append(
            {
                "path": path.relative_to(root.parent).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} entries to {output}")


if __name__ == "__main__":
    main()
