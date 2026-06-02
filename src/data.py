"""Recursive tensor-size survey for a dataset directory.

Walks DATA_PATH, loads every supported image, prints its tensor shape, and writes a
shapes table (path, depth, channels, height, width) to REPORT_PATH for analysis.
"""
from pathlib import Path

import pandas as pd

from main import ImageLoader, _is_supported

DATA_PATH = Path("data/")
REPORT_PATH = Path("report") / "tensor_sizes.csv"


def analyze(data_path: Path, report_path: Path) -> pd.DataFrame:
    rows: list[dict] = []

    if not data_path.exists():
        print(f"Path does not exist: {data_path}")
        return pd.DataFrame(columns=["path", "depth", "channels", "height", "width"])

    paths = [data_path] if data_path.is_file() else sorted(data_path.rglob("*"))

    for p in paths:
        if not (p.is_file() and _is_supported(p)):
            continue
        try:
            loader = ImageLoader(p)
            shape = loader.log_tensor_shape()
            d, c, h, w = (tuple(shape) + (None,) * 4)[:4]
            rows.append({"path": str(p), "depth": d, "channels": c, "height": h, "width": w})
        except Exception as exc:
            print(f"[{p}] load failed: {exc}")

    df = pd.DataFrame(rows, columns=["path", "depth", "channels", "height", "width"])

    if not df.empty:
        print(f"\n{len(df)} images")
        print(f"unique (D,1,H,W) shapes: "
              f"{df[['depth', 'channels', 'height', 'width']].drop_duplicates().shape[0]}")
        print(df[["depth", "height", "width"]].describe())

    report_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(report_path, index=False)
    print(f"\nShapes table written: {report_path}")
    return df


def main():
    analyze(DATA_PATH, REPORT_PATH)


if "__main__" == __name__:
    main()
