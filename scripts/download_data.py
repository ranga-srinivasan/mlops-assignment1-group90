"""
Download the UCI Heart Disease dataset zip into data/raw.

This script is used for reproducibility (local + CI + AWS).
It avoids committing raw dataset files into Git.
"""

from __future__ import annotations

from pathlib import Path
import urllib.request

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "heart+disease.zip"
    if zip_path.exists() and zip_path.stat().st_size > 0:
        print(f"Dataset already exists: {zip_path}")
        return

    print(f"Downloading dataset from: {UCI_ZIP_URL}")
    print(f"Saving to: {zip_path}")
    urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)
    print("Download complete.")


if __name__ == "__main__":
    main()
