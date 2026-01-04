from pathlib import Path

from src.data_processing import DatasetPaths, load_processed_dataset


def test_dataset_loads():
    project_root = Path(__file__).resolve().parents[1]
    zip_path = project_root / "data" / "raw" / "heart+disease.zip"
    extracted_dir = project_root / "data" / "processed" / "uci_heart_extracted"

    # If the raw zip is missing (common on fresh machines/CI/AWS),
    # download it to keep tests reproducible without committing raw data.
    if not zip_path.exists():
        import subprocess
        subprocess.check_call(["python", "scripts/download_data.py"])

    df = load_processed_dataset(
        DatasetPaths(zip_path=zip_path, processed_dir=extracted_dir),
        which="cleveland",
    )
    assert len(df) > 0
