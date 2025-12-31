from pathlib import Path
from src.data_processing import DatasetPaths, load_processed_dataset

def test_dataset_loads():
    project_root = Path(__file__).resolve().parents[1]
    zip_path = project_root / "data" / "raw" / "heart+disease.zip"
    extracted_dir = project_root / "data" / "processed" / "uci_heart_extracted"

    df = load_processed_dataset(DatasetPaths(zip_path=zip_path, processed_dir=extracted_dir), which="cleveland")
    assert "target" in df.columns
    assert df.shape[0] > 0
