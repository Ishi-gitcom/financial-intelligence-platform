from pathlib import Path

# ==========================
# PROJECT PATHS
# ==========================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "Data" / "Raw" / "Stocks"

MODEL_DIR = BASE_DIR / "Models"

OUTPUT_DIR = BASE_DIR / "Output"

RESULTS_DIR = BASE_DIR / "results"

# ==========================
# CREATE FOLDERS
# ==========================

MODEL_DIR.mkdir(exist_ok=True)

OUTPUT_DIR.mkdir(exist_ok=True)

RESULTS_DIR.mkdir(exist_ok=True)