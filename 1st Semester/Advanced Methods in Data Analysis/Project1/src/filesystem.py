from pathlib import Path


class FileSystem:
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    DATASETS_DIR: Path = PROJECT_ROOT / "datasets"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
