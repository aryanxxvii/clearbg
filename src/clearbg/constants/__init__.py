from pathlib import Path

# Define the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent # Adjust as necessary based on your file structure

# Define the paths to your configuration and parameters files
CONFIG_FILE_PATH = PROJECT_ROOT / "config/config.yaml"
PARAMS_FILE_PATH = PROJECT_ROOT / "params.yaml"

print(CONFIG_FILE_PATH)