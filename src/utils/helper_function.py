import yaml
import os

import random
import string

import logging
from pathlib import Path
par_dir = Path(__file__).parent.parent.parent

def get_params(config_name: str) -> dict:
    """
    Load a YAML config file from configs/<config_name>.yml,
    expand environment variables, and return a Python dict.
    """
    config_path = par_dir / "configs" / f"{config_name}.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw_text = f.read()

    # Expand environment variables like ${HOME}
    expanded_text = os.path.expandvars(raw_text)

    params = yaml.safe_load(expanded_text)

    if not isinstance(params, dict):
        raise ValueError(f"Invalid YAML structure in {config_path}")

    return params


ALPHABET = string.ascii_letters + string.digits

def nano_id(size: int = 8) -> str:
    return ''.join(random.choice(ALPHABET) for _ in range(size))



def setup_logging(level: int = logging.INFO) -> None:
    """
    Set up logging with console + file handlers.
    Log file is stored in logs/run.log.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / "run.log"

    # Prevent duplicate handlers in Jupyter / reruns
    if logging.getLogger().handlers:
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="a"),
        ],
    )

    logging.info("Logging initialized.")
