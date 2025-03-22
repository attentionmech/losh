import os
import sys
from src.flows.backend_flow import backend_workflow
from src.utils.logger_setup import setup_logger


# TODO: find better solution
os.environ["WANDB_DISABLED"] = "true"

logger = setup_logger()

if __name__ == "__main__":
    with open("banner.txt") as f:
        print(f.read())
        print("\n")
    if len(sys.argv) != 2:
        logger.error("Usage: python main.py <config_file>")
        sys.exit(1)
    config_file = sys.argv[1]
    backend_workflow(config_file)