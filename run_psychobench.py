import sys

import hydra
from loguru import logger
from omegaconf import DictConfig

from example_generator import example_generator
from utils import run_psychobench


def _configure_logging():
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    _configure_logging()
    run_psychobench(cfg, example_generator)


if __name__ == "__main__":
    main()
