import sys

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from example_generator import example_generator

# Custom resolver: join a list into a comma-separated string for Hydra sweeper params.
# Usage: model: ${join:${models}}
OmegaConf.register_new_resolver(
    "join",
    lambda seq: ",".join(str(x) for x in (seq or [])),
)
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
