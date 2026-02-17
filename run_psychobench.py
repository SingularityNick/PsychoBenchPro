import hydra
from omegaconf import DictConfig

from utils import run_psychobench
from example_generator import example_generator


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_psychobench(cfg, example_generator)


if __name__ == "__main__":
    main()
