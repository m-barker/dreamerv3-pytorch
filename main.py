import hydra
import os
from omegaconf import DictConfig, OmegaConf
from dreamer.dreamer import Dreamer


@hydra.main(
    version_base=None,
    config_path=os.path.join("dreamer", "config"),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    dreamer = Dreamer(cfg)
    dreamer.train()


if __name__ == "__main__":
    main()
