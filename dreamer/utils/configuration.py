from omegaconf import DictConfig

from dreamer.envs.minigrid_wrapper import MiniGridFullObsWrapper
from dreamer.envs.dmc import DMCWrapper
from dreamer.envs.atari import AtariWrapper
from dreamer.envs.crafter import CrafterWrapper


def configure_environments(cfg: DictConfig):
    if cfg.env.suite_name == "minigrid":
        train_env = MiniGridFullObsWrapper(
            task_name=cfg.env.task_name,
            seed=cfg.seed,
            max_steps=cfg.env.max_steps,
            image_res=cfg.env.image_res,
        )
        eval_env = MiniGridFullObsWrapper(
            task_name=cfg.env.task_name,
            seed=cfg.seed,
            max_steps=cfg.env.max_steps,
            image_res=cfg.env.image_res,
        )
    elif cfg.env.suite_name == "dmc":
        train_env = DMCWrapper(
            task_name=cfg.env.task_name,
            image_res=cfg.env.image_res,
            seed=cfg.seed,
        )
        eval_env = DMCWrapper(
            task_name=cfg.env.task_name,
            image_res=cfg.env.image_res,
            seed=cfg.seed,
            return_high_res_img=True,
        )
    elif cfg.env.suite_name == "atari":
        train_env = AtariWrapper(
            task_name=cfg.env.task_name,
            image_res=cfg.env.image_res,
            seed=cfg.seed,
        )
        eval_env = AtariWrapper(
            task_name=cfg.env.task_name,
            image_res=cfg.env.image_res,
            seed=cfg.seed,
            return_high_res_image=True,
        )
    elif cfg.env.suite_name == "crafter":
        train_env = CrafterWrapper(
            task=cfg.env.task_name,
            image_res=cfg.env.image_res,
            seed=cfg.seed,
        )
        eval_env = CrafterWrapper(
            task=cfg.env.task_name,
            image_res=cfg.env.image_res,
            seed=cfg.seed,
        )
    else:
        raise ValueError("Unhandled environment in config")
    return train_env, eval_env
