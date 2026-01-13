from omegaconf import DictConfig

from dreamer.envs.minigrid_wrapper import MiniGridFullObsWrapper
from dreamer.envs.dmc import DMCWrapper
from dreamer.envs.wrappers import UnscaleAction


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
        # Unscale the action from actor's [-1,1] range back
        # to environments range.
        train_env = UnscaleAction(
            DMCWrapper(
                task_name=cfg.env.task_name,
                image_res=cfg.env.image_res,
                seed=cfg.seed,
            )
        )
        eval_env = UnscaleAction(
            DMCWrapper(
                task_name=cfg.env.task_name,
                image_res=cfg.env.image_res,
                seed=cfg.seed,
                return_high_res_img=True,
            )
        )
    else:
        raise ValueError("Unhandled environment in config")
    return train_env, eval_env


def configure_buffer(cfg: DictConfig):
    pass


def configure_logger(cfg: DictConfig):
    pass


def configure_wm(cfg: DictConfig):
    pass


def configure_behaviour(cfg: DictConfig):
    pass
