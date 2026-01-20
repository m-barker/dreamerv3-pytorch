from typing import Tuple, List
import torch
from omegaconf import DictConfig, open_dict

from dreamer.envs.minigrid_wrapper import MiniGridFullObsWrapper
from dreamer.envs.dmc import DMCWrapper
from dreamer.envs.atari import AtariWrapper
from dreamer.envs.crafter import CrafterWrapper

from dreamer.networks.actor import ActorParams
from dreamer.networks.critic import CriticParams
from dreamer.networks.decoder import DecoderCNNParams, DecoderParams
from dreamer.networks.encoder import CNNParams, EncoderParams
from dreamer.networks.rssm import RSSMParams
from dreamer.networks.shared import (
    BernouliDistParams,
    BoundedNormalParams,
    MLPParams,
    MLPandHeadParams,
    OneHotParams,
    TwoHotDistParams,
)
from dreamer.utils.optimiser import SimpleDreamerOptimizer
from dreamer.world_model import WorldModel, WorldModelTrainingParams
from dreamer.behaviour import Behaviour, BehaviourTrainingParams
from dreamer.utils.replay import Buffer


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
            sticky=cfg.env.sticky,
        )
        eval_env = AtariWrapper(
            task_name=cfg.env.task_name,
            image_res=cfg.env.image_res,
            seed=cfg.seed,
            return_high_res_image=True,
            sticky=cfg.env.sticky,
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


def configure_world_model(
    config: DictConfig,
    action_dim: int,
    n_actions: int,
    device: torch.device,
    env_observation_space,
) -> WorldModel:
    """
    Creates and returns a WorldModel based on the provided configuration.

    NOTE: this does NOT load any weights from a checkpoint. This must be done elsewhere if desired.

    Args:
        config (DictcConfig): Hydra-parsed configuration.

        action_dim (int): dimensionality of each action.

        n_actions (int): number of action dimensions.

        device (torch.device): device to load the world model on.

        env_observation_space (Dict): the obs space of the environment the world
        model is to be trained on.
    """

    training_params = WorldModelTrainingParams(**config.world_model_training)
    image_keys = config.env.image_keys
    image_shapes = []
    vector_keys = config.env.vector_keys
    vector_shapes = []

    obs_space = env_observation_space
    for k in image_keys:
        if k in obs_space.keys():
            image_shapes.append(obs_space[k].shape)
        else:
            raise ValueError(
                f"Required image key: {k} not found in environment observation space"
            )

    if vector_keys:
        for k in vector_keys:
            if k in obs_space.keys():
                vector_shapes.append(obs_space[k].shape)
            else:
                raise ValueError(
                    f"Required vector key: {k} not found in environment observation space"
                )

    cnn_encoder_params = CNNParams(**config.cnn_encoder)
    encoder_params = EncoderParams(
        image_keys=image_keys,
        image_shapes=image_shapes,
        cnn_params=cnn_encoder_params,
    )
    if vector_keys:
        encoder_params.vector_keys = vector_keys
        encoder_params.vector_shapes = vector_shapes
        encoder_params.mlp_params = MLPParams(**config.mlp_encoder)

    # Each vector is embedded as hidden_size dimensions
    encoded_dim = config.encoder_embed_dim + (len(vector_keys) * config.hidden_size)

    cnn_decoder_params = DecoderCNNParams(**config.cnn_decoder)
    decoder_params = DecoderParams(
        image_keys=image_keys,
        image_shapes=image_shapes,
        cnn_params=cnn_decoder_params,
    )
    if vector_keys:
        decoder_params.vector_keys = vector_keys
        decoder_params.vector_shapes = vector_shapes
        decoder_params.mlp_params = MLPParams(**config.mlp_decoder)

    rssm_params = config.rssm
    with open_dict(rssm_params):
        # We want the flattened action dimensionality
        rssm_params.action_dim = action_dim * n_actions
    rssm_params = RSSMParams(**rssm_params)
    rssm_params.compile = config.compile
    rssm_params.encoded_size = encoded_dim
    rssm_params.device = device

    reward_network_params = MLPParams(**config.reward_mlp)
    reward_head_params = TwoHotDistParams(**config.reward_head)

    continue_network_params = MLPParams(**config.continue_mlp)
    continue_head_params = BernouliDistParams()

    world_model = WorldModel(
        training_params=training_params,
        rssm_params=rssm_params,
        encoder_params=encoder_params,
        decoder_params=decoder_params,
        grad_components=config.world_model_grad_components,
        reward_params=MLPandHeadParams(reward_network_params, reward_head_params),
        continue_params=MLPandHeadParams(continue_network_params, continue_head_params),
        device=device,
        compile=config.compile,
    )
    return world_model


def configure_behaviour(
    config: DictConfig,
    n_actions: int,
    action_dim: int,
    device: torch.device,
    discrete_actor: bool,
) -> Behaviour:
    """
    Creates and returns a Behaviour instance based on the provided config.

    NOTE: this does not load any model weights. This must be done separately.

    Args:
        config (DictConfig): Hydra-parsed configuarion.

        n_actions (int): number of possible actions.

        action_dim (int): the dimensionality of each action.

        device (torch.device): device to load the instance on.

        discrete_actor (bool): whether the actor's policy should be a discrete
        or continuous distribution.
    """

    actor_params = config.actor
    with open_dict(actor_params):
        actor_params.n_actions = n_actions
        actor_params.action_dim = action_dim

    if discrete_actor:
        actor_dist = OneHotParams(**config.actor_discrete_head)
    else:
        actor_dist = BoundedNormalParams(**config.actor_cont_head)

    actor_params = ActorParams(dist_params=actor_dist, **actor_params)

    critic_params = config.critic
    critic_dist_params = TwoHotDistParams(**config.critic_head)
    critic_params = CriticParams(two_hot_params=critic_dist_params, **critic_params)

    training_params = BehaviourTrainingParams(**config.behaviour_training)

    return Behaviour(
        actor_params=actor_params,
        critic_params=critic_params,
        training_params=training_params,
        device=device,
        compile=config.compile,
    )


def configure_buffer(config: DictConfig) -> Tuple[Buffer, List[str]]:
    """
    Configures and returns the replay buffer.

    Returns a tuple of the buffer and a list of the keys which must be
    added to the buffer.
    """

    keys_to_sample = ["reward", "is_first", "prev_action", "continue", "episode_id"]
    keys_to_sample.extend(config.env.image_keys)
    keys_to_sample.extend(config.env.vector_keys)
    keys_to_sample.extend(config.env.extra_keys)

    return Buffer(
        capacity=config.replay.capacity,
        keys_to_sample=keys_to_sample,
        disk_path=config.replay.disk_path,
        load_existing=config.replay.load_existing,
        save_every=config.replay.save_every,
    ), keys_to_sample


def configure_optimiser(
    world_model_params,
    behaviour_params,
    config,
) -> Tuple[SimpleDreamerOptimizer, SimpleDreamerOptimizer]:
    """
    Configures and returns two optimisers, one for training the world model,
    and another for training the agent's behaviour. Can probably be merged
    into a single optimiser in the future.
    """
    print(
        f"Total parameters: {sum(p.numel() for p in world_model_params + behaviour_params)}"
    )
    return SimpleDreamerOptimizer(
        parameters=world_model_params, **config.world_model_optim
    ), SimpleDreamerOptimizer(parameters=behaviour_params, **config.behaviour_optim)
