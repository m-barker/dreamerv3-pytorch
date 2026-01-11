import time
import random
import torch
from tqdm import tqdm

from dreamer.envs.dmc import DMCWrapper
from dreamer.envs.minigrid_wrapper import MiniGridFullObsWrapper
from dreamer.world_model import (
    WorldModel,
    WorldModelTrainingParams,
    RSSMParams,
    EncoderParams,
    DecoderParams,
)
from dreamer.networks.encoder import CNNParams
from dreamer.networks.decoder import DecoderCNNParams
from dreamer.behaviour import (
    Behaviour,
    BehaviourTrainingParams,
    ActorParams,
    CriticParams,
)
from dreamer.networks.shared import (
    BoundedNormalParams,
    TwoHotDistParams,
    MLPandHeadParams,
    MLPParams,
    BernouliDistParams,
    OneHotParams,
)
from dreamer.utils.utils import combine_det_and_stoch
from dreamer.utils.optimiser import SimpleDreamerOptimizer
from dreamer.utils.replay import Buffer


def test_run():
    replay_buffer = Buffer(
        capacity=1000000,
        keys_to_sample=[
            "image",
            "reward",
            "is_first",
            "prev_action",
            "continue",
            "episode_id",
        ],
        disk_path="./storage/replay_buffer/",
    )
    train_env = DMCWrapper("acrobot-swingup-v0", (64, 64), 1)
    # train_env = MiniGridFullObsWrapper("MiniGrid-DoorKey-6x6-v0", (64, 64), 1)
    # eval_env = MiniGridFullObsWrapper(
    #    "MiniGrid-DoorKey-6x6-v0", (64, 64), 1, render_mode="human"
    # )
    eval_env = DMCWrapper("acrobot-swingup_sparse-v0", (64, 64), 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent_dim = 2048 + (16 * 32)
    action_dim = 1
    eval_every = 500

    training_params = WorldModelTrainingParams()
    rssm_params = RSSMParams(
        deter_size=2048,
        n_stoch_dists=16,
        n_stoch_cats=32,
        encoded_size=256 * 4,
        act_func="SiLU",
        n_prior_layers=2,
        n_post_layers=1,
        n_deter_layers=1,
        layer_norm=True,
        bias=True,
        unimix=0.01,
        winit_scale=1.0,
        n_blocks=8,
        hidden_size=256,
        action_dim=action_dim,
        device=device,
    )
    encoder_cnn_params = CNNParams(
        initial_depth=16,
        kernel_size=5,
        stride=1,
        min_res=4,
        bias=True,
        norm=True,
        act_func="SiLU",
        max_pool=True,
        depth_mult=[2, 3, 4, 4],
        max_pool_stride=2,
        max_pool_kernel=2,
    )
    encoder_params = EncoderParams(
        image_keys=["image"],
        image_shapes=[(64, 64, 3)],
        cnn_params=encoder_cnn_params,
    )
    decoder_cnn_params = DecoderCNNParams(
        starting_res=4,
        latent_dim=latent_dim,
        kernel_size=5,
        starting_depth=16,
        depth_mults=[2, 3, 4, 4],
        bias=True,
        norm=True,
        act_func="SiLU",
        final_sigmoid=True,
    )
    decoder_params = DecoderParams(
        image_keys=["image"],
        image_shapes=[(64, 64, 3)],
        cnn_params=decoder_cnn_params,
    )
    reward_network_params = MLPParams(
        input_dim=latent_dim,
        out_dim=255,
        n_layers=1,
        layer_width=256,
        act_func="SiLU",
        layer_norm=True,
        bias=True,
        winit_scale=0.0,
    )
    reward_head_params = TwoHotDistParams()
    reward_params = MLPandHeadParams(reward_network_params, reward_head_params)
    continue_network_params = MLPParams(
        input_dim=latent_dim,
        out_dim=1,
        n_layers=1,
        layer_width=256,
        act_func="SiLU",
        layer_norm=True,
        bias=True,
        winit_scale=1.0,
    )
    continue_head_params = BernouliDistParams()
    continue_params = MLPandHeadParams(continue_network_params, continue_head_params)
    grad_components = ["encoder", "decoder", "reward", "continue"]
    policy_dist_params = BoundedNormalParams(
        min_std=0.1,
        max_std=1.0,
    )
    # policy_dist_params = OneHotParams(0.01)
    actor_params = ActorParams(
        n_actions=1,
        action_dim=action_dim,
        latent_state_size=latent_dim,
        n_layers=3,
        layer_width=256,
        act_func="SiLU",
        layer_norm=True,
        bias=True,
        winit_scale=0.01,
        dist_params=policy_dist_params,
    )
    critic_dist_params = TwoHotDistParams()
    critic_params = CriticParams(
        latent_state_size=latent_dim,
        n_layers=3,
        layer_width=256,
        act_func="SiLU",
        layer_norm=True,
        bias=True,
        winit_scale=0.0,
        two_hot_params=critic_dist_params,
    )
    behaviour_training_params = BehaviourTrainingParams(lam=0.95)

    world_model = WorldModel(
        training_params,
        rssm_params,
        encoder_params,
        decoder_params,
        grad_components,
        device=device,
        reward_params=reward_params,
        continue_params=continue_params,
    )
    agent = Behaviour(
        actor_params, critic_params, behaviour_training_params, device=device
    )

    wm_optim = SimpleDreamerOptimizer(
        world_model.get_parameters(), lr=4e-5, grad_clip=1000
    )
    be_optim = SimpleDreamerOptimizer(agent.get_parameters(), lr=4e-5)

    # Fill buffer with X random steps.
    done = True
    total_env_training_steps = 0
    episode_id = 0
    for step in tqdm(range(2500)):
        if done:
            obs, info = train_env.reset()
            episode_id += 1
            first_step = True
            terminated = False
            truncated = False
            done = False
            reward = 0.0
            prev_action = torch.zeros(action_dim)
        else:
            # action = random.randint(0, action_dim - 1)
            action = train_env.action_space.sample()
            obs, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated
            prev_action = torch.zeros(action_dim)
            prev_action[action] = 1.0
            first_step = False
        # Add to buffer
        image = obs["image"]

        transition = {
            "image": image,
            "prev_action": prev_action,
            "reward": reward,
            "is_first": first_step,
            "continue": float(not (terminated)),
            "episode_id": episode_id,
        }
        replay_buffer.add(transition)
        total_env_training_steps += 1
    done = True
    episode_reward = 0.0
    # Train - step - eval loop.
    while True:
        should_eval = total_env_training_steps % eval_every == 0
        if should_eval:
            with torch.no_grad():
                eval_obs, eval_info = eval_env.reset()
                eval_first_step = True
                eval_terminated = False
                eval_truncated = False
                eval_done = False
                eval_prev_action = torch.zeros(action_dim)
                eval_prev_deter = None
                eval_prev_stoch = None
                eval_reward = 0.0
                while not eval_done:
                    eval_tensor_obs = {
                        k: torch.tensor(v).unsqueeze(0).to(device)
                        for k, v in eval_obs.items()
                    }
                    eval_deter, eval_stoch = world_model.get_posterior(
                        eval_tensor_obs,
                        torch.tensor(eval_prev_action).to(device),
                        eval_first_step,
                        eval_prev_deter,
                        eval_prev_stoch,
                    )
                    eval_latent_state = combine_det_and_stoch(eval_deter, eval_stoch)
                    eval_action = (
                        agent.act(eval_latent_state).squeeze(-1).detach().cpu().numpy()
                    )
                    (
                        eval_obs,
                        eval_r,
                        eval_terminated,
                        eval_truncated,
                        eval_info,
                    ) = eval_env.step(eval_action)
                    eval_done = eval_terminated or eval_truncated
                    eval_reward += eval_r
                    eval_prev_action = eval_action
                    eval_first_step = False
                print(f"Eval Reward: {eval_reward}")

        start_time = time.perf_counter()
        data = replay_buffer.sample(8, 64, device)
        loss, starting_deter, starting_stoch = world_model.train(data)
        starting_deter = starting_deter.reshape((-1, 2048))
        starting_stoch = starting_stoch.reshape((-1, 16, 32))
        recon_deter = starting_deter[20].unsqueeze(0)
        recon_stoch = starting_stoch[20].unsqueeze(0)
        recon_latent = combine_det_and_stoch(recon_deter, recon_stoch)
        world_model.decode_images_and_save(recon_latent)
        actor_loss, critic_loss = agent.imag_train(
            world_model,
            starting_deter.detach(),
            starting_stoch.detach(),
        )
        agent_loss = actor_loss + critic_loss
        world_model_loss = sum([v for k, v in loss.items()])
        wm_optim(world_model_loss)
        be_optim(agent_loss)
        for step in range(2):
            with torch.no_grad():
                if done:
                    obs, info = train_env.reset()
                    episode_id += 1
                    first_step = True
                    terminated = False
                    truncated = False
                    reward = 0.0
                    done = False
                    prev_action = torch.zeros(action_dim)
                    prev_deter = None
                    prev_stoch = None
                    print("+--------------------------------------------------+")
                    print(f"Step: {total_env_training_steps}")
                    print(f"Episode reward: {episode_reward}")
                    print(f"World Model Loss: {world_model_loss}")
                    print(f"Detailed world model loss: {loss}")
                    print(f"Actor Loss: {actor_loss}")
                    print(f"Critic Loss: {critic_loss}")
                    print("+--------------------------------------------------+")
                    episode_reward = 0.0
                else:
                    tensor_obs = {
                        k: torch.tensor(v).unsqueeze(0).to(device)
                        for k, v in obs.items()
                    }
                    deter, stoch = world_model.get_posterior(
                        tensor_obs,
                        torch.tensor(prev_action).to(device),
                        first_step,
                        prev_deter,
                        prev_stoch,
                    )
                    latent_state = combine_det_and_stoch(deter, stoch)
                    world_model.decode_images_and_save(latent_state)
                    action = agent.act(latent_state).squeeze(-1).detach().cpu().numpy()
                    obs, reward, terminated, truncated, info = train_env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    prev_action = action
                    first_step = False
                # Add to buffer
                image = obs["image"]

                transition = {
                    "image": image,
                    "prev_action": prev_action,
                    "reward": reward,
                    "is_first": int(first_step),
                    "continue": float(not terminated),
                    "episode_id": episode_id,
                }
                replay_buffer.add(transition)
                total_env_training_steps += 1
        end_time = time.perf_counter()
        # print(f"FPS: {2 / (end_time - start_time)}")
