from typing import Tuple, List, Optional, Dict
import time
import os
from pathlib import Path
import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
import pandas as pd

from gymnasium.spaces.discrete import Discrete
from omegaconf import DictConfig, open_dict, OmegaConf

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
from dreamer.utils.configuration import (
    configure_environments,
    configure_world_model,
    configure_behaviour,
    configure_buffer,
    configure_optimiser,
)
from dreamer.utils.optimiser import SimpleDreamerOptimizer
from dreamer.utils.utils import (
    set_seed_everywhere,
    combine_det_and_stoch,
    set_torch_precision,
)
from dreamer.world_model import WorldModel, WorldModelTrainingParams
from dreamer.behaviour import Behaviour, BehaviourTrainingParams
from dreamer.utils.replay import Buffer


class Dreamer:
    def __init__(self, config: DictConfig) -> None:
        """
        Loads and instantiates the components needed for the Dreamer
        model, based on the specified config that was parsed by Hydra.
        """

        self._config = config

        set_seed_everywhere(self._config.seed)
        set_torch_precision(self._config.matmul_prec, self._config.cudnn_bench)

        self._train_env, self._eval_env = configure_environments(self._config)

        if self._config.wandb:
            wandb_run_name = f"dreamerv3-{self._config.env.suite_name}-{self._config.env.task_name}-seed-{self._config.seed}"
            self._wandb_run = wandb.init(
                project=self._config.wandb_project,
                name=wandb_run_name,
                config=OmegaConf.to_container(self._config),
            )
        else:
            self._wandb_run = wandb.init(mode="disabled")

        self._logs = []

        self._discrete_actor = isinstance(self._train_env.action_space, Discrete)

        if self._discrete_actor:
            self._action_dim = int(self._train_env.action_space.n)
            self._n_actions = 1
        else:
            action_sample = self._train_env.action_space.sample()
            assert len(action_sample.shape) < 3, (
                "Can currently only handle 1D or 2D continuous actions"
            )
            if len(action_sample.shape) == 2:
                self._n_actions = action_sample.shape[0]
                self._action_dim = action_sample.shape[1]
            else:
                self._n_actions = 1
                self._action_dim = action_sample.shape[0]

        self._device = torch.device(self._config.device)
        self._action_repeat = 1

        # By default, all Atari envs have an action repeat of 4
        if self._config.env.suite_name == "atari":
            self._action_repeat = 4

        self._replay_buffer, self._keys_to_store = configure_buffer(self._config)

        self._world_model = configure_world_model(
            self._config,
            self._action_dim,
            self._n_actions,
            self._device,
            self._train_env.observation_space,
        )
        self._behaviour = configure_behaviour(
            self._config,
            self._n_actions,
            self._action_dim,
            self._device,
            self._discrete_actor,
        )
        self._wm_optim, self._behaviour_optim = configure_optimiser(
            self._world_model.get_parameters(),
            self._behaviour.get_parameters(),
            self._config,
        )

        self._world_model_loss = torch.tensor(0.0)
        self._world_model_loss_detailed = {}
        self._world_model_metrics = {}
        self._behaviour_metrics = {}
        self._actor_loss = 0.0
        self._critic_loss = 0.0

        if self._config.load_existing:
            self.load_checkpoint(self._config.load_existing_path)

        self._total_env_training_steps = len(self._replay_buffer) * self._action_repeat
        self._episode_id = 0

        self._train_first_step = True
        self._train_terminated = False
        self._train_truncated = False
        self._train_done = True
        self._train_prev_action = torch.zeros(self._n_actions * self._action_dim)
        self._train_prev_deter = self._world_model.get_init_deter()
        self._train_prev_stoch = self._world_model.get_init_stoch()
        self._train_obs = {}
        self._train_episode_reward = 0.0

    def _print_losses(self):
        """Prints the agent and world model losses from the latest batch
        of training data.
        """
        print("+--------------------------------------------------+")
        print(f"Step: {self._total_env_training_steps}")
        print(f"Episode reward: {self._train_episode_reward}")
        print(f"World Model Loss: {self._world_model_loss}")
        print(f"Detailed world model loss: {self._world_model_loss_detailed}")
        print(f"Actor Loss: {self._actor_loss}")
        print(f"Critic Loss: {self._critic_loss}")
        print("+--------------------------------------------------+")

    def _train_log(self):
        """
        Creates and uploads a log of training information to weights and biases
        from the latest batch of training data.
        """
        log_dict = {}
        for k, v in self._world_model_loss_detailed.items():
            log_dict[f"{k}_loss"] = float(v)
        for k, v in self._world_model_metrics.items():
            log_dict[k] = float(v)
        for k, v in self._behaviour_metrics.items():
            log_dict[k] = float(v)

        log_dict["actor_loss"] = float(self._actor_loss)
        log_dict["critic_loss"] = float(self._critic_loss)
        log_dict["fps"] = self._fps
        log_dict["world_model_train_time"] = self._wm_batch_train_time
        log_dict["behaviour_train_time"] = self._behaviour_train_time
        log_dict["grad_update_time"] = self._grad_update_time
        log_dict["env_step_time"] = self._env_step_time

        print(f"Train logs at step {self._total_env_training_steps}: {log_dict}")

        self._logs.append({"env_step": self._total_env_training_steps, **log_dict})

        self._wandb_run.log(log_dict, step=self._total_env_training_steps)

    def _log_video(self, image_sequence: List[np.ndarray]) -> None:
        video_array = np.array(image_sequence)
        # (T, H, W, C) -> (T, C, H, W)
        video_array = video_array.transpose(0, 3, 1, 2)
        self._wandb_run.log(
            {"evaluation/video_policy": wandb.Video(video_array, fps=4, format="mp4")},
            step=self._total_env_training_steps,
        )

    def _log_wm_predictions_plot(
        self, data: Dict[str, torch.Tensor], horizon: int = 15
    ) -> None:
        """
        Creates a plot visualising the world model's prediction versus the ground
        truth.

        Args:
            data (Dict[str, torch.Tensor]): replay buffer sample. Each value is
            of shape (B, T, ...)

            horizon (int, optional): number of timesteps to predict/visualise. Defaults
            to 15.
        """

        with torch.no_grad():
            encoded_obs = self._world_model.encode_obs(data)
            latent_components = self._world_model.observe_sequence(
                data["prev_action"],
                encoded_obs,
                data["prev_deter"][:, 0],
                data["prev_stoch"][:, 0],
            )
            # All of shape (B, T, D)
            post_latent = combine_det_and_stoch(
                latent_components["deter"], latent_components["post_sample"]
            )

            post_decoded_images = self._world_model.decode_images(
                latent_components["deter"], latent_components["post_sample"]
            )[0, : horizon + 1]
            post_predicted_reward = (
                self._world_model.predict_reward(post_latent).squeeze().cpu().numpy()
            )[0, : horizon + 1]
            post_predicted_cont = (
                self._world_model.predict_cont(post_latent).squeeze().cpu().numpy()
            )[0, : horizon + 1]
            post_predicted_val = (
                self._behaviour.predict_values(post_latent).squeeze().cpu().numpy()
            )[0, : horizon + 1]

            imagined_components = self._world_model.imagine_sequence(
                latent_components["deter"][:, 0],
                latent_components["post_sample"][:, 0],
                horizon,
                actions=data["prev_action"][
                    :, 1:
                ],  # These are actions to take, not prev action
            )

            imagined_latent = combine_det_and_stoch(
                imagined_components["deter"], imagined_components["prior_sample"]
            )

            imagined_images = self._world_model.decode_images(
                latent_components["deter"], latent_components["post_sample"]
            )[0]
            imagined_reward = (
                self._world_model.predict_reward(imagined_latent)
                .squeeze()
                .cpu()
                .numpy()
            )[0]
            imagined_cont = (
                self._world_model.predict_cont(imagined_latent).squeeze().cpu().numpy()
            )[0]
            imagined_val = (
                self._behaviour.predict_values(imagined_latent).squeeze().cpu().numpy()
            )[0]

            true_images = (
                data[self._config.env.image_keys[0]][0, : horizon + 1].cpu().numpy()
            )
            true_reward = data["reward"][0, : horizon + 1].cpu().numpy()
            true_cont = data["continue"][0, : horizon + 1].cpu().numpy()

        T = horizon + 1

        def draw_caption(ax, text, color="white"):
            ax.text(
                0.5,
                -0.08,
                text,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
                color=color,
                bbox=dict(facecolor="black", alpha=0.6, pad=2),
                clip_on=False,
            )

        fig, axes = plt.subplots(3, T, figsize=(2.5 * T, 5), squeeze=False)

        for t in range(T):
            ax = axes[0, t]
            img = imagined_images[t]
            ax.imshow(img)
            ax.axis("off")

            ax.set_title(f"t={t}", fontsize=10)

            draw_caption(
                ax,
                f"r={imagined_reward[t]:.2f}  "
                f"c={imagined_cont[t]:.2f}  "
                f"v={imagined_val[t]:.2f}",
            )

            ax = axes[1, t]
            img = post_decoded_images[t]
            ax.imshow(img)
            ax.axis("off")

            ax.set_title(f"t={t}", fontsize=10)

            draw_caption(
                ax,
                f"r={post_predicted_reward[t]:.2f}  "
                f"c={post_predicted_cont[t]:.2f}  "
                f"v={post_predicted_val[t]:.2f}",
            )

            ax = axes[2, t]
            img = true_images[t]
            ax.imshow(img)
            ax.axis("off")

            draw_caption(
                ax,
                f"r={true_reward[t]:.2f}  c={true_cont[t]}",
            )

        # Row labels
        axes[0, 0].set_ylabel("Imagined", fontsize=12)
        axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
        axes[2, 0].set_ylabel("True", fontsize=12)

        plt.tight_layout()
        self._wandb_run.log(
            {"evaluation/wm_pred": wandb.Image(fig)},
            step=self._total_env_training_steps,
        )
        plt.close(fig)

    def _save_logs(self) -> None:
        """
        Saves the logs as a csv and json to the logdirectory
        """
        path_lib_path = Path(self._config.log_dir)
        path_lib_path.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self._logs).set_index("env_step").sort_index()
        out_path = os.path.join(path_lib_path, "metrics")

        df.to_csv(f"{out_path}.csv")
        df.reset_index().to_json(f"{out_path}.jsonl", orient="records", lines=True)

    @torch.no_grad()
    def step_environment_train(self, random_actions: bool, n_steps: int):
        """
        Steps the agent in the training environment, to add transitions to
        the replay buffer. Updates relevant class data variables to store
        the world model state across multiple calls to this function.

        Also evaluates the agent and world model if the required number
        of environment training steps is reached whilst inside of this
        function.

        Args:
            random_actions (bool): whether to use random actions to step
            the environment (if True), or whether to sample actions from
            the agent's policy (if False).

            n_steps (int): number of environment steps to take.
        """
        for _ in range(n_steps):
            if (
                self._total_env_training_steps % self._config.eval_every == 0
                and self._total_env_training_steps >= self._config.prefill_steps
            ):
                self._log_wm_predictions_plot(
                    self._replay_buffer.sample(2, 16, self._device), 15
                )
            if self._total_env_training_steps % self._config.eval_every == 0:
                self.step_environment_eval(self._config.n_eval_episodes)
            if (
                self._total_env_training_steps % self._config.log_every == 0
                and self._config.save_replay
            ):
                self._train_log()
            if (
                self._total_env_training_steps % self._config.save_every == 0
                and self._config.save_model
            ):
                self.save_checkpoint(
                    self._config.weights_dir, self._total_env_training_steps
                )
                self._save_logs()
            if self._train_done:
                reward = 0.0
                self._train_obs, info = self._train_env.reset()
                self._episode_id += 1
                self._train_first_step = True
                self._train_terminated = False
                self._train_truncated = False
                self._train_prev_action = torch.zeros(
                    self._n_actions * self._action_dim
                )
                self._train_prev_deter = self._world_model.get_init_deter()
                self._train_prev_stoch = self._world_model.get_init_stoch()

                self._print_losses()
                self._wandb_run.log(
                    {"train_ret": self._train_episode_reward},
                    step=self._total_env_training_steps,
                )
                self._logs.append(
                    {
                        "env_step": self._total_env_training_steps,
                        "train_ret": self._train_episode_reward,
                    }
                )
                self._train_episode_reward = 0.0
                self._train_done = False
            else:
                tensor_obs = {
                    k: torch.tensor(v).unsqueeze(0).to(self._device)
                    for k, v in self._train_obs.items()
                    if k in self._keys_to_store
                }
                deter, stoch = self._world_model.get_posterior(
                    tensor_obs,
                    self._train_prev_action.clone().detach().to(self._device),
                    self._train_prev_deter,
                    self._train_prev_stoch,
                )
                self._train_prev_deter = deter
                self._train_prev_stoch = stoch
                latent_state = combine_det_and_stoch(deter, stoch)
                if random_actions:
                    action = self._train_env.action_space.sample()
                    # We assume that discrete environments take a one-hot discrete
                    # array as input
                    if self._discrete_actor:
                        action_arr = np.zeros((self._action_dim * self._n_actions))
                        action_arr[action] = 1.0
                        action = action_arr
                        action = torch.tensor(action).to(torch.float32)
                else:
                    action = (
                        self._behaviour.act(latent_state)
                        .squeeze(-1)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                (
                    self._train_obs,
                    reward,
                    self._train_terminated,
                    self._train_truncated,
                    info,
                ) = self._train_env.step(action)
                self._train_done = self._train_terminated or self._train_truncated
                self._train_episode_reward += reward
                self._train_prev_action = (
                    action
                    if isinstance(action, torch.Tensor)
                    else torch.tensor(action).to(torch.float32)
                )
                self._train_first_step = False

            self._total_env_training_steps += 1 * self._action_repeat
            transition = {
                "prev_action": self._train_prev_action,
                "reward": reward,
                "is_first": int(self._train_first_step),
                "continue": float(not self._train_terminated),
                "episode_id": self._episode_id,
                "prev_deter": (
                    self._train_prev_deter
                    if self._train_prev_deter is not None
                    else self._world_model.get_init_deter()
                ).squeeze(),
                "prev_stoch": (
                    self._train_prev_stoch
                    if self._train_prev_stoch is not None
                    else self._world_model.get_init_stoch()
                ).squeeze(),
            }

            for k in self._keys_to_store:
                if k not in transition.keys():
                    if k in self._train_obs.keys():
                        transition[k] = self._train_obs[k]
                    elif k in info.keys():
                        transition[k] = info[k]
                    else:
                        raise ValueError(
                            f"Cannot find key: {k} in environment observation or info"
                        )

            self._replay_buffer.add(transition)

    @torch.no_grad()
    def step_environment_eval(self, n_episodes: int):
        """
        Steps the evaluation environment for a fixed number of episodes, following
        the current agent's policy. Reports the mean return across all episodes, and
        logs a video of the agent's behaviour policy.

        Args:
            n_episodes (int): number of episodes to evaluate the policy over.
        """
        total_eval_reward = 0.0
        for episode in range(n_episodes):
            obs, info = self._eval_env.reset()
            first_step = True
            prev_deter = self._world_model.get_init_deter()
            prev_stoch = self._world_model.get_init_stoch()
            prev_action = torch.zeros((self._n_actions * self._action_dim)).squeeze()
            done = False
            episode_reward = 0.0
            video_obs = []
            if "high_res_image" in obs.keys():
                video_obs.append(obs["high_res_image"])
            else:
                video_obs.append(obs[self._config.env.image_keys[0]])
            while not done:
                tensor_obs = {
                    k: torch.tensor(v).unsqueeze(0).to(self._device)
                    for k, v in obs.items()
                    if k in self._keys_to_store
                }
                deter, stoch = self._world_model.get_posterior(
                    tensor_obs,
                    torch.tensor(prev_action).to(self._device),
                    prev_deter,
                    prev_stoch,
                )
                prev_deter, prev_stoch = deter, stoch
                latent_state = combine_det_and_stoch(deter, stoch)
                action = (
                    self._behaviour.act(latent_state).squeeze(-1).detach().cpu().numpy()
                )
                obs, reward, terminated, truncated, info = self._eval_env.step(action)
                if "high_res_image" in obs:
                    video_obs.append(obs["high_res_image"])
                else:
                    video_obs.append(obs[self._config.env.image_keys[0]])
                episode_reward += reward
                done = terminated or truncated
                first_step = False
                prev_action = action
            total_eval_reward += episode_reward
        mean_eval_reward = total_eval_reward / n_episodes
        print("+---------EVALUATION RESULTS---------+")
        print(f"Step: {self._total_env_training_steps}")
        print(f"Mean eval reward: {mean_eval_reward}")
        self._wandb_run.log(
            {"evaluation/eval_ret": mean_eval_reward},
            step=self._total_env_training_steps,
        )
        self._logs.append(
            {"env_step": self._total_env_training_steps, "eval_ret": mean_eval_reward}
        )
        self._log_video(video_obs)

    def train(self) -> None:
        """
        Main training loop.

        Alternates between training the world model and agent behaviour from
        replay buffer samples, and stepping the agent in the training environment
        using the current policy, to add more data to the replay buffer.

        Evaluates the world model and policy after every self._eval_every environment
        training steps.
        """
        self._fps = 0.0
        self._env_step_time = 0.0
        self._wm_batch_train_time = 0.0
        self._behaviour_train_time = 0.0
        self._grad_update_time = 0.0
        if len(self._replay_buffer) < self._config.prefill_steps:
            self.step_environment_train(
                random_actions=True,
                n_steps=self._config.prefill_steps - len(self._replay_buffer),
            )
        self._train_done = True
        model_steps_per_batch = self._config.batch_size * self._config.batch_length

        env_steps_per_model_batch = (
            model_steps_per_batch // self._config.env.model_steps_per_env_step
        )
        while self._total_env_training_steps < self._config.total_steps:
            start_time = time.perf_counter()
            training_data = self._replay_buffer.sample(
                self._config.batch_size, self._config.batch_length, self._device
            )
            wm_start_time = time.perf_counter()
            wm_loss, starting_deter, starting_stoch, self._world_model_metrics = (
                self._world_model.train(training_data)
            )
            wm_end_time = time.perf_counter()
            self._wm_batch_train_time = wm_end_time - wm_start_time

            self._world_model_loss_detailed = wm_loss
            self._world_model_loss = sum([v for k, v in wm_loss.items()])

            starting_deter = starting_deter.reshape((-1, self._config.deter_size))
            starting_stoch = starting_stoch.reshape(
                (-1, self._config.rssm.n_stoch_dists, self._config.n_stoch_cats)
            )

            behaviour_start_time = time.perf_counter()
            actor_loss, critic_loss, self._behaviour_metrics = (
                self._behaviour.imag_train(
                    self._world_model, starting_deter.detach(), starting_stoch.detach()
                )
            )
            behaviour_end_time = time.perf_counter()
            self._behaviour_train_time = behaviour_end_time - behaviour_start_time
            self._actor_loss = actor_loss
            self._critic_loss = critic_loss

            grad_start_time = time.perf_counter()
            self._wm_optim(self._world_model_loss)
            self._behaviour_optim(self._actor_loss + self._critic_loss)
            grad_end_time = time.perf_counter()
            self._grad_update_time = grad_end_time - grad_start_time

            env_start_time = time.perf_counter()
            with torch.no_grad():
                self.step_environment_train(
                    random_actions=False, n_steps=env_steps_per_model_batch
                )
            env_end_time = time.perf_counter()
            self._env_step_time = env_end_time - env_start_time
            end_time = time.perf_counter()
            self._fps = (
                env_steps_per_model_batch
                * self._action_repeat
                / (end_time - start_time)
            )

    def save_checkpoint(self, path: str, step: Optional[int] = None) -> None:
        """
        Save world model, behaviour, optimisers, and training metadata.

        Args:
            path: Directory path to save checkpoint.
            step: Optional step number (used for naming).
        """
        path_lib_path = Path(path)
        path_lib_path.mkdir(parents=True, exist_ok=True)

        if step is None:
            step = self._total_env_training_steps

        checkpoint = {
            "step": step,
            "episode_id": self._episode_id,
            "world_model": self._world_model.state_dict(),
            "behaviour": self._behaviour.state_dict(),
            "wm_optim": self._wm_optim.state_dict(),
            "behaviour_optim": self._behaviour_optim.state_dict(),
            "config": OmegaConf.to_container(self._config),
        }

        if self._config.checkpoint:
            path_lib_path = path_lib_path / f"checkpoint_{step}.pt"
        else:
            path_lib_path = path_lib_path / "latest.pt"

        torch.save(checkpoint, path_lib_path)
        print(f"[Dreamer] Saved checkpoint to {path_lib_path}")

    def load_checkpoint(self, path: str) -> None:
        """
        Loads a saved model checkpoint from a single .pt file.

        Args:
            path (str): full path to the .pt file to load weights from.
        """
        checkpoint = torch.load(path, self._device)

        self._world_model.load_state_dict(checkpoint["world_model"])
        self._behaviour.load_state_dict(checkpoint["behaviour"])
        self._wm_optim.load_state_dict(checkpoint["wm_optim"])
        self._behaviour_optim.load_state_dict(checkpoint["behaviour_optim"])

        self._total_env_training_steps = checkpoint["step"]
        self._episode_id = checkpoint["episode_id"]

        print(f"[Dreamer] Loaded checkpoint from {path}")
