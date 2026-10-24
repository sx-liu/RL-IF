import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback

class SaveStatesCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super(SaveStatesCallback, self).__init__(verbose)
        self.save_path = save_path
        self.all_states = []
        self.all_actions = []
        self.all_rewards = []
        self.all_advantages = []
        self.all_episode_starts = []

    def _clear_all(self) -> None:
        self.all_states.clear()
        self.all_actions.clear()
        self.all_rewards.clear()
        self.all_advantages.clear()
        self.all_episode_starts.clear()

    def _on_step(self) -> bool:
        return True

    def on_rollout_end(self) -> None:
        if hasattr(self.model.rollout_buffer, "observations"):
            obs = self.model.rollout_buffer.observations.copy()
            self.all_states.extend(obs)

            actions = self.model.rollout_buffer.actions.copy()
            self.all_actions.extend(actions)

            rewards = self.model.rollout_buffer.rewards.copy()
            self.all_rewards.extend(rewards)

            advantages = self.model.rollout_buffer.advantages.copy()
            self.all_advantages.extend(advantages)

            episode_starts = self.model.rollout_buffer.episode_starts.copy()
            self.all_episode_starts.extend(episode_starts)

    def on_training_end(self) -> None:
        rollout_idx = self.model.num_timesteps // (self.model.n_steps * self.model.n_envs) - 1
        prefix = f"r{rollout_idx}"

        if self.save_path is not None:
            all_states_array = np.array(self.all_states)
            save_file = os.path.join(self.save_path, f"{prefix}_states.npy")
            np.save(save_file, all_states_array)
            print(f"Saved all states to {save_file}")

            all_actions_array = np.array(self.all_actions)
            save_file = os.path.join(self.save_path, f"{prefix}_actions.npy")
            np.save(save_file, all_actions_array)
            print(f"Saved all actions to {save_file}")

            all_rewards_array = np.array(self.all_rewards)
            save_file = os.path.join(self.save_path, f"{prefix}_rewards.npy")
            np.save(save_file, all_rewards_array)
            print(f"Saved all rewards to {save_file}")

            all_advantages_array = np.array(self.all_advantages)
            save_file = os.path.join(self.save_path, f"{prefix}_advantages.npy")
            np.save(save_file, all_advantages_array)
            print(f"Saved all advantages to {save_file}")

            all_episode_starts_array = np.array(self.all_episode_starts)
            save_file = os.path.join(self.save_path, f"{prefix}_episode_starts.npy")
            np.save(save_file, all_episode_starts_array)
            print(f"Saved all episode starts to {save_file}")

        self._clear_all()

class SaveAllStatesCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super(SaveStatesCallback, self).__init__(verbose)
        self.save_path = save_path
        self.all_states = []
        self.all_actions = []
        self.all_rewards = []
        self.all_advantages = []
        self.all_episode_starts = []

    def _clear_all(self) -> None:
        self.all_states.clear()
        self.all_actions.clear()
        self.all_rewards.clear()
        self.all_advantages.clear()
        self.all_episode_starts.clear()

    def _on_step(self) -> bool:
        return True

    def on_rollout_end(self) -> None:
        if hasattr(self.model.rollout_buffer, "observations"):
            obs = self.model.rollout_buffer.observations.copy()
            self.all_states.extend(obs)

            actions = self.model.rollout_buffer.actions.copy()
            self.all_actions.extend(actions)

            rewards = self.model.rollout_buffer.rewards.copy()
            self.all_rewards.extend(rewards)

            advantages = self.model.rollout_buffer.advantages.copy()
            self.all_advantages.extend(advantages)

            episode_starts = self.model.rollout_buffer.episode_starts.copy()
            self.all_episode_starts.extend(episode_starts)

    def on_training_end(self) -> None:
        all_states_array = np.array(self.all_states)
        save_file = os.path.join(self.save_path, "all_states.npy")
        np.save(save_file, all_states_array)
        print(f"Saved all states to {save_file}")

        all_actions_array = np.array(self.all_actions)
        save_file = os.path.join(self.save_path, "all_actions.npy")
        np.save(save_file, all_actions_array)
        print(f"Saved all actions to {save_file}")

        all_rewards_array = np.array(self.all_rewards)
        save_file = os.path.join(self.save_path, "all_rewards.npy")
        np.save(save_file, all_rewards_array)
        print(f"Saved all rewards to {save_file}")

        all_advantages_array = np.array(self.all_advantages)
        save_file = os.path.join(self.save_path, "all_advantages.npy")
        np.save(save_file, all_advantages_array)
        print(f"Saved all advantages to {save_file}")

        all_episode_starts_array = np.array(self.all_episode_starts)
        save_file = os.path.join(self.save_path, "all_episode_starts.npy")
        np.save(save_file, all_episode_starts_array)
        print(f"Saved all episode starts to {save_file}")

        self._clear_all()
