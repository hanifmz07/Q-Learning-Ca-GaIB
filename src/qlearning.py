import numpy as np
import random
import os
import time
from tqdm import tqdm


class QLearning1DGame:
    def __init__(
        self,
        learning_rate=0.7,
        min_epsilon=0.05,
        max_epsilon=1.0,
        episodes=100,
        decay_rate=0.05,
        discount_value=0.95,
    ):
        self._state_space = np.array([i for i in range(8)])
        self._action_space = np.array([0, 1])  # 0 is left, 1 is right
        self._reward_table = np.array(
            [
                [-100, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, 100],
            ]
        )
        self._learning_rate = learning_rate
        self._min_epsilon = min_epsilon
        self._max_epsilon = max_epsilon
        self._episodes = episodes
        self._decay_rate = decay_rate
        self._discount_value = discount_value
        self._current_state = 1
        self._current_reward_sum = 0
        self._current_epsilon = max_epsilon

        self.q_table = np.zeros((len(self._state_space), len(self._action_space)))

    def _greedy_policy(self):
        return self._action_space[np.argmax(self.q_table[self._current_state])]

    def _e_greedy_policy(self, epsilon):
        if random.random() > epsilon:
            return self._greedy_policy()
        else:
            return self._action_space[random.randint(0, 1)]

    def _is_episode_done(self):
        return self._current_reward_sum <= -200 or self._current_reward_sum >= 500

    def _reset_env(self):
        self._current_state = 1
        self._current_reward_sum = 0

    def _visualize(self):
        state_vis = [" " for i in range(8)]
        state_vis[self._current_state] = "O"
        print("||||||||||||||||||||||||||||||||||||||||||||||||||||")
        print(
            f"|| H || {state_vis[0]} || {state_vis[1]} || {state_vis[2]} || {state_vis[3]} || {state_vis[4]} || {state_vis[5]} || {state_vis[6]} || {state_vis[7]} || A ||"
        )
        print("||||||||||||||||||||||||||||||||||||||||||||||||||||")

    def train(self, visualize_mode=False, slow_mode=False):
        win_count = 0
        lost_count = 0
        for episode in tqdm(range(0, self._episodes), desc="Episode"):
            self._reset_env()
            epsilon = self._min_epsilon + (
                self._max_epsilon - self._min_epsilon
            ) * np.exp(-self._decay_rate * episode)
            while True:
                action = self._e_greedy_policy(epsilon)
                reward = self._reward_table[self._current_state][action]
                if reward == -1:
                    new_state = (
                        self._current_state + 1
                        if action == 1
                        else self._current_state - 1
                    )
                else:
                    new_state = 1

                self.q_table[self._current_state][action] = (
                    1 - self._learning_rate
                ) * self.q_table[self._current_state][action] + self._learning_rate * (
                    reward + self._discount_value * np.max(self.q_table[new_state])
                )

                self._current_state = new_state
                self._current_reward_sum += reward
                if visualize_mode:
                    print(f"Episode: {episode + 1}/{self._episodes}")
                    self._visualize()
                    print(f"Current total reward: {self._current_reward_sum}")
                    print(f"Win = {win_count} | Lost = {lost_count}")

                    # Only for better observation of visualize mode
                    if slow_mode == True:
                        time.sleep(0.1)
                    if episode != self._episodes - 1:
                        os.system("cls")

                if self._is_episode_done():
                    if self._current_reward_sum >= 500:
                        win_count += 1
                    else:
                        lost_count += 1
                    break
        self._current_epsilon = self._max_epsilon
