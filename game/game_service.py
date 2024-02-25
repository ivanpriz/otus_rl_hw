import datetime
import os.path
import time
import logging

import gymnasium as gym
import numpy as np
import requests
from matplotlib import pyplot as plt
from tqdm import tqdm

from game_config import HOST, AGENT_PORT


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class GameService:
    def __init__(
        self,
        episodes_num: int,
        max_steps_per_episode: int,
        graphs_dir_path: str,
    ):
        self.graphs_dir_path = graphs_dir_path
        self.env = gym.make("CartPole-v1")
        self.state = {}
        self.n_actions = None
        self.n_dimensions = None
        self.reset_state()

        self.episodes_num = episodes_num
        self.max_steps_per_episode = max_steps_per_episode
        self.scores_per_episodes = []

    def send_env_info_to_players(self):
        requests.post(
            f"http://{HOST}:{AGENT_PORT}/initialize",
            json={
                "n_actions": int(self.n_actions),
                "n_dimensions": int(self.n_dimensions),
                "game_state": self.state["game_state"],
            }
        )

    def send_state_to_players(self):
        requests.post(f"http://{HOST}:{AGENT_PORT}/state", json=self.state)

    def get_players_input(self) -> int:
        """Currently we support only 1 player"""
        resp = requests.get(f"http://{HOST}:{AGENT_PORT}/action")
        return int(resp.content)

    def update_state(self, action: int):
        observation, reward, terminated, truncated, _ = self.env.step(action)
        # Important note: we are not sending agent the real reward, we are sending game data.
        # How it parses reward from it is defined on agent side.
        self.state = {
            "game_state": observation.tolist(),
            "scores": reward,
            "terminated": terminated,
            "truncated": truncated,
        }

    def reset_state(self):
        state, info = self.env.reset()
        self.n_actions = self.env.action_space.n
        self.n_dimensions = len(state)
        self.state = {
            "game_state": state.tolist(),
            "scores": 0,
            "terminated": False,
            "truncated": False,
        }
        _logger.info("Reseted state")

    def tell_agent_to_save(self):
        requests.post(f"http://{HOST}:{AGENT_PORT}/save")

    def run(self):
        start_time = time.time()
        self.send_env_info_to_players()
        for i in tqdm(range(self.episodes_num)):
            self.reset_state()
            ep_steps, ep_scores = self.run_episode()
            self.scores_per_episodes.append(ep_scores)
            _logger.info(f"Episode {i + 1} finished, scores: {ep_scores}, steps: {ep_steps}")
        time_taken = time.time() - start_time
        print(f"Total time taken for {self.episodes_num}: {time_taken} seconds")
        self.tell_agent_to_save()
        self.draw_plot()

    def draw_plot(self):
        # Вычисление скользящего среднего
        window_size = 50
        moving_avg = np.convolve(
            self.scores_per_episodes,
            np.ones(window_size)/window_size,
            mode='valid',
        )
        plt.plot(moving_avg)
        plt.title('Scores per episodes')
        plt.xlabel('Episode')
        plt.ylabel('Scores')
        plt.show()

        if not os.path.exists(self.graphs_dir_path):
            os.makedirs(self.graphs_dir_path)

        plt.savefig(
            os.path.join(
                self.graphs_dir_path,
                f"run_eps_{self.episodes_num}_{datetime.datetime.utcnow().isoformat()}.png"
            )
        )

    def run_episode(self) -> (int, int):
        self.send_state_to_players()
        steps_in_curr_episode_run = 0
        episode_scores = 0
        _logger.debug("Episode running...")
        while (
            not self.state["terminated"]
            and not self.state["truncated"]
        ):
            steps_in_curr_episode_run += 1
            _logger.debug("Getting players input...")
            action = self.get_players_input()
            _logger.debug(f"Received player action: {action}")
            self.update_state(action)
            episode_scores += self.state["scores"]
            _logger.debug(f"New episode total reward: {episode_scores}")
            _logger.debug("State updated, sending to players...")
            self.send_state_to_players()
            _logger.debug("State sent")
            # time.sleep(0.05)
        return steps_in_curr_episode_run, episode_scores
