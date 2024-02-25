import gymnasium as gym
import matplotlib.pyplot as plt
from itertools import count

import numpy as np
from torch import Tensor
from tqdm import tqdm

import torch

from distributed.agents.dqn import DQNAgent
from distributed.replay_buffers import (
    Transition, TransitionBatch, ReplayMemoryRandom
)


class FakeEnvService:
    def __init__(self, name: str):
        self.env = gym.make(name)
        self.curr_state = {}
        # Here we imitate like we were running a game
        self.reset()

    def get_game_state(self) -> dict:
        # Here we would have a call to a remote game server
        return self.curr_state

    def read_input(self, action: int):
        new_gym_state, gym_rew, gym_term, gym_trunc, _ = self.env.step(action)
        self.curr_state = {
            "game_state": new_gym_state,
            "scores": self.curr_state["scores"] + gym_rew,
            "game_over": gym_term,
            "trunc": gym_trunc,  # guess it is win condition?
        }

    def reset(self):
        gym_state, _info = self.env.reset()

        # This is a format we have game state for gym.
        # The idea is to show that we have some game state we later
        # will parse state and reward for agent
        self.curr_state = {
            "game_state": gym_state,
            "scores": 0,
            "game_over": False,
            "trunc": False,  # guess it is win condition?
        }


class FakeAgentService:
    def __init__(self, agent: DQNAgent):
        self.agent = agent

        # массив длительности эпизода - пойдет в отчет о том, сколько продержался агент
        self.episode_durations = []
        self.total_reward = []

        # todo: перенести в нормальный сервис

    def parse_game_data(self, game_data: dict) -> tuple[Tensor, int, bool, bool]:
        return (
            torch.tensor(game_data["game_state"], device=self.agent.device),
            game_data["scores"],
            game_data["game_over"],
            game_data["trunc"],
        )

    def train_one_episode(
        self,
        memory: ReplayMemoryRandom,
        env_service: FakeEnvService,
    ):
        episode_reward = 0

        # выполняем действия пока не получим флаг done
        # t - считает сколько шагов успели сделать пока шест не упал
        for t in count():
            game_data = env_service.get_game_state()
            state, _, _, _ = self.parse_game_data(game_data)

            # выбираем действие [0, 1]
            action = self.agent.select_action(state)

            # Делаем шаг, посылаем его на сервер
            env_service.read_input(action)

            # Получаем новое состояние
            game_data = env_service.get_game_state()

            observation, reward, terminated, truncated = self.parse_game_data(game_data)
            episode_reward += reward

            # Преобразуем в тензор
            reward = torch.tensor([reward], device=self.agent.device)

            # Объединяем done по двум конечным состояниям
            done = terminated or truncated

            # присваиваем следующее состояние
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation,
                    dtype=torch.float32,
                    device=self.agent.device
                ).unsqueeze(0)

            # отправляем в память
            memory.push(state, action, next_state, reward)

            # переходим на следующее состояние
            # state = next_state

            # запускаем обучение сети
            self.agent.optimize_model(batch_size=128, memory=memory)
            self.agent.update_weights_soft()

            # Если получили terminated or truncated завершаем эпизод обучения
            if done:
                # добавляем в массив продолжительность эпизода
                self.episode_durations.append(t + 1)
                self.total_reward.append(episode_reward)
                return


def train_agent(
        agent_service: FakeAgentService,
        memory: ReplayMemoryRandom,
        env_service: FakeEnvService,
):
    print("Training agent...")
    for _ in tqdm(range(Config.NUM_EPISODES)):

        env_service.reset()
        agent_service.train_one_episode(memory, env_service)
    print("Training finished")


def draw_plot(total_reward):
    # Вычисление скользящего среднего
    window_size = 50
    moving_avg = np.convolve(
        total_reward,
        np.ones(window_size)/window_size,
        mode='valid'
    )
    plt.plot(moving_avg)
    plt.title('AgentNN + ReplayBuffer')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()


def main():
    # Среда
    env = gym.make(Config.ENV_NAME)
    print(f"Env {Config.ENV_NAME} initialized")

    # Получить число действий
    n_actions = env.action_space.n
    # Получить число степеней свободы состояний
    state, info = env.reset()
    print(type(state))
    n_observations = len(state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DQNAgent(device, n_actions, n_observations)




if __name__ == '__main__':
    main()

# todo:
# реализовать очередь с возможность перемешивать элементы.
# простая реализация - перемешать все один раз на входе, положить в обычную очередь
# новые резты класть в в очередь. Сложная реализация - возможность перемешивать элементы, получая новую очередь.
# Можно попробовать сначала делать случайные пары из элементов, потом случайны образом соединять пары, потом четверки и т.д.

