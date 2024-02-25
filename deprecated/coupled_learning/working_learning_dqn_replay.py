import sys
import time
from queue import Queue

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import numpy
import numpy as np
from torch import Tensor
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
TransitionBatch = namedtuple('TransitionBatch', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    pass


class ReplayMemoryRandom(ReplayMemory):
    def __init__(self, capacity: int):
        self.memory: deque[Transition] = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size) -> TransitionBatch:
        s = random.sample(self.memory, batch_size)
        return TransitionBatch(*zip(*s))

    def __len__(self):
        return len(self.memory)


class ReplayMemoryRandomList(ReplayMemory):
    def __init__(self, capacity: int):
        self.memory: list[Transition] = []

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size) -> TransitionBatch:
        s = random.sample(self.memory, batch_size)
        return TransitionBatch(*zip(*s))

    def __len__(self):
        return len(self.memory)


class ReplayMemoryQueue(ReplayMemory):
    def __init__(self, capacity: int):
        self.memory: Queue[Transition] = Queue(maxsize=capacity)

    def push(self, *args):
        self.memory.put(Transition(*args))

    def sample(self, batch_size: int) -> TransitionBatch:
        s = [self.memory.get() for _ in range(batch_size)]
        return TransitionBatch(*zip(*s))

    def __len__(self):
        return self.memory.qsize()


class AgentNN(nn.Module):
    def __init__(self, n_observations: int, n_actions: int):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        """Called with either one element to determine next action, or a batch
        during optimization. Returns tensor([[left0exp,right0exp]...])."""
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class Config:
    BATCH_SIZE = 128  # количество эпизодов, отобранных из буфера воспроизведения
    GAMMA = 0.99  # коэффициент дисконтирования
    EPS_START = 0.9  # начальное значение эпсилон
    EPS_END = 0.05  # конечное значение эпсилон
    EPS_DECAY = 1000  # скорость экспоненциального спада эпсилон, чем больше - тем медленнее падение
    TAU = 0.005  # скорость обновления целевой сети
    LR = 1e-4  # скорость обучения оптимизатора ``AdamW``.
    FULL_MEMORY_LENGTH = 10000


class DQN:
    def __init__(
            self,
            env_name: str = "CartPole-v1",
            episodes_num_cpu: int = 500,
            episodes_num_gpu: int = 500,
    ):
        # Среда
        self.env = gym.make(env_name)
        print(f"Env {env_name} initialized")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device is: {self.device}")
        if torch.cuda.is_available():
            self.num_episodes = episodes_num_gpu
        else:
            self.num_episodes = episodes_num_cpu

        # Получить число действий
        self.n_actions = self.env.action_space.n
        # Получить число степеней свободы состояний
        self.state, self.info = self.env.reset()
        print(type(self.state))
        self.n_observations = len(self.state)

        # Инициилизировать сети: целевую и политики
        self.policy_net = AgentNN(self.n_observations, self.n_actions).to(self.device)
        print("Policy net created")
        self.target_net = AgentNN(self.n_observations, self.n_actions).to(self.device)
        print("Target net created")

        # Подгрузить в целевую сеть коэффициенты из сети политики
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print("Coefs loaded")

        # Задать оптимайзер
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=Config.LR, amsgrad=True)

        # Инициализировать Replay Memory buffer
        self.memory = ReplayMemoryRandom(Config.FULL_MEMORY_LENGTH)

        self.steps_done = 0

        # массив длительности эпизода - пойдет в отчет о том, сколько продержался агент
        self.episode_durations = []
        self.total_reward = []

    def select_action(self, state: Tensor) -> Tensor:
        """еpsilon-жадная стратегия выбора действия"""
        #     случайное значение для определения какой шаг будем делать жадный или случайный
        sample = random.random()

        # установка порога принятия решения - уровня epsilon
        eps_threshold = Config.EPS_END + (Config.EPS_START - Config.EPS_END) * \
            math.exp(-1. * self.steps_done / Config.EPS_DECAY)

        # увеличиваем счетчик шагов
        self.steps_done += 1

        # если случайный порог больше epsilon-порога
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) вернет наибольшее значение столбца в каждой строке.
                # Второй столбец в результате max - это индекс того места,
                # где был найден максимальный элемент,
                # поэтому мы выбираем действие с наибольшим ожидаемым вознаграждением.
                res = self.policy_net(state)
                # print(f"Result we got from policy net: {res}, for state: {state}")
                return res.max(1)[1].view(1, 1)
        else:
            # Иначы выбираем случайное дайствие
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def get_non_final_mask(self, batch: TransitionBatch) -> Tensor:
        return torch.tensor(
            tuple(
                map(
                    lambda s: s is not None,
                    batch.next_state
                )
            ),
            device=self.device,
            dtype=torch.bool,
        )

    def optimize_model(self, batch_size: int):
        if len(self.memory) < batch_size:
            return

        # Получить из памяти батч
        batch = self.memory.sample(batch_size)

        # Вычислить маску нефинальных состояний и соединить элементы батча
        # (финальным состоянием должно быть то, после которого моделирование закончилось)
        non_final_mask = self.get_non_final_mask(batch)

        non_final_next_states = torch.cat(
            [
                s for s in batch.next_state
                if s is not None
            ]
        )

        # Собираем батчи для состояний, действий и наград
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Вычислить Q(s_t, a) - модель вычисляет Q(s_t),
        # затем мы выбираем столбцы предпринятых действий.
        # Это те действия, которые были бы предприняты для каждого состояния партии в соответствии с policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Вычислить V(s_{t+1}) для всех следующих состояний.
        # Ожидаемые значения действий для не_финальных_следующих_состояний вычисляются
        # на основе "старшей" целевой_сети; выбирается их наилучшее вознаграждение с помощью max(1)[0].
        # Это объединяется по маске, так что мы будем иметь либо ожидаемое значение состояния,
        # либо 0, если состояние было финальным.
        next_state_values = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        # Вычисляем ожидаемые Q значения
        expected_state_action_values = (next_state_values * Config.GAMMA) + reward_batch

        # Объединяем все в общий лосс
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Готовим градиент
        self.optimizer.zero_grad()
        loss.backward()
        # Обрезаем значения градиента - проблемма исчезающего/взрывающего градиента
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)

        self.optimizer.step()

    def update_weights_soft(self, tau: float = Config.TAU):
        """делаем "мягкое" обновление весов
        θ′ ← τ θ + (1 −τ )θ′"""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * tau \
                                         + target_net_state_dict[key] * (1 - tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def train_agent(self):
        """Вот эту функцию будет выполнять среда, посылая запросы агенту"""
        print("Training agent...")
        start = time.time()
        for _ in tqdm(range(self.num_episodes)):
            episode_reward = 0
            # Для каждого эпизода инициализируем начальное состояние
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            # выполняем действия пока не получим флаг done
            # t - считает сколько шагов успели сделать пока шест не упал
            for t in count():
                # выбираем действие [0, 1]
                action = self.select_action(state)
                # Делаем шаг
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                episode_reward += reward

                # Преобразуем в тензор
                reward = torch.tensor([reward], device=self.device)

                # Объединяем done по двум конечным состояниям
                done = terminated or truncated

                # присваиваем следующее состояние
                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                # отправляем в память
                self.memory.push(state, action, next_state, reward)

                # переходим на следующее состояние
                state = next_state

                # запускаем обучение сети
                self.optimize_model(batch_size=128)

                self.update_weights_soft()

                # Если получили terminated or truncated завершаем эпизод обучения
                if done:
                    # добавляем в массив продолжительность эпизода
                    self.episode_durations.append(t + 1)
                    self.total_reward.append(episode_reward)
                    break

        total_time = time.time() - start
        print(f"Time taken {total_time} seconds")
        print("Training finished")

    def draw_plot(self):
        # Вычисление скользящего среднего
        window_size = 50
        moving_avg = np.convolve(self.total_reward, np.ones(window_size)/window_size, mode='valid')
        plt.plot(moving_avg)
        plt.title('AgentNN + ReplayBuffer')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.show()


if __name__ == '__main__':
    dqn = DQN()
    dqn.train_agent()
    dqn.draw_plot()
    # env = gym.make("CartPole-v1")
    # n_actions = env.action_space.n
    # # Получить число степеней свободы состояний
    # state, info = env.reset()
    # print(type(state)) 4:56
    # n_observations = len(state)
    # print(n_actions, n_observations)

# todo:
# реализовать очередь с возможность перемешивать элементы.
# простая реализация - перемешать все один раз на входе, положить в обычную очередь
# новые резты класть в в очередь. Сложная реализация - возможность перемешивать элементы, получая новую очередь.
# Можно попробовать сначала делать случайные пары из элементов, потом случайны образом соединять пары, потом четверки и т.д.

