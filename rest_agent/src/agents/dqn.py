import datetime
import math
import os
import random

from torch import Tensor

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rest_agent.src.replay_buffers import (
    Transition,
    TransitionBatch,
    ReplayMemoryRandom
)
from rest_agent.src.learningconfig import LearningConfig


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


class DQNAgent:
    def __init__(
            self,
            device,
            n_actions: int,
            n_observations: int,
    ):
        self.device = device
        print(f"Device is: {self.device}")

        self.n_actions = n_actions
        self.n_observations = n_observations

        # Инициилизировать сети: целевую и политики
        self.policy_net = AgentNN(self.n_observations, self.n_actions).to(self.device)
        print("Policy net created")
        self.target_net = AgentNN(self.n_observations, self.n_actions).to(self.device)
        print("Target net created")

        # Подгрузить в целевую сеть коэффициенты из сети политики
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print("Coefs loaded")

        # Задать оптимайзер
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LearningConfig.LR, amsgrad=True)

        self.steps_done = 0

    def select_action(self, state: Tensor) -> Tensor:
        """еpsilon-жадная стратегия выбора действия"""
        #     случайное значение для определения какой шаг будем делать жадный или случайный
        sample = random.random()

        # установка порога принятия решения - уровня epsilon
        eps_threshold = LearningConfig.EPS_END + (LearningConfig.EPS_START - LearningConfig.EPS_END) * \
            math.exp(-1. * self.steps_done / LearningConfig.EPS_DECAY)

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
            return torch.tensor([[random.randint(0, self.n_actions-1)]], device=self.device, dtype=torch.long)

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

    def optimize_model(self, batch_size: int, memory: ReplayMemoryRandom):
        if len(memory) < batch_size:
            return

        # Получить из памяти батч
        batch = memory.sample(batch_size)

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
        expected_state_action_values = (next_state_values * LearningConfig.GAMMA) + reward_batch

        # Объединяем все в общий лосс
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Готовим градиент
        self.optimizer.zero_grad()
        loss.backward()
        # Обрезаем значения градиента - проблемма исчезающего/взрывающего градиента
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)

        self.optimizer.step()

    def update_weights_soft(self, tau: float = LearningConfig.TAU):
        """делаем "мягкое" обновление весов
        θ′ ← τ θ + (1 −τ )θ′"""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * tau \
                                         + target_net_state_dict[key] * (1 - tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def save_model(self, save_dir_path: str):
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)

        curr_run_dir_name = f"run_{datetime.datetime.utcnow().isoformat()}"
        os.makedirs(os.path.join(save_dir_path, curr_run_dir_name))

        torch.save(
            self.policy_net,
            os.path.join(
                save_dir_path,
                curr_run_dir_name,
                f"policy.json",
            )
        )

        torch.save(
            self.target_net,
            os.path.join(
                save_dir_path,
                curr_run_dir_name,
                f"target.json",
            )
        )
