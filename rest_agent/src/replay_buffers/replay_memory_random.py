import random
from collections import deque

from .utils import Transition, TransitionBatch


class ReplayMemoryRandom:
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
