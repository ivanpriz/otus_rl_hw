from collections import namedtuple

import torch

from rest_agent.src.schemas import StateSchema


ParsedGameData = namedtuple(
    "ParsedGameData",
    [
        "game_state",
        "reward",
        "terminated",
        "truncated",
    ]
)


class GameStateDataParserGym:
    def __init__(self, device):
        self.device = device

    """Class for parsing state and reward to feed the agent"""
    def parse_game_data(self, game_data: StateSchema) -> ParsedGameData:
        return ParsedGameData(
            torch.tensor(game_data.game_state, device=self.device, dtype=torch.float32).unsqueeze(0),
            self.reward_function(game_data),
            game_data.terminated,
            game_data.truncated,
        )

    def reward_function(self, game_data: StateSchema) -> torch.Tensor:
        # for gym we have it directly. For other envs we may have scores.
        return torch.tensor([game_data.scores], device=self.device)
