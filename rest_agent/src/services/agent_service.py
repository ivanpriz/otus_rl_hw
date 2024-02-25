import torch
from torch import Tensor

from rest_agent.src.agents import DQNAgent
from rest_agent.src.replay_buffers import ReplayMemoryRandom
from rest_agent.src.data_parsers import GameStateDataParserGym, ParsedGameData
from rest_agent.src.schemas import StateSchema


class AgentService:
    """Here we maintain current model.
    It supports getting actions for given state with the model."""
    def __init__(
            self,
            n_actions: int,
            n_dimensions: int,
            initial_game_state: list,
            save_dir_path: str,
    ):
        self.save_dir_path = save_dir_path
        self.n_actions = n_actions
        self.n_dimensions = n_dimensions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_game_data: ParsedGameData | None = ParsedGameData(
            game_state=torch.tensor(initial_game_state, device=self.device, dtype=torch.float32).unsqueeze(0),
            reward=0,
            terminated=False,
            truncated=False,
        )
        self.action_for_current_state: Tensor | None = None
        self.agent = None
        self.storage = ReplayMemoryRandom(capacity=1000)
        self.data_parser = GameStateDataParserGym(self.device)

    def build_agent(self):
        self.agent = DQNAgent(
            device=self.device,
            n_actions=self.n_actions,
            n_observations=self.n_dimensions,
        )

    def get_action(self):
        return self.action_for_current_state.item()

    def update_state(self, game_data: StateSchema):
        parsed_data = self.data_parser.parse_game_data(game_data)

        if self.current_game_data is not None and self.action_for_current_state is not None:
            if parsed_data.truncated or parsed_data.terminated:
                self.storage.push(
                    self.current_game_data.game_state,
                    self.action_for_current_state,
                    None,
                    parsed_data.reward,
                )
            else:
                self.storage.push(
                    self.current_game_data.game_state,
                    self.action_for_current_state,
                    parsed_data.game_state,
                    parsed_data.reward,
                )
            # print("Pushed to memory")
        else:
            pass
            # print("!!!!ERROR: NOT UPDATED MEMORY!!!!")

        self.current_game_data = parsed_data
        # print("Updated current game data")

        # запускаем обучение сети
        self.agent.optimize_model(batch_size=128, memory=self.storage)
        # print("Optimized model")

        self.agent.update_weights_soft()
        # print("Updated weights")

        # Выбираем новое действие
        self.action_for_current_state = self.agent.select_action(self.current_game_data.game_state)
        # print(f"Got action for current state: {self.action_for_current_state}")

    def save_agent(self):
        self.agent.save_model(self.save_dir_path)
