from pydantic import BaseModel


class StateSchema(BaseModel):
    game_state: list
    scores: int
    terminated: bool
    truncated: bool


class GameSetupSchema(BaseModel):
    n_actions: int
    n_dimensions: int
    game_state: list
