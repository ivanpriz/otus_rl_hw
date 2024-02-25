from distributed.config import HOST, PORT_GAME_STATE, HEADER_SIZE_BYTES, ACTION_SIZE_BYTES
from .client_base import ClientBase


class AgentClient(ClientBase):
    """Class for reading actions from client"""

    def read_action(self) -> int:
        return int.from_bytes(
            self.socket.recv(ACTION_SIZE_BYTES),
            byteorder="big",
        )
