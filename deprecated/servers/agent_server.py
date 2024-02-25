from .server_base import ServerBase

from distributed.config import ACTION_SIZE_BYTES


class AgentActionsServer(ServerBase):
    def send_action(self, action: int):
        self.conn.sendall(action.to_bytes(ACTION_SIZE_BYTES, byteorder="big"))
