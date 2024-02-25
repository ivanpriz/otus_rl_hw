import json
import socket

from distributed.config import HOST, PORT_GAME_STATE, HEADER_SIZE_BYTES
from .client_base import ClientBase


class GameStateClient:
    """Class for reading state from server"""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.connected = False
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.state_buffer = bytearray()

    def __enter__(self):
        self.socket.__enter__()
        self.socket.connect((self.host, self.port))
        self.connected = True
        return self.socket

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.socket.__exit__(exc_type, exc_val, exc_tb)

    def read_state(self) -> dict:
        incoming_state_size_bytes = int.from_bytes(
            self.socket.recv(HEADER_SIZE_BYTES),
            byteorder="big",
        )
        bytes_for_current_state_left = incoming_state_size_bytes
        while bytes_for_current_state_left > 0:
            if bytes_for_current_state_left < 1024:
                data_chunk = self.socket.recv(1024)
                bytes_for_current_state_left -= 1024
            else:
                data_chunk = self.socket.recv(bytes_for_current_state_left)
                bytes_for_current_state_left = 0

            self.state_buffer.extend(data_chunk)

        print(bytes(self.state_buffer))
        return json.loads(bytes(self.state_buffer))

    def run(self):
        if not self.connected:
            raise Exception("Trying to run no connected client. Use it as context manager.")

        print("Started!")
        while True:
            print("Reading state...")
            new_state = self.read_state()
            print(f"State read: {new_state}")
