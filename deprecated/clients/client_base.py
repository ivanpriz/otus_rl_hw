import socket


class ClientBase:
    """Class for reading state from server"""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.connected = False
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def __enter__(self):
        self.socket.__enter__()
        self.socket.connect((self.host, self.port))
        self.connected = True
        return self.socket

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.socket.__exit__(exc_type, exc_val, exc_tb)
