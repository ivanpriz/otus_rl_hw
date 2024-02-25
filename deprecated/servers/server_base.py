import socket


class ServerBase:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.connected = False
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn = None

    def __enter__(self):
        self.socket.bind((self.host, self.port))
        self.socket.listen()
        self.conn, addr = self.socket.accept()
        self.conn.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.__exit__(exc_type, exc_val, exc_tb)
