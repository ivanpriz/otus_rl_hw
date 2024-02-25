import json
import sys

from .server_base import ServerBase

HOST = "localhost"
PORT = 14500

# u32 representing how much bytes next state will be
HEADER_SIZE_BYTES: int = 4


class GameStateServer(ServerBase):
    def send_state_to_client(self, state: dict):
        data = json.dumps(state).encode("utf-8")
        size = sys.getsizeof(data)
        print(f"Size of state to send (bytes): {size}")
        self.conn.sendall(size.to_bytes(HEADER_SIZE_BYTES, byteorder="big"))
        print("Sending state data..")
        self.conn.sendall(data)
        print("State sent!")


if __name__ == '__main__':
    server = GameStateServer(host=HOST, port=PORT)
    print("Starting server...")
    with server as s:
        print("Server accepted connection!")
        s.send_state_to_client({"hello": "world"})

# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     s.bind((HOST, PORT_GAME_STATE))
#     s.listen()
#     conn, addr = s.accept()
#     with conn:
#         print(f"Connected by {addr}")
#         while True:
#             data = conn.recv(1024)
#             print(data)
#             time.sleep(1)
#             # if not data:
#             #     break
#             # conn.sendall(data)
