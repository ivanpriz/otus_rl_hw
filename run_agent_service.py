import uvicorn

from rest_agent.src.api import app
from game_config import AGENT_PORT


if __name__ == '__main__':
    uvicorn.run(app, port=AGENT_PORT)
