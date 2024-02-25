import torch


HOST = "localhost"
AGENT_PORT = 8000
EPISODES_NUM_GPU = 1500
EPISODES_NUM_CPU = 500
if torch.cuda.is_available():
    NUM_EPISODES = EPISODES_NUM_GPU
else:
    NUM_EPISODES = EPISODES_NUM_CPU
ENV_NAME = "CartPole-v1"
