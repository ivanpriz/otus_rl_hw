import os

from game.game_service import GameService


if __name__ == '__main__':
    game_service = GameService(
        episodes_num=100,
        max_steps_per_episode=5000000,
        graphs_dir_path=os.path.join(
            os.curdir,
            "dqn_replay_http"
        )
    )
    game_service.run()
