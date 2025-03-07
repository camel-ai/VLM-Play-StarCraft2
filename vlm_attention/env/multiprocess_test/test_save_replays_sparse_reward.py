import os
import multiprocessing
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from absl import app, flags
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
save replay & sparse reward test code for starcraft 2

"""

FLAGS = flags.FLAGS
map_list = ["2m_vs_1z_vlm_attention",
            "3m_vlm_attention",
            "2s_vs_1sc_vlm_attention"]


class SimpleAgent(base_agent.BaseAgent):
    def step(self, obs):
        super(SimpleAgent, self).step(obs)
        if obs.observation.game_loop[0] < 2:
            return actions.RAW_FUNCTIONS.raw_move_camera([32, 32])
        return actions.FUNCTIONS.no_op()


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def run_episode(args):
    episode, map_name = args
    agent = SimpleAgent()

    current_dir = os.path.abspath(os.path.dirname(__file__))
    replay_dir = os.path.join(current_dir, 'replays', f'worker_{episode}')
    ensure_dir(replay_dir)

    logger.info(f"Worker {episode} starting with map {map_name}")
    logger.info(f"Replay directory (absolute path): {replay_dir}")

    try:
        with sc2_env.SC2Env(
                map_name=map_name,
                players=[sc2_env.Agent(sc2_env.Race.terran),
                         sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    use_feature_units=True,
                    raw_crop_to_playable_area=True,
                    use_camera_position=True
                ),
                step_mul=8,
                game_steps_per_episode=200,
                visualize=False,
        ) as env:

            logger.info(f"Worker {episode}: Environment created")

            timesteps = env.reset()
            agent.reset()

            while True:
                step_actions = [agent.step(timesteps[0])]
                if timesteps[0].last():
                    # 获取结果
                    outcome = timesteps[0].reward  # 因为我们没有设置score_index，所以reward就是outcome
                    result_str = "victory" if outcome == 1 else "defeat" if outcome == -1 else "tie"

                    logger.info(f"Worker {episode}: Episode finished with result: {result_str}")
                    try:
                        # 在replay名称中包含结果
                        save_path = env.save_replay(
                            replay_dir,
                            prefix=f"test_replay_{episode}_{result_str}_"
                        )
                        logger.info(f"Worker {episode}: Replay saved to {save_path}")
                    except Exception as e:
                        logger.error(f"Worker {episode}: Failed to save replay: {e}")
                    break
                timesteps = env.step(step_actions)

    except Exception as e:
        logger.error(f"Worker {episode}: Error during episode: {e}")


def init_worker():
    import sys
    sys.argv = sys.argv[:1]
    flags.FLAGS(sys.argv)


def main(argv):
    logger.info("Starting main process")
    flags.FLAGS(argv)

    current_dir = os.path.abspath(os.path.dirname(__file__))
    main_replay_dir = os.path.join(current_dir, 'replays')
    ensure_dir(main_replay_dir)
    logger.info(f"Created main replay directory (absolute path): {main_replay_dir}")

    num_episodes = 3
    map_name = map_list[1]
    args_list = [(i, map_name) for i in range(num_episodes)]
    with multiprocessing.Pool(num_episodes, initializer=init_worker) as pool:
        logger.info(f"Created process pool with {num_episodes} workers")
        pool.map(run_episode, args_list)

    logger.info("All episodes completed")

    # 检查replay文件
    for i in range(num_episodes):
        replay_dir = os.path.join(main_replay_dir, f'worker_{i}')
        if os.path.exists(replay_dir):
            replays = [f for f in os.listdir(replay_dir) if f.endswith('.SC2Replay')]
            logger.info(f"Worker {i} replays: {replays}")
        else:
            logger.warning(f"Replay directory for worker {i} not found")


if __name__ == "__main__":
    app.run(main)