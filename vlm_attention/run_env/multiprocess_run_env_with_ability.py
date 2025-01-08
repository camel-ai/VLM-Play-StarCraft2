import os
import time
import psutil
import logging
import multiprocessing as mp
from absl import app
from absl import flags
from datetime import datetime
from typing import Type, Union

from agent.RandomAgent import RandomAgent
from agent.vlm_agent_without_move_v5 import VLMAgentWithoutMove
from agent.vlm_agent_v6 import VLMAgent
from agent.test_agent_with_ability import TestAgent
from vlm_attention import ROOT_DIR, CONFIG_FILE_RELATIVE_PATH
from vlm_attention.env.env_core_with_ability import SC2MultimodalEnv

map_list = ["ability_map_8marine_3marauder_1medivac_1tank"]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string("map", map_list[0], "Name of the map to use,we can get from map_list")
flags.DEFINE_string("config_path", os.path.join(ROOT_DIR, CONFIG_FILE_RELATIVE_PATH), "Path to the configuration file")
flags.DEFINE_boolean("draw_grid", False, "Whether to draw grid on screenshots")
flags.DEFINE_boolean("annotate_units", True, "Whether to annotate units on screenshots")
flags.DEFINE_string("agent", "TestAgent", "Agent to use:RandomAgent, VLMAgentWithoutMove, TestAgent,VLMAgent")
flags.DEFINE_boolean("use_self_attention", True, "Whether to use self-attention in the agent")
flags.DEFINE_boolean("use_rag", True, "Whether to use RAG in the agent")
flags.DEFINE_integer("max_steps", 2000, "Maximum steps per episode")
flags.DEFINE_integer("num_processes", 4, "Number of parallel processes to use")

# Screen and map size flags
flags.DEFINE_integer('feature_screen_width', 256, 'Width of feature screen')
flags.DEFINE_integer('feature_screen_height', 256, 'Height of feature screen')
flags.DEFINE_integer('rgb_screen_width', 1920, 'Width of RGB screen')
flags.DEFINE_integer('rgb_screen_height', 1080, 'Height of RGB screen')
flags.DEFINE_integer('map_size_width', 64, 'Width of the map')
flags.DEFINE_integer('map_size_height', 64, 'Height of the map')




def get_agent_class(agent_name: str) -> Type[Union[VLMAgentWithoutMove, RandomAgent, TestAgent]]:
    """Get the agent class based on the agent name."""
    agent_classes = {
        "VLMAgentWithoutMove": VLMAgentWithoutMove,
        "RandomAgent": RandomAgent,
        "TestAgent": TestAgent,
        "VLMAgent": VLMAgent
    }
    if agent_name not in agent_classes:
        raise ValueError(f"Unknown agent: {agent_name}. Valid options are: {', '.join(agent_classes.keys())}")
    return agent_classes[agent_name]


def run_episode(config: dict) -> dict:
    """Run a single episode in a subprocess."""
    env = None
    episode_id = config['episode_id']

    try:
        # 创建日志目录
        base_log_dir = config['base_log_dir']
        save_dir = os.path.join(base_log_dir, f"episode_{episode_id}", "logs_file")
        replay_dir = os.path.join(base_log_dir, f"episode_{episode_id}", "replays")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(replay_dir, exist_ok=True)

        # 创建环境
        env = SC2MultimodalEnv(
            map_name=config['map_name'],
            save_dir=save_dir,
            replay_dir=replay_dir,
            timestamp=config['timestamp'],
            feature_dims=config['feature_dims'],
            rgb_dims=config['rgb_dims'],
            map_size=config['map_size'],
        )

        # 创建智能体
        agent_class = get_agent_class(config['agent_name'])
        agent = agent_class(
            action_space=env.action_space,
            config_path=config['config_path'],
            draw_grid=config['draw_grid'],
            annotate_units=config['annotate_units'],
            save_dir=save_dir,
            use_rag=config['use_rag'],
            use_self_attention=config['use_self_attention'],
        )

        # 运行单个episode
        observation = env.reset()
        total_reward = 0
        done = False
        step = 0

        # 初始化等待
        time.sleep(5)

        while not done and step < config['max_steps']:
            if step > 0:
                time.sleep(0.1)

            action = agent.get_action(observation)
            observation, reward, done, info = env.step(action)

            # 检查是否有环境错误
            if info.get("error"):
                logger.error(f"Environment error: {info['error']}")
                break

            total_reward += reward
            step += 1

            logger.info(f"Episode {episode_id}, Step {step}, Reward: {reward}, Total Reward: {total_reward}")

        return {
            'episode_id': episode_id,
            'total_reward': total_reward,
            'steps': step,
            'success': True
        }

    except Exception as e:
        logger.error(f"Error in episode {episode_id}: {str(e)}")
        return {
            'episode_id': episode_id,
            'total_reward': None,
            'steps': step,
            'success': False,
            'error': str(e)
        }

    finally:
        if env is not None:
            try:
                env.close()
                time.sleep(2)
            except Exception as e:
                logger.error(f"Error closing environment: {str(e)}")



def init_worker():
    """Initialize worker process."""
    import sys
    sys.argv = sys.argv[:1]
    flags.FLAGS(sys.argv)


def main(argv):
    """Main function to run multiple episodes in parallel."""
    flags.FLAGS(argv)




    # 创建日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = os.path.abspath(os.path.join("log", FLAGS.agent, timestamp))
    os.makedirs(base_log_dir, exist_ok=True)

    # 准备每个进程的配置
    configs = []
    for i in range(FLAGS.num_processes):
        config = {
            'episode_id': i,
            'base_log_dir': base_log_dir,
            'map_name': FLAGS.map,
            'timestamp': timestamp,
            'feature_dims': (FLAGS.feature_screen_width, FLAGS.feature_screen_height),
            'rgb_dims': (FLAGS.rgb_screen_width, FLAGS.rgb_screen_height),
            'map_size': (FLAGS.map_size_width, FLAGS.map_size_height),
            'agent_name': FLAGS.agent,
            'config_path': FLAGS.config_path,
            'draw_grid': FLAGS.draw_grid,
            'annotate_units': FLAGS.annotate_units,
            'use_rag': FLAGS.use_rag,
            'use_self_attention': FLAGS.use_self_attention,
            'max_steps': FLAGS.max_steps
        }
        configs.append(config)

    # 使用进程池运行episodes
    with mp.Pool(FLAGS.num_processes, initializer=init_worker) as pool:
        results = pool.map(run_episode, configs)

    # 处理结果
    for result in results:
        if result['success']:
            logger.info(f"Episode {result['episode_id']} completed with total reward {result['total_reward']} in {result['steps']} steps")
        else:
            logger.error(f"Episode {result['episode_id']} failed: {result.get('error', 'Unknown error')}")


if __name__ == '__main__':
    app.run(main)