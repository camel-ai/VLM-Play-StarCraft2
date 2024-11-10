import copy
import gc
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from datetime import datetime

import psutil
import websocket
from absl import app
from absl import flags

from pysc2.lib import protocol

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from agent.RandomAgent import RandomAgent
from agent.vlm_agent_without_move_v5 import VLMAgentWithoutMove
from vlm_attention import ROOT_DIR, CONFIG_FILE_RELATIVE_PATH
from vlm_attention.env.env_core import SC2MultimodalEnv
from agent.test_agent import TestAgent

map_list = ["vlm_attention_1",
            "2c_vs_64zg_vlm_attention",
            "2m_vs_1z_vlm_attention",
            "2s_vs_1sc_vlm_attention",
            "2s3z_vlm_attention",
            "3m_vlm_attention",
            "3s_vs_3z_vlm_attention"]
# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string("map", map_list[0], "Name of the map to use")
flags.DEFINE_string("config_path", os.path.join(ROOT_DIR, CONFIG_FILE_RELATIVE_PATH), "Path to the configuration file")
flags.DEFINE_boolean("draw_grid", False, "Whether to draw grid on screenshots")
flags.DEFINE_boolean("annotate_units", True, "Whether to annotate units on screenshots")
flags.DEFINE_string("agent", "TestAgent", "Agent to use:RandomAgent, VLMAgentWithoutMove, TestAgent")
flags.DEFINE_integer("num_processes", 4, "Number of parallel processes to use")
flags.DEFINE_boolean("use_self_attention", True, "Whether to use self-attention in the agent")
flags.DEFINE_boolean("use_rag", True, "Whether to use RAG in the agent")

# Screen and map size flags
flags.DEFINE_integer('feature_screen_width', 256, 'Width of feature screen')
flags.DEFINE_integer('feature_screen_height', 256, 'Height of feature screen')
flags.DEFINE_integer('rgb_screen_width', 1920, 'Width of RGB screen')
flags.DEFINE_integer('rgb_screen_height', 1080, 'Height of RGB screen')
flags.DEFINE_integer('map_size_width', 64, 'Width of the map')
flags.DEFINE_integer('map_size_height', 64, 'Height of the map')


def terminate_process_safely(pid: int):
    """Safely terminate a process"""
    try:
        proc = psutil.Process(pid)
        proc.terminate()
        proc.wait(timeout=5)
    except:
        try:
            os.kill(pid, signal.SIGTERM)
        except:
            pass





def run_episode(config: dict) -> dict:
    """简化的episode运行函数"""
    env = None
    episode_id = config['episode_id']

    try:
        # 保持原有的目录结构和命名
        base_log_dir = config['base_log_dir']
        env_config = config['env_config']
        agent_name = config['agent_name']
        agent_args = config['agent_args']

        save_dir = os.path.join(base_log_dir, f"episode_{episode_id}", "logs_file")
        replay_dir = os.path.join(base_log_dir, f"episode_{episode_id}", "replays")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(replay_dir, exist_ok=True)

        # 更新环境配置
        env_config = copy.deepcopy(env_config)
        env_config.update({
            'save_dir': save_dir,
            'replay_dir': replay_dir,
        })

        # 创建环境和智能体
        env = SC2MultimodalEnv(**env_config)
        agent_class = {
            "VLMAgentWithoutMove": VLMAgentWithoutMove,
            "RandomAgent": RandomAgent,
            "TestAgent": TestAgent
        }[agent_name]

        agent_args = copy.deepcopy(agent_args)
        agent_args.update({
            'save_dir': save_dir,
            'action_space': env.action_space
        })
        agent = agent_class(**agent_args)

        # 运行episode
        observation = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(observation)
            observation, reward, done, info = env.step(action)

            if info.get("error"):
                raise RuntimeError(f"Environment error: {info['error']}")

            total_reward += reward
            logger.info(f"Episode {episode_id}, Reward: {reward}, Total: {total_reward}")

        return {
            'episode_id': episode_id,
            'reward': total_reward,
            'success': True
        }

    except Exception as e:
        logger.error(f"Error in episode {episode_id}: {str(e)}")
        return {
            'episode_id': episode_id,
            'reward': None,
            'success': False,
            'error': str(e)
        }

    finally:
        if env is not None:
            env.close()

def init_worker():
    """Initialize worker process"""
    import sys
    sys.argv = sys.argv[:1]
    flags.FLAGS(sys.argv)
def main(argv):
    # 保持原有的flag定义和配置
    flags.FLAGS(argv)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = os.path.abspath(os.path.join("log", FLAGS.agent, timestamp))
    os.makedirs(base_log_dir, exist_ok=True)

    # 保持原有的环境配置
    env_config = {
        'map_name': FLAGS.map,
        'timestamp': timestamp,
        'feature_dims': (FLAGS.feature_screen_width, FLAGS.feature_screen_height),
        'rgb_dims': (FLAGS.rgb_screen_width, FLAGS.rgb_screen_height),
        'map_size': (FLAGS.map_size_width, FLAGS.map_size_height)
    }

    agent_args = {
        'config_path': FLAGS.config_path,
        'draw_grid': FLAGS.draw_grid,
        'annotate_units': FLAGS.annotate_units,
        'use_self_attention': FLAGS.use_self_attention,
        'use_rag': FLAGS.use_rag
    }

    # 准备每个进程的配置
    configs = []
    for i in range(FLAGS.num_processes):
        config = {
            'episode_id': i,
            'base_log_dir': base_log_dir,
            'env_config': env_config,
            'agent_name': FLAGS.agent,
            'agent_args': agent_args
        }
        configs.append(config)

    # 使用进程池运行episodes
    with mp.Pool(FLAGS.num_processes, initializer=init_worker) as pool:
        results = pool.map(run_episode, configs)

    # 处理结果
    for result in results:
        if result['success']:
            logger.info(f"Episode {result['episode_id']} completed with reward {result['reward']}")
        else:
            logger.error(f"Episode {result['episode_id']} failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    app.run(main)