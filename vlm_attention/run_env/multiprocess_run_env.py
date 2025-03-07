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
from agent.vlm_agent_v6 import VLMAgent
from agent.test_agent import TestAgent



"""
multi process run the environment with one player, without ability support

"""


map_list = ["2c_vs_64zg_vlm_attention",
            "2m_vs_1z_vlm_attention",
            "2s_vs_1sc_vlm_attention",
            "3s_vs_3z_vlm_attention",
            "6reaper_vs8zealot_vlm_attention",
            "8marine_1medvac_vs_2tank",
            "8marine_2tank_vs_zerglings_banelings_vlm_attention",
            "2bc1prism_vs_8m_vlm_attention",
            "2s3z_vlm_attention",
            "3m_vlm_attention",
            "vlm_attention_1",
            "ability_map_8marine_3marauder_1medivac_1tank"]
# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string("map", map_list[6], "Name of the map to use")
flags.DEFINE_string("config_path", os.path.join(ROOT_DIR, CONFIG_FILE_RELATIVE_PATH), "Path to the configuration file")
flags.DEFINE_boolean("draw_grid", True, "Whether to draw grid on screenshots")
flags.DEFINE_boolean("annotate_units", True, "Whether to annotate units on screenshots")
flags.DEFINE_string("agent", "VLMAgent", "Agent to use:RandomAgent, VLMAgentWithoutMove, TestAgent,VLMAgent")
flags.DEFINE_integer("num_processes", 4, "Number of parallel processes to use")
flags.DEFINE_boolean("use_self_attention", True, "Whether to use self-attention in the agent")
flags.DEFINE_boolean("use_rag", True, "Whether to use RAG in the agent")
flags.DEFINE_boolean("use_proxy", True, "Whether to use proxy in the agent, in china, gpt models need proxy")

# Screen and map size flags
flags.DEFINE_integer('feature_screen_width', 256, 'Width of feature screen')
flags.DEFINE_integer('feature_screen_height', 256, 'Height of feature screen')
flags.DEFINE_integer('rgb_screen_width', 1920, 'Width of RGB screen')
flags.DEFINE_integer('rgb_screen_height', 1080, 'Height of RGB screen')
flags.DEFINE_integer('map_size_width', 64, 'Width of the map')
flags.DEFINE_integer('map_size_height', 64, 'Height of the map')

flags.DEFINE_string('model_name', default="openai", help="which model we used ")


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

        logger.info(f"Creating environment with config: {env_config}")
        env = SC2MultimodalEnv(**env_config)

        # 创建智能体
        agent_class = {
            "VLMAgentWithoutMove": VLMAgentWithoutMove,
            "RandomAgent": RandomAgent,
            "TestAgent": TestAgent,
            "VLMAgent": VLMAgent
        }[agent_name]

        agent_args = copy.deepcopy(agent_args)
        agent_args.update({
            'save_dir': save_dir,
            'action_space': env.action_space
        })
        
        logger.info(f"Creating agent {agent_name} with args: {agent_args}")
        agent = agent_class(**agent_args)

        # 运行episode
        logger.info(f"Starting episode {episode_id}")
        observation = env.reset()
        total_reward = 0
        done = False

        while not done:
            try:
                action = agent.get_action(observation)
                observation, reward, done, info = env.step(action)

                if info.get("error"):
                    logger.error(f"Environment error in episode {episode_id}: {info['error']}")
                    raise RuntimeError(f"Environment error: {info['error']}")

                total_reward += reward
                logger.info(f"Episode {episode_id}, Reward: {reward}, Total: {total_reward}")

            except Exception as step_error:
                logger.error(f"Error during step in episode {episode_id}:")
                logger.error(traceback.format_exc())  # 打印完整的错误堆栈
                raise step_error

        return {
            'episode_id': episode_id,
            'reward': total_reward,
            'success': True
        }

    except Exception as e:
        error_msg = f"Error in episode {episode_id}:\n{traceback.format_exc()}"
        logger.error(error_msg)
        return {
            'episode_id': episode_id,
            'reward': None,
            'success': False,
            'error': error_msg
        }

    finally:
        if env is not None:
            logger.info(f"Closing environment for episode {episode_id}")
            env.close()


def init_worker():
    """Initialize worker process"""
    import sys
    sys.argv = sys.argv[:1]
    flags.FLAGS(sys.argv)


def main(argv):
    # 保持原有的flag定义和配置
    flags.FLAGS(argv)

    # 修改这里：添加地图名称到目录路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = os.path.abspath(os.path.join(
        "log",
        FLAGS.agent,
        f"{FLAGS.model_name}_{timestamp}_{FLAGS.map}"  # 在时间戳后添加地图名称
    ))
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
        'use_rag': FLAGS.use_rag,
        'use_proxy': FLAGS.use_proxy
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
    success_count = 0
    failure_count = 0
    for result in results:
        if result['success']:
            success_count += 1
            logger.info(f"Episode {result['episode_id']} completed with reward {result['reward']}")
        else:
            failure_count += 1
            logger.error(f"Episode {result['episode_id']} failed:")
            logger.error(result['error'])  # 这里会显示完整的错误堆栈

    logger.info(f"Run completed. Successful episodes: {success_count}, Failed episodes: {failure_count}")

    # 如果有失败的episode，返回非零退出码
    if failure_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    app.run(main)
