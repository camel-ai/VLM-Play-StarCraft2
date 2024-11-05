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

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from agent.RandomAgent import RandomAgent
from agent.vlm_agent_without_move_v5 import VLMAgentWithoutMove
from vlm_attention import ROOT_DIR, CONFIG_FILE_RELATIVE_PATH
from vlm_attention.env.env_core import SC2MultimodalEnv
from agent.test_agent import TestAgent

# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string("map", "vlm_attention_1", "Name of the map to use")
flags.DEFINE_string("config_path", os.path.join(ROOT_DIR, CONFIG_FILE_RELATIVE_PATH), "Path to the configuration file")
flags.DEFINE_boolean("draw_grid", False, "Whether to draw grid on screenshots")
flags.DEFINE_boolean("annotate_units", True, "Whether to annotate units on screenshots")
flags.DEFINE_string("agent", "RandomAgent", "Agent to use:RandomAgent, VLMAgentWithoutMove, TestAgent")
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


def cleanup_sc2_processes():
    """Clean up SC2 processes"""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if 'SC2' in proc.info['name']:
                terminate_process_safely(proc.info['pid'])
        except:
            continue


def run_episode(config: dict) -> dict:
    """Run a single episode with retry mechanism"""
    env = None
    sc2_pid = None
    episode_id = config['episode_id']
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Reset flags
            flags.FLAGS(sys.argv[:1])

            base_log_dir = config['base_log_dir']
            env_config = config['env_config']
            agent_name = config['agent_name']
            agent_args = config['agent_args']

            # Create directories
            save_dir = os.path.join(base_log_dir, f"episode_{episode_id}", "logs_file")
            replay_dir = os.path.join(base_log_dir, f"episode_{episode_id}", "replays")
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(replay_dir, exist_ok=True)

            # Update environment configuration
            env_config = copy.deepcopy(env_config)
            env_config.update({
                'save_dir': save_dir,
                'replay_dir': replay_dir,
            })

            # Create environment
            start_time = time.time()
            env = SC2MultimodalEnv(**env_config)

            # Find SC2 process (but don't fail if not found)
            time.sleep(2)
            for proc in psutil.process_iter(['pid', 'name', 'create_time']):
                try:
                    if 'SC2' in proc.info['name'] and proc.create_time() > start_time:
                        sc2_pid = proc.info['pid']
                        break
                except:
                    continue

            # Create agent
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

            # Run episode
            observation = env.reset()
            total_reward = 0
            done = False
            step = 0
            max_steps = 1000

            time.sleep(5)

            while not done and step < max_steps:
                try:
                    if step > 0:
                        time.sleep(0.1)

                    action = agent.get_action(observation)
                    observation, reward, done, info = env.step(action)

                    if info.get("error"):
                        # 给SC2一些时间保存回放
                        time.sleep(5)
                        raise RuntimeError(f"Environment error: {info['error']}")

                    total_reward += reward
                    step += 1
                    logger.info(f"Episode {episode_id}, Step: {step}, Reward: {reward}, Total: {total_reward}")

                except Exception as e:
                    logger.warning(f"Error during step execution in episode {episode_id}: {str(e)}")
                    # 在清理环境前等待回放保存
                    time.sleep(5)
                    if env is not None:
                        try:
                            # 确保环境正确关闭
                            env.close()
                            time.sleep(2)  # 给予额外时间完成清理
                        except:
                            pass
                    env = None
                    retry_count += 1
                    if retry_count >= max_retries:
                        # 最后一次重试也失败时，再次等待确保回放保存
                        time.sleep(5)
                        raise
                    time.sleep(2 * retry_count)
                    break

            # 如果正常完成游戏，也等待足够时间保存回放
            if done or step >= max_steps:
                time.sleep(5)  # 等待回放保存
                return {
                    'episode_id': episode_id,
                    'reward': total_reward,
                    'sc2_pid': sc2_pid,
                    'success': True
                }

        except Exception as e:
            logger.error(f"Error in episode {episode_id}: {str(e)}")
            logger.error(traceback.format_exc())
            retry_count += 1
            # 确保有足够时间保存回放
            time.sleep(5)
            if retry_count >= max_retries:
                return {
                    'episode_id': episode_id,
                    'reward': None,
                    'sc2_pid': sc2_pid,
                    'success': False,
                    'error': str(e)
                }
            time.sleep(2 * retry_count)

        finally:
            if env is not None:
                try:
                    # 在关闭环境前给予足够时间保存回放
                    time.sleep(5)
                    env.close()
                    time.sleep(2)  # 额外等待时间确保清理完成
                except:
                    pass
            gc.collect()

        return {
            'episode_id': episode_id,
            'reward': None,
            'sc2_pid': sc2_pid,
            'success': False,
            'error': "Max retries exceeded"
        }

def run_episode_with_queue(queue, config):
    """Wrapper to put episode result in queue"""
    result = run_episode(config)
    queue.put(result)


def main(argv):
    flags.FLAGS(argv)

    # Create base directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = os.path.abspath(os.path.join("log", FLAGS.agent, timestamp))
    os.makedirs(base_log_dir, exist_ok=True)

    # Initialize environment configuration
    env_config = {
        'map_name': FLAGS.map,
        'timestamp': timestamp,
        'feature_dims': (FLAGS.feature_screen_width, FLAGS.feature_screen_height),
        'rgb_dims': (FLAGS.rgb_screen_width, FLAGS.rgb_screen_height),
        'map_size': (FLAGS.map_size_width, FLAGS.map_size_height)
    }

    # Set agent configuration
    agent_args = {
        'config_path': FLAGS.config_path,
        'draw_grid': FLAGS.draw_grid,
        'annotate_units': FLAGS.annotate_units,
        'use_self_attention': FLAGS.use_self_attention,
        'use_rag': FLAGS.use_rag
    }

    try:
        # Start processes
        processes = []
        queues = []

        for i in range(FLAGS.num_processes):
            config = {
                'episode_id': i,
                'base_log_dir': base_log_dir,
                'env_config': env_config,
                'agent_name': FLAGS.agent,
                'agent_args': agent_args
            }

            queue = mp.Queue()
            process = mp.Process(target=run_episode_with_queue, args=(queue, config))

            processes.append(process)
            queues.append(queue)

            process.start()
            logger.info(f"Started process {i}")
            time.sleep(2)  # Delay between process starts

        # Wait for all processes to complete
        results = []
        for i, (process, queue) in enumerate(zip(processes, queues)):
            process.join()
            try:
                result = queue.get()
                if result['success']:
                    results.append(result['reward'])
                    logger.info(f"Process {i} completed successfully")
                else:
                    logger.error(f"Process {i} failed: {result.get('error', 'Unknown error')}")
            except:
                logger.error(f"Failed to get result from process {i}")

        logger.info(f"All processes completed with results: {results}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        logger.error(traceback.format_exc())
        raise

    finally:
        # Clean up
        cleanup_sc2_processes()
        gc.collect()


if __name__ == "__main__":
    try:
        app.run(main)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
        cleanup_sc2_processes()
        sys.exit(1)