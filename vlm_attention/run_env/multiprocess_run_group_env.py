import os
import time
import signal
import psutil
import shutil
from absl import app
from absl import flags
from datetime import datetime
import multiprocessing as mp
import traceback
import logging
import sys
import queue
import gc
from typing import Dict, List, Optional, Set
import copy

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from agent.vlm_agent_without_move_v5 import VLMAgentWithoutMove
from vlm_attention import ROOT_DIR, CONFIG_FILE_RELATIVE_PATH
from vlm_attention.env.env_core import SC2MultimodalEnv
from agent.test_agent import TestAgent

# 定义flags (保持不变)
FLAGS = flags.FLAGS
flags.DEFINE_string("map", "vlm_attention_1", "Name of the map to use")
flags.DEFINE_string("config_path", os.path.join(ROOT_DIR, CONFIG_FILE_RELATIVE_PATH), "Path to the configuration file")
flags.DEFINE_boolean("draw_grid", False, "Whether to draw grid on screenshots")
flags.DEFINE_boolean("annotate_units", True, "Whether to annotate units on screenshots")
flags.DEFINE_string("agent", "TestAgent", "Agent to use: random, flexible_random, VLMAgentWithoutMove, TestAgent")
flags.DEFINE_integer("groups", 3, "Number of groups to run")
flags.DEFINE_integer("num_processes", 2, "Number of processes to use per group")
flags.DEFINE_boolean("use_self_attention", True, "Whether to use self-attention in the agent")
flags.DEFINE_boolean("use_rag", True, "Whether to use RAG in the agent")

# Screen and map size flags (保持不变)
flags.DEFINE_integer('feature_screen_width', 256, 'Width of feature screen')
flags.DEFINE_integer('feature_screen_height', 256, 'Height of feature screen')
flags.DEFINE_integer('rgb_screen_width', 1920, 'Width of RGB screen')
flags.DEFINE_integer('rgb_screen_height', 1080, 'Height of RGB screen')
flags.DEFINE_integer('map_size_width', 64, 'Width of the map')
flags.DEFINE_integer('map_size_height', 64, 'Height of the map')


def terminate_process_safely(pid: int, process_type: str = "SC2") -> bool:
    """安全地终止进程，处理所有可能的异常"""
    try:
        proc = psutil.Process(pid)
        if not proc.is_running():
            logger.debug(f"{process_type} process {pid} is not running")
            return True

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except psutil.TimeoutExpired:
            logger.warning(f"Timeout waiting for {process_type} process {pid} to terminate, forcing kill")
            if proc.is_running():
                proc.kill()
        return True

    except psutil.NoSuchProcess:
        logger.debug(f"{process_type} process {pid} no longer exists")
        return True
    except psutil.AccessDenied:
        logger.warning(f"Access denied when terminating {process_type} process {pid}")
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)
            return True
        except ProcessLookupError:
            return True
        except Exception as e:
            logger.error(f"Error killing {process_type} process {pid}: {e}")
            return False
    except Exception as e:
        logger.error(f"Unexpected error terminating {process_type} process {pid}: {e}")
        return False


def cleanup_sc2_processes(pids: Set[int] = None):
    """清理SC2进程，可以指定特定的进程ID集合"""
    # 如果提供了特定的PID，先清理这些进程
    if pids:
        for pid in pids:
            terminate_process_safely(pid)

    # 清理所有剩余的SC2进程
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if 'SC2' in proc.info['name']:
                terminate_process_safely(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def run_episode(config: dict) -> dict:
    """运行单个episode的函数"""
    # 初始化变量
    env = None
    sc2_pid = None
    episode_id = config['episode_id']

    try:
        # 重新初始化flags
        flags.FLAGS(sys.argv[:1])

        # 解析配置
        group_id = config['group_id']
        base_log_dir = config['base_log_dir']
        env_config = config['env_config']
        agent_name = config['agent_name']
        agent_args = config['agent_args']

        # 创建目录
        save_dir = os.path.join(base_log_dir, f"group_{group_id}", f"episode_{episode_id}", "logs_file")
        replay_dir = os.path.join(base_log_dir, f"group_{group_id}", f"episode_{episode_id}", "replays")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(replay_dir, exist_ok=True)

        # 更新环境配置
        env_config = copy.deepcopy(env_config)
        env_config.update({
            'save_dir': save_dir,
            'replay_dir': replay_dir,
        })

        # 记录启动时间
        start_time = time.time()

        # 创建环境
        env = SC2MultimodalEnv(**env_config)

        # 查找SC2进程
        time.sleep(2)  # 等待SC2进程启动
        for proc in psutil.process_iter(['pid', 'name', 'create_time']):
            try:
                if 'SC2' in proc.info['name'] and proc.create_time() > start_time:
                    sc2_pid = proc.info['pid']
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # 创建agent
        agent_class = {
            "VLMAgentWithoutMove": VLMAgentWithoutMove,
            "FlexibleRandomAgent": FlexibleRandomAgent,
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
        step = 0
        max_steps = 1000

        time.sleep(5)  # 初始化延迟

        while not done and step < max_steps:
            if step > 0:
                time.sleep(0.1)

            action = agent.get_action(observation)
            observation, reward, done, _ = env.step(action)
            total_reward += reward

            step += 1
            logger.info(f"Group {group_id}, Episode {episode_id}, "
                        f"Step: {step}, Reward: {reward}, Total: {total_reward}")

        return {
            'episode_id': episode_id,
            'reward': total_reward,
            'sc2_pid': sc2_pid,
            'success': True
        }

    except Exception as e:
        logger.error(f"Error in episode {episode_id}: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'episode_id': episode_id,
            'reward': None,
            'sc2_pid': sc2_pid,
            'success': False,
            'error': str(e)
        }

    finally:
        if env is not None:
            try:
                env.close()
            except Exception as e:
                logger.error(f"Error closing environment: {e}")
        gc.collect()


def run_episode_with_queue(queue, config):
    """包装run_episode函数，将结果放入队列"""
    result = run_episode(config)
    queue.put(result)


def run_group(group_id: int, num_processes: int, base_log_dir: str,
              env_config: dict, agent_name: str, agent_args: dict) -> List[float]:
    """运行一个组的episodes"""
    results = []
    sc2_pids = set()
    processes = []
    queues = []

    try:
        logger.info(f"Starting group {group_id} with {num_processes} processes")

        # 同时启动组内所有进程
        for i in range(num_processes):
            config = {
                'episode_id': i,
                'group_id': group_id,
                'base_log_dir': base_log_dir,
                'env_config': env_config,
                'agent_name': agent_name,
                'agent_args': agent_args
            }

            queue = mp.Queue()
            process = mp.Process(target=run_episode_with_queue, args=(queue, config))

            processes.append(process)
            queues.append(queue)

            process.start()
            logger.info(f"Started process {i} in group {group_id}")
            time.sleep(2)  # 进程启动间隔

        # 等待所有进程完成
        completed = [False] * num_processes
        while not all(completed):
            for i in range(num_processes):
                if completed[i]:
                    continue

                process = processes[i]
                queue = queues[i]

                if not process.is_alive():
                    # 进程结束，获取结果
                    try:
                        result = queue.get_nowait()
                        if result['success']:
                            results.append(result['reward'])
                            if result['sc2_pid']:
                                sc2_pids.add(result['sc2_pid'])
                        else:
                            logger.error(
                                f"Episode {result['episode_id']} failed: {result.get('error', 'Unknown error')}")
                            if result['sc2_pid']:
                                sc2_pids.add(result['sc2_pid'])

                        completed[i] = True
                        process.join()
                        logger.info(f"Process {i} in group {group_id} completed with result")
                    except queue.Empty:
                        logger.warning(f"Process {i} ended but no result available")
                        completed[i] = True
                        process.join()

                elif process.exitcode is not None:
                    # 进程已结束但没有正常退出
                    logger.warning(f"Process {i} exited with code {process.exitcode}")
                    completed[i] = True
                    process.join()

            if not all(completed):
                time.sleep(1)  # 等待检查间隔

        logger.info(f"All processes in group {group_id} have completed")
        return results

    except Exception as e:
        logger.error(f"Error in group {group_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise

    finally:
        # 确保所有进程都被终止
        for i, process in enumerate(processes):
            if process.is_alive():
                logger.warning(f"Force terminating process {i} in group {group_id}")
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)

        # 清理SC2进程
        logger.info(f"Cleaning up SC2 processes for group {group_id}")
        try:
            cleanup_sc2_processes(sc2_pids)
        except Exception as e:
            logger.error(f"Error during SC2 process cleanup: {e}")
            logger.error(traceback.format_exc())

        # 清理资源
        gc.collect()
        logger.info(f"Cleanup completed for group {group_id}")


def main(argv):
    flags.FLAGS(argv)

    # 创建基础目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = os.path.abspath(os.path.join("log", FLAGS.agent, timestamp))
    os.makedirs(base_log_dir, exist_ok=True)

    # 初始化环境配置
    env_config = {
        'map_name': FLAGS.map,
        'timestamp': timestamp,
        'feature_dims': (FLAGS.feature_screen_width, FLAGS.feature_screen_height),
        'rgb_dims': (FLAGS.rgb_screen_width, FLAGS.rgb_screen_height),
        'map_size': (FLAGS.map_size_width, FLAGS.map_size_height)
    }

    # 设置agent配置
    agent_args = {
        'config_path': FLAGS.config_path,
        'draw_grid': FLAGS.draw_grid,
        'annotate_units': FLAGS.annotate_units,
        'use_self_attention': FLAGS.use_self_attention,
        'use_rag': FLAGS.use_rag
    }

    try:
        # 串行运行每个组
        for group in range(FLAGS.groups):
            logger.info(f"Starting group {group + 1}/{FLAGS.groups}")

            try:
                # 运行当前组
                results = run_group(
                    group_id=group,
                    num_processes=FLAGS.num_processes,
                    base_log_dir=base_log_dir,
                    env_config=env_config,
                    agent_name=FLAGS.agent,
                    agent_args=agent_args
                )

                logger.info(f"Group {group + 1} completed with results: {results}")

            except Exception as e:
                logger.error(f"Error in group {group}: {str(e)}")
                logger.error(traceback.format_exc())
                # 继续下一个组
                continue

            finally:
                # 组间清理和延迟
                logger.info(f"Cleaning up after group {group + 1}...")
                cleanup_sc2_processes()  # 清理所有SC2进程
                time.sleep(10)  # 组间延迟
                gc.collect()

        logger.info("All groups completed")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        logger.error(traceback.format_exc())
        raise

    finally:
        # 最终清理
        cleanup_sc2_processes()
        gc.collect()


if __name__ == "__main__":
    try:
        app.run(main)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc())

        # 确保清理所有SC2进程
        for proc in psutil.process_iter(['name']):
            try:
                if 'SC2' in proc.info['name']:
                    proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        sys.exit(1)