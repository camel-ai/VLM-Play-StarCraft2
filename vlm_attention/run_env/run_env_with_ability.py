import os
import time
import psutil
import logging
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
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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

# Screen and map size flags
flags.DEFINE_integer('feature_screen_width', 256, 'Width of feature screen')
flags.DEFINE_integer('feature_screen_height', 256, 'Height of feature screen')
flags.DEFINE_integer('rgb_screen_width', 1920, 'Width of RGB screen')
flags.DEFINE_integer('rgb_screen_height', 1080, 'Height of RGB screen')
flags.DEFINE_integer('map_size_width', 64, 'Width of the map')
flags.DEFINE_integer('map_size_height', 64, 'Height of the map')


def terminate_sc2_processes():
    """Terminate any running SC2 processes."""
    for proc in psutil.process_iter(['name']):
        if 'SC2' in proc.info['name']:
            try:
                proc.kill()
                proc.wait(timeout=3)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                pass
    time.sleep(1)


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


def main(argv):
    """Main function to run a single episode."""
    flags.FLAGS(argv)

    # 确保没有残留的SC2进程
    terminate_sc2_processes()

    # 创建日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = os.path.abspath(os.path.join("log", FLAGS.agent, timestamp))
    save_dir = os.path.join(base_log_dir, "logs_file")
    replay_dir = os.path.join(base_log_dir, "replays")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(replay_dir, exist_ok=True)

    env = None
    try:
        # 创建环境
        env = SC2MultimodalEnv(
            map_name=FLAGS.map,
            save_dir=save_dir,
            replay_dir=replay_dir,
            timestamp=timestamp,
            feature_dims=(FLAGS.feature_screen_width, FLAGS.feature_screen_height),
            rgb_dims=(FLAGS.rgb_screen_width, FLAGS.rgb_screen_height),
            map_size=(FLAGS.map_size_width, FLAGS.map_size_height),
        )

        # 创建智能体
        agent_class = get_agent_class(FLAGS.agent)
        agent = agent_class(
            action_space=env.action_space,
            config_path=FLAGS.config_path,
            draw_grid=FLAGS.draw_grid,
            annotate_units=FLAGS.annotate_units,
            save_dir=save_dir,
            use_rag=FLAGS.use_rag,
            use_self_attention=FLAGS.use_self_attention,
        )

        # 运行单个episode
        observation = env.reset()
        total_reward = 0
        done = False
        step = 0

        # 初始化等待
        time.sleep(5)

        while not done and step < FLAGS.max_steps:
            if step > 0:
                time.sleep(0.1)

            action = agent.get_action(observation)
            observation, reward, done, info = env.step(action)

            # 检查是否有环境错误
            if info.get("error"):
                import traceback
                logging.error(f"Environment error: {info['error']}")
                logging.error(f"Full traceback:\n{traceback.format_exc()}")
                break

            total_reward += reward
            step += 1

            print(f"Step: {step}, Reward: {reward}, Total Reward: {total_reward}")

        print(f"\nEpisode finished. Total Steps: {step}, Total Reward: {total_reward}")

    except Exception as e:
        import traceback
        logging.error(f"Critical error occurred: {str(e)}")
        logging.error(f"Full traceback:\n{traceback.format_exc()}")
        raise  # 重新抛出异常以显示完整的堆栈跟踪

    finally:
        # 清理资源
        if env is not None:
            try:
                env.close()
                time.sleep(2)
            except Exception as e:
                import traceback
                logging.error(f"Error closing environment: {str(e)}")
                logging.error(f"Close environment traceback:\n{traceback.format_exc()}")

        terminate_sc2_processes()


if __name__ == '__main__':
    app.run(main)
