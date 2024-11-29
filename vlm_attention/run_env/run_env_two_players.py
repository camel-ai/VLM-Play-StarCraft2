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
from agent.test_agent import TestAgent
from vlm_attention import ROOT_DIR, CONFIG_FILE_RELATIVE_PATH
from vlm_attention.env.two_players_env_core import SC2MultimodalTwoPlayerEnv  # 改用两人版环境

map_list = ["vlm_attention_1_two_players", "vlm_attention_2_terran_vs_terran_two_players"]

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string("map", map_list[1], "Name of the map to use,we can get from map_list")
flags.DEFINE_string("config_path", os.path.join(ROOT_DIR, CONFIG_FILE_RELATIVE_PATH), "Path to the configuration file")
flags.DEFINE_boolean("draw_grid", False, "Whether to draw grid on screenshots")
flags.DEFINE_boolean("annotate_units", True, "Whether to annotate units on screenshots")
# 为两个agent分别定义flags
flags.DEFINE_string("agent1", "TestAgent", "First agent type: RandomAgent, VLMAgentWithoutMove, TestAgent, VLMAgent")
flags.DEFINE_string("agent2", "TestAgent", "Second agent type: RandomAgent, VLMAgentWithoutMove, TestAgent, VLMAgent")

flags.DEFINE_boolean("draw_grid1", False, "Whether to draw grid on screenshots for agent 1")
flags.DEFINE_boolean("draw_grid2", False, "Whether to draw grid on screenshots for agent 2")

flags.DEFINE_boolean("annotate_units1", True, "Whether to annotate units on screenshots for agent 1")
flags.DEFINE_boolean("annotate_units2", True, "Whether to annotate units on screenshots for agent 2")

flags.DEFINE_boolean("use_self_attention1", True, "Whether to use self-attention in agent 1")
flags.DEFINE_boolean("use_self_attention2", True, "Whether to use self-attention in agent 2")

flags.DEFINE_boolean("use_rag1", True, "Whether to use RAG in agent 1")
flags.DEFINE_boolean("use_rag2", True, "Whether to use RAG in agent 2")

# Screen and map size flags
flags.DEFINE_integer('feature_screen_width', 256, 'Width of feature screen')
flags.DEFINE_integer('feature_screen_height', 256, 'Height of feature screen')
flags.DEFINE_integer('rgb_screen_width', 1920, 'Width of RGB screen')
flags.DEFINE_integer('rgb_screen_height', 1080, 'Height of RGB screen')
flags.DEFINE_integer('map_size_width', 64, 'Width of the map')
flags.DEFINE_integer('map_size_height', 64, 'Height of the map')
flags.DEFINE_integer("max_steps", 2000, "Maximum steps per episode")


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
    flags.FLAGS(argv)
    terminate_sc2_processes()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = os.path.abspath(os.path.join("log", f"{FLAGS.agent1}_vs_{FLAGS.agent2}", timestamp))
    save_dir = os.path.join(base_log_dir, "logs_file")
    replay_dir = os.path.join(base_log_dir, "replays")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(replay_dir, exist_ok=True)

    env = None
    try:
        env = SC2MultimodalTwoPlayerEnv(
            map_name=FLAGS.map,
            save_dir=save_dir,
            replay_dir=replay_dir,
            timestamp=timestamp,
            feature_dims=(FLAGS.feature_screen_width, FLAGS.feature_screen_height),
            rgb_dims=(FLAGS.rgb_screen_width, FLAGS.rgb_screen_height),
            map_size=(FLAGS.map_size_width, FLAGS.map_size_height),
        )

        # 创建两个不同的智能体
        agent_configs = [
            {
                'agent_type': FLAGS.agent1,
                'draw_grid': FLAGS.draw_grid1,
                'annotate_units': FLAGS.annotate_units1,
                'use_self_attention': FLAGS.use_self_attention1,
                'use_rag': FLAGS.use_rag1,
                'save_dir': os.path.join(save_dir, "agent1")
            },
            {
                'agent_type': FLAGS.agent2,
                'draw_grid': FLAGS.draw_grid2,
                'annotate_units': FLAGS.annotate_units2,
                'use_self_attention': FLAGS.use_self_attention2,
                'use_rag': FLAGS.use_rag2,
                'save_dir': os.path.join(save_dir, "agent2")
            }
        ]

        agents = []
        for i, config in enumerate(agent_configs):
            agent_class = get_agent_class(config['agent_type'])
            agent = agent_class(
                action_space=env.action_space,
                config_path=FLAGS.config_path,
                draw_grid=config['draw_grid'],
                annotate_units=config['annotate_units'],
                save_dir=config['save_dir'],
                use_rag=config['use_rag'],
                use_self_attention=config['use_self_attention'],
            )
            agents.append(agent)
            logging.info(f"Agent {i + 1} created: {config['agent_type']}")

        # 运行单个episode
        observations = env.reset()  # 现在返回两个观察值
        total_rewards = [0, 0]  # 跟踪两个玩家的奖励
        done = False
        step = 0

        # 初始化等待
        time.sleep(5)

        while not done and step < FLAGS.max_steps:
            try:
                if step > 0:
                    time.sleep(0.1)

                # 获取两个玩家的动作
                actions = [
                    agents[0].get_action(observations[0]),
                    agents[1].get_action(observations[1])
                ]

                # 执行动作
                observations, rewards, done, info = env.step(actions)

                # 检查是否有环境错误
                if info.get("error"):
                    logging.error(f"Environment error: {info['error']}")
                    break

                # 更新总奖励
                total_rewards = [total + r for total, r in zip(total_rewards, rewards)]
                step += 1

                print(f"Step: {step}")
                print(f"Player 1 - Reward: {rewards[0]}, Total: {total_rewards[0]}")
                print(f"Player 2 - Reward: {rewards[1]}, Total: {total_rewards[1]}")

            except Exception as e:
                logging.error(f"Error in game loop: {e}")
                break

        print(f"\nEpisode finished after {step} steps")
        print(f"Player 1 Total Reward: {total_rewards[0]}")
        print(f"Player 2 Total Reward: {total_rewards[1]}")
        if info.get('game_result'):
            print(f"Game Result: {info['game_result']}")

    except Exception as e:
        logging.error(f"Critical error occurred: {e}")

    finally:
        # 清理资源
        if env is not None:
            try:
                env.close()
                time.sleep(2)
            except Exception as e:
                logging.error(f"Error closing environment: {e}")

        terminate_sc2_processes()


if __name__ == '__main__':
    app.run(main)
