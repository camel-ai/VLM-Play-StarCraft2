import time
from absl import app
from absl import flags
import os
import cv2
from vlm_attention.env.env_core import SC2MultimodalEnv
from vlm_attention import ROOT_DIR, CONFIG_FILE_RELATIVE_PATH

FLAGS = flags.FLAGS
flags.DEFINE_string("map", "test_for_cluster", "Name of the map to use")
flags.DEFINE_string("replay_dir", "./replays", "Directory to save replays")
flags.DEFINE_string("config_path", os.path.join(ROOT_DIR, CONFIG_FILE_RELATIVE_PATH), "Path to the configuration file")


class PureImageAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.screenshot_dir = "./pure_screenshots"
        self.screenshot_count = 0
        os.makedirs(self.screenshot_dir, exist_ok=True)

    def get_action(self, observation):
        self._save_pure_image(observation)

        # 返回一个空的动作集
        return {
            'attack': [],
            'move': []
        }

    def _save_pure_image(self, observation):
        self.screenshot_count += 1
        frame = observation['image']

        filename = os.path.join(self.screenshot_dir, f"pure_screenshot_{self.screenshot_count:04d}.png")
        cv2.imwrite(filename, frame)
        print(f"Pure screenshot saved: {filename}")


def run_episode(env, agent, max_steps=1000):
    observation = env.reset()
    total_reward = 0
    done = False
    step = 0

    while not done and step < max_steps:
        if step == 0:
            time.sleep(5)  # 给游戏一些时间来初始化
        else:
            time.sleep(0.1)  # 控制步骤速度

        action = agent.get_action(observation)
        observation, reward, done, _ = env.step(action)
        total_reward += reward
        step += 1

        print(f"Step: {step}, Reward: {reward}, Total Reward: {total_reward}")

    return total_reward


def main(argv):
    del argv  # Unused

    env = SC2MultimodalEnv(
        map_name=FLAGS.map,
        replay_dir=FLAGS.replay_dir,
        config_path=FLAGS.config_path,
        save_replay_episodes=5
    )

    agent = PureImageAgent(env.action_space)

    num_episodes = 1
    for episode in range(num_episodes):
        print(f"\nStarting Episode {episode + 1}")
        total_reward = run_episode(env, agent)
        print(f"Episode {episode + 1} finished. Total Reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    app.run(main)