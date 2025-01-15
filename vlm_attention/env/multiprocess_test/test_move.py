import os
import multiprocessing
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from absl import app, flags
import cv2
import numpy as np

FLAGS = flags.FLAGS
map_list = ["vlm_attention_1",
            "2c_vs_64zg_vlm_attention",
            "2m_vs_1z_vlm_attention",
            "2s_vs_1sc_vlm_attention",
            "2s3z_vlm_attention",
            "3m_vlm_attention",
            "3s_vs_3z_vlm_attention"]


class MoveScreenAgent(base_agent.BaseAgent):
    def __init__(self):
        super(MoveScreenAgent, self).__init__()
        self.move_positions = [(255,255)]
        self.current_pos = 0
        self.map_info_printed = False

    def step(self, obs):
        super(MoveScreenAgent, self).step(obs)

        if not self.map_info_printed:
            print("\n=== Map Information ===")
            if 'feature_minimap' in obs.observation:
                minimap_layers = obs.observation['feature_minimap']
                height_map = minimap_layers[0]
                real_map_size = height_map.shape
                print(f"Real Map Size: {real_map_size[0]} x {real_map_size[1]}")

            if 'feature_screen' in obs.observation:
                feature_screen = obs.observation['feature_screen']
                print(f"\nFeature Screen Resolution: {feature_screen.shape[1]} x {feature_screen.shape[2]}")

            print("=====================")
            self.map_info_printed = True

        # Move_screen 逻辑
        if actions.FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            target = self.move_positions[self.current_pos]
            self.current_pos = (self.current_pos + 1) % len(self.move_positions)
            return actions.FUNCTIONS.Move_screen("now", target)

        if actions.FUNCTIONS.select_army.id in obs.observation.available_actions:
            return actions.FUNCTIONS.select_army("select")

        return actions.FUNCTIONS.no_op()


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_rgb_image(image, directory, filename):
    ensure_dir(directory)
    full_path = os.path.join(directory, filename)
    cv2.imwrite(full_path, image)


def run_episode(episode):
    agent = MoveScreenAgent()

    with sc2_env.SC2Env(
            map_name=map_list[0],
            players=[sc2_env.Agent(sc2_env.Race.terran),
                     sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=256, minimap=64),
                rgb_dimensions=features.Dimensions(screen=(1920, 1080), minimap=(64, 64)),
                action_space=actions.ActionSpace.FEATURES,
                use_feature_units=True,
                use_raw_units=True,
                use_unit_counts=True,
                use_camera_position=True,
                show_cloaked=True,
                show_burrowed_shadows=True,
                show_placeholders=True,
                raw_crop_to_playable_area=True,
                raw_resolution=64
            ),
            step_mul=8,
            game_steps_per_episode=1000,
            visualize=True) as env:

        agent.setup(env.observation_spec(), env.action_spec())

        print(f"Starting episode {episode + 1}")
        obs = env.reset()
        agent.reset()

        total_reward = 0
        step_count = 0

        while True:
            step_actions = [agent.step(obs[0])]

            if 'rgb_screen' in obs[0].observation:
                rgb_screen = obs[0].observation['rgb_screen']
                rgb_screen = np.clip(rgb_screen, 0, 255).astype(np.uint8)
                save_rgb_image(rgb_screen,
                               f'log_screen_episode_{episode}',
                               f'screen_step{obs[0].observation["game_loop"][0]}.png')

            step_count += 1
            total_reward += obs[0].reward

            if obs[0].last():
                print(f"\nEpisode {episode + 1} completed in {step_count} steps")
                break

            obs = env.step(step_actions)


def init_worker():
    import sys
    sys.argv = sys.argv[:1]
    flags.FLAGS(sys.argv)


def main(argv):
    flags.FLAGS(argv)
    num_episodes = 1

    with multiprocessing.Pool(num_episodes, initializer=init_worker) as pool:
        pool.map(run_episode, range(num_episodes))


if __name__ == "__main__":
    app.run(main)