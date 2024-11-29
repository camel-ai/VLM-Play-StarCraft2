import os
import multiprocessing
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from absl import app, flags
import cv2
import numpy as np
import json
import datetime

FLAGS = flags.FLAGS
map_list = ["vlm_attention_1",
            "2c_vs_64zg_vlm_attention",
            "2m_vs_1z_vlm_attention",
            "2s_vs_1sc_vlm_attention",
            "2s3z_vlm_attention",
            "3m_vlm_attention",
            "3s_vs_3z_vlm_attention"]


class SimpleAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SimpleAgent, self).__init__()
        self.map_info_printed = False

    def setup(self, obs_spec, action_spec):
        super(SimpleAgent, self).setup(obs_spec, action_spec)

    def reset(self):
        super(SimpleAgent, self).reset()
        self.map_info_printed = False

    def step(self, obs):
        super(SimpleAgent, self).step(obs)

        if not self.map_info_printed:
            print("\n=== Map Information ===")

            if 'feature_minimap' in obs.observation:
                minimap_layers = obs.observation['feature_minimap']
                height_map = minimap_layers[0]
                real_map_size = height_map.shape
                print(f"Real Map Size: {real_map_size[0]} x {real_map_size[1]}")

                visibility_map = minimap_layers[1]
                playable_areas = np.where(visibility_map > 0)
                if len(playable_areas[0]) > 0 and len(playable_areas[1]) > 0:
                    min_y, max_y = np.min(playable_areas[0]), np.max(playable_areas[0])
                    min_x, max_x = np.min(playable_areas[1]), np.max(playable_areas[1])
                    print(f"Playable Area: from ({min_x}, {min_y}) to ({max_x}, {max_y})")
                    print(f"Playable Area Size: {max_x - min_x + 1} x {max_y - min_y + 1}")

            if 'feature_screen' in obs.observation:
                feature_screen = obs.observation['feature_screen']
                print(f"\nFeature Screen Resolution: {feature_screen.shape[1]} x {feature_screen.shape[2]}")

            if 'rgb_screen' in obs.observation:
                rgb_screen = obs.observation['rgb_screen']
                print(f"RGB Screen Resolution: {rgb_screen.shape[1]} x {rgb_screen.shape[0]}")

            if 'camera_position' in obs.observation:
                cam_pos = obs.observation['camera_position']
                print(f"Camera Position: ({cam_pos[0]}, {cam_pos[1]})")

            if 'camera_size' in obs.observation:
                cam_size = obs.observation['camera_size']
                print(f"Camera View Size: {cam_size[0]} x {cam_size[1]}")

            if 'feature_minimap' in obs.observation:
                height_map = obs.observation['feature_minimap'][0]
                min_height = np.min(height_map)
                max_height = np.max(height_map)
                print(f"\nTerrain Height Range: {min_height} to {max_height}")

            if 'map_name' in obs.observation:
                print(f"Map Name: {obs.observation['map_name']}")

            print("=====================")
            self.map_info_printed = True

        return actions.RAW_FUNCTIONS.raw_move_camera([30, 30])


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_log_path(episode, agent_id):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = os.path.join("logs", f"episode_{episode}_{timestamp}")
    agent_log_dir = os.path.join(base_log_dir, f"agent_{agent_id}")
    ensure_dir(agent_log_dir)
    return agent_log_dir


def save_rgb_image(image, directory, filename):
    ensure_dir(directory)
    full_path = os.path.join(directory, filename)
    cv2.imwrite(full_path, image)


def get_structure(obj):
    if isinstance(obj, (int, float, str, bool)):
        return type(obj).__name__
    elif isinstance(obj, list):
        return [get_structure(obj[0])] if obj else "empty_list"
    elif isinstance(obj, dict):
        return {k: get_structure(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return f"ndarray(shape={obj.shape}, dtype={obj.dtype})"
    else:
        return type(obj).__name__


def save_obs_structure(obs, episode, agent_id, log_dir):
    obs_structure = get_structure(obs.observation)
    structure_file = os.path.join(log_dir, f'obs_structure_agent_{agent_id}.json')
    with open(structure_file, 'w') as f:
        json.dump(obs_structure, f, indent=2)


def print_score(score):
    score_names = [
        "Total Score",
        "Collected Minerals",
        "Collected Vespene",
        "Collected Resources",
        "Spent Minerals",
        "Spent Vespene",
        "Food Used (Supply)",
        "Killed Unit Score",
        "Killed Building Score",
        "Killed Minerals",
        "Killed Vespene",
        "Lost Minerals",
        "Lost Vespene"
    ]

    for i, name in enumerate(score_names):
        if i < len(score):
            print(f"  {name}: {score[i]}")


def run_episode(episode):
    agent1 = SimpleAgent()
    agent2 = SimpleAgent()

    with sc2_env.SC2Env(
            map_name="vlm_attention_1_new",
            players=[sc2_env.Agent(sc2_env.Race.terran),
                     sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=256, minimap=64),
                rgb_dimensions=features.Dimensions(screen=(1920, 1080), minimap=(64, 64)),
                action_space=actions.ActionSpace.RAW,
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

        agent1.setup(env.observation_spec(), env.action_spec())
        agent2.setup(env.observation_spec(), env.action_spec())

        print(f"Starting episode {episode + 1}")
        obs = env.reset()
        agent1.reset()
        agent2.reset()

        log_dir_agent1 = get_log_path(episode, 1)
        log_dir_agent2 = get_log_path(episode, 2)

        save_obs_structure(obs[0], episode, 1, log_dir_agent1)
        save_obs_structure(obs[1], episode, 2, log_dir_agent2)

        total_reward1 = 0
        total_reward2 = 0
        step_count = 0

        while True:
            step_actions = [
                agent1.step(obs[0]),
                agent2.step(obs[1])
            ]

            # Save Agent 1 observations
            if 'rgb_screen' in obs[0].observation:
                rgb_screen = obs[0].observation['rgb_screen']
                rgb_screen = np.clip(rgb_screen, 0, 255).astype(np.uint8)
                save_rgb_image(rgb_screen,
                               os.path.join(log_dir_agent1, 'screens'),
                               f'screen_step{obs[0].observation["game_loop"][0]}.png')

            if 'rgb_minimap' in obs[0].observation:
                rgb_minimap = obs[0].observation['rgb_minimap']
                rgb_minimap = np.clip(rgb_minimap, 0, 255).astype(np.uint8)
                save_rgb_image(rgb_minimap,
                               os.path.join(log_dir_agent1, 'minimaps'),
                               f'minimap_step{obs[0].observation["game_loop"][0]}.png')

            # Save Agent 2 observations
            if 'rgb_screen' in obs[1].observation:
                rgb_screen = obs[1].observation['rgb_screen']
                rgb_screen = np.clip(rgb_screen, 0, 255).astype(np.uint8)
                save_rgb_image(rgb_screen,
                               os.path.join(log_dir_agent2, 'screens'),
                               f'screen_step{obs[1].observation["game_loop"][0]}.png')

            if 'rgb_minimap' in obs[1].observation:
                rgb_minimap = obs[1].observation['rgb_minimap']
                rgb_minimap = np.clip(rgb_minimap, 0, 255).astype(np.uint8)
                save_rgb_image(rgb_minimap,
                               os.path.join(log_dir_agent2, 'minimaps'),
                               f'minimap_step{obs[1].observation["game_loop"][0]}.png')

            step_count += 1
            total_reward1 += obs[0].reward
            total_reward2 += obs[1].reward

            if obs[0].last():
                for agent_id, agent_obs, total_reward in [
                    (1, obs[0], total_reward1),
                    (2, obs[1], total_reward2)
                ]:
                    print(f"\nAgent {agent_id} Episode {episode + 1} Results:")
                    print(f"  Steps: {step_count}")
                    print(f"  Total Reward: {total_reward}")
                    print_score(agent_obs.observation["score_cumulative"])

                    result = "Victory" if agent_obs.reward > 0 else "Defeat" if agent_obs.reward < 0 else "Tie"
                    print(f"  Result: {result}")
                break

            obs = env.step(step_actions)


def init_worker():
    import sys
    sys.argv = sys.argv[:1]
    flags.FLAGS(sys.argv)


def main(argv):
    flags.FLAGS(argv)
    num_episodes = 1

    # 直接循环运行episodes，不使用多进程
    for episode in range(num_episodes):
        run_episode(episode)

if __name__ == "__main__":
    app.run(main)