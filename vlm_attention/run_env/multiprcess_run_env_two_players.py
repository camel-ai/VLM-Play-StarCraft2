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
from absl import app, flags

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from agent.RandomAgent import RandomAgent
from agent.vlm_agent_without_move_v5 import VLMAgentWithoutMove
from agent.vlm_agent_v6 import VLMAgent
from vlm_attention import ROOT_DIR, CONFIG_FILE_RELATIVE_PATH
from vlm_attention.env.two_players_env_core import SC2MultimodalTwoPlayerEnv
from agent.test_agent import TestAgent

# Available maps
map_list = [
    "vlm_attention_1_two_players",
    "vlm_attention_2_terran_vs_terran_two_players",
    "MMM_vlm_attention_two_players"
]

# Define flags
FLAGS = flags.FLAGS

# Map and agent selection flags
flags.DEFINE_string("map", map_list[0], "Name of the map to use")
flags.DEFINE_string("agent1", "TestAgent", "First agent: RandomAgent, VLMAgentWithoutMove, TestAgent, VLMAgent")
flags.DEFINE_string("agent2", "TestAgent", "Second agent: RandomAgent, VLMAgentWithoutMove, TestAgent, VLMAgent")

# Configuration flags
flags.DEFINE_string("config_path", os.path.join(ROOT_DIR, CONFIG_FILE_RELATIVE_PATH), "Path to configuration file")
flags.DEFINE_integer("num_processes", 4, "Number of parallel processes")
flags.DEFINE_integer("max_steps", 2000, "Maximum steps per episode")

# Agent feature flags
flags.DEFINE_boolean("use_self_attention1", True, "Whether to use self-attention in agent 1")
flags.DEFINE_boolean("use_self_attention2", True, "Whether to use self-attention in agent 2")
flags.DEFINE_boolean("use_rag1", True, "Whether to use RAG in agent 1")
flags.DEFINE_boolean("use_rag2", True, "Whether to use RAG in agent 2")
flags.DEFINE_boolean("use_proxy_1", False, "Whether to use proxy for API calls for agent 1")
flags.DEFINE_boolean("use_proxy_2", False, "Whether to use proxy for API calls for agent 2")
flags.DEFINE_string('model_name_1', default="qwen", help="Model to be used by agent 1")
flags.DEFINE_string('model_name_2', default="qwen", help="Model to be used by agent 2")

# Visualization flags
flags.DEFINE_boolean("draw_grid", True, "Whether to draw grid on screenshots")
flags.DEFINE_boolean("annotate_units", True, "Whether to annotate units on screenshots")

# Screen and map dimensions
flags.DEFINE_integer('feature_screen_width', 256, 'Width of feature screen')
flags.DEFINE_integer('feature_screen_height', 256, 'Height of feature screen')
flags.DEFINE_integer('rgb_screen_width', 1920, 'Width of RGB screen')
flags.DEFINE_integer('rgb_screen_height', 1080, 'Height of RGB screen')
flags.DEFINE_integer('map_size_width', 64, 'Width of the map')
flags.DEFINE_integer('map_size_height', 64, 'Height of the map')


def terminate_process_safely(pid: int):
    """Safely terminate a process with error handling"""
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
    """Run a single episode with two agents"""
    env = None
    episode_id = config['episode_id']

    try:
        # Setup directories
        base_log_dir = config['base_log_dir']
        save_dir = os.path.join(base_log_dir, f"episode_{episode_id}", "logs_file")
        replay_dir = os.path.join(base_log_dir, f"episode_{episode_id}", "replays")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(replay_dir, exist_ok=True)

        # Initialize environment
        env_config = copy.deepcopy(config['env_config'])
        env_config.update({
            'save_dir': save_dir,
            'replay_dir': replay_dir,
        })
        env = SC2MultimodalTwoPlayerEnv(**env_config)

        # Initialize agents
        agent_classes = {
            "VLMAgentWithoutMove": VLMAgentWithoutMove,
            "RandomAgent": RandomAgent,
            "TestAgent": TestAgent,
            "VLMAgent": VLMAgent
        }

        # Setup agent directories and configurations
        agent1_save_dir = os.path.join(save_dir, "agent1")
        agent2_save_dir = os.path.join(save_dir, "agent2")
        os.makedirs(agent1_save_dir, exist_ok=True)
        os.makedirs(agent2_save_dir, exist_ok=True)

        # Update agent configurations
        agent1_args = copy.deepcopy(config['agent1_args'])
        agent2_args = copy.deepcopy(config['agent2_args'])

        # Update agent 1 configuration
        agent1_args.update({
            'save_dir': agent1_save_dir,
            'action_space': env.action_space,
            'use_proxy': FLAGS.use_proxy_1,
            'model_name': FLAGS.model_name_1
        })

        # Update agent 2 configuration
        agent2_args.update({
            'save_dir': agent2_save_dir,
            'action_space': env.action_space,
            'use_proxy': FLAGS.use_proxy_2,
            'model_name': FLAGS.model_name_2
        })

        # Create agents
        agent1 = agent_classes[config['agent1_name']](**agent1_args)
        agent2 = agent_classes[config['agent2_name']](**agent2_args)

        # Run episode
        observations = env.reset()
        total_rewards = [0, 0]
        step = 0
        done = False

        while not done and step < FLAGS.max_steps:
            # Get actions from both agents
            actions = [
                agent1.get_action(observations[0]),
                agent2.get_action(observations[1])
            ]

            # Environment step
            observations, rewards, done, info = env.step(actions)

            # Check for errors
            if info.get("error"):
                raise RuntimeError(f"Environment error: {info['error']}")

            # Update rewards and step counter
            total_rewards = [total + r for total, r in zip(total_rewards, rewards)]
            step += 1

            # Log progress
            logger.info(f"Episode {episode_id}, Step {step}/{FLAGS.max_steps}, "
                        f"Rewards: [{rewards[0]:.2f}, {rewards[1]:.2f}], "
                        f"Total: [{total_rewards[0]:.2f}, {total_rewards[1]:.2f}]")

        return {
            'episode_id': episode_id,
            'rewards': total_rewards,
            'steps': step,
            'success': True
        }

    except Exception as e:
        logger.error(f"Error in episode {episode_id}: {str(e)}")
        traceback.print_exc()
        return {
            'episode_id': episode_id,
            'rewards': None,
            'steps': None,
            'success': False,
            'error': str(e)
        }

    finally:
        if env is not None:
            env.close()


def init_worker():
    """Initialize worker process"""
    sys.argv = sys.argv[:1]
    flags.FLAGS(sys.argv)


def main(argv):
    flags.FLAGS(argv)

    # Setup logging directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = os.path.abspath(os.path.join(
        "log",
        f"{FLAGS.agent1}_vs_{FLAGS.agent2}",
        f"{FLAGS.model_name_1}_vs_{FLAGS.model_name_2}_{timestamp}_{FLAGS.map}"
    ))
    os.makedirs(base_log_dir, exist_ok=True)

    # Environment configuration
    env_config = {
        'map_name': FLAGS.map,
        'timestamp': timestamp,
        'feature_dims': (FLAGS.feature_screen_width, FLAGS.feature_screen_height),
        'rgb_dims': (FLAGS.rgb_screen_width, FLAGS.rgb_screen_height),
        'map_size': (FLAGS.map_size_width, FLAGS.map_size_height)
    }

    # Base agent configuration
    agent_base_config = {
        'config_path': FLAGS.config_path,
        'draw_grid': FLAGS.draw_grid,
        'annotate_units': FLAGS.annotate_units,
        'action_space': None  # Will be set in run_episode
    }

    # Agent specific configurations
    agent1_args = {**agent_base_config, 'use_self_attention': FLAGS.use_self_attention1, 'use_rag': FLAGS.use_rag1}
    agent2_args = {**agent_base_config, 'use_self_attention': FLAGS.use_self_attention2, 'use_rag': FLAGS.use_rag2}

    # Prepare configurations for each process
    configs = [
        {
            'episode_id': i,
            'base_log_dir': base_log_dir,
            'env_config': env_config,
            'agent1_name': FLAGS.agent1,
            'agent2_name': FLAGS.agent2,
            'agent1_args': agent1_args,
            'agent2_args': agent2_args
        }
        for i in range(FLAGS.num_processes)
    ]

    # Run episodes using process pool
    with mp.Pool(FLAGS.num_processes, initializer=init_worker) as pool:
        results = pool.map(run_episode, configs)

    # Process and log results
    successes = 0
    failures = 0
    total_rewards = [0, 0]

    for result in results:
        if result['success']:
            successes += 1
            rewards = result['rewards']
            total_rewards = [t + r for t, r in zip(total_rewards, rewards)]
            logger.info(f"Episode {result['episode_id']} completed: steps={result['steps']}, "
                        f"rewards={rewards}")
        else:
            failures += 1
            logger.error(f"Episode {result['episode_id']} failed: {result.get('error', 'Unknown error')}")

    # Log summary
    logger.info(f"\nRun Summary:")
    logger.info(f"Successful episodes: {successes}")
    logger.info(f"Failed episodes: {failures}")
    if successes > 0:
        avg_rewards = [r / successes for r in total_rewards]
        logger.info(f"Average rewards: {avg_rewards}")


if __name__ == "__main__":
    app.run(main)