import os
import time
import psutil
import logging
import multiprocessing as mp
import traceback
from absl import app, flags
from datetime import datetime
from typing import Type, Union

from agent.RandomAgent import RandomAgent
from agent.vlm_agent_without_move_v5 import VLMAgentWithoutMove
from agent.vlm_agent_v6 import VLMAgent
from agent.test_agent_with_ability import TestAgent
from vlm_attention import ROOT_DIR, CONFIG_FILE_RELATIVE_PATH
from vlm_attention.env.two_players_with_ability_env_core import SC2MultimodalTwoPlayerEnv

# Available maps
map_list = [
    "ability_test_map",
    "ability_map_8marine_3marauder_1medivac_1tank_2_players",
    "ability_8stalker_vs_8marine_3marauder_1medivac_tank_map_2_players",
    "ability_7stalker_vs_11marine_1medivac_1tank_map_2_players"
]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define flags
FLAGS = flags.FLAGS

# Map and agent selection flags
flags.DEFINE_string("map", map_list[1], "Name of the map to use")
flags.DEFINE_string("agent1", "TestAgent", "First agent type: RandomAgent, VLMAgentWithoutMove, TestAgent, VLMAgent")
flags.DEFINE_string("agent2", "TestAgent", "Second agent type: RandomAgent, VLMAgentWithoutMove, TestAgent, VLMAgent")

# Configuration flags
flags.DEFINE_string("config_path", os.path.join(ROOT_DIR, CONFIG_FILE_RELATIVE_PATH), "Configuration file path")
flags.DEFINE_integer("num_processes", 2, "Number of parallel processes")
flags.DEFINE_integer("max_steps", 2000, "Maximum steps per episode")

# Agent-specific flags
flags.DEFINE_boolean("use_self_attention1", True, "Whether to use self-attention in agent 1")
flags.DEFINE_boolean("use_self_attention2", True, "Whether to use self-attention in agent 2")
flags.DEFINE_boolean("use_rag1", True, "Whether to use RAG in agent 1")
flags.DEFINE_boolean("use_rag2", True, "Whether to use RAG in agent 2")
flags.DEFINE_boolean("use_proxy_1", False, "Whether to use proxy for API calls for agent 1")
flags.DEFINE_boolean("use_proxy_2", False, "Whether to use proxy for API calls for agent 2")
flags.DEFINE_string("model_name_1", default="qwen", help="Model to be used by agent 1")
flags.DEFINE_string("model_name_2", default="qwen", help="Model to be used by agent 2")

# Visualization flags
flags.DEFINE_boolean("draw_grid1", False, "Whether to draw grid for agent 1")
flags.DEFINE_boolean("draw_grid2", False, "Whether to draw grid for agent 2")
flags.DEFINE_boolean("annotate_units1", True, "Whether to annotate units for agent 1")
flags.DEFINE_boolean("annotate_units2", True, "Whether to annotate units for agent 2")

# Screen and map dimensions
flags.DEFINE_integer('feature_screen_width', 256, 'Width of feature screen')
flags.DEFINE_integer('feature_screen_height', 256, 'Height of feature screen')
flags.DEFINE_integer('rgb_screen_width', 1920, 'Width of RGB screen')
flags.DEFINE_integer('rgb_screen_height', 1080, 'Height of RGB screen')
flags.DEFINE_integer('map_size_width', 64, 'Width of the map')
flags.DEFINE_integer('map_size_height', 64, 'Height of the map')



def get_agent_class(agent_name: str) -> Type[Union[VLMAgentWithoutMove, RandomAgent, TestAgent, VLMAgent]]:
    """Get the agent class based on the agent name"""
    agent_classes = {
        "VLMAgentWithoutMove": VLMAgentWithoutMove,
        "RandomAgent": RandomAgent,
        "TestAgent": TestAgent,
        "VLMAgent": VLMAgent
    }
    if agent_name not in agent_classes:
        raise ValueError(f"Unknown agent: {agent_name}. Valid options are: {', '.join(agent_classes.keys())}")
    return agent_classes[agent_name]

def run_episode(config: dict) -> dict:
    """Run a single episode with two agents and ability support"""
    env = None
    episode_id = config['episode_id']
    step = 0

    try:
        # Setup directories
        base_log_dir = config['base_log_dir']
        save_dir = os.path.join(base_log_dir, f"episode_{episode_id}", "logs_file")
        replay_dir = os.path.join(base_log_dir, f"episode_{episode_id}", "replays")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(replay_dir, exist_ok=True)

        # Initialize environment
        env = SC2MultimodalTwoPlayerEnv(
            map_name=config['map_name'],
            save_dir=save_dir,
            replay_dir=replay_dir,
            timestamp=config['timestamp'],
            feature_dims=config['feature_dims'],
            rgb_dims=config['rgb_dims'],
            map_size=config['map_size']
        )

        # Setup agent configurations
        agent_configs = [
            {
                'agent_type': config['agent1_name'],
                'draw_grid': config['draw_grid1'],
                'annotate_units': config['annotate_units1'],
                'use_self_attention': config['use_self_attention1'],
                'use_rag': config['use_rag1'],
                'use_proxy': config['use_proxy_1'],
                'model_name': config['model_name_1'],
                'save_dir': os.path.join(save_dir, "agent1")
            },
            {
                'agent_type': config['agent2_name'],
                'draw_grid': config['draw_grid2'],
                'annotate_units': config['annotate_units2'],
                'use_self_attention': config['use_self_attention2'],
                'use_rag': config['use_rag2'],
                'use_proxy': config['use_proxy_2'],
                'model_name': config['model_name_2'],
                'save_dir': os.path.join(save_dir, "agent2")
            }
        ]

        # Initialize agents
        agents = []
        for i, agent_config in enumerate(agent_configs, 1):
            os.makedirs(agent_config['save_dir'], exist_ok=True)
            agent_class = get_agent_class(agent_config['agent_type'])
            agent = agent_class(
                action_space=env.action_space,
                config_path=config['config_path'],
                **{k: v for k, v in agent_config.items() if k not in ['agent_type']}
            )
            agents.append(agent)
            logger.info(f"Agent {i} initialized: {agent_config['agent_type']}")

        # Initial setup wait
        time.sleep(5)

        # Run episode
        observations = env.reset()
        total_rewards = [0, 0]
        done = False

        while not done and step < config['max_steps']:
            if step > 0:
                time.sleep(0.1)

            # Get actions from both agents
            actions = [
                agents[0].get_action(observations[0]),
                agents[1].get_action(observations[1])
            ]

            # Environment step
            observations, rewards, done, info = env.step(actions)

            # Check for environment errors
            if info.get("error"):
                logger.error(f"Episode {episode_id} - Environment error: {info['error']}")
                break

            # Update metrics
            total_rewards = [total + r for total, r in zip(total_rewards, rewards)]
            step += 1

            # Log progress (every 10 steps to reduce output)
            if step % 10 == 0:
                logger.info(f"Episode {episode_id}, Step {step}/{config['max_steps']}, "
                          f"Rewards: [{rewards[0]:.2f}, {rewards[1]:.2f}], "
                          f"Total: [{total_rewards[0]:.2f}, {total_rewards[1]:.2f}]")

        return {
            'episode_id': episode_id,
            'total_rewards': total_rewards,
            'steps': step,
            'success': True
        }

    except Exception as e:
        logger.error(f"Error in episode {episode_id}: {str(e)}")
        traceback.print_exc()
        return {
            'episode_id': episode_id,
            'total_rewards': None,
            'steps': step,
            'success': False,
            'error': str(e)
        }

    finally:
        if env is not None:
            try:
                env.close()
                time.sleep(2)
            except Exception as e:
                logger.error(f"Error closing environment in episode {episode_id}: {str(e)}")


def init_worker():
    """Initialize worker process"""
    import sys
    sys.argv = sys.argv[:1]
    flags.FLAGS(sys.argv)

def main(argv):
    """Main function to run multiple episodes in parallel"""
    flags.FLAGS(argv)

    # Clear any existing SC2 processes

    # Setup logging directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = os.path.abspath(os.path.join(
        "log",
        f"{FLAGS.agent1}_vs_{FLAGS.agent2}",
        f"{FLAGS.model_name_1}_vs_{FLAGS.model_name_2}_{timestamp}_{FLAGS.map}"
    ))
    os.makedirs(base_log_dir, exist_ok=True)

    # Prepare base configuration
    base_config = {
        'map_name': FLAGS.map,
        'timestamp': timestamp,
        'feature_dims': (FLAGS.feature_screen_width, FLAGS.feature_screen_height),
        'rgb_dims': (FLAGS.rgb_screen_width, FLAGS.rgb_screen_height),
        'map_size': (FLAGS.map_size_width, FLAGS.map_size_height),
        'config_path': FLAGS.config_path,
        'agent1_name': FLAGS.agent1,
        'agent2_name': FLAGS.agent2,
        'draw_grid1': FLAGS.draw_grid1,
        'draw_grid2': FLAGS.draw_grid2,
        'annotate_units1': FLAGS.annotate_units1,
        'annotate_units2': FLAGS.annotate_units2,
        'use_self_attention1': FLAGS.use_self_attention1,
        'use_self_attention2': FLAGS.use_self_attention2,
        'use_rag1': FLAGS.use_rag1,
        'use_rag2': FLAGS.use_rag2,
        'use_proxy_1': FLAGS.use_proxy_1,
        'use_proxy_2': FLAGS.use_proxy_2,
        'model_name_1': FLAGS.model_name_1,
        'model_name_2': FLAGS.model_name_2,
        'max_steps': FLAGS.max_steps
    }

    # Prepare process-specific configurations
    configs = [
        {**base_config, 'episode_id': i, 'base_log_dir': base_log_dir}
        for i in range(FLAGS.num_processes)
    ]

    # Run episodes using process pool
    with mp.Pool(FLAGS.num_processes, initializer=init_worker) as pool:
        results = pool.map(run_episode, configs)

    # Process and log results
    successes = 0
    failures = 0
    total_rewards = [0, 0]
    total_steps = 0

    for result in results:
        if result['success']:
            successes += 1
            total_rewards = [t + r for t, r in zip(total_rewards, result['total_rewards'])]
            total_steps += result['steps']
            logger.info(f"Episode {result['episode_id']} completed: "
                       f"steps={result['steps']}, rewards={result['total_rewards']}")
        else:
            failures += 1
            logger.error(f"Episode {result['episode_id']} failed: {result.get('error', 'Unknown error')}")

    # Log summary
    logger.info("\nRun Summary:")
    logger.info(f"Successful episodes: {successes}")
    logger.info(f"Failed episodes: {failures}")
    if successes > 0:
        avg_rewards = [r/successes for r in total_rewards]
        avg_steps = total_steps / successes
        logger.info(f"Average rewards: {avg_rewards}")
        logger.info(f"Average steps per episode: {avg_steps:.2f}")

if __name__ == "__main__":
    app.run(main)