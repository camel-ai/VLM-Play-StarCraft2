# StarCraft II Multimodal Agent Project

## Project Overview

This project aims to develop a multimodal agent for StarCraft II. The agent is capable of processing visual information and text descriptions from the game and making appropriate decisions.

## Key Features

- Custom environment based on OpenAI Gym
- Support for multimodal inputs (images and text)
- Automatic unit annotation system
- Configurable unit identification and color schemes
- Random behavior agent for testing

## Environment Setup

### Dependencies

- Python 3.10
- Operating System: Windows 11 (Linux doesn't have StarCraft's graphical interface, but our current code can directly read RGB values, so Linux should theoretically work. Linux and Mac haven't been tested - our server has some issues, so we hope others can test on their servers)
- Requirements: `vlm_attention/requirements.txt`

### Installation Steps

1. Clone the repository
2. Run the following command in the project root directory for editable mode installation:
   ```bash
   pip install -e . --no-deps
   ```
3. Install dependencies:
   ```bash
   pip install -r vlm_attention/requirements.txt
   ```
   
   **Note**: During installation, you may encounter protobuf version conflicts. This is a legacy issue from Google's development, and we need to use protobuf==3.20.0. If you encounter conflicts, please resolve them using the following steps:
   
   ```bash
   pip uninstall protobuf
   pip install protobuf==3.20.0
   pip install pysc2 --no-deps
   ```

4. Download StarCraft II. Please log in to the Asia Battle.net to download. For Linux, download version 4.10.0 from https://github.com/Blizzard/s2client-proto#downloads
3. Map setup: In the StarCraft II game folder, locate the Maps folder (create one if it doesn't exist), then place the entire `pysc2/maps/SMAC` folder inside. The map for our experiment is `vlm_attention_1.SC2Map`
4. If you want to use custom maps, you need to check the file `pysc2/maps/readme.md`

## Maps

1. Single player without ability, run: `vlm_attention/run_env/multiprocess_run_env.py`

Available maps:

`[2bc1prism_vs_8m_vlm_attention,2c_vs_64zg_vlm_attention,2m_vs_1z_vlm_attention,2s3z_vlm_attention,2s_vs_1sc_vlm_attention,3m_vlm_attention.SC2Map,3s_vs_3z_vlm_attention,6reaper_vs8zealot_vlm_attention,8marine_1medvac_vs_2tank,8marine_2tank_vs_zerglings_banelings_vlm_attention,vlm_attention_1.SC2Map]`

2. Two players without ability, run `vlm_attention/run_env/run_env_two_players.py`

Available maps: `[vlm_attention_1_two_players,vlm_attention_2_terran_vs_terran_two_players,MMM_vlm_attention_two_players]`

3. Single player with ability, run: `vlm_attention/run_env/run_env_with_ability.py`

Available maps: `[ability_map_8marine_3marauder_1medivac_1tank]`

4. Two players with ability, run `vlm_attention/run_env/run_env_two_players_with_ability.py`

Available maps: `[ability_7stalker_vs_11marine_1medivac_1tank_map_2_players,ability_8stalker_vs_8marine_3marauder_1medivac_tank_map_2_players,ability_map_8marine_3marauder_1medivac_1tank_2_players]`

## Map Details

### Single Player Maps (Without Abilities)

| Map Name | Unit Control | Multi-unit | Terrain Use | Kiting | Dispersion | Mirror Match | Unit Configuration | Source |
|----------|-------------|------------|-------------|--------|------------|--------------|-------------------|--------|
| 2c_vs_64zg_vlm_attention | ✓ | ✓ | ✓ | ✓ | | | Player: 2 Colossus<br>Enemy: 64 Zergling | SMAC |
| 2m_vs_1z_vlm_attention | ✓ | ✓ | | | | | Player: 2 Marine<br>Enemy: 1 Zealot | SMAC |
| 2s_vs_1sc_vlm_attention | ✓ | ✓ | | | | | Player: 2 Stalker<br>Enemy: 1 Spinecrawler | SMAC |
| 3s_vs_3z_vlm_attention | ✓ | ✓ | | | | | Player: 3 Stalker<br>Enemy: 3 Zealot | SMAC |
| 6reaper_vs8zealot_vlm_attention | ✓ | ✓ | ✓ | ✓ | | | Player: 6 Reaper<br>Enemy: 8 Zealot | NEW |
| 8marine_1medvac_vs_2tank | ✓ | ✓ | | | | | Player: 8 Marine, 1 Medivac<br>Enemy: 2 Siege Tank | NEW |
| 8marine_2tank_vs_zerglings_banelings_vlm_attention | ✓ | ✓ | ✓ | | | | Player: 8 Marine, 2 Siege Tank<br>Enemy: 35 Zergling, 4 Baneling | NEW |
| 2bc1prism_vs_8m_vlm_attention | ✓ | | | | | | Player: 8 Marine<br>Enemy: 1 Warp Prism, 2 Photon Cannon | NEW |
| 2s3z_vlm_attention | ✓ | ✓ | ✓ | | | ✓ | Player: 2 Stalker, 3 Zealot<br>Enemy: 2 Stalker, 3 Zealot | SMAC |
| 3m_vlm_attention | ✓ | ✓ | | | | ✓ | Player: 3 Marine<br>Enemy: 3 Marine | SMAC |
| vlm_attention_1 | ✓ | ✓ | | | | | Player: 1 Zealot, 1 Immortal, 1 Archon, 1 Stalker, 1 Phoenix<br>Enemy: 1 Marine, 1 Marauder, 1 Reaper, 1 Hellbat, 1 Medivac, 1 Viking, 1 Ghost, 1 Banshee | NEW |

### Single Player Maps (With Abilities)

| Map Name | Unit Control | Multi-unit | Terrain Use | Kiting | Dispersion | Ability Use | Unit Configuration | Source |
|----------|-------------|------------|-------------|--------|------------|-------------|-------------------|--------|
| ability_map_8marine_3marauder_1medivac_1tank | ✓ | ✓ | | | ✓ | ✓ | Player: 8 Marine, 3 Marauder, 1 Medivac, 1 Siege Tank<br>Enemy: 8 Marine, 3 Marauder, 1 Medivac, 1 Siege Tank | NEW |
| ability_8stalker_vs_8marine_3marauder_1medivac_tank_map | ✓ | | | | ✓ | ✓ | Player: 8 Stalker<br>Enemy: 8 Marine, 3 Marauder, 1 Medivac, 1 Siege Tank | NEW |
| ability_map_8marine_3marauder_1medivac_1tank_vs_5stalker_2colossis | ✓ | ✓ | | | ✓ | ✓ | Player: 8 Marine, 3 Marauder, 1 Medivac, 1 Siege Tank<br>Enemy: 5 Stalker, 2 Colossus | NEW |
| pvz_task6_level3 | ✓ | ✓ | | | | ✓ | Player: 12 Stalker, 1 Archon, 4 Sentry, 6 High Templar<br>Enemy: 64 Zergling, 32 Baneling, 3 Ultralisk, 3 Queen | LLM-PYSC2 |

### Two Player Maps (Without Abilities)

| Map Name | Unit Control | Multi-unit | Terrain Use | Kiting | Dispersion | Mirror Match | Unit Configuration | Source |
|----------|-------------|------------|-------------|--------|------------|--------------|-------------------|--------|
| MMM_vlm_attention_two_players | ✓ | ✓ | | ✓ | ✓ | ✓ | Player 1: 8 Marine, 3 Marauder, 1 Medivac<br>Player 2: 8 Marine, 3 Marauder, 1 Medivac | SMAC |
| vlm_attention_1_two_players | ✓ | ✓ | | | | | Player 1: 1 Zealot, 1 Immortal, 1 Archon, 1 Stalker, 1 Phoenix<br>Player 2: 1 Marine, 1 Marauder, 1 Reaper, 1 Hellbat, 1 Medivac, 1 Viking, 1 Ghost, 1 Banshee | NEW |
| vlm_attention_2_terran_vs_terran_two_players | ✓ | ✓ | | | | ✓ | Player 1: 1 Marine, 1 Marauder, 1 Reaper, 1 Hellbat, 1 Medivac, 1 Viking, 1 Ghost, 1 Banshee<br>Player 2: 1 Marine, 1 Marauder, 1 Reaper, 1 Hellbat, 1 Medivac, 1 Viking, 1 Ghost, 1 Banshee | NEW |

### Two Player Maps (With Abilities)

| Map Name | Unit Control | Multi-unit | Terrain Use | Kiting | Dispersion | Ability Use | Unit Configuration | Source |
|----------|-------------|------------|-------------|--------|------------|-------------|-------------------|--------|
| ability_7stalker_vs_11marine_1medivac_1tank_map_2_players | ✓ | | | ✓ | ✓ | ✓ | Player 1: 7 Stalker<br>Player 2: 11 Marine, 1 Medivac, 1 Siege Tank | NEW |
| ability_8stalker_vs_8marine_3marauder_1medivac_tank_map_2_players | ✓ | | | ✓ | ✓ | ✓ | Player 1: 8 Stalker<br>Player 2: 8 Marine, 3 Marauder, 1 Medivac, 1 Siege Tank | NEW |
| ability_map_8marine_3marauder_1medivac_1tank_2_players | ✓ | ✓ | | ✓ | ✓ | ✓ | Player 1: 8 Marine, 3 Marauder, 1 Medivac, 1 Siege Tank<br>Player 2: 8 Marine, 3 Marauder, 1 Medivac, 1 Siege Tank | NEW |

## Configuration

The project uses the `vlm_attention/env/config.py` file for configuration. Main configuration items include:

```python
# Color definitions (BGR format)
COLORS = {
    "self_color": [0, 255, 0],  # Green
    "enemy_color": [255, 255, 0]  # Yellow
}

# Get standard unit names from pysc2
def get_unit_name(unit_type: int) -> str:
    """
    Get unit name via unit_type
    Using enumerations defined in pysc2.lib.units

    Args:
        unit_type: Unit type ID

    Returns:
        str: Formatted unit name, returns Unknown if not found
    """
    # Try to get unit type by race order
    for race in [Protoss, Terran, Zerg, Neutral]:
        try:
            unit_enum = race(unit_type)
            return format_unit_name(unit_enum.name)
        except ValueError:
            continue

    return f"Unknown Unit {unit_type}"
```

## Usage

1. When installing, make sure to run vlm_attention/knowledge_data/database/generate_config_yaml.py to ensure the database connection is correct

```python
def main():
    folder_path = r"Path to processed JSON folder" # This needs to be modified
    output_file = "sc2_unit_data_index.yaml"

.......
```

2. Run the main program, with single-process and multi-process options:

```bash
python run_env.py # Single process
python multiprocess_run_env.py # Multi-process

# Note: Currently please use `vlm_agent_without_move_v5` and `test_agent` for testing and modification.
```

The default configuration file provides relevant parameters, you can run directly or modify according to your needs.
For example:

```bash
python run_env.py --map vlm_attention_1
```

## Project Structure

1. `env_bot.py`: Defines the Multimodal_bot class for interacting with StarCraft II
2. `env_core.py`: Implements the custom SC2MultimodalEnv environment
3. `run_env.py`: Main running script, includes RandomAgent and TestAgent for testing
4. `utils.py`: Contains helper functions, including calling LLM, VLM and testing API. Note that since I'm in mainland China, I used a proxy to ensure operation. Please modify `vlm_attention/utils/call_vlm.py` as needed when deploying

```python
class BaseChatbot:
    def __init__(self, model_name):
        # Set proxy
        proxy_url = get_config("proxy", "url")
        os.environ["http_proxy"] = proxy_url # Set proxy
        os.environ["https_proxy"] = proxy_url # Set proxy
```

5. `knowledge_data`: Contains knowledge data, where `\database` is the database and corresponding calling code, and `\firecrawl_test` is the code for crawling and cleaning data

## Customization and Extension

1. Modify `config.json` to adapt to different game scenarios and unit configurations
2. Extend the Multimodal_bot class in `env_bot.py` to implement more complex behaviors. Currently only attack behavior is supported, such as unit 3 attacking unit 14
3. Replace FlexibleRandomAgent and TestAgent in `run_env.py` to implement more advanced decision logic

## Environment Setup

### Action Space

- `attack`: Attack a specified unit
- `move`: Move to a specified position

### Observation Space

The observation space is a dictionary containing three key elements:

```python
{
    'text': text_description,
    'image': raw_image,
    'unit_info': unit_info
}
```

1. **Text Observation** (`'text'`)

- Description: Provides a text description of the game state, including information about friendly and enemy units. Text observation is only available in the first step.

2. **Image Observation** (`'image'`)

- Description: Screenshot of the game screen. Image observation is available in every step, obtained from Obs.

3. **Unit Information** (`'unit_info'`)

- Description: Contains detailed information about units on the field, used for annotation and decision-making. Obtained using feature screen.
- Specific information as follows:

```python
{
    'tag_index': tag_index, # Tag
    'position': (screen_x, screen_y), # Position
    'color': color, # Color
    'alliance': unit.alliance # Alliance
}
```

### Action Space

The `_generate_random_actions` method returns a dictionary with two keys: attack and move, each key corresponding to a list containing one or more tuples.

- Attack Action (attack)

  - Each tuple contains two integers (attacker tag, target tag).
  - For example: [(1, 6)] means the friendly unit with tag 1 attacks the enemy unit with tag 6.
  - An empty list [] means no attack action is executed.
- Move Action (move)

  - move_type:
    - 0: No movement
    - 1: Original movement (using grid coordinates); Default 10 x 10
    - 2: SMAC movement (using directions, 0,1,2,3 corresponding to up, down, left, right)
  - unit_index: Index of the unit to move
  - target:
    - For original movement (move_type=1): [x, y] grid coordinates
    - For SMAC movement (move_type=2): [direction], directions defined as follows:
      - 0: Move up
      - 1: Move right
      - 2: Move down
      - 3: Move left
    - For no movement (move_type=0): [0, 0]

### Unit Annotation Selection

`annotate_units_on_image` requires a dictionary as input, containing unit tag and color information. Returns an annotated image for displaying unit information.

For example:

```python
units_to_annotate = [
    {'tag_index': 1, 'position': (100, 100), 'color': (0, 255, 0)},  # Green annotation
    {'tag_index': 2, 'position': (200, 200), 'color': (0, 0, 255)},  # Red annotation
    {'tag_index': 3, 'position': (300, 300), 'color': (255, 0, 0)}   # Blue annotation
]
```

### Agent Setup

Currently, due to poor feedback on movement and grid functionality, we provide an agent that only calls attack without moving.

Additionally, some configuration information and helper functions for the agent are in `\run_env\agent\agent_utils.py` and `run_env\utils.py`

Specific parameters that need to be set:

1. use_self_attention: Whether to use self-attention
2. use_rag: Whether to use RAG

```python
class VLMAgentWithoutMove:
    def __init__(self, action_space: Dict[str, Any], config_path: str, save_dir: str, draw_grid: bool = False,
                 annotate_units: bool = True, grid_size: Tuple[int, int] = (10, 10),
                 use_self_attention: bool = False, use_rag: bool = False):
      
        """
        Initialize VLMAgentWithoutMove agent.
        :param action_space: Action space dictionary
        :param config_path: Configuration file path
        :param save_dir: Save directory
        :param draw_grid: Whether to draw grid on screenshots
        :param annotate_units: Whether to annotate units on screenshots
        :param grid_size: Grid size
        :param use_self_attention: Whether to use self-attention
        :param use_rag: Whether to use RAG
        """
```

### StarCraft II Observation Information

We test in `vlm_attention/env/multiprocess_test/test.py`, process the observation space, and obtain corresponding reward and score information.

```python
# Reward & score

            step_count += 1
            total_reward += obs[0].reward

            if obs[0].last():
                score = obs[0].observation["score_cumulative"]
                print(f"Episode {episode + 1} Results:")
                print(f"  Steps: {step_count}")
                print(f"  Total Reward: {total_reward}")
                print(f"  Total Score: {score[0]}")
                print(f"  Collected Minerals: {score[1]}")
                print(f"  Collected Vespene: {score[2]}")
                print(f"  Collected Resources: {score[3]}")
                print(f"  Spent Minerals: {score[4]}")
                print(f"  Spent Vespene: {score[5]}")
                print(f"  Food Used (Supply): {score[6]}")
                print(f"  Killed Unit Score: {score[7]}")
                print(f"  Killed Building Score: {score[8]}")
                print(f"  Killed Minerals: {score[9]}")
                print(f"  Killed Vespene: {score[10]}")
                print(f"  Lost Minerals: {score[11]}")
                print(f"  Lost Vespene: {score[12]}")
                print(f"  Friendly Fire Minerals: {score[13]}")
                print(f"  Friendly Fire Vespene: {score[14]}")
                print(f"  Used Minerals: {score[15]}")
                print(f"  Used Vespene: {score[16]}")
                print(f"  Total Used Resources: {score[17]}")
                print(f"  Total Items Score: {score[18]}")
                print(f"  Score: {score[19]}")
```

The complete observation structure is saved in `vlm_attention/env/multiprocess_test/obs_structure.json`

### RAG

1. Currently using firecrawl to first crawl data from the website `https://liquipedia.net/starcraft2/Main_Page`, then clean and process it. The crawled data is stored in `vlm_attention\knowledge_data\firecrawl_test\sc2_unit_info`. The cleaned data is stored in `vlm_attention\knowledge_data\firecrawl_test\sc2_unit_info_processed`. The cleaning code is `vlm_attention\knowledge_data\firecrawl_test\clean_sc2_unit_info_file.py`.
2. Please note that we pre-crawl and then clean, not involving new additions at runtime. The URLs needed for crawling are stored in `vlm_attention\knowledge_data\url.py`. If you need to crawl new data, please add it yourself.
3. When installing for the first time, please run the `vlm_attention\knowledge_data\database\generate_config_yaml.py` file. Note that you need to replace the corresponding `folder_path` with the absolute location of `vlm_attention\knowledge_data\firecrawl_test\sc2_unit_info` on your local computer.
4. After installation, please run `vlm_attention/knowledge_data/database/test_llm_use_database.py` for testing.

### 2024-10-25

Currently implemented: obtaining RGB images from the game engine, obtaining unit.tag from the game engine, supporting operations on multiple units of the same type (e.g., 3 stalkers).

### 2024-11-05

Updated new maps, please place the maps from pysc2/maps/VLM_ATTENTION in the SC2/MAPS folder. The new map is the vlm_attention version of 2cvs64zg.

```python
Currently available maps:
map_list = ["vlm_attention_1",
            "2c_vs_64zg_vlm_attention",
            "2m_vs_1z_vlm_attention",
            "2s_vs_1sc_vlm_attention",
            "2s3z_vlm_attention",
            "3m_vlm_attention",
            "3s_vs_3z_vlm_attention"]
```

Updated the model calling in vlm_attention/utils/call_vlm.py. Now the default is:

```python
class TextChatbot(BaseChatbot):
    def query(self, system_prompt, user_input, maintain_history=True):

class MultimodalChatbot(BaseChatbot):
    def query(self, system_prompt, user_input, image_path=None, maintain_history=True):
```

## Running Instructions

The project provides four main running scripts for different scenarios:

1. Single Player Without Ability:
```bash
python vlm_attention/run_env/multiprocess_run_env.py --agent [agent_name]
```
Available agents: RandomAgent, VLMAgentWithoutMove, TestAgent, VLMAgent

2. Two Players Without Ability:
```bash
python vlm_attention/run_env/multiprocess_run_env_two_players.py --agent1 [agent1_name] --agent2 [agent2_name]
```
Available agents: RandomAgent, VLMAgentWithoutMove, TestAgent, VLMAgent

3. Single Player With Ability:
```bash
python vlm_attention/run_env/multiprocess_run_env_with_ability.py --agent [agent_name]
```
Available agents: VLMAgentWithAbility, TestAgent_With_Ability

4. Two Players With Ability:
```bash
python vlm_attention/run_env/multi_process_run_env_two_players_with_ability.py --agent1 [agent1_name] --agent2 [agent2_name]
```
Available agents: VLMAgentWithoutMove, RandomAgent, TestAgent, VLMAgent

Common Parameters:
- `--map`: Select map
- `--num_processes`: Set number of parallel processes
- `--use_self_attention`: Whether to use self-attention mechanism
- `--use_rag`: Whether to use RAG
- `--use_proxy`: Whether to use proxy (required for accessing GPT models in mainland China)
- `--model_name`: Select model to use (e.g., "openai", "qwen", etc.)

Examples:
```bash
# Run single player without ability using VLMAgentWithoutMove
python vlm_attention/run_env/multiprocess_run_env.py --agent VLMAgentWithoutMove --map vlm_attention_1 --use_self_attention True --use_rag True

# Run two players with ability using different agents
python vlm_attention/run_env/multi_process_run_env_two_players_with_ability.py --agent1 TestAgent --agent2 VLMAgent --map ability_test_map
```

## Acknowledgments

We would like to extend our special thanks to:

[@LLM-PySC2](https://github.com/NKAI-Decision-Team/LLM-PySC2) - This project provided significant help in our environment design, including:
- Providing rich map resources
- Sharing knowledge and experience with PySC2 interfaces
- Offering reference for multi-agent environment design
