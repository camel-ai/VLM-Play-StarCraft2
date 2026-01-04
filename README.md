# StarCraft II Multimodal Agent Project

[English](READMEEN.md) | [中文](READMECN.md)


## Overview

This project develops a multimodal agent for StarCraft II that processes both visual information and text descriptions to make strategic decisions. The agent leverages vision-language models and reinforcement learning techniques to understand and interact with the game environment.

## Key Features

- 🎮 Custom environment based on OpenAI Gym
- 🖼️ Support for multimodal inputs (images and text)
- 🤖 Automatic unit annotation system
- 🎯 Multiple agent types (Random, VLM, Test agents)
- 🗺️ Rich map selection for different scenarios
- 🛠️ Support for both ability and non-ability based gameplay
- 🤝 Support for both single-player and two-player modes

## Quick Start

### Prerequisites

- Python 3.10
- Windows 11 (primary support)
- StarCraft II (Asia Battle.net version)

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install in editable mode:
```bash
pip install -e . --no-deps
```

3. Install dependencies:
```bash
pip install -r vlm_attention/requirements.txt
```

4. Fix potential protobuf conflicts:
```bash
pip uninstall protobuf
pip install protobuf==3.20.0
pip install pysc2 --no-deps
```

### Running the Project

1. Single Player (No Abilities):
```bash
python vlm_attention/run_env/multiprocess_run_env.py --agent VLMAgentWithoutMove --map vlm_attention_1
```

2. Two Players (No Abilities):
```bash
python vlm_attention/run_env/multiprocess_run_env_two_players.py --agent1 TestAgent --agent2 VLMAgent
```

3. Single Player (With Abilities):
```bash
python vlm_attention/run_env/multiprocess_run_env_with_ability.py --agent VLMAgentWithAbility
```

4. Two Players (With Abilities):
```bash
python vlm_attention/run_env/multi_process_run_env_two_players_with_ability.py --agent1 TestAgent --agent2 VLMAgent
```

## Available Maps

### Single Player Maps
- Basic maps (vlm_attention_1, 2m_vs_1z, etc.)
- Complex scenarios (2c_vs_64zg, 8marine_2tank_vs_zerglings_banelings, etc.)
- Mirror matches (2s3z, 3m)

### Two Player Maps
- Balanced matchups (MMM_vlm_attention_two_players)
- Asymmetric battles (vlm_attention_1_two_players)
- Mirror matches (vlm_attention_2_terran_vs_terran_two_players)

## Project Structure

```
vlm_attention/
├── env/                 # Environment implementation
├── run_env/            # Running scripts
├── knowledge_data/     # RAG and knowledge base
├── utils/              # Utility functions
└── requirements.txt    # Dependencies
```

## Configuration

Key configuration files:
- `vlm_attention/env/config.py`: Environment settings
- `vlm_attention/knowledge_data/database/config.yaml`: Database configuration
- `vlm_attention/utils/call_vlm.py`: VLM model settings

## Features Update

### 2024-11-05
- Added new maps including 2cvs64zg vlm_attention version
- Updated model calling interface in vlm_attention/utils/call_vlm.py
- Enhanced support for multiple units of same type operations

### 2024-10-25
- Implemented RGB image acquisition from game engine
- Added unit.tag retrieval functionality
- Improved multi-unit control support

## Acknowledgments

Special thanks to [@LLM-PySC2](https://github.com/NKAI-Decision-Team/LLM-PySC2) for providing valuable resources and references for our environment design.

## Documentation

For detailed documentation:
- [English Documentation](READMEEN.md)
- [中文文档](READMECN.md)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

```
Copyright 2024 CAMEL-AI.org

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. 
