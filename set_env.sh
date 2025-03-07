 #!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set PYTHONPATH to include the project root
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Set project specific environment variables
export VLM_ATTENTION_ROOT="${SCRIPT_DIR}"
export VLM_ATTENTION_DATA="${SCRIPT_DIR}/vlm_attention/knowledge_data"
export VLM_ATTENTION_MAPS="${SCRIPT_DIR}/pysc2/maps"

# Create necessary directories if they don't exist
mkdir -p "${VLM_ATTENTION_DATA}"
mkdir -p "${VLM_ATTENTION_MAPS}"

# Echo the current settings
echo "Environment variables have been set:"
echo "PYTHONPATH=${PYTHONPATH}"
echo "VLM_ATTENTION_ROOT=${VLM_ATTENTION_ROOT}"
echo "VLM_ATTENTION_DATA=${VLM_ATTENTION_DATA}"
echo "VLM_ATTENTION_MAPS=${VLM_ATTENTION_MAPS}"

# Instructions for use
echo
echo "To activate these settings, source this script:"
echo "source set_env.sh"
echo
echo "To run the environments, use commands like:"
echo "python vlm_attention/run_env/multiprocess_run_env.py"
echo "python vlm_attention/run_env/run_env_two_players.py"
echo "python vlm_attention/run_env/run_env_with_ability.py"
echo