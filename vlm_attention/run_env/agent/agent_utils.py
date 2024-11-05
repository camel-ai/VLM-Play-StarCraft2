import re
import logging
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

"""

辅助函数和常量，用于处理VLM任务中的文本数据与格式化,标记任务
"""



def generate_unit_selection_prompt(available_units: List[str], important_units_response: str) -> str:
    return f"""
    Based on the image analysis and the available units in our database, select the most relevant units to query for more information. 
    Available units: {', '.join(available_units)}
    Image analysis result: {important_units_response}

    Please provide a list of unit names (comma-separated) that we should query from the database.
    """

def summarize_unit_info(unit_info: Dict[str, Dict]) -> str:
    summary_prompt = "Summarize the key information for the following StarCraft 2 units, including their strengths, weaknesses, and special abilities:\n\n"
    for unit, info in unit_info.items():
        summary_prompt += f"Unit: {unit}\n"
        summary_prompt += f"Description: {info.get('Description', 'N/A')}\n"
        summary_prompt += f"Strong against: {', '.join(info.get('Strong against', ['N/A']))}\n"
        summary_prompt += f"Weak against: {', '.join(info.get('Weak against', ['N/A']))}\n"
        summary_prompt += f"Special ability: {info.get('Ability', 'N/A')}\n\n"
    return summary_prompt

def generate_important_units_prompt() -> str:
    return """
    You are an AI assistant specialized in analyzing StarCraft II game states. Your task is to identify and explain 
    the most strategically important enemy units based on the provided game information and screenshot. 

    When analyzing, consider the following:
    1. Focus on units that significantly impact the battle due to their abilities or potential threat.
    2. Prioritize only those units that are key to the current game dynamics.
    3. Avoid selecting all enemy units; choose only the most crucial ones.

    For each important unit you identify, provide:
    - The unit's name
    - Its tag number
    - A detailed reason explaining its strategic importance in the current scenario

    Use the following format for each unit:

    ## Important Units ##
    Unit: [Unit Name]
    Tag: [Tag Number]
    Reason: [Explanation of why the unit is important]

    Repeat this format for each important unit. Do not include any other text or explanations outside of this format.
    """

def generate_decision_prompt() -> str:
    return """
    You are an AI assistant specialized in StarCraft II micro-management strategy. Your task is to suggest the best actions for our units based on the current game state, strategic principles, and unit information provided.

    Consider the following micro-management principles:
    1. Focus fire to quickly eliminate key enemy units.
    2. Prioritize attacking crucial enemy units that can turn the tide of battle.
    3. Maintain consistent targeting to avoid wasting attacks.
    4. Consider unit counter relationships when assigning targets.
    5. Each of our units must have exactly one attack action.
    6. Different friendly units can attack the same enemy unit if strategically beneficial.

    Provide attack actions for each of our units using the following format:

    ## Actions ##
    Attack: [Attacker Tag] -> [Target Tag]
    Reasoning: [Brief explanation of your decision, referencing the above principles, unit counter information, and provided unit summaries]

    Adhere to these guidelines:
    - Provide one attack action for each of our units.
    - Use the unit tags provided in the Units information section.
    - Each action should be on a new line.
    - Do not include any other text or explanations outside of this format.
    - Ensure that every friendly unit has exactly one attack action.
    """

def format_units_info(unit_info: List[Dict[str, Any]], predefined_tags: Dict[str, int], unit_align_dict: Dict[str, str]) -> str:
    our_units = []
    enemy_units = []

    for unit in unit_info:
        try:
            tag_index = unit['tag_index']
            alliance = unit['alliance']

            unit_type = next((key.split(',')[0] for key, value in predefined_tags.items() if value == tag_index), None)

            if unit_type:
                unit_name = unit_align_dict.get(unit_type, "Unknown")
                unit_str = f"{unit_name}: Tag {tag_index}"

                if alliance == 1:  # Self
                    our_units.append(unit_str)
                elif alliance == 4:  # Enemy
                    enemy_units.append(unit_str)
                else:
                    logger.warning(f"Unknown alliance for unit: {unit}")
            else:
                logger.warning(f"Unknown unit type for tag: {tag_index}")

        except Exception as e:
            logger.error(f"Error processing unit: {unit}. Error: {e}", exc_info=True)

    return f"Our units:\n" + "\n".join(our_units) + "\n\nEnemy units:\n" + "\n".join(enemy_units)

def parse_vlm_response(response: str) -> List[int]:
    important_units = []
    unit_blocks = re.findall(r'## Important Units ##(.*?)(?=## Important Units ##|\Z)', response, re.DOTALL)

    for block in unit_blocks:
        lines = block.strip().split('\n')
        unit_name = ""
        tag = None
        for line in lines:
            if line.startswith("Unit:"):
                unit_name = line.split(":")[1].strip()
            elif line.startswith("Tag:"):
                tag_str = line.split(":")[1].strip()
                try:
                    # 尝试从字符串中提取数字
                    tag_match = re.search(r'\d+', tag_str)
                    if tag_match:
                        tag = int(tag_match.group())
                        important_units.append(tag)
                        logger.info(f"Successfully parsed tag {tag} for unit {unit_name}")
                    else:
                        logger.warning(f"No valid tag found in string: '{tag_str}' for unit {unit_name}")
                except ValueError as e:
                    logger.warning(f"Failed to parse tag for unit {unit_name}. Tag string: '{tag_str}'. Error: {e}")

        if tag is None:
            logger.warning(f"No valid tag found for unit: {unit_name}")

    return important_units

def parse_vlm_decision(response: str) -> Dict[str, List[Tuple]]:
    attack_actions = []

    if "## Actions ##" not in response:
        logger.warning("VLM response does not contain the expected '## Actions ##' section")
        return {'attack': attack_actions, 'move': []}

    actions_section = response.split("## Actions ##")[1].strip()

    attack_pattern = re.compile(r'Attack:\s*(\d+)\s*->\s*(\d+)\s*\nReasoning:\s*(.*?)(?=Attack:|$)', re.DOTALL)
    matches = attack_pattern.findall(actions_section)

    for match in matches:
        attacker, target, reasoning = match
        try:
            attacker = int(attacker)
            target = int(target)
            attack_actions.append((attacker, target))
            logger.debug(f"Parsed attack action: {attacker} -> {target}")
            logger.debug(f"Reasoning: {reasoning.strip()}")
        except ValueError:
            logger.warning(f"Failed to parse attack action: {match}")

    return {
        'attack': attack_actions,
        'move': []  # Always return an empty list for move actions
    }

def format_history_for_prompt(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "No previous steps available."

    formatted_history = []
    for entry in history:
        step_info = f"Step {entry['step']}:\n"
        step_info += f"Important Units: {', '.join(map(str, entry['important_units']))}\n"
        step_info += "Attack Actions:\n"
        for attack in entry['attack_actions']:
            step_info += f"  Attack: {attack[0]} -> {attack[1]}\n"
        formatted_history.append(step_info)

    return "\n".join(formatted_history)


def generate_enhanced_unit_selection_prompt(available_units: List[str], important_units_response: str,
                                            text_observation: str) -> str:
    return f"""
    You are a StarCraft 2 strategist specializing in micro-management tasks for the Protoss race. 
    We are currently in a micro-management scenario where all units are visible (no fog of war).

    Based on the image analysis and the available units in our database, select the most relevant units to query for more information. 
    Focus on units that are crucial for micro-management decisions.

    Available units: {', '.join(available_units)}

    Image analysis result: {important_units_response}

    Current game state observation:
    {text_observation}

    Please provide a list of unit names (comma-separated) that we should query from the database to assist in micro-management decision-making.
    """


def generate_unit_info_summary_prompt(unit_info: Dict[str, Dict]) -> str:
    summary_prompt = """
    You are a StarCraft 2 expert focusing on Protoss micro-management. 
    Summarize the key information for the following units, emphasizing aspects crucial for micro-management decision-making.
    Keep the summary concise and directly applicable to our current micro-management task.

    For each unit, briefly highlight:
    1. Its primary role in combat
    2. Key strengths in micro-management scenarios
    3. Critical weaknesses to be aware of
    4. Any special abilities that could be game-changing in micro-management

    Units to summarize:
    """
    for unit, info in unit_info.items():
        summary_prompt += f"\n{unit}:\n"
        summary_prompt += f"Description: {info.get('Description', 'N/A')}\n"
        summary_prompt += f"Strong against: {', '.join(info.get('Strong against', ['N/A']))}\n"
        summary_prompt += f"Weak against: {', '.join(info.get('Weak against', ['N/A']))}\n"
        summary_prompt += f"Special ability: {info.get('Ability', 'N/A')}\n"

    return summary_prompt


def generate_action_normalization_prompt(text_observation: str, unit_info: str,
                                         raw_action: Dict[str, List[Tuple]]) -> str:
    return f"""
    As a StarCraft 2 expert, review the current game state and generate optimal attack actions for our units.

    Current game state:
    {text_observation}

    Unit information:
    {unit_info}

    Please provide attack actions for each of our units using the following format:
    ## Actions ##
    Attack: [Attacker Tag] -> [Target Tag]
    Reasoning: [Brief explanation of the strategic choice]

    Ensure that:
    1. All attacker tags correspond to our units.
    2. All target tags correspond to enemy units.
    3. Each of our units has exactly one attack action.
    4. The actions make strategic sense given the current game state.

    Example:
    ## Actions ##
    Attack: 1 -> 9
    Reasoning: The Stalker targets the Ghost to neutralize its high damage potential and disruptive abilities.

    Attack: 2 -> 14
    Reasoning: The Phoenix focuses on the damaged Banshee to quickly remove its air-to-ground threat.
    """